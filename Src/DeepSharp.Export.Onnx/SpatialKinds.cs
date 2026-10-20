// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A convolution along a series, over an image or through a volume: ONNX's Conv with its kernel written channels out by channels in by the
/// window's places, as ONNX writes it, the window's lengths, stride, dilation and border, and its channels in groups.
/// </summary>
/// <typeparam name="TLayer">The convolution's class.</typeparam>
/// <param name="stand">The window the convolution walks, for examples of the lengths given.</param>
internal sealed class ConvolutionKind<TLayer>(Func<TLayer, int[], Standing> stand) : ExportKind<TLayer>
    where TLayer : Convolution
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var window = stand(layer, Spatial.Extents(at));
        var places = window.Sizes.Aggregate(1, (all, length) => all * length);
        var kernel = at.Graph.Number(
            at.Number(layer.Weight.Name),
            [layer.OutChannels, layer.InChannels / layer.Groups, .. window.Sizes],
            OnnxLayouts.SlotToKernel(layer.Weight.Value.Values.ToArray(), layer.OutChannels, layer.InChannels / layer.Groups, places));

        return at.Giving(at.Graph.Node(
            at.Node("Conv", 1) with
            {
                Attributes = node =>
                {
                    window.State(node);
                    node.State("group", (long)layer.Groups);
                },
            },
            Spatial.WithBias(at, layer, kernel),
            at.Path));
    }
}

/// <summary>
/// A convolution that gives the places back: ONNX's ConvTranspose with its kernel written channels in by the channels out of one group by
/// the window's places, the places taken off each end, and the places kept at the end of each axis as <c>output_padding</c>.
/// </summary>
/// <typeparam name="TLayer">The convolution's class.</typeparam>
/// <param name="stand">The window it walks, with the border it states.</param>
internal sealed class TransposedConvolutionKind<TLayer>(Func<TLayer, Standing> stand) : ExportKind<TLayer>
    where TLayer : TransposedConvolution
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var window = stand(layer);
        var places = window.Sizes.Aggregate(1, (all, length) => all * length);
        var outputs = layer.OutChannels / layer.Groups;
        var kernel = at.Graph.Number(
            at.Number(layer.Weight.Name),
            [layer.InChannels, outputs, .. window.Sizes],
            OnnxLayouts.SlotToGivenBack(layer.Weight.Value.Values.ToArray(), layer.InChannels, outputs, places));

        return at.Giving(at.Graph.Node(
            at.Node("ConvTranspose", 1) with
            {
                Attributes = node =>
                {
                    window.State(node);
                    node.State("group", (long)layer.Groups);
                    node.State("output_padding", layer.OutputPadding.Select(each => (long)each).ToArray());
                },
            },
            Spatial.WithBias(at, layer, kernel),
            at.Path));
    }
}

/// <summary>
/// A pooling of the values a window covers: ONNX's MaxPool or AveragePool with the window's lengths, stride, dilation and border, rounding
/// the places it stands at up where the window does (ceil_mode, from the tenth operator set), and, for an average, whether the border is
/// counted in each.
/// </summary>
/// <typeparam name="TLayer">The pooling's class.</typeparam>
/// <param name="op">ONNX's name for the operator.</param>
/// <param name="stand">The window it walks, for examples of the lengths given.</param>
internal sealed class PoolingKind<TLayer>(string op, Func<TLayer, int[], Standing> stand) : ExportKind<TLayer>
    where TLayer : Pooling
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var window = stand(layer, Spatial.Extents(at));
        var average = layer is AveragePooling;
        var since = Math.Max(window.RoundsUp ? 10 : 1, window.Dilated ? (average ? 19 : 10) : 1);

        return at.Giving(at.Graph.Node(
            at.Node(op, Math.Max(since, average ? 7 : 1)) with
            {
                Attributes = node =>
                {
                    window.State(node);

                    if (window.RoundsUp)
                    {
                        node.State("ceil_mode", 1L);
                    }

                    if (layer is AveragePooling { CountsPadding: true })
                    {
                        node.State("count_include_pad", 1L);
                    }
                },
            },
            [at.Input.Name],
            at.Path));
    }
}

/// <summary>
/// A pooling of every place of each channel into one: ONNX's GlobalAveragePool or GlobalMaxPool where the axes it pools are kept as axes of
/// one place, and a ReduceMean or ReduceMax over those axes, dropping them, where they are not.
/// </summary>
/// <typeparam name="TLayer">The pooling's class.</typeparam>
/// <param name="kept">ONNX's operator that keeps the axes.</param>
/// <param name="reduced">ONNX's operator that reduces over them.</param>
internal sealed class GlobalPoolingKind<TLayer>(string kept, string reduced) : ExportKind<TLayer>
    where TLayer : GlobalPooling
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var spatial = Spatial.Extents(at);

        if (layer.KeepsAxes)
        {
            return at.Giving(at.Graph.Node(at.Node(kept, 1), [at.Input.Name], at.Path));
        }

        // The channels come first and then the places the pooling pools, which are the axes after them.
        long[] axes = [.. Enumerable.Range(2, spatial.Length).Select(axis => (long)axis)];

        return at.Giving(at.Graph.Node(
            at.Node(reduced, 1) with
            {
                Attributes = node =>
                {
                    node.State("axes", axes);
                    node.State("keepdims", 0L);
                },
            },
            [at.Input.Name],
            at.Path));
    }
}

/// <summary>
/// Places added around each example: ONNX's Pad, from the eleventh operator set where its pads and its value are inputs, one place before
/// each axis and then one after each, holding a value (constant) or the example's own values mirrored back (reflect), repeated from the
/// edge (edge) or gone round (wrap, from the nineteenth set).
/// </summary>
/// <typeparam name="TLayer">The padding's class.</typeparam>
/// <param name="border">The places it adds before each axis, then after each.</param>
internal sealed class PaddingKind<TLayer>(Func<TLayer, int[]> border) : ExportKind<TLayer>
    where TLayer : Padding
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var spatial = Spatial.Extents(at);
        var sides = border(layer);
        var pads = at.Graph.Wholes(at.Number("pads"), [0, 0, .. sides[..spatial.Length].Select(each => (long)each), 0, 0, .. sides[spatial.Length..].Select(each => (long)each)]);
        var mode = layer.Filling switch { Filling.Mirror => "reflect", Filling.Repeat => "edge", Filling.Wrap => "wrap", _ => "constant" };
        string[] inputs = layer.Filling == Filling.Constant
            ? [at.Input.Name, pads, at.Graph.Number(at.Number("value"), [], [layer.Value])]
            : [at.Input.Name, pads];

        return at.Giving(at.Graph.Node(
            at.Node("Pad", layer.Filling == Filling.Wrap ? 19 : 11) with { Attributes = node => node.State("mode", mode) },
            inputs,
            at.Path));
    }
}

/// <summary>
/// An instance normalisation of a series, an image or a volume: ONNX's InstanceNormalization with the scale and the shift of each channel
/// — ones and noughts where it keeps none, since the operator takes both — and its epsilon.
/// </summary>
internal sealed class InstanceNormKind : ExportKind<InstanceNorm>
{
    /// <inheritdoc />
    protected override Value Write(InstanceNorm layer, Writing at)
    {
        return at.Giving(at.Graph.Node(
            at.Node("InstanceNormalization", 6) with { Attributes = node => node.State("epsilon", layer.Epsilon) },
            [at.Input.Name, LayerNormKind.Scale(at, layer), Spatial.Shift(at, layer)],
            at.Path));
    }
}

/// <summary>
/// A group normalisation of the channels of a row, a series, an image or a volume: ONNX's GroupNormalization as the twenty-first operator set
/// defines it, a scale and a shift for each channel, with the groups and the epsilon.
/// </summary>
internal sealed class GroupNormKind : ExportKind<GroupNorm>
{
    /// <inheritdoc />
    protected override Value Write(GroupNorm layer, Writing at)
    {
        if (at.Input.Image is not null)
        {
            throw new NotSupportedException("A group normalisation of a row a flatten made of an image: the channels the groups are made of are no longer an axis of the row.");
        }

        return at.Giving(at.Graph.Node(
            at.Node("GroupNormalization", 21) with
            {
                Attributes = node =>
                {
                    node.State("num_groups", (long)layer.Groups);
                    node.State("epsilon", layer.Epsilon);
                },
            },
            [at.Input.Name, LayerNormKind.Scale(at, layer), Spatial.Shift(at, layer)],
            at.Path));
    }
}

/// <summary>What the spatial kinds share: the lengths a series, an image or a volume has beside its channels, and the bias of a layer that walks one.</summary>
internal static class Spatial
{
    /// <summary>The lengths of the places an example holds beside its channels: a layer that walks them is handed a series, an image or a volume.</summary>
    /// <param name="at">The step.</param>
    public static int[] Extents(Writing at) => [.. at.Input.Each.Axes[..^1].ToArray()];

    /// <summary>The inputs of a node that takes a kernel and, where the layer has one, a bias.</summary>
    /// <param name="at">The step.</param>
    /// <param name="layer">The layer.</param>
    /// <param name="kernel">The kernel's name in the graph.</param>
    public static string[] WithBias(Writing at, Convolution layer, string kernel) =>
        layer.Bias is { } bias
            ? [at.Input.Name, kernel, at.Graph.Number(at.Number(bias.Name), [layer.OutChannels], bias.Value.Values)]
            : [at.Input.Name, kernel];

    /// <summary>The shift of a normalisation of channels, or noughts where it keeps none.</summary>
    /// <param name="at">The step.</param>
    /// <param name="layer">The normalisation.</param>
    public static string Shift(Writing at, Normalisation layer) =>
        layer.Bias is { } bias
            ? at.Graph.Number(at.Number(bias.Name), [layer.Features], bias.Value.Values)
            : at.Graph.Number(at.Number("bias"), [layer.Features], new float[layer.Features]);
}

/// <summary>Each example laid out as one row: ONNX's Flatten from the axis after the batch.</summary>
internal sealed class FlattenKind : ExportKind<Flatten>
{
    /// <inheritdoc />
    protected override Value Write(Flatten layer, Writing at) =>
        at.Input.Spatial
            ? at.Giving(at.Graph.Node(at.Node("Flatten", 1) with { Attributes = node => node.State("axis", 1L) }, [at.Input.Name], at.Path)) with { Image = Flattened.Of(at.Input.Each) }
            : at.Input;
}

/// <summary>Each example laid out as one row of the values it has: ONNX's Reshape into the batch's length and the row's.</summary>
internal sealed class ReshapeKind : ExportKind<Reshape>
{
    /// <inheritdoc />
    protected override Value Write(Reshape layer, Writing at)
    {
        if (layer.Each.Rank != 1)
        {
            throw new NotSupportedException(
                $"A reshape into {layer.Each} lays each example out as more than one row of numbers, and the graph holds a series, an image or a volume channels first where the network holds it channels last: only a reshape into one row is written.");
        }

        if (!at.Input.Spatial)
        {
            return at.Input;
        }

        var target = at.Graph.Wholes(at.Number("shape"), 0, layer.Each[0]);

        return at.Giving(at.Graph.Node(at.Node("Reshape", 5), [at.Input.Name, target], at.Path)) with { Image = Flattened.Of(at.Input.Each) };
    }
}
