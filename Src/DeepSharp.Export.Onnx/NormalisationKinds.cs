// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A layer normalisation of a row over its last axis: ONNX's LayerNormalization, which first appeared in the seventeenth operator set,
/// with the scale it keeps — ones where it keeps none, since the operator takes one — and the shift it keeps, left out where it keeps none.
/// </summary>
internal sealed class LayerNormKind : ExportKind<LayerNorm>
{
    /// <inheritdoc />
    protected override Value Write(LayerNorm layer, Writing at)
    {
        RequireRow(at, "A layer normalisation");

        var scale = Scale(at, layer);
        string[] inputs = layer.Bias is { } shift
            ? [at.Input.Name, scale, at.Graph.Number(at.Number(shift.Name), [layer.Features], at.Input.Ordered(shift.Value.Values))]
            : [at.Input.Name, scale];

        return at.Keeping(at.Graph.Node(
            at.Node("LayerNormalization", 17) with
            {
                Attributes = node =>
                {
                    node.State("axis", -1L);
                    node.State("epsilon", layer.Epsilon);
                },
            },
            inputs,
            at.Path));
    }

    // The scale it keeps, or ones where it keeps none, in the order the graph's row has them.
    internal static string Scale(Writing at, Normalisation layer) =>
        layer.Weight is { } weight
            ? at.Graph.Number(at.Number(weight.Name), [layer.Features], at.Input.Ordered(weight.Value.Values))
            : at.Graph.Number(at.Number("weight"), [layer.Features], [.. Enumerable.Repeat(1f, layer.Features)]);

    // A normalisation over the last axis of a row: ONNX's operators for it normalise from an axis to the end, which over a series, an
    // image or a volume with its channels first would take the places in with the channels.
    internal static void RequireRow(Writing at, string kind)
    {
        if (at.Input.Spatial)
        {
            throw new NotSupportedException(
                $"{kind} normalises the channels of every place of a series, an image or a volume, and ONNX's operator for it normalises every axis from the one it names to the last, which would take the places in with the channels.");
        }
    }
}

/// <summary>A root-mean-square normalisation of a row: ONNX's RMSNormalization, which first appeared in the twenty-third operator set, with its scale.</summary>
internal sealed class RmsNormKind : ExportKind<RmsNorm>
{
    /// <inheritdoc />
    protected override Value Write(RmsNorm layer, Writing at)
    {
        LayerNormKind.RequireRow(at, "A root-mean-square normalisation");

        return at.Keeping(at.Graph.Node(
            at.Node("RMSNormalization", 23) with
            {
                Attributes = node =>
                {
                    node.State("axis", -1L);
                    node.State("epsilon", layer.Epsilon);
                },
            },
            [at.Input.Name, LayerNormKind.Scale(at, layer)],
            at.Path));
    }
}

/// <summary>
/// A batch normalisation answering from its running statistics: ONNX's BatchNormalization with the scale and the shift it keeps — ones and
/// noughts where it keeps none — its running mean and variance, its epsilon and its momentum.
/// </summary>
internal sealed class BatchNormKind : ExportKind<BatchNorm>
{
    /// <inheritdoc />
    protected override Value Write(BatchNorm layer, Writing at)
    {
        var graph = at.Graph;
        var scale = LayerNormKind.Scale(at, layer);
        var shift = layer.Bias is { } kept
            ? graph.Number(at.Number(kept.Name), [layer.Features], at.Input.Ordered(kept.Value.Values))
            : graph.Number(at.Number("bias"), [layer.Features], new float[layer.Features]);
        var mean = graph.Number(at.Number(layer.RunningMean.Name), [layer.Features], at.Input.Ordered(layer.RunningMean.Value.Values));
        var variance = graph.Number(at.Number(layer.RunningVariance.Name), [layer.Features], at.Input.Ordered(layer.RunningVariance.Value.Values));

        return at.Keeping(graph.Node(
            at.Node("BatchNormalization", 9) with
            {
                Attributes = node =>
                {
                    node.State("epsilon", layer.Epsilon);
                    node.State("momentum", layer.Momentum);
                },
            },
            [at.Input.Name, scale, shift, mean, variance],
            at.Path));
    }
}
