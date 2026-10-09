// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Onnx;

/// <summary>How a window along a series, over an image or through a volume is worded in a refusal: along how many axes it walks, and what each is called.</summary>
/// <param name="Axes">How many axes the window walks: one, two or three.</param>
internal readonly record struct Walking(int Axes)
{
    /// <summary>Where the window walks: <c>along a series</c>, <c>over an image</c> or <c>through a volume</c>.</summary>
    public string Over => Axes switch { 1 => "along a series", 2 => "over an image", _ => "through a volume" };

    /// <summary>How many numbers ONNX writes for a setting of the window, one for each axis.</summary>
    public string Writes => Axes switch { 1 => "one", 2 => "a pair", _ => "three" };

    /// <summary>What the pads of the window run along, before and after.</summary>
    public string Places => Axes switch { 1 => "steps", 2 => "rows and columns", _ => "planes, rows and columns" };

    /// <summary>What the window walks, in the plural.</summary>
    public string Things => Axes switch { 1 => "series", 2 => "images", _ => "volumes" };

    /// <summary>The words for the axes of an image and of a volume, in the order ONNX writes them; a series has the one axis and needs none.</summary>
    private string[] Directions => Axes == 2 ? ["down", "across"] : ["deep", "down", "across"];

    /// <summary>A setting said axis by axis: <c>2 down and 1 across</c>; the one number of a series said alone.</summary>
    /// <param name="lengths">The setting, one number for each axis.</param>
    public string Said(int[] lengths) =>
        Axes == 1
            ? lengths[0].ToString(CultureInfo.InvariantCulture)
            : Listed(lengths.Zip(Directions, (length, direction) => string.Create(CultureInfo.InvariantCulture, $"{length} {direction}")));

    /// <summary>The axes said all together: <c>down and across</c>. Of a window along more than one.</summary>
    public string Alike => Listed(Directions);

    private static string Listed(IEnumerable<string> items)
    {
        var all = items.ToArray();

        return $"{string.Join(", ", all[..^1])} and {all[^1]}";
    }
}

/// <summary>
/// A window as an ONNX node states it: its length along each axis, one stride on every axis, and the border it pads with —
/// and, written as words of a description, the layer it makes along one, two or three axes.
/// </summary>
/// <param name="Sizes">The window's length along each axis.</param>
/// <param name="Stride">How far it moves between one place and the next, on every axis.</param>
/// <param name="Mode">How its border is had.</param>
/// <param name="Padding">The places of nothing on every side, read when the border is stated.</param>
internal sealed record WindowPlan(int[] Sizes, int Stride, PaddingMode Mode, int Padding)
{
    private Window1D Line => new(Sizes[0]) { Stride = Stride, PaddingMode = Mode, Padding = Padding };

    private Window Plane => new(Sizes[0], Sizes[1]) { Stride = Stride, PaddingMode = Mode, Padding = Padding };

    private Window3D Block => new(Sizes[0], Sizes[1], Sizes[2]) { Stride = Stride, PaddingMode = Mode, Padding = Padding };

    /// <summary>A convolution along its axes, making so many channels.</summary>
    /// <param name="filters">How many channels it makes.</param>
    public Action<Sequential> Convolve(int filters) => Sizes.Length switch
    {
        1 => description => description.Conv1D(filters, Line),
        2 => description => description.Conv2D(filters, Plane),
        _ => description => description.Conv3D(filters, Block),
    };

    /// <summary>The largest value the window covers.</summary>
    public Action<Sequential> Largest() => Sizes.Length switch
    {
        1 => description => description.MaxPool1D(Line),
        2 => description => description.MaxPool2D(Plane),
        _ => description => description.MaxPool3D(Block),
    };

    /// <summary>The average of the values the window covers.</summary>
    /// <param name="countsPadding">Whether the border it pads with counts as values of nought in each average.</param>
    public Action<Sequential> Average(bool countsPadding) => Sizes.Length switch
    {
        1 => description => description.AvgPool1D(Line, countsPadding),
        2 => description => description.AvgPool2D(Plane, countsPadding),
        _ => description => description.AvgPool3D(Block, countsPadding),
    };
}

/// <summary>
/// The pads a node writes out one side at a time — before each axis, then after each — or says as SAME_LOWER, to be read as
/// TensorFlow's 'same': when they are the border it gives the images that reach the node.
/// </summary>
/// <param name="Sizes">The window's length along each axis.</param>
/// <param name="Stride">How far the window moves between one place and the next.</param>
/// <param name="Pads">The pads as ONNX writes them: the places before each axis, then the places after each; none for SAME_LOWER.</param>
internal sealed record DeclaredPads(int[] Sizes, int Stride, int[] Pads)
{
    /// <summary>Whether the node says SAME_LOWER, which pads as 'same' does with the odd place before the axis instead of after it.</summary>
    public bool Lower { get; init; }

    private int Axes => Sizes.Length;

    /// <summary>
    /// Whether they can be the border 'same' gives some images: along no axis does it pad fewer places after than before, or
    /// more than one place more.
    /// </summary>
    public bool Possible => Enumerable.Range(0, Axes).All(axis => Pads[Axes + axis] - Pads[axis] is 0 or 1);

    /// <summary>Whether they are the border 'same' gives images so long along each axis.</summary>
    /// <param name="lengths">How long the images are along each axis.</param>
    public bool SameOver(int[] lengths) => Written(lengths).SequenceEqual(Same(lengths));

    /// <summary>What is wrong with them for the examples that reach the layer once the network is lowered; nothing when they are the border 'same' gives them.</summary>
    /// <param name="reaching">The shape of what reaches the layer: the batch first, the channels last.</param>
    public string? Mismatch(Shape reaching)
    {
        var lengths = reaching.Axes[1..^1].ToArray();

        if (SameOver(lengths))
        {
            return null;
        }

        var words = new Walking(Axes);
        var pads = string.Join(", ", Written(lengths));
        var says = Lower ? $"it pads as SAME_LOWER, [{pads}]" : $"it pads [{pads}]";

        return $"{says} — {words.Places} before, then after — where TensorFlow's 'same' pads the {string.Join('x', lengths)} {words.Things} reaching it [{string.Join(", ", Same(lengths))}], and a window here pads every side alike, or as that 'same'.";
    }

    // The pads the node writes, or SAME_LOWER's for images so long: the places 'same' pads after each axis come before it.
    private int[] Written(int[] lengths)
    {
        if (!Lower)
        {
            return Pads;
        }

        var same = Same(lengths);

        return [.. same.Skip(Axes), .. same.Take(Axes)];
    }

    // TensorFlow's 'same' for images so long, as ONNX writes pads: the places before each axis, then the places after each.
    private int[] Same(int[] lengths)
    {
        var borders = lengths.Select((length, axis) => new Window(1, Sizes[axis]) { Stride = Stride, PaddingMode = PaddingMode.Same }.BordersOver(1, length)).ToArray();

        return [.. borders.Select(border => border.Left), .. borders.Select(border => border.Right)];
    }
}

/// <summary>A window as a node states it, and the pads it writes out as TensorFlow's 'same' when it does.</summary>
/// <param name="Plan">The window.</param>
/// <param name="Declared">The pads written out one side at a time, to be held to the images reaching the layer; nothing when the border is stated alike on every side, or declared.</param>
internal readonly record struct StatedWindow(WindowPlan Plan, DeclaredPads? Declared);

/// <summary>
/// An ONNX operator that walks a window along a series, over an image or through a volume: its lengths, its stride, its
/// dilation and its border, read for as many axes as the window has and refused at the node where a window here cannot be
/// as the node says.
/// </summary>
internal abstract class WindowKind : OnnxKind
{
    private const string Pads = "and a window here pads every side alike, or as SAME_UPPER, TensorFlow's 'same'.";

    /// <summary>The window a node states with the lengths given, each setting of it that no window here can be refused at the node.</summary>
    /// <param name="node">The node.</param>
    /// <param name="sizes">The window's length along each axis.</param>
    /// <param name="unit">What the node is, as a refusal says it: <c>convolution</c> or <c>pooling</c>.</param>
    protected static StatedWindow Walk(OnnxNode node, int[] sizes, string unit)
    {
        var words = new Walking(sizes.Length);
        var strides = Lengths(node, "strides", words, unit);
        var dilations = Lengths(node, "dilations", words, unit);
        var padded = Padding(node, new WindowPlan(sizes, strides[0], PaddingMode.Stated, 0), words);

        if (strides.Distinct().Count() > 1)
        {
            node.Refuse($"it strides {words.Said(strides)}, and a window here walks one stride {words.Alike} alike.");
        }

        if (dilations.Any(length => length != 1))
        {
            node.Refuse($"its window is dilated by {words.Said(dilations)}, and a window here covers neighbouring places.");
        }

        return new StatedWindow(padded.Plan, padded.Declared);
    }

    // A setting ONNX writes for each axis of the window, one each when it is left out.
    private static int[] Lengths(OnnxNode node, string name, Walking words, string unit)
    {
        var ones = Enumerable.Repeat(1, words.Axes).ToArray();
        var lengths = node.Wholes(name, ones);

        if (lengths.Length == words.Axes)
        {
            return lengths;
        }

        node.Refuse($"its '{name}' is written as [{string.Join(", ", lengths)}], and a {unit} {words.Over} writes {words.Writes} there.");

        return ones;
    }

    // The border the node declares: SAME_UPPER's, none for VALID, the one its pads state for every side alike, or TensorFlow's
    // 'same' written out one side at a time, as tf2onnx writes it. At a stride of one 'same' pads every image alike — half
    // the window's reach before, the odd place after — so pads written otherwise are refused here; at a longer stride the
    // border 'same' gives depends on the image, and the pads are held to it once the network is lowered.
    private static StatedWindow Padding(OnnxNode node, WindowPlan plan, Walking words)
    {
        var declared = node.Text("auto_pad", "NOTSET");

        if (declared == "SAME_UPPER")
        {
            return new StatedWindow(plan with { Mode = PaddingMode.Same }, null);
        }

        if (declared == "VALID")
        {
            return new StatedWindow(plan, null);
        }

        if (declared == "SAME_LOWER")
        {
            var lower = new DeclaredPads(plan.Sizes, plan.Stride, []) { Lower = true };

            if (plan.Stride == 1 && !lower.SameOver(plan.Sizes))
            {
                node.Refuse($"it pads as {declared.Quoted()}, {Pads}");

                return new StatedWindow(plan, null);
            }

            return new StatedWindow(plan with { Mode = PaddingMode.Same }, lower);
        }

        if (declared != "NOTSET")
        {
            node.Refuse($"it pads as {declared.Quoted()}, {Pads}");

            return new StatedWindow(plan, null);
        }

        var pads = node.Wholes("pads", new int[2 * words.Axes]);

        if (pads.Length == 2 * words.Axes)
        {
            if (pads.Distinct().Count() == 1)
            {
                return new StatedWindow(plan with { Padding = pads[0] }, null);
            }

            var written = new DeclaredPads(plan.Sizes, plan.Stride, pads);

            if (plan.Stride == 1 ? written.SameOver(plan.Sizes) : written.Possible)
            {
                return new StatedWindow(plan with { Mode = PaddingMode.Same }, written);
            }
        }

        node.Refuse($"it pads [{string.Join(", ", pads)}] — {words.Places} before, then after — {Pads}");

        return new StatedWindow(plan, null);
    }
}

/// <summary>
/// ONNX's <c>Conv</c> along a series, over an image or through a volume: a window walking one stride on every axis, padded
/// alike on every side or as TensorFlow's 'same' — SAME_UPPER, or its pads written out — its kernel channels out by channels
/// in by the window's steps, or its rows and columns, or its planes, rows and columns.
/// </summary>
internal sealed class ConvKind : WindowKind
{
    /// <inheritdoc />
    public override string Name => "Conv";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!TakesImages(node, reaching))
        {
            return new OnnxLayer(null, Flow.Images);
        }

        var kernel = numbers.Floats(node.Input(1));

        if (kernel.Lengths.Length is < 3 or > 5)
        {
            node.Refuse(
                $"its kernel, '{node.Input(1).Quoted()}', is written as {Written(kernel.Lengths)}, and a convolution here slides a window along a series, an image or a volume: channels out by channels in by steps, or by rows by columns, or by planes by rows by columns.");

            return new OnnxLayer(null, Flow.Images);
        }

        var window = Walk(node, kernel.Lengths[2..], "convolution");
        var groups = node.Whole("group", 1);

        if (groups != 1)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it splits its channels into {groups} groups, and a convolution here takes every channel of a place at once."));
        }

        if (node.Input(2).Length == 0)
        {
            node.Refuse(NoBias);
        }

        return new OnnxLayer(window.Plan.Convolve(kernel.Lengths[0]), Flow.Images)
        {
            Numbers = [new LayerNumber("weight", kernel, Laying.Kernel), new LayerNumber("bias", numbers.Floats(node.Input(2)), Laying.AsWritten)],
            Declared = window.Declared,
        };
    }
}

/// <summary>
/// An ONNX pooling along a series, over an image or through a volume: its window the <c>kernel_shape</c> it states, the same
/// stride on every axis, padded alike on every side or as TensorFlow's 'same', rounding the places it stands at down.
/// </summary>
internal abstract class PoolKind : WindowKind
{
    /// <inheritdoc />
    public sealed override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!TakesImages(node, reaching))
        {
            return new OnnxLayer(null, Flow.Images);
        }

        var sizes = node.Wholes("kernel_shape", []);

        if (sizes.Length is < 1 or > 3)
        {
            node.Refuse($"its 'kernel_shape' is written as [{string.Join(", ", sizes)}], and a pooling here slides a window along one, two or three axes.");

            return new OnnxLayer(null, Flow.Images);
        }

        var window = Walk(node, sizes, "pooling");
        var rounding = node.Whole("ceil_mode", 0);

        if (rounding != 0)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it lets its window overhang the end of an axis, ceil_mode {rounding}, and a window here stands only where it fits whole."));
        }

        return new OnnxLayer(Pooling(node, window.Plan), Flow.Images) { Declared = window.Declared };
    }

    /// <summary>The word of the pooling for the window it states, each setting of it that is the pooling's own refused at the node.</summary>
    /// <param name="node">The node.</param>
    /// <param name="plan">The window.</param>
    protected abstract Action<Sequential> Pooling(OnnxNode node, WindowPlan plan);
}

/// <summary>ONNX's <c>MaxPool</c>: the largest value its window covers, the places of those values given by no second output.</summary>
internal sealed class MaxPoolKind : PoolKind
{
    /// <inheritdoc />
    public override string Name => "MaxPool";

    /// <inheritdoc />
    protected override Action<Sequential> Pooling(OnnxNode node, WindowPlan plan)
    {
        var order = node.Whole("storage_order", 0);

        if (order != 0)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it numbers the places of its largest values column by column, storage_order {order}, and a max pooling here gives the largest values alone."));
        }

        if (node.Taken(1) is { } places)
        {
            node.Refuse($"it gives the places of its largest values as a second value, '{places.Quoted()}', which the graph uses, and a max pooling here gives the largest values alone.");
        }

        return plan.Largest();
    }
}

/// <summary>ONNX's <c>AveragePool</c>: the average of the values its window covers, its padding left out of the average unless the node counts it.</summary>
internal sealed class AveragePoolKind : PoolKind
{
    /// <inheritdoc />
    public override string Name => "AveragePool";

    /// <inheritdoc />
    protected override Action<Sequential> Pooling(OnnxNode node, WindowPlan plan) => plan.Average(node.Whole("count_include_pad", 0) != 0);
}

/// <summary>
/// An ONNX pooling of every place of each channel into one — a series, an image or a volume the graph keeps as one place
/// of each channel, as ONNX's global poolings do.
/// </summary>
internal abstract class GlobalPoolKind : OnnxKind
{
    /// <inheritdoc />
    public sealed override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers) =>
        new(TakesImages(node, reaching) ? Pool(reaching.Axes, keepsAxes: true) : null, Flow.Images);

    /// <summary>The word of the pooling, for a series, an image or a volume.</summary>
    /// <param name="axes">How many axes the examples that reach it have beside their channels: one, two or three.</param>
    /// <param name="keepsAxes">Whether the axes it pools are kept, each as an axis of one place.</param>
    internal abstract Action<Sequential> Pool(int axes, bool keepsAxes);
}

/// <summary>ONNX's <c>GlobalAveragePool</c>: the average of every channel, its axes kept as axes of one place.</summary>
internal sealed class GlobalAveragePoolKind : GlobalPoolKind
{
    /// <inheritdoc />
    public override string Name => "GlobalAveragePool";

    /// <inheritdoc />
    internal override Action<Sequential> Pool(int axes, bool keepsAxes) => axes switch
    {
        1 => description => description.GlobalAvgPool1D(keepsAxes),
        2 => description => description.GlobalAvgPool2D(keepsAxes),
        _ => description => description.GlobalAvgPool3D(keepsAxes),
    };
}

/// <summary>ONNX's <c>GlobalMaxPool</c>: the largest value of every channel, its axes kept as axes of one place.</summary>
internal sealed class GlobalMaxPoolKind : GlobalPoolKind
{
    /// <inheritdoc />
    public override string Name => "GlobalMaxPool";

    /// <inheritdoc />
    internal override Action<Sequential> Pool(int axes, bool keepsAxes) => axes switch
    {
        1 => description => description.GlobalMaxPool1D(keepsAxes),
        2 => description => description.GlobalMaxPool2D(keepsAxes),
        _ => description => description.GlobalMaxPool3D(keepsAxes),
    };
}
