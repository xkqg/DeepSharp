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
}

/// <summary>
/// A window as an ONNX node states it: its length along each axis, one stride on every axis, and the border it pads with —
/// and, written as words of a description, the layer it makes along one, two or three axes.
/// </summary>
/// <param name="Sizes">The window's length along each axis.</param>
/// <param name="Stride">How far it moves between one place and the next, along each axis.</param>
/// <param name="Dilation">How far apart its cells are, along each axis.</param>
/// <param name="Mode">How its border is had.</param>
/// <param name="Padding">The places of nothing on every side, read when the border is stated alike.</param>
internal sealed record WindowPlan(int[] Sizes, int[] Stride, int[] Dilation, PaddingMode Mode, int Padding)
{
    /// <summary>Whether the window also stands at a last place whose patch overhangs the end: ONNX's <c>ceil_mode</c>.</summary>
    public bool RoundsUp { get; init; }

    /// <summary>The places of nothing at each end of each axis, as ONNX writes its pads: before each axis, then after each; nothing where the border is alike or worked out.</summary>
    public int[]? Pads { get; init; }

    /// <summary>How many groups the channels are split into: ONNX's <c>group</c>.</summary>
    public int Groups { get; init; } = 1;

    /// <summary>How many places of nothing are kept at the end of each axis: ONNX's <c>output_padding</c>, none where it says none.</summary>
    public int[] Grown { get; init; } = [];

    /// <summary>How many axes the window walks.</summary>
    public int Axes => Sizes.Length;

    private Sides Along(int axis) => Pads is { } pads ? new Sides(pads[axis], pads[Axes + axis]) : new Sides(Padding, Padding);

    private Window1D Line => new(Sizes[0])
    {
        Stride = Stride[0],
        Dilation = Dilation[0],
        RoundsUp = RoundsUp,
        PaddingMode = Pads is null ? Mode : PaddingMode.Stated,
        Padding = Pads is null ? Padding : 0,
        Border = Pads is null ? null : Along(0),
    };

    private Window Plane => new(Sizes[0], Sizes[1])
    {
        Stride = new Steps(Stride[0], Stride[1]),
        Dilation = new Steps(Dilation[0], Dilation[1]),
        RoundsUp = RoundsUp,
        PaddingMode = Pads is null ? Mode : PaddingMode.Stated,
        Padding = Pads is null ? Padding : 0,
        Border = Pads is null ? null : new Margins(Along(0), Along(1)),
    };

    private Window3D Block => new(Sizes[0], Sizes[1], Sizes[2])
    {
        Stride = new Steps3D(Stride[0], Stride[1], Stride[2]),
        Dilation = new Steps3D(Dilation[0], Dilation[1], Dilation[2]),
        RoundsUp = RoundsUp,
        PaddingMode = Pads is null ? Mode : PaddingMode.Stated,
        Padding = Pads is null ? Padding : 0,
        Border = Pads is null ? null : new Margins3D(Along(0), Along(1), Along(2)),
    };

    /// <summary>A convolution along its axes, making so many channels.</summary>
    /// <param name="filters">How many channels it makes.</param>
    public Action<Sequential> Convolve(int filters) => Sizes.Length switch
    {
        1 => description => description.Conv1D(filters, Line, Groups),
        2 => description => description.Conv2D(filters, Plane, Groups),
        _ => description => description.Conv3D(filters, Block, Groups),
    };

    /// <summary>A convolution along its axes that gives the places back, making so many channels.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <remarks>The places kept at the end are read for every axis, so a node that says none of them says a nought for each.</remarks>
    public Action<Sequential> GiveBack(int filters) => Sizes.Length switch
    {
        1 => description => description.Conv1DTranspose(filters, Line, Grown[0]),
        2 => description => description.Conv2DTranspose(filters, Plane, new Steps(Grown[0], Grown[1])),
        _ => description => description.Conv3DTranspose(filters, Block, new Steps3D(Grown[0], Grown[1], Grown[2])),
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

        return Padding(node, new WindowPlan(sizes, strides, dilations, PaddingMode.Stated, 0), words);
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

    // The border the node declares: SAME_UPPER's, none for VALID, or the pads it writes out one side at a time, which a
    // window here states at each end of each axis. SAME_LOWER is the one border that still depends on the images reaching the
    // node — 'same' with the odd place before the axis instead of after it — so it is held to them once the network is
    // lowered; a negative pad is a crop, which a window here never does.
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
            var lower = new DeclaredPads(plan.Sizes, plan.Stride[0], []) { Lower = true };

            if (plan.Stride.Distinct().Count() == 1 && plan.Stride[0] == 1 && !lower.SameOver(plan.Sizes))
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

        if (pads.Length != 2 * words.Axes)
        {
            node.Refuse($"it pads [{string.Join(", ", pads)}] — {words.Places} before, then after — and a window {words.Over} states {2 * words.Axes} of them.");

            return new StatedWindow(plan, null);
        }

        if (pads.Any(places => places < 0))
        {
            node.Refuse($"it pads [{string.Join(", ", pads)}] places, and a padding here adds places and never takes them away.");

            return new StatedWindow(plan, null);
        }

        if (pads.Distinct().Count() == 1)
        {
            return new StatedWindow(plan with { Padding = pads[0] }, null);
        }

        // Where the pads ARE what TensorFlow's 'same' gives, they are read as that word rather than as the numbers they
        // came out as: that is what tf2onnx means by writing them out, and a window padded so pads every example it is
        // handed rather than the one the graph was exported for. Where they are anything else, they are exactly what they
        // say, at each end of each axis.
        var written = new DeclaredPads(plan.Sizes, plan.Stride[0], pads);

        return plan.Stride.Distinct().Count() == 1 && (plan.Stride[0] == 1 ? written.SameOver(plan.Sizes) : written.Possible)
            ? new StatedWindow(plan with { Mode = PaddingMode.Same }, written)
            : new StatedWindow(plan with { Pads = pads }, null);
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

        if (node.Input(2).Length == 0)
        {
            node.Refuse(NoBias);
        }

        return new OnnxLayer((window.Plan with { Groups = groups }).Convolve(kernel.Lengths[0]), Flow.Images)
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
        var rounded = window.Plan with { RoundsUp = node.Whole("ceil_mode", 0) != 0 };

        return new OnnxLayer(Pooling(node, rounded), Flow.Images) { Declared = window.Declared };
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

/// <summary>
/// ONNX's <c>ConvTranspose</c>: a convolution that gives back the places an ordinary one takes away, its kernel channels in
/// by the channels out of one group by the window's lengths, and the places it keeps at the end of each axis written as
/// <c>output_padding</c>.
/// </summary>
internal sealed class ConvTransposeKind : WindowKind
{
    /// <inheritdoc />
    public override string Name => "ConvTranspose";

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
                $"its kernel, '{node.Input(1).Quoted()}', is written as {Written(kernel.Lengths)}, and a transposed convolution here gives the places back along a series, an image or a volume: channels in by the channels out of one group by steps, or by rows by columns, or by planes by rows by columns.");

            return new OnnxLayer(null, Flow.Images);
        }

        var window = Walk(node, kernel.Lengths[2..], "transposed convolution");
        var words = new Walking(window.Plan.Axes);
        var groups = node.Whole("group", 1);
        var shaped = node.Wholes("output_shape", []);

        if (shaped.Length != 0)
        {
            node.Refuse(
                $"it says the shape it makes, [{string.Join(", ", shaped)}], and a transposed convolution here is told how many places to keep at the end of each axis instead, as 'output_padding' says them.");
        }

        var grown = node.Wholes("output_padding", new int[window.Plan.Axes]);

        if (grown.Length != window.Plan.Axes)
        {
            node.Refuse($"its 'output_padding' is written as [{string.Join(", ", grown)}], and a transposed convolution {words.Over} writes {words.Writes} there.");

            grown = new int[window.Plan.Axes];
        }

        if (node.Input(2).Length == 0)
        {
            node.Refuse(NoBias);
        }

        return new OnnxLayer((window.Plan with { Groups = groups, Grown = grown }).GiveBack(kernel.Lengths[1] * groups), Flow.Images)
        {
            Numbers = [new LayerNumber("weight", kernel, Laying.GivenBack), new LayerNumber("bias", numbers.Floats(node.Input(2)), Laying.AsWritten)],
            Declared = window.Declared,
        };
    }
}

/// <summary>
/// ONNX's <c>Pad</c> of a constant: a layer that adds places of one value around each example and does nothing else. The
/// places are a value of the graph, one before each axis and then one after each, and the axes they run along are every axis
/// unless the node names them.
/// </summary>
internal sealed class PadKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Pad";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!TakesImages(node, reaching))
        {
            return new OnnxLayer(null, Flow.Images);
        }

        var mode = node.Text("mode", "constant");

        if (mode != "constant")
        {
            node.Refuse($"it pads by {mode.Quoted()}, which mirrors or repeats the example, and a padding here pads with one value.");

            return new OnnxLayer(null, Flow.Images);
        }

        var held = numbers.Wholes(node.Input(1));

        if (held.Values is not { } written)
        {
            node.Refuse($"its pads, '{node.Input(1).Quoted()}', {held.Fault} A padding here reads the places it pads from the graph's own numbers, never from what the graph works out.");

            return new OnnxLayer(null, Flow.Images);
        }

        // Channels first, as ONNX lays a batch out: the batch's axis, the channels', then the axes a window would walk.
        var axes = reaching.Axes;
        var named = node.Input(3).Length == 0 ? null : numbers.Wholes(node.Input(3)).Values;
        var sides = Sided(node, written, named, axes);

        if (sides is null)
        {
            return new OnnxLayer(null, Flow.Images);
        }

        var value = node.Input(2).Length == 0 ? 0f : Valued(node, numbers, node.Input(2));

        return new OnnxLayer(Padding(axes, sides, value), Flow.Images);
    }

    // The places before and after each axis a window walks, from the pads as the node writes them; nothing where they are not
    // places a padding here can add.
    private static int[]? Sided(OnnxNode node, long[] written, long[]? named, int axes)
    {
        var all = axes + 2;
        var pads = new long[2 * all];

        if (named is null)
        {
            if (written.Length != 2 * all)
            {
                node.Refuse($"it pads [{string.Join(", ", written)}], and a padding of a batch of {axes + 1} axes and its channels writes {2 * all} places: one before each axis, then one after each.");

                return null;
            }

            written.CopyTo(pads, 0);
        }
        else
        {
            if (written.Length != 2 * named.Length)
            {
                node.Refuse($"it pads [{string.Join(", ", written)}] along {named.Length} axes, and writes one place before each and one after each.");

                return null;
            }

            for (var at = 0; at < named.Length; at++)
            {
                var axis = named[at] < 0 ? named[at] + all : named[at];

                if (axis < 0 || axis >= all)
                {
                    node.Refuse($"it pads along axis {named[at]}, and a batch of {axes + 1} axes and its channels has none there.");

                    return null;
                }

                pads[axis] = written[at];
                pads[all + axis] = written[named.Length + at];
            }
        }

        if (pads.Any(places => places < 0))
        {
            node.Refuse($"it pads [{string.Join(", ", pads)}] places, and a padding here adds places and never takes them away.");

            return null;
        }

        if (pads[0] != 0 || pads[1] != 0 || pads[all] != 0 || pads[all + 1] != 0)
        {
            node.Refuse(
                $"it pads the batch's own axis or its channels, [{string.Join(", ", pads)}], and a padding here adds places around each example and leaves the examples and their channels as they are.");

            return null;
        }

        return [.. Enumerable.Range(0, axes).SelectMany(axis => new[] { (int)pads[axis + 2], (int)pads[all + axis + 2] })];
    }

    // The one value the places hold, as the graph writes it; nothing where it is not one value.
    private static float Valued(OnnxNode node, GraphNumbers numbers, string input)
    {
        var held = numbers.Floats(input);

        if (held.Values is { Length: 1 } one)
        {
            return one[0];
        }

        node.Refuse($"the value it pads with, '{input.Quoted()}', is written as {Written(held.Lengths)}, and a padding here pads with one value.");

        return 0f;
    }

    // The word of the padding for the places it adds along each axis.
    private static Action<Sequential> Padding(int axes, int[] sides, float value) => axes switch
    {
        1 => description => description.Pad1D(new Sides(sides[0], sides[1]), value),
        2 => description => description.Pad2D(new Margins(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3])), value),
        _ => description => description.Pad3D(
            new Margins3D(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3]), new Sides(sides[4], sides[5])),
            value),
    };
}
