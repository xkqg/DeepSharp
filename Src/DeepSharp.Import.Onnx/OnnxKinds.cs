// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Onnx;

/// <summary>What reaches a node, as the graph lays it out.</summary>
internal enum Flow
{
    /// <summary>A batch of rows: batch by values.</summary>
    Rows,

    /// <summary>A batch of images: batch by channels by rows by columns in the graph, with its channels last here.</summary>
    Images,

    /// <summary>A batch of images each flattened into a row, channel by channel in the graph and place by place here.</summary>
    FlattenedImages,

    /// <summary>
    /// A batch of images with their channels last in the graph too — batch by rows by columns by channels — as TensorFlow
    /// lays them out and as images are here: what a Transpose moving an image's channels last hands on.
    /// </summary>
    ImagesLast,
}

/// <summary>What reaches a node: the value's layout, and the batch's length when the graph states one.</summary>
/// <param name="Flow">How the value is laid out.</param>
/// <param name="Batch">The batch's length the graph's input states; nothing when it leaves it open.</param>
internal readonly record struct Reaching(Flow Flow, long? Batch);

/// <summary>A number a node holds for one slot of the layer it becomes, and how it is laid out for that slot.</summary>
/// <param name="Slot">The slot's name in the layer: <c>weight</c>, <c>running_var</c>.</param>
/// <param name="Held">The number, as the graph holds it.</param>
/// <param name="Laying">How its node declares it laid out.</param>
internal readonly record struct LayerNumber(string Slot, HeldTensor Held, Laying Laying)
{
    /// <summary>Whether it lies along the row an image was flattened into, one value of it for each of the image's values.</summary>
    public bool AlongFlattenedImage { get; init; }
}

/// <summary>What one node of a graph becomes here: the word it is written as, the layout it hands on, and its numbers.</summary>
/// <param name="Words">What it writes into a description; nothing for a softmax, which only a loss that applies it takes over.</param>
/// <param name="After">How the value it gives is laid out.</param>
internal readonly record struct OnnxLayer(Action<Sequential>? Words, Flow After)
{
    /// <summary>The numbers it holds, one for each slot of the layer it becomes, in the layer's order.</summary>
    public IReadOnlyList<LayerNumber> Numbers { get; init; } = [];

    /// <summary>The output activation it is, which a loss that applies that activation itself takes over when it is the last.</summary>
    public OutputActivation? Owned { get; init; }

    /// <summary>Whether it flattens an image into a row.</summary>
    public bool Flattens { get; init; }

    /// <summary>The length of the row a reshape states it lays each example out as; nothing when it leaves it to be worked out.</summary>
    public long? RowLength { get; init; }

    /// <summary>How wide the bias a MatMul still waits for is — the Add after it adds that bias; nothing for any other layer.</summary>
    public int? Unbiased { get; init; }

    /// <summary>The bias an Add adds to what the MatMul before it made; nothing for any other layer.</summary>
    public AddedBias? Adds { get; init; }

    /// <summary>
    /// The border a convolution's pads write out, one side at a time, read as TensorFlow's 'same': held after lowering to the
    /// border 'same' gives the image that reaches it. Nothing for a border stated alike on every side, or declared 'same'.
    /// </summary>
    public Borders? Declared { get; init; }
}

/// <summary>A bias an Add adds, as the graph holds it, and its name there.</summary>
/// <param name="Name">The value's name in the graph.</param>
/// <param name="Held">The value.</param>
internal readonly record struct AddedBias(string Name, HeldTensor Held);

/// <summary>An ONNX operator read here: what a node of it becomes, and what of it is refused.</summary>
/// <remarks>The kinds listed here are every operator read, with Identity and Constant; a node of any other is refused, naming them.</remarks>
internal abstract class OnnxKind
{
    /// <summary>What a dense layer or a convolution without a bias is refused with.</summary>
    internal const string NoBias = "it adds no bias, and a dense layer or a convolution here always adds one.";

    // Every operator read, in the order a refusal names them.
    private static readonly OnnxKind[] Covered =
    [
        new GemmKind(), new MatMulKind(), new AddKind(), new ConvKind(), new BatchNormalizationKind(), new LayerNormalizationKind(), new FlattenKind(), new ReshapeKind(),
        new ActivationKind("Relu", description => description.Relu(), null), new ActivationKind("Tanh", description => description.Tanh(), null),
        new ActivationKind("Sigmoid", description => description.Sigmoid(), OutputActivation.Sigmoid), new SoftmaxKind(),
    ];

    /// <summary>Every operator read, by ONNX's name for it.</summary>
    public static IReadOnlyDictionary<string, OnnxKind> ByName { get; } = Covered.ToDictionary(kind => kind.Name, StringComparer.Ordinal);

    /// <summary>The operators read, named as a refusal names them: <c>Gemm, Conv, … or Softmax</c>.</summary>
    public static string Listed { get; } = $"{string.Join(", ", Covered[..^1].Select(kind => kind.Name))} or {Covered[^1].Name}";

    /// <summary>ONNX's name for the operator.</summary>
    public abstract string Name { get; }

    /// <summary>What a node of this operator becomes; each fault of it noted at the node.</summary>
    /// <param name="node">The node.</param>
    /// <param name="reaching">What reaches it.</param>
    /// <param name="numbers">The values the graph holds, the node's weights among them.</param>
    public abstract OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers);

    /// <summary>
    /// A flatten of what reaches it, as Flatten or a reshape into a row: an image with its channels after the batch flattened
    /// channel by channel, one with them last place by place, as here.
    /// </summary>
    protected static OnnxLayer Flattened(Flow flow) => flow switch
    {
        Flow.Images => new(description => description.Flatten(), Flow.FlattenedImages) { Flattens = true },
        Flow.ImagesLast => new(description => description.Flatten(), Flow.Rows),
        _ => new(description => description.Flatten(), flow),
    };

    /// <summary>Whether what reaches a node is an image, whichever axis its channels stand on in the graph.</summary>
    protected static bool IsImage(Flow flow) => flow is Flow.Images or Flow.ImagesLast;

    /// <summary>Numbers of a normalisation, each read from the node's input at its place and laid out as written.</summary>
    protected static IReadOnlyList<LayerNumber> Along(OnnxNode node, Reaching reaching, GraphNumbers numbers, string[] slots) =>
        [.. slots.Select((slot, at) => new LayerNumber(slot, numbers.Floats(node.Input(at + 1)), Laying.AsWritten) { AlongFlattenedImage = reaching.Flow == Flow.FlattenedImages })];

    /// <summary>A shape as a fault names it: <c>16x14</c>, or <c>scalar</c>.</summary>
    internal static string Written(int[] lengths) => lengths.Length == 0 ? "scalar" : string.Join('x', lengths);
}

/// <summary>ONNX's <c>Gemm</c>: a dense layer, its weights turned round when the node says they are written outputs by inputs.</summary>
internal sealed class GemmKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Gemm";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var transposed = node.Whole("transB", 0) != 0;
        var alpha = node.Number("alpha", 1);
        var beta = node.Number("beta", 1);
        var weights = numbers.Floats(node.Input(1));

        if (node.Whole("transA", 0) != 0)
        {
            node.Refuse("it takes its input turned round, transA, and a dense layer here takes each example as a row.");
        }

        if (alpha != 1 || beta != 1)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it scales its product by {alpha} and its bias by {beta}, and a dense layer here scales neither."));
        }

        if (node.Input(2).Length == 0)
        {
            node.Refuse(NoBias);
        }

        if (weights.Lengths is not [var first, var second])
        {
            node.Refuse($"its weights, '{node.Input(1).Quoted()}', are written as {Written(weights.Lengths)}, and a dense layer's are a matrix.");

            return new OnnxLayer(null, Flow.Rows);
        }

        var units = transposed ? first : second;

        return new OnnxLayer(description => description.Dense(units), Flow.Rows)
        {
            Numbers =
            [
                new LayerNumber("weight", weights, transposed ? Laying.Transposed : Laying.AsWritten) { AlongFlattenedImage = reaching.Flow == Flow.FlattenedImages },
                new LayerNumber("bias", numbers.Floats(node.Input(2)), Laying.AsWritten),
            ],
        };
    }
}

/// <summary>
/// ONNX's <c>MatMul</c> of what reaches it by weights the graph holds: a dense layer, its weights written inputs by outputs
/// as the product declares them, its bias the <c>Add</c> after it — as Keras's own export and tf2onnx write a dense layer.
/// </summary>
internal sealed class MatMulKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "MatMul";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (!numbers.Holds(node.Input(1)))
        {
            node.Refuse($"its right side, '{node.Input(1).Quoted()}', is no value the graph holds, and a dense layer here multiplies each row by weights it holds.");

            return new OnnxLayer(null, Flow.Rows);
        }

        var weights = numbers.Floats(node.Input(1));

        if (weights.Lengths is not [_, var units])
        {
            node.Refuse($"its weights, '{node.Input(1).Quoted()}', are written as {Written(weights.Lengths)}, and a dense layer's are a matrix.");

            return new OnnxLayer(null, Flow.Rows);
        }

        return new OnnxLayer(description => description.Dense(units), Flow.Rows)
        {
            Numbers = [new LayerNumber("weight", weights, Laying.AsWritten) { AlongFlattenedImage = reaching.Flow == Flow.FlattenedImages }],
            Unbiased = units,
        };
    }
}

/// <summary>ONNX's <c>Add</c> of a value the graph holds: the bias of the MatMul before it, which it makes a dense layer with.</summary>
internal sealed class AddKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Add";

    /// <inheritdoc />
    /// <remarks>What it adds is the one of its two inputs the graph holds; the other is what the node before it made.</remarks>
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var bias = numbers.Holds(node.Input(1)) ? node.Input(1) : node.Input(0);

        return new OnnxLayer(null, reaching.Flow) { Adds = new AddedBias(bias, numbers.Floats(bias)) };
    }
}

/// <summary>
/// ONNX's <c>Conv</c> over images: a window walking one stride down and across alike, padded alike on every side or as
/// TensorFlow's 'same' — SAME_UPPER, or its pads written out — its kernel channels out by channels in by rows by columns.
/// </summary>
internal sealed class ConvKind : OnnxKind
{
    private const string Pads = "and a window here pads every side alike, or as SAME_UPPER, TensorFlow's 'same'.";

    /// <inheritdoc />
    public override string Name => "Conv";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (reaching.Flow == Flow.ImagesLast)
        {
            node.Refuse("it takes images with their channels after the batch, as ONNX's Conv does, and the images reaching it have them last.");

            return new OnnxLayer(null, Flow.Images);
        }

        var kernel = numbers.Floats(node.Input(1));

        if (kernel.Lengths is not [var filters, _, var rows, var columns])
        {
            node.Refuse(
                $"its kernel, '{node.Input(1).Quoted()}', is written as {Written(kernel.Lengths)}, and a convolution here slides a window along an image's rows and columns: channels out by channels in by rows by columns.");

            return new OnnxLayer(null, Flow.Images);
        }

        var strides = Axes(node, "strides");
        var dilations = Axes(node, "dilations");
        var groups = node.Whole("group", 1);
        var padding = Padding(node, new Window(rows, columns) { Stride = strides.Down });

        if (strides.Down != strides.Across)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it strides {strides.Down} down and {strides.Across} across, and a window here walks one stride down and across alike."));
        }

        if (dilations != new Pair(1, 1))
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"its window is dilated by {dilations.Down} down and {dilations.Across} across, and a window here covers neighbouring places."));
        }

        if (groups != 1)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it splits its channels into {groups} groups, and a convolution here takes every channel of a place at once."));
        }

        if (node.Input(2).Length == 0)
        {
            node.Refuse(NoBias);
        }

        var window = new Window(rows, columns) { Stride = strides.Down, PaddingMode = padding.Mode, Padding = padding.Sides };

        return new OnnxLayer(description => description.Conv2D(filters, window), Flow.Images)
        {
            Numbers = [new LayerNumber("weight", kernel, Laying.Kernel), new LayerNumber("bias", numbers.Floats(node.Input(2)), Laying.AsWritten)],
            Declared = padding.Declared,
        };
    }

    // A setting ONNX writes for each of an image's two axes, down and across; one each, when it is left out.
    private static Pair Axes(OnnxNode node, string name)
    {
        var lengths = node.Wholes(name, [1, 1]);

        if (lengths is [var down, var across])
        {
            return new Pair(down, across);
        }

        node.Refuse($"its '{name}' is written as [{string.Join(", ", lengths)}], and a convolution over an image writes a pair there.");

        return new Pair(1, 1);
    }

    // The border the node declares: SAME_UPPER's, none for VALID, the one its pads state for every side alike, or TensorFlow's
    // 'same' written out one side at a time, as tf2onnx writes it. At a stride of one 'same' pads every image alike — half
    // the window's reach before, the odd place after — so pads written otherwise are refused here; at a longer stride the
    // border 'same' gives depends on the image, and the pads are held to it once the network is lowered.
    private static Padded Padding(OnnxNode node, Window window)
    {
        var declared = node.Text("auto_pad", "NOTSET");

        if (declared == "SAME_UPPER")
        {
            return new Padded(PaddingMode.Same, 0);
        }

        if (declared == "VALID")
        {
            return new Padded(PaddingMode.Stated, 0);
        }

        if (declared != "NOTSET")
        {
            node.Refuse($"it pads as {declared.Quoted()}, {Pads}");

            return new Padded(PaddingMode.Stated, 0);
        }

        var pads = node.Wholes("pads", [0, 0, 0, 0]);

        if (pads.Length == 4 && pads.Distinct().Count() == 1)
        {
            return new Padded(PaddingMode.Stated, pads[0]);
        }

        if (pads is [var top, var left, var bottom, var right])
        {
            var sides = new Borders(top, bottom, left, right);
            var same = window.Stride == 1
                ? (window with { PaddingMode = PaddingMode.Same }).BordersOver(window.Height, window.Width) == sides
                : bottom - top is 0 or 1 && right - left is 0 or 1;

            if (same)
            {
                return new Padded(PaddingMode.Same, 0) { Declared = sides };
            }
        }

        node.Refuse($"it pads [{string.Join(", ", pads)}] — rows and columns before, then after — {Pads}");

        return new Padded(PaddingMode.Stated, 0);
    }

    /// <summary>A window's border, as a node declares it, and the sides its pads write out when they are 'same' written so.</summary>
    private readonly record struct Padded(PaddingMode Mode, int Sides)
    {
        public Borders? Declared { get; init; }
    }
}

/// <summary>Two lengths ONNX writes for an image's axes: down, along the rows, and across, along the columns.</summary>
/// <param name="Down">Along the rows.</param>
/// <param name="Across">Along the columns.</param>
internal readonly record struct Pair(int Down, int Across);

/// <summary>
/// ONNX's <c>BatchNormalization</c>, answering from its running statistics: its momentum the share of them a batch leaves
/// as they were — Keras's meaning, as the words in Keras's meaning take it — and its epsilon.
/// </summary>
internal sealed class BatchNormalizationKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "BatchNormalization";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (reaching.Flow == Flow.ImagesLast)
        {
            node.Refuse("it normalises along the axis after the batch, the rows of the images reaching it, whose channels come last: a batch normalisation here normalises each channel.");
        }

        if (node.Whole("training_mode", 0) != 0)
        {
            node.Refuse("it measures each batch as it answers, in training mode, and a batch normalisation here answers from its running statistics.");
        }

        if (node.Whole("spatial", 1) != 1)
        {
            node.Refuse("it normalises every value on its own, spatial 0, and a batch normalisation here normalises each feature, or each channel of an image.");
        }

        var momentum = node.Number("momentum", 0.9);
        var epsilon = node.Number("epsilon", 1e-5);

        return new OnnxLayer(description => description.BatchNorm(momentum, epsilon), reaching.Flow)
        {
            Numbers = Along(node, reaching, numbers, ["weight", "bias", "running_mean", "running_var"]),
        };
    }
}

/// <summary>ONNX's <c>LayerNormalization</c> of a row, over its last axis, with its scale, its shift and its epsilon.</summary>
internal sealed class LayerNormalizationKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "LayerNormalization";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        if (IsImage(reaching.Flow))
        {
            node.Refuse("it normalises an image, and a layer normalisation here normalises the values of a row.");

            return new OnnxLayer(null, reaching.Flow);
        }

        var axis = node.Whole("axis", -1);

        if (axis is not (-1 or 1))
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it normalises from axis {axis}, and a layer normalisation here normalises over the last axis of a row."));
        }

        if (node.Input(2).Length == 0)
        {
            node.Refuse("it leaves out its shift, and a layer normalisation here learns one.");
        }

        var epsilon = node.Number("epsilon", 1e-5);

        return new OnnxLayer(description => description.LayerNorm(epsilon), reaching.Flow) { Numbers = Along(node, reaching, numbers, ["weight", "bias"]) };
    }
}

/// <summary>ONNX's <c>Flatten</c> from the axis after the batch: each example as one row.</summary>
internal sealed class FlattenKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Flatten";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var axis = node.Whole("axis", 1);

        if (axis != 1)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it flattens from axis {axis}, and a flatten here keeps each example as one row."));
        }

        return Flattened(reaching.Flow);
    }
}

/// <summary>
/// ONNX's <c>Reshape</c> into a row of each example's values, as PyTorch's default exporter writes a flatten: the batch kept —
/// its length left to be worked out, stated as the graph states it, or copied — and the row's length stated or left to be
/// worked out.
/// </summary>
internal sealed class ReshapeKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Reshape";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var target = numbers.Wholes(node.Input(1));

        if (target.Values is not { } lengths)
        {
            node.Refuse($"its target, '{node.Input(1).Quoted()}', {target.Fault}");

            return new OnnxLayer(null, reaching.Flow);
        }

        var copies = node.Whole("allowzero", 0) == 0;

        if (lengths is not [var batch, var row]
            || !(batch == -1 || batch == reaching.Batch || (batch == 0 && copies))
            || !(row > 0 || (row == -1 && batch != -1)))
        {
            node.Refuse($"it lays each batch out as [{string.Join(", ", lengths)}], and a reshape is read here when it flattens each example into one row, as Flatten does.");

            return new OnnxLayer(null, reaching.Flow);
        }

        return Flattened(reaching.Flow) with { RowLength = row > 0 ? row : null };
    }
}

/// <summary>An activation ONNX writes as an operator of its own: a relu, a tanh or a sigmoid, a layer each here.</summary>
/// <param name="name">ONNX's name for it.</param>
/// <param name="word">The word it is written as.</param>
/// <param name="owned">The output activation it is, when a loss can apply it itself; nothing otherwise.</param>
internal sealed class ActivationKind(string name, Action<Sequential> word, OutputActivation? owned) : OnnxKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers) => new(word, reaching.Flow) { Owned = owned };
}

/// <summary>
/// ONNX's <c>Softmax</c> over a row: no layer here, read as the last of a graph whose cross-entropy applies it itself.
/// </summary>
internal sealed class SoftmaxKind : OnnxKind
{
    /// <inheritdoc />
    public override string Name => "Softmax";

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var axis = node.Whole("axis", -1);

        if (IsImage(reaching.Flow))
        {
            node.Refuse("it takes shares over an image, and a softmax is read here over the values of a row.");
        }
        else if (axis is not (-1 or 1))
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it takes shares along axis {axis}, and a softmax is read here over the values of a row."));
        }

        return new OnnxLayer(null, reaching.Flow) { Owned = OutputActivation.Softmax };
    }
}
