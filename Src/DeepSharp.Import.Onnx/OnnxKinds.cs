// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Import.Onnx;

/// <summary>What reaches a node, as the graph lays it out.</summary>
internal enum Flow
{
    /// <summary>A batch of rows: batch by values.</summary>
    Rows,

    /// <summary>
    /// A batch of series, images or volumes: batch by channels by steps, or by rows by columns, or by planes by rows by
    /// columns in the graph, with its channels last here.
    /// </summary>
    Images,

    /// <summary>A batch of series, images or volumes each flattened into a row, channel by channel in the graph and place by place here.</summary>
    FlattenedImages,

    /// <summary>
    /// A batch of images with their channels last in the graph too — batch by rows by columns by channels — as TensorFlow
    /// lays them out and as images are here: what a Transpose moving an image's channels last hands on.
    /// </summary>
    ImagesLast,
}

/// <summary>What reaches a node: the value's layout, the batch's length when the graph states one, and along how many axes it walks.</summary>
/// <param name="Flow">How the value is laid out.</param>
/// <param name="Batch">The batch's length the graph's input states; nothing when it leaves it open.</param>
/// <param name="Axes">How many axes a series, an image or a volume has beside its batch and its channels: one, two or three; none for rows.</param>
/// <param name="Lifted">Whether a series has one more axis of one place than it has, made by the Unsqueeze before the pooling that takes it.</param>
internal readonly record struct Reaching(Flow Flow, long? Batch, int Axes, bool Lifted = false);

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
    /// The border a convolution's or a pooling's pads write out, one side at a time, read as TensorFlow's 'same': held after
    /// lowering to the border 'same' gives the images that reach it. Nothing for a border stated alike on every side, or
    /// declared 'same'.
    /// </summary>
    public DeclaredPads? Declared { get; init; }
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
        new GemmKind(), new MatMulKind(), new AddKind(), new ConvKind(), new MaxPoolKind(), new AveragePoolKind(), new GlobalAveragePoolKind(), new GlobalMaxPoolKind(),
        new ReduceKind("ReduceMean", new GlobalAveragePoolKind()), new ReduceKind("ReduceMax", new GlobalMaxPoolKind()), new BatchNormalizationKind(), new LayerNormalizationKind(), new FlattenKind(), new ReshapeKind(),
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

    /// <summary>
    /// Whether what reaches an operator over a series, an image or a volume is one with its channels after the batch, as ONNX's
    /// operators over them take it; the fault noted at the node when it is not.
    /// </summary>
    /// <param name="node">The node.</param>
    /// <param name="reaching">What reaches it.</param>
    protected static bool TakesImages(OnnxNode node, Reaching reaching)
    {
        if (reaching.Flow == Flow.Images)
        {
            return true;
        }

        node.Refuse(reaching.Flow switch
        {
            Flow.ImagesLast => $"it takes images with their channels after the batch, as ONNX's {node.Operator} does, and the images reaching it have them last.",
            Flow.Rows => $"it takes a series, an image or a volume with its channels after the batch, as ONNX's {node.Operator} does, and what reaches it is a row of values.",
            _ => $"it takes a series, an image or a volume with its channels after the batch, as ONNX's {node.Operator} does, and what reaches it is an image flattened into a row.",
        });

        return false;
    }

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

        if (lengths is not [var batch, var row] || !KeepsTheBatch(batch) || !MakesARow(row, batch))
        {
            node.Refuse($"it lays each batch out as [{string.Join(", ", lengths)}], and a reshape is read here when it flattens each example into one row, as Flatten does.");

            return new OnnxLayer(null, reaching.Flow);
        }

        return Flattened(reaching.Flow) with { RowLength = row > 0 ? row : null };

        // The batch's length left to be worked out, stated as the graph's input states it, or copied from the input.
        bool KeepsTheBatch(long length) => length == -1 || length == reaching.Batch || (length == 0 && copies);

        // The row's length stated, or left to be worked out once the batch's length is not.
        static bool MakesARow(long length, long batchLength) => length > 0 || (length == -1 && batchLength != -1);
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
