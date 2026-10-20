// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Export.Onnx;

/// <summary>An image, a series or a volume as a flatten found it: so many rows, columns and channels.</summary>
/// <param name="Rows">Its rows; the planes' rows one after another for a volume, one for a series.</param>
/// <param name="Columns">Its columns.</param>
/// <param name="Channels">Its channels.</param>
internal readonly record struct Flattened(int Rows, int Columns, int Channels)
{
    /// <summary>The image a flatten finds in examples of the shape given, the channels last.</summary>
    /// <param name="each">The shape of one example, without the batch's axis.</param>
    public static Flattened Of(Shape each) => new(each.Axes[..^2].ToArray().Aggregate(1, (all, length) => all * length), each[each.Rank - 2], each[each.Rank - 1]);
}

/// <summary>
/// A value of the graph being written: the name it goes by, and the shape of one example of it as the network holds it — the channels
/// last. The graph holds an image, a series or a volume with its channels first, as ONNX's operators over them take it.
/// </summary>
/// <param name="Name">The value's name in the graph.</param>
/// <param name="Each">The shape of one example, without the batch's axis.</param>
internal readonly record struct Value(string Name, Shape Each)
{
    /// <summary>The image a flatten made this row of, when it did: ONNX flattens it channel by channel and the network place by place.</summary>
    public Flattened? Image { get; init; }

    /// <summary>Whether it is a series, an image or a volume the graph holds with its channels first.</summary>
    public bool Spatial => Image is null && Each.Rank > 1;

    /// <summary>The numbers that lie along this value's row, in the order the graph's row has them.</summary>
    /// <param name="numbers">The numbers in the order the network's row has them: a dense layer's weights by row, a normalisation's scale.</param>
    /// <returns>The same numbers, channel by channel where the row is a flattened image.</returns>
    public float[] Ordered(ReadOnlySpan<float> numbers) =>
        Image is { } image ? OnnxLayouts.ChannelByChannel(numbers.ToArray(), image.Rows, image.Columns, image.Channels) : numbers.ToArray();
}

/// <summary>One step being written: the layer, where it stands, the values it takes, and what it makes.</summary>
/// <param name="Graph">The graph being written.</param>
/// <param name="Layer">The layer.</param>
/// <param name="Path">The path of its step: the name its slots' numbers go by in the graph.</param>
/// <param name="Inputs">The values of the steps it takes, in the order it takes them.</param>
/// <param name="Made">The shape of one example of what it makes.</param>
internal sealed record Writing(GraphDraft Graph, Layer Layer, string Path, IReadOnlyList<Value> Inputs, Shape Made)
{
    /// <summary>The value of the one step it takes.</summary>
    public Value Input => Inputs[0];

    /// <summary>A node of the step, named under its path.</summary>
    /// <param name="op">ONNX's name for the operator.</param>
    /// <param name="since">The first operator set that carries it with the meaning written here.</param>
    /// <param name="part">What tells it from the step's other nodes of the same operator; nothing for the only one.</param>
    public NodeOf Node(string op, int since, string? part = null) => new(op, since, $"{Path}/{part ?? op}");

    /// <summary>The name a number of the step goes by in the graph.</summary>
    /// <param name="slot">The slot's name: <c>weight</c>.</param>
    public string Number(string slot) => $"{Path}.{slot}";

    /// <summary>The value the step makes, given the name it goes by.</summary>
    /// <param name="name">The name of the graph's value.</param>
    public Value Giving(string name) => new(name, Made);

    /// <summary>A series, an image or a volume the graph holds with its channels first, turned so its channels are last, which is where ONNX's operators over the last axis take them.</summary>
    /// <param name="value">The value's name.</param>
    /// <param name="part">What tells the node from the step's others.</param>
    /// <returns>The turned value's name.</returns>
    public string ChannelsLast(string value, string part) =>
        Graph.Node(Node("Transpose", 1, part) with { Attributes = node => node.States("perm", [0L, .. Enumerable.Range(2, Input.Each.Rank - 1).Select(axis => (long)axis), 1L]) }, [value], $"{Path}/{part}");

    /// <summary>A value turned with its channels last, turned back so its channels come first; the step's own value, given the name the step goes by.</summary>
    /// <param name="value">The value's name.</param>
    /// <returns>The step's value.</returns>
    public Value ChannelsFirst(string value) =>
        Giving(Graph.Node(Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", [0L, Made.Rank, .. Enumerable.Range(1, Made.Rank - 1).Select(axis => (long)axis)]) }, [value], Path));

    /// <summary>The value the step makes, given the name it goes by, as the row of the image the step it takes made one of, when it did.</summary>
    /// <param name="name">The name of the graph's value.</param>
    public Value Keeping(string name) => new(name, Made) { Image = Input.Image };
}

/// <summary>How one kind of layer is written as ONNX nodes, or left out of the graph where a trained network does nothing with it.</summary>
/// <remarks>
/// A kind is added by writing a class and listing it in <see cref="ExportKinds"/>; nothing else decides which layers are written.
/// A layer that is of no kind listed is refused by name, never written as something it is not.
/// </remarks>
internal abstract class ExportKind
{
    /// <summary>The layer this kind writes.</summary>
    public abstract Type Layer { get; }

    /// <summary>Writes the step.</summary>
    /// <param name="at">The step.</param>
    /// <returns>The value the step makes.</returns>
    public abstract Value Write(Writing at);
}

/// <summary>A kind that writes one class of layer.</summary>
/// <typeparam name="TLayer">The layer's class.</typeparam>
internal abstract class ExportKind<TLayer> : ExportKind
    where TLayer : Layer
{
    /// <inheritdoc />
    public sealed override Type Layer => typeof(TLayer);

    /// <inheritdoc />
    public sealed override Value Write(Writing at) => Write((TLayer)at.Layer, at);

    /// <summary>Writes the step.</summary>
    /// <param name="layer">The layer.</param>
    /// <param name="at">The step.</param>
    protected abstract Value Write(TLayer layer, Writing at);
}

/// <summary>A layer that makes the value of one operator over what it takes, with nothing to state.</summary>
/// <typeparam name="TLayer">The layer's class.</typeparam>
/// <param name="op">ONNX's name for the operator.</param>
/// <param name="since">The first operator set that carries it.</param>
internal sealed class OperatorKind<TLayer>(string op, int since) : ExportKind<TLayer>
    where TLayer : Layer
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at) =>
        at.Keeping(at.Graph.Node(at.Node(op, since), [at.Input.Name], at.Path));
}

/// <summary>A layer a trained network does nothing with: a dropout, or a variation of the rows made while training. It is left out of the graph.</summary>
/// <typeparam name="TLayer">The layer's class.</typeparam>
internal sealed class OmittedKind<TLayer> : ExportKind<TLayer>
    where TLayer : Layer
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at) => at.Input;
}
