// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Google.Protobuf;
using Onnx;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A graph being written: its nodes, its numbers as initializers, and the operator set the nodes need, which is the oldest that
/// holds every one of them.
/// </summary>
/// <remarks>
/// Every node says the operator set it first appeared in with the meaning written here, and the graph asks for the greatest of
/// those. The graph's IR version is the one ONNX's table pairs with that operator set, so a runtime that reads the oldest sets
/// reads a network that needs no more.
/// </remarks>
internal sealed class GraphDraft
{
    // ONNX's table of IR versions by operator set: the first operator set each IR version carries.
    private static readonly IrAt[] IrByOpset = [new(1, 3), new(9, 4), new(10, 5), new(11, 6), new(12, 7), new(14, 8), new(19, 9), new(21, 10), new(23, 11), new(24, 12)];

    private readonly GraphProto _graph = new() { Name = "network" };
    private int _opset = 1;

    // The operators that took their axes as an attribute in the older sets and take them as an input from a later one.
    private static readonly AxesInput[] AxesAsInput =
        [new("Squeeze", 13), new("Unsqueeze", 13), new("ReduceSum", 13), new("ReduceMean", 18), new("ReduceMax", 18), new("ReduceMin", 18)];

    /// <summary>The operator set the nodes written so far need.</summary>
    public int Opset => _opset;

    /// <summary>
    /// Settles the nodes to the operator set the whole graph needs. A node that states its axes as an attribute, as the operator set it first
    /// appeared in has them, states them as an input where another node of the graph asks for a later set that has them so.
    /// </summary>
    /// <remarks>Called once the last node is written: the set a graph needs is the greatest of its nodes', which a node written first cannot know.</remarks>
    public void Settle()
    {
        foreach (var node in _graph.Node)
        {
            if (AxesAsInput.Any(each => each.Operator == node.OpType && _opset >= each.From) && node.Attribute.FirstOrDefault(attribute => attribute.Name == "axes") is { } axes)
            {
                node.Attribute.Remove(axes);
                node.Input.Add(Wholes($"{node.Name}.axes", [.. axes.Ints]));
            }
        }
    }

    /// <summary>The IR version ONNX pairs with <see cref="Opset"/>.</summary>
    public long IrVersion => IrByOpset.Last(each => each.Opset <= _opset).Ir;

    /// <summary>The graph.</summary>
    public GraphProto Graph => _graph;

    /// <summary>Declares the one value the graph takes: a batch of examples of the lengths given.</summary>
    /// <param name="name">The value's name.</param>
    /// <param name="lengths">The length of each axis after the batch.</param>
    public void Takes(string name, ReadOnlySpan<int> lengths) => _graph.Input.Add(Value(name, lengths));

    /// <summary>Declares the one value the graph gives: a batch of rows of the lengths given.</summary>
    /// <param name="name">The value's name.</param>
    /// <param name="lengths">The length of each axis after the batch.</param>
    public void Gives(string name, ReadOnlySpan<int> lengths) => _graph.Output.Add(Value(name, lengths));

    /// <summary>Adds numbers to the graph as an initializer of single-precision numbers.</summary>
    /// <param name="name">The initializer's name: the slot's path.</param>
    /// <param name="lengths">The length of each axis.</param>
    /// <param name="values">The numbers, row-major.</param>
    /// <returns>The initializer's name.</returns>
    public string Number(string name, ReadOnlySpan<int> lengths, ReadOnlySpan<float> values)
    {
        var tensor = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Float };

        foreach (var length in lengths)
        {
            tensor.Dims.Add(length);
        }

        var bytes = new byte[values.Length * sizeof(float)];

        for (var at = 0; at < values.Length; at++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(at * sizeof(float)), values[at]);
        }

        tensor.RawData = ByteString.CopyFrom(bytes);
        _graph.Initializer.Add(tensor);

        return name;
    }

    /// <summary>Adds whole numbers to the graph as an initializer of 64-bit integers: the axes, the starts and the ends a node reads.</summary>
    /// <param name="name">The initializer's name.</param>
    /// <param name="values">The numbers.</param>
    /// <returns>The initializer's name.</returns>
    public string Wholes(string name, params long[] values)
    {
        var tensor = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Int64 };

        tensor.Dims.Add(values.Length);
        tensor.RawData = ByteString.CopyFrom([.. values.SelectMany(BitConverter.GetBytes)]);
        _graph.Initializer.Add(tensor);

        return name;
    }

    /// <summary>Adds a node.</summary>
    /// <param name="at">What the node is: its operator, the operator set it first appeared in, and its name.</param>
    /// <param name="inputs">The names of the values it takes.</param>
    /// <param name="output">The name of the value it gives.</param>
    /// <returns>The name of the value it gives.</returns>
    public string Node(NodeOf at, string[] inputs, string output)
    {
        var node = new NodeProto { OpType = at.Operator, Name = at.Name };

        node.Input.AddRange(inputs);
        node.Output.Add(output);
        at.Attributes?.Invoke(node);
        _graph.Node.Add(node);
        _opset = Math.Max(_opset, at.Since);

        return output;
    }

    /// <summary>Adds a node that gives several values.</summary>
    /// <param name="at">What the node is.</param>
    /// <param name="inputs">The names of the values it takes.</param>
    /// <param name="outputs">The names of the values it gives, in order.</param>
    /// <returns>The names of the values it gives.</returns>
    public string[] Nodes(NodeOf at, string[] inputs, string[] outputs)
    {
        var node = new NodeProto { OpType = at.Operator, Name = at.Name };

        node.Input.AddRange(inputs);
        node.Output.AddRange(outputs);
        at.Attributes?.Invoke(node);
        _graph.Node.Add(node);
        _opset = Math.Max(_opset, at.Since);

        return outputs;
    }

    private static ValueInfoProto Value(string name, ReadOnlySpan<int> lengths)
    {
        var shape = new TensorShapeProto();

        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimParam = "batch" });

        foreach (var length in lengths)
        {
            shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = length });
        }

        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorProto.Types.DataType.Float, Shape = shape } },
        };
    }
}

/// <summary>An operator that states its axes as an input from an operator set on, and as an attribute before it.</summary>
/// <param name="Operator">ONNX's name for the operator.</param>
/// <param name="From">The first operator set that takes the axes as an input.</param>
internal readonly record struct AxesInput(string Operator, int From);

/// <summary>The first operator set an IR version of ONNX carries.</summary>
/// <param name="Opset">The operator set.</param>
/// <param name="Ir">The IR version.</param>
internal readonly record struct IrAt(int Opset, long Ir);

/// <summary>A node to be written: its operator, the operator set it first appeared in, its name, and what it states of itself.</summary>
/// <param name="Operator">ONNX's name for the operator.</param>
/// <param name="Since">The first operator set that carries the operator with the meaning written here.</param>
/// <param name="Name">The node's name.</param>
internal readonly record struct NodeOf(string Operator, int Since, string Name)
{
    /// <summary>What it states: its attributes.</summary>
    public Action<NodeProto>? Attributes { get; init; }
}

/// <summary>The attributes a node states, written the way ONNX writes them.</summary>
internal static class NodeAttributes
{
    extension(NodeProto node)
    {
        /// <summary>States a whole number.</summary>
        public void State(string name, long value) =>
            node.Attribute.Add(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value });

        /// <summary>States a number, as the single-precision number ONNX keeps it as.</summary>
        public void State(string name, double value) =>
            node.Attribute.Add(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = (float)value });

        /// <summary>States a word.</summary>
        public void State(string name, string value) =>
            node.Attribute.Add(new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.String, S = ByteString.CopyFromUtf8(value) });

        /// <summary>States a list of whole numbers.</summary>
        public void States(string name, params long[] values)
        {
            var attribute = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };

            attribute.Ints.Add(values);
            node.Attribute.Add(attribute);
        }
    }

}
