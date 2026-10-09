// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using Onnx;
using static Onnx.AttributeProto.Types;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// One node of an ONNX graph, read in the graph's own words: where it stands, what it takes and gives, and its attributes,
/// each one that cannot be read, or says what nothing here does, noted as a fault at the node.
/// </summary>
/// <remarks>
/// A fault is noted and the reading goes on, so every fault of a graph is named at once; an attribute that could not be read
/// stands at the value ONNX gives it when it is left out, which nothing is built from while a fault is noted.
/// </remarks>
internal sealed class OnnxNode
{
    private readonly NodeProto _node;
    private readonly List<string> _faults;
    private readonly HashSet<string> _used;

    /// <summary>A node, its place among the graph's nodes, the faults of the graph it is noted among, and the names of the values the graph takes.</summary>
    /// <param name="node">The node.</param>
    /// <param name="place">Its place among the graph's nodes.</param>
    /// <param name="faults">The faults of the graph, where the node's are noted.</param>
    /// <param name="used">The names of the values something takes: another node, or the graph as what it gives.</param>
    public OnnxNode(NodeProto node, int place, List<string> faults, HashSet<string> used)
    {
        _node = node;
        _faults = faults;
        _used = used;
        Operator = node.Domain is "" or "ai.onnx" ? node.OpType : $"{node.Domain}.{node.OpType}";
        Address = node.Name.Length > 0 ? $"node '{node.Name}' ({Operator})" : string.Create(CultureInfo.InvariantCulture, $"node {place} ({Operator})");
    }

    /// <summary>What it computes: ONNX's operator by its name, a domain of another's before it.</summary>
    public string Operator { get; }

    /// <summary>Where it stands, as a fault names it: <c>node '/0/Gemm' (Gemm)</c>, or its place when it has no name.</summary>
    public string Address { get; }

    /// <summary>The names of the values it takes, in order; an optional one left out is empty.</summary>
    public IReadOnlyList<string> Inputs => _node.Input;

    /// <summary>The value it gives: its first output, the only one a network here uses.</summary>
    public string Output => string.Concat(_node.Output.Take(1));

    /// <summary>The name of the value it takes at a place; empty when it takes none there.</summary>
    public string Input(int at) => at < _node.Input.Count ? _node.Input[at] : string.Empty;

    /// <summary>
    /// The axes it states: as an input the graph holds — an Unsqueeze's and a Squeeze's from ONNX 13, a reduce's from ONNX 18 —
    /// or as an attribute before; none when it states none; nothing, its fault noted, when they cannot be read.
    /// </summary>
    /// <param name="numbers">The values the graph holds.</param>
    public int[]? Axes(GraphNumbers numbers)
    {
        var input = Input(1);

        if (input.Length == 0)
        {
            return Wholes("axes", []);
        }

        if (Says("axes"))
        {
            Refuse($"it says its axes as an input, '{input.Quoted()}', and as an attribute, and ONNX writes them one way or the other.");

            return null;
        }

        var held = numbers.Wholes(input);

        if (held.Values is not { } values)
        {
            Refuse($"its axes, '{input.Quoted()}', {held.Fault}");

            return null;
        }

        if (values.Any(value => value is < int.MinValue or > int.MaxValue))
        {
            Refuse($"its axes, '{input.Quoted()}', are written as [{string.Join(", ", values)}], which is not what ONNX writes there.");

            return null;
        }

        return [.. values.Select(value => (int)value)];
    }

    /// <summary>The name of the value it gives at a place, when it names one that something takes — another node, or the graph; nothing otherwise.</summary>
    /// <param name="at">Its place among the values the node gives.</param>
    public string? Taken(int at) => at < _node.Output.Count && _node.Output[at].Length > 0 && _used.Contains(_node.Output[at]) ? _node.Output[at] : null;

    /// <summary>A whole-number attribute; what ONNX gives it when it is left out, or cannot be read.</summary>
    public int Whole(string name, int otherwise) =>
        Said(name, AttributeType.Int) is { } attribute ? WholeOf(name, attribute.I, otherwise) : otherwise;

    /// <summary>A list of whole numbers; what ONNX gives it when it is left out, or cannot be read.</summary>
    public int[] Wholes(string name, int[] otherwise)
    {
        if (Said(name, AttributeType.Ints) is not { } attribute)
        {
            return otherwise;
        }

        if (attribute.Ints.Any(value => value is < int.MinValue or > int.MaxValue))
        {
            Refuse($"its '{name}' is written as [{string.Join(", ", attribute.Ints)}], which is not what ONNX writes there.");

            return otherwise;
        }

        return [.. attribute.Ints.Select(value => (int)value)];
    }

    /// <summary>
    /// A number attribute, as the shortest decimal the single-precision number ONNX keeps it as stands for — an epsilon
    /// written as 1e-5 is read as 1e-5, not as the single nearest it; what ONNX gives it when it is left out, or cannot be read.
    /// </summary>
    public double Number(string name, double otherwise) =>
        Said(name, AttributeType.Float) is { } attribute
            ? double.Parse(attribute.F.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture)
            : otherwise;

    /// <summary>A text attribute; what ONNX gives it when it is left out, or cannot be read.</summary>
    public string Text(string name, string otherwise) => Said(name, AttributeType.String) is { } attribute ? attribute.S.ToStringUtf8() : otherwise;

    /// <summary>A tensor attribute; nothing when it is left out, or cannot be read.</summary>
    public TensorProto? Tensor(string name) => Said(name, AttributeType.Tensor)?.T;

    /// <summary>Whether it writes an attribute of the name, of whatever kind.</summary>
    public bool Says(string name) => _node.Attribute.Any(attribute => attribute.Name == name);

    /// <summary>The names of its attributes, as a fault lists them: <c>'value_ints'</c>.</summary>
    public string Attributes => $"'{string.Join("', '", _node.Attribute.Select(attribute => attribute.Name.Quoted()))}'";

    /// <summary>Notes a fault at the node.</summary>
    public void Refuse(string fault) => _faults.Add($"{Address.Quoted()}: {fault}");

    // The attribute of a name when it is of the kind ONNX writes it as; nothing, its fault noted, when it is of another, and
    // nothing when there is none.
    private AttributeProto? Said(string name, AttributeType kind)
    {
        var attribute = _node.Attribute.FirstOrDefault(each => each.Name == name);

        if (attribute is null || attribute.Type == kind)
        {
            return attribute;
        }

        Refuse($"its '{name}' is written as {attribute.Type.ToString().ToUpperInvariant()}, which is not what ONNX writes there.");

        return null;
    }

    private int WholeOf(string name, long value, int otherwise)
    {
        if (value is >= int.MinValue and <= int.MaxValue)
        {
            return (int)value;
        }

        Refuse(string.Create(CultureInfo.InvariantCulture, $"its '{name}' is written as {value}, which is not what ONNX writes there."));

        return otherwise;
    }
}
