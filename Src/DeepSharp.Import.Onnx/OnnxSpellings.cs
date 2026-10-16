// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Onnx;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// A word an exporter wrote out of several operators, recognised as the one layer it spells: the nodes it took, the value it
/// gives, the layer, and the constant nodes among the nodes it took, whose values the graph keeps.
/// </summary>
/// <param name="Last">The place of the last node the spelling took.</param>
/// <param name="Output">The value it gives: what its last node makes.</param>
/// <param name="Layer">The layer it spells.</param>
/// <param name="Constants">The places of the constant nodes among the nodes it took.</param>
internal readonly record struct Spelled(int Last, string Output, OnnxLayer Layer, IReadOnlyList<int> Constants);

/// <summary>Where a spelling is looked for: the place of its first node, the value flowing into it, and what that value is.</summary>
/// <param name="Place">The place of the first node.</param>
/// <param name="Flowing">The value flowing into it.</param>
/// <param name="Reaching">What that value is.</param>
internal readonly record struct SpellingAt(int Place, string Flowing, Reaching Reaching);

/// <summary>
/// The spellings PyTorch's exporters write a word in when they are asked for an operator set that has no operator of its own for
/// it — read as the one layer each spells, every number in it held to the word's own.
/// </summary>
/// <remarks>
/// Each spelling is the chain the exporter writes, node by node, as onnx-vocabulary.txt shows them, from the TorchScript exporter
/// and from the dynamo one where they differ: a GELU as Div, Erf, Add and two Muls, or as its hyperbolic tangent's chain; a SiLU
/// as a Sigmoid and a Mul; a mish as a Softplus, a Tanh and a Mul; a group normalisation as a Reshape into groups, an
/// InstanceNormalization of ones and noughts, a Reshape back, then its scale and its shift; a root-mean-square normalisation of a
/// row as Pow, ReduceMean, Add, Sqrt, Reciprocal and two Muls. A chain is read only where every node of it takes the value the one
/// before it made, every number in it is the word's own — the root of two of a GELU, not some other — and nothing outside it takes
/// a value made inside it: anything else is no spelling, and its nodes are read, or refused, one by one.
/// </remarks>
internal static class OnnxSpellings
{
    private static readonly Func<Cursor, Reaching, OnnxLayer?>[] Every =
    [
        GeluByErf(lastByTheInput: true), GeluByErf(lastByTheInput: false), GeluByTanhAsTorchScript, GeluByTanhAsDynamo, Silu, Mish,
        GroupNorm(asTorchScript: true), GroupNorm(asTorchScript: false), RmsNorm,
    ];

    /// <summary>The word the nodes from a place on spell, taking the value flowing into that place; nothing when they spell none.</summary>
    /// <param name="graph">The graph.</param>
    /// <param name="numbers">The values the graph holds.</param>
    /// <param name="at">The place of the first node, the value flowing into it, and what that value is.</param>
    public static Spelled? Read(GraphProto graph, GraphNumbers numbers, SpellingAt at)
    {
        foreach (var spelling in Every)
        {
            var cursor = new Cursor(graph, at.Place, at.Flowing, numbers);

            if (spelling(cursor, at.Reaching) is { } layer && cursor.Closed())
            {
                return new Spelled(cursor.Last, cursor.Value, layer, cursor.Constants);
            }
        }

        return null;
    }

    // x · ½(1 + erf(x / √2)): the TorchScript exporter multiplies by the input and then by a half, the dynamo one the other way.
    private static Func<Cursor, Reaching, OnnxLayer?> GeluByErf(bool lastByTheInput) => (cursor, reaching) =>
        cursor.Ordered("Div", Operand.Value(Math.Sqrt(2))) && cursor.Step("Erf") && cursor.Step("Add", Operand.Value(1))
        && (lastByTheInput
            ? cursor.Step("Mul", Operand.Input) && cursor.Step("Mul", Operand.Value(0.5))
            : cursor.Step("Mul", Operand.Value(0.5)) && cursor.Step("Mul", Operand.Input))
            ? new OnnxLayer(description => description.Gelu(), reaching.Flow)
            : null;

    // ½x(1 + tanh(√(2/π)(x + 0.044715x³))), the cube as x times x times x.
    private static OnnxLayer? GeluByTanhAsTorchScript(Cursor cursor, Reaching reaching) =>
        cursor.Step("Mul", Operand.Input) && cursor.Step("Mul", Operand.Input) && cursor.Step("Mul", Operand.Value(0.044715))
        && cursor.Step("Add", Operand.Input) && cursor.Step("Mul", Operand.Value(Math.Sqrt(2 / Math.PI))) && cursor.Step("Tanh")
        && cursor.Step("Add", Operand.Value(1)) && cursor.Step("Mul", Operand.Input) && cursor.Step("Mul", Operand.Value(0.5))
            ? new OnnxLayer(description => description.Gelu(approximate: true), reaching.Flow)
            : null;

    // The same curve, the cube as a power of three and the half before the input.
    private static OnnxLayer? GeluByTanhAsDynamo(Cursor cursor, Reaching reaching) =>
        cursor.Ordered("Pow", Operand.Value(3)) && cursor.Step("Mul", Operand.Value(0.044715)) && cursor.Step("Add", Operand.Input)
        && cursor.Step("Mul", Operand.Value(Math.Sqrt(2 / Math.PI))) && cursor.Step("Tanh") && cursor.Step("Add", Operand.Value(1))
        && cursor.Step("Mul", Operand.Value(0.5)) && cursor.Step("Mul", Operand.Input)
            ? new OnnxLayer(description => description.Gelu(approximate: true), reaching.Flow)
            : null;

    // x · sigmoid(x).
    private static OnnxLayer? Silu(Cursor cursor, Reaching reaching) =>
        cursor.Step("Sigmoid") && cursor.Step("Mul", Operand.Input) ? new OnnxLayer(description => description.Silu(), reaching.Flow) : null;

    // x · tanh(softplus(x)).
    private static OnnxLayer? Mish(Cursor cursor, Reaching reaching) =>
        cursor.Step("Softplus") && cursor.Step("Tanh") && cursor.Step("Mul", Operand.Input) ? new OnnxLayer(description => description.Mish(), reaching.Flow) : null;

    // A Reshape into [0, groups, -1], an InstanceNormalization of a scale of ones and a shift of noughts, a Reshape back — to the
    // input's own Shape as TorchScript writes it, to stated lengths as dynamo does — and the scale and the shift of each channel.
    private static Func<Cursor, Reaching, OnnxLayer?> GroupNorm(bool asTorchScript) => (cursor, reaching) =>
    {
        if (reaching.Flow is not (Flow.Rows or Flow.Images)
            || !cursor.Ordered("Reshape", Operand.Wholes("groups", lengths => lengths is [0, > 0, -1]))
            || !cursor.Ordered("InstanceNormalization", Operand.All(1), Operand.All(0)))
        {
            return null;
        }

        var epsilon = cursor.Number("epsilon", 1e-5);

        if (!(asTorchScript ? cursor.Aside("Shape") && cursor.Ordered("Reshape", Operand.Aside) : cursor.Ordered("Reshape", Operand.Wholes("back", _ => true)))
            || !cursor.Step("Mul", Operand.Held("weight"))
            || !cursor.Step("Add", Operand.Held("bias")))
        {
            return null;
        }

        var groups = (int)cursor.Captured("groups")[1];

        return new OnnxLayer(description => description.GroupNorm(groups, epsilon, default), reaching.Flow)
        {
            Numbers = [cursor.Flat("weight"), cursor.Flat("bias")],
        };
    };

    // x · 1/√(mean(x², over the last axis) + ε) · scale, of a row.
    private static OnnxLayer? RmsNorm(Cursor cursor, Reaching reaching)
    {
        if (reaching.Flow != Flow.Rows || !cursor.Ordered("Pow", Operand.Value(2)) || !cursor.ReducedOverTheLast()
            || !cursor.Step("Add", Operand.Held("epsilon")) || !cursor.Step("Sqrt") || !cursor.Step("Reciprocal")
            || !cursor.Step("Mul", Operand.Input) || !cursor.Step("Mul", Operand.Held("weight")))
        {
            return null;
        }

        if (cursor.Floats("epsilon") is not [var epsilon])
        {
            return null;
        }

        return new OnnxLayer(description => description.RmsNorm(epsilon), reaching.Flow) { Numbers = [cursor.Flat("weight")] };
    }
}

/// <summary>What one input of a node of a spelling must be, beside the value the node before it made.</summary>
internal abstract class Operand
{
    /// <summary>The value the spelling began with: the input of the word.</summary>
    public static Operand Input { get; } = new Of((cursor, name) => name == cursor.Start);

    /// <summary>What the spelling's Shape aside gave.</summary>
    public static Operand Aside { get; } = new Of((cursor, name) => name == cursor.Beside);

    /// <summary>A single number the graph holds, within a millionth of the word's own.</summary>
    public static Operand Value(double expected) =>
        new Of((cursor, name) => cursor.FloatsOf(name) is [var held] && Math.Abs(held - expected) <= 1e-6 * Math.Max(1, Math.Abs(expected)));

    /// <summary>Numbers the graph holds, every one of them the given value.</summary>
    public static Operand All(float value) => new Of((cursor, name) => cursor.FloatsOf(name) is { Length: > 0 } held && held.All(each => each == value));

    /// <summary>Numbers the graph holds, kept under a name for the layer.</summary>
    public static Operand Held(string key) => new Of((cursor, name) => cursor.Capture(key, name));

    /// <summary>Whole numbers the graph holds that the spelling accepts, kept under a name.</summary>
    public static Operand Wholes(string key, Func<long[], bool> accepts) => new Of((cursor, name) => cursor.CaptureWholes(key, name, accepts));

    /// <summary>Whether the input of this name is what the spelling asks for here.</summary>
    public abstract bool Accepts(Cursor cursor, string name);

    private sealed class Of(Func<Cursor, string, bool> accepts) : Operand
    {
        public override bool Accepts(Cursor cursor, string name) => accepts(cursor, name);
    }
}

/// <summary>A spelling being read: the nodes it has taken, the value its last one made, and the numbers it kept.</summary>
internal sealed class Cursor(GraphProto graph, int place, string start, GraphNumbers numbers)
{
    private readonly HashSet<int> _taken = [];
    private readonly List<string> _made = [];
    private readonly List<int> _constants = [];
    private readonly Dictionary<string, HeldTensor> _held = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long[]> _wholes = new(StringComparer.Ordinal);
    private NodeProto? _node;

    /// <summary>The value the spelling began with.</summary>
    public string Start { get; } = start;

    /// <summary>The value its last node made.</summary>
    public string Value { get; private set; } = start;

    /// <summary>The place of its last node; the place before it began, until it takes one.</summary>
    public int Last { get; private set; } = place - 1;

    /// <summary>What its Shape aside gave, when it has one.</summary>
    public string? Beside { get; private set; }

    /// <summary>The places of the constant nodes among the nodes it took.</summary>
    public IReadOnlyList<int> Constants => _constants;

    /// <summary>Takes the next node, when it is of the operator and takes the value the last one made, and each operand besides, in either order.</summary>
    public bool Step(string op, params Operand[] operands) => Took(op, ordered: false, operands);

    /// <summary>Takes the next node, when it is of the operator and takes the value the last one made first, and each operand after it.</summary>
    public bool Ordered(string op, params Operand[] operands) => Took(op, ordered: true, operands);

    private bool Took(string op, bool ordered, Operand[] operands)
    {
        if (Next(op) is not { } node)
        {
            return false;
        }

        var inputs = node.Input.Where(name => name.Length > 0).ToList();
        var at = inputs.IndexOf(Value);

        if (at < 0 || (ordered && at != 0) || inputs.Count != operands.Length + 1)
        {
            return false;
        }

        inputs.RemoveAt(at);

        if (!operands.Select((operand, index) => operand.Accepts(this, inputs[index])).All(accepted => accepted))
        {
            return false;
        }

        Took(node);

        return true;
    }

    /// <summary>Takes the next node, when it is a node of the operator that takes the spelling's input alone, beside the chain.</summary>
    public bool Aside(string op)
    {
        if (Next(op) is not { } node || node.Input is not [var only] || only != Start)
        {
            return false;
        }

        _taken.Add(Last);
        Beside = node.Output[0];
        _made.Add(Beside);

        return true;
    }

    /// <summary>Takes the next node, when it is a ReduceMean of the value the last made over its last axis, keeping that axis.</summary>
    public bool ReducedOverTheLast()
    {
        if (Next("ReduceMean") is not { } node || node.Input.Count == 0 || node.Input[0] != Value)
        {
            return false;
        }

        var axes = node.Input.Count > 1 && node.Input[1].Length > 0
            ? WholesOf(node.Input[1])
            : [.. node.Attribute.Where(attribute => attribute.Name == "axes").SelectMany(attribute => attribute.Ints)];
        var keeps = node.Attribute.FirstOrDefault(attribute => attribute.Name == "keepdims") is not { } keepdims || keepdims.I != 0;

        if (axes is not [-1 or 1] || !keeps)
        {
            return false;
        }

        Took(node);

        return true;
    }

    /// <summary>A number the last node says as an attribute; the given one where it says none.</summary>
    public double Number(string name, double otherwise) =>
        _node?.Attribute.FirstOrDefault(attribute => attribute.Name == name) is { } said ? said.F : otherwise;

    /// <summary>Whether every value made inside the spelling but its last is taken by nothing outside it.</summary>
    public bool Closed()
    {
        var inside = _made.Where(name => name != Value).ToHashSet(StringComparer.Ordinal);

        return !graph.Node.Where((_, at) => !_taken.Contains(at)).Any(node => node.Input.Any(inside.Contains))
            && !graph.Output.Any(output => inside.Contains(output.Name));
    }

    /// <summary>The numbers kept under a name.</summary>
    public float[]? Floats(string key) => _held.TryGetValue(key, out var held) ? held.Values : null;

    /// <summary>The whole numbers kept under a name.</summary>
    public long[] Captured(string key) => _wholes[key];

    /// <summary>Numbers kept under a name as one number for each channel, however many axes of one place the graph writes them with.</summary>
    public LayerNumber Flat(string key)
    {
        var held = _held[key];

        return new LayerNumber(key, held with { Lengths = [held.Lengths.Aggregate(1, (all, length) => all * length)] }, Laying.AsWritten);
    }

    /// <summary>Keeps the numbers of a value under a name, when the graph holds them; false otherwise.</summary>
    public bool Capture(string key, string name)
    {
        if (Held(name) is not { Values: not null } held)
        {
            return false;
        }

        _held[key] = held;

        return true;
    }

    /// <summary>Keeps the whole numbers of a value under a name, when the graph holds them and the spelling accepts them; false otherwise.</summary>
    public bool CaptureWholes(string key, string name, Func<long[], bool> accepts)
    {
        if (WholesOf(name) is not { } wholes || !accepts(wholes))
        {
            return false;
        }

        _wholes[key] = wholes;

        return true;
    }

    /// <summary>The numbers of a value the graph holds, or a constant node of the spelling gives; nothing otherwise.</summary>
    public float[]? FloatsOf(string name) => Held(name)?.Values;

    // The next node that is no constant, when it is of the operator; the constant nodes passed on the way counted as the spelling's.
    private NodeProto? Next(string op)
    {
        var at = Last + 1;
        var passed = new List<int>();

        while (at < graph.Node.Count && graph.Node[at].OpType == "Constant")
        {
            passed.Add(at++);
        }

        if (at >= graph.Node.Count || graph.Node[at].OpType != op || graph.Node[at].Output.Count == 0)
        {
            return null;
        }

        _constants.AddRange(passed);
        _taken.UnionWith(passed);
        Last = at;

        return graph.Node[at];
    }

    private void Took(NodeProto node)
    {
        _node = node;
        _taken.Add(Last);
        Value = node.Output[0];
        _made.Add(Value);
    }

    // A value the graph holds, or one a constant node gives that the walk has not kept yet: read without keeping it, so a spelling
    // that does not hold leaves the graph's values as they were.
    private HeldTensor? Held(string name)
    {
        if (numbers.Holds(name))
        {
            return numbers.Floats(name);
        }

        return Given(name) is { } given ? given.Floats(name) : null;
    }

    private long[]? WholesOf(string name) =>
        (numbers.Holds(name) ? numbers.Wholes(name) : Given(name)?.Wholes(name))?.Values;

    private GraphNumbers? Given(string name)
    {
        if (graph.Node.FirstOrDefault(node => node.OpType == "Constant" && node.Output.Contains(name))?.Attribute.FirstOrDefault(attribute => attribute.Name == "value")?.T is not { } tensor)
        {
            return null;
        }

        var scratch = new GraphNumbers(null);
        scratch.Keep(name, tensor, $"constant '{name}'");

        return scratch;
    }
}
