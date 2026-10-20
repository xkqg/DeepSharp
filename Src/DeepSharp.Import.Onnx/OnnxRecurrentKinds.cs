// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// ONNX's <c>LSTM</c> and <c>GRU</c> over a series taken step by batch by channel: one layer here, the weights turned round and their
/// gates put in this library's order, in one direction or both, giving either every step or the last state.
/// </summary>
/// <remarks>
/// A node is read when it starts from nought — the state a graph spells out as an Expand of a constant nought — and reads series of
/// one length; lengths for each series of a batch, another starting state, other activations, a clip, peepholes, a coupled forget gate and a
/// batch-first layout are refused by name. A node whose steps and last states are both taken is refused, for a layer makes one of them.
/// </remarks>
internal abstract class RecurrentOnnxKind : OnnxKind
{
    /// <summary>How many gates the operator has: four for an LSTM, three for a GRU.</summary>
    protected abstract int Gates { get; }

    /// <summary>For each gate in ONNX's order, which gate of this library's order it is.</summary>
    protected abstract int[] Order { get; }

    /// <summary>The word the layer is written as.</summary>
    /// <param name="description">The description it is written into.</param>
    /// <param name="hidden">How many numbers each direction's state holds.</param>
    /// <param name="ways">Whether it reads each series both ways, and whether it gives every step.</param>
    protected abstract void Write(Sequential description, int hidden, Ways ways);

    /// <summary>Notes a fault at a node for what only one of the operators states; nothing to note by default.</summary>
    /// <param name="node">The node.</param>
    protected virtual void Check(OnnxNode node)
    {
    }

    /// <inheritdoc />
    public override OnnxLayer Read(OnnxNode node, Reaching reaching, GraphNumbers numbers)
    {
        var before = node.Faults;

        if (reaching.Flow != Flow.TimeMajor)
        {
            node.Refuse($"it takes a series one step after another, step by batch by channel, as ONNX's {Name} does, and what reaches it is not laid out so.");
        }

        var hidden = node.Whole("hidden_size", 0);
        var direction = node.Text("direction", "forward");
        var directions = direction == "bidirectional" ? 2 : 1;
        var steps = node.Taken(0) is not null;
        var last = node.Taken(1) is not null;

        Refusals(node, hidden, direction, new Taking(steps, last));

        if (node.Faults > before)
        {
            return new OnnxLayer(null, reaching.Flow);
        }

        var ways = new Ways(directions == 2, steps);

        return new OnnxLayer(description => Write(description, hidden, ways), steps ? Flow.Directed : Flow.States)
        {
            Width = hidden,
            Directions = directions,
            OutputAt = steps ? 0 : 1,
            Numbers = Held(node, numbers, hidden, directions),
        };
    }

    // Everything about the node that no layer here does, each noted at the node.
    private void Refusals(OnnxNode node, int hidden, string direction, Taking taken)
    {
        if (hidden < 1)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"its hidden_size is {hidden}, and a recurrent layer holds a state of one number or more."));
        }

        if (direction is not ("forward" or "bidirectional"))
        {
            node.Refuse($"it reads in the direction '{direction.Quoted()}', and a recurrent layer here reads forward or both ways, forward and back, laid side by side.");
        }

        if (node.Whole("layout", 0) != 0)
        {
            node.Refuse("it takes its series batch first, layout 1, and a recurrent layer here is read from ONNX's own layout, the steps first.");
        }

        if (node.Input(4).Length > 0)
        {
            node.Refuse($"it is given the lengths of its series, sequence_lens '{node.Input(4).Quoted()}', and a recurrent layer here reads series of one length.");
        }

        if (node.Input(7).Length > 0)
        {
            node.Refuse("it has peepholes, and a recurrent layer here has none.");
        }

        if (node.Says("activations") || node.Says("activation_alpha") || node.Says("activation_beta") || node.Says("clip") || node.Whole("input_forget", 0) != 0)
        {
            node.Refuse("it states activations, a clip or a coupled forget gate, and a recurrent layer here has the activations, the unclipped gates and the uncoupled forget gate its operator starts from.");
        }

        if (taken.Steps == taken.Last)
        {
            node.Refuse(taken.Steps
                ? "it gives its steps and its last states both to nodes, and a recurrent layer here makes one of them."
                : "it gives nothing a node or the graph takes, and a recurrent layer here makes its steps or its last states.");
        }

        if (node.Taken(2) is not null)
        {
            node.Refuse("it gives its last cell state to a node, and a recurrent layer here gives out the state it carries and not the cell.");
        }

        Check(node);
    }

    // The numbers of the node for the slots of a layer, a direction at a time: weights in, weights between steps, and the two biases.
    private List<LayerNumber> Held(OnnxNode node, GraphNumbers numbers, int hidden, int directions)
    {
        var wide = Gates * hidden;
        var kept = new List<LayerNumber>();
        var read = numbers.Floats(node.Input(1));
        var between = numbers.Floats(node.Input(2));
        var bias = node.Input(3).Length > 0 ? numbers.Floats(node.Input(3)) : (HeldTensor?)null;
        var inputs = read.Lengths is [_, _, var channels] ? channels : 0;

        Shaped(node, read, [directions, wide, inputs], "weights W");
        Shaped(node, between, [directions, wide, hidden], "recurrent weights R");

        if (bias is { } given)
        {
            Shaped(node, given, [directions, 2 * wide], "biases B");
        }

        for (var direction = 0; direction < directions; direction++)
        {
            var suffix = $"_l0{(direction == 1 ? "_reverse" : string.Empty)}";

            kept.Add(Weighed($"weight_ih{suffix}", read, direction, new Reads(inputs, hidden)));
            kept.Add(Weighed($"weight_hh{suffix}", between, direction, new Reads(hidden, hidden)));
            kept.Add(Biased($"bias_ih{suffix}", bias, direction * 2, hidden));
            kept.Add(Biased($"bias_hh{suffix}", bias, (direction * 2) + 1, hidden));
        }

        return kept;
    }

    // A held value written as the lengths the operator takes, or its fault noted at the node.
    private static void Shaped(OnnxNode node, HeldTensor held, int[] lengths, string what)
    {
        if (held.Values is not null && !held.Lengths.AsSpan().SequenceEqual(lengths))
        {
            node.Refuse($"its {what}, '{held.Source.Quoted()}', are written as {Written(held.Lengths)}, and the operator takes {Written(lengths)}.");
        }
    }

    // One direction's weights, from the gates' rows of ONNX's order to the slot's inputs by gates, in the library's gate order.
    private LayerNumber Weighed(string slot, HeldTensor held, int direction, Reads reads)
    {
        var (inputs, hidden) = reads;
        var wide = Gates * hidden;
        var each = wide * inputs;

        if (held.Values is not { } values || values.Length < (direction + 1) * each)
        {
            return new LayerNumber(slot, held with { Values = null, Fault = held.Fault ?? "is not as many numbers as the operator takes." }, Laying.AsWritten);
        }

        var laid = OnnxLayouts.GatesToSlot(values[(direction * each)..((direction + 1) * each)], inputs, hidden, Order);

        return new LayerNumber(slot, new HeldTensor(held.Source, [inputs, wide]) { Values = laid }, Laying.AsWritten);
    }

    // One of a direction's two biases; nought where the node leaves its biases out, as ONNX does.
    private LayerNumber Biased(string slot, HeldTensor? held, int block, int hidden)
    {
        var wide = Gates * hidden;

        if (held is null)
        {
            return new LayerNumber(slot, new HeldTensor("the operator's default", [wide]) { Values = new float[wide] }, Laying.AsWritten);
        }

        if (held.Value.Values is not { } values || values.Length < (block + 1) * wide)
        {
            return new LayerNumber(slot, held.Value with { Values = null, Fault = held.Value.Fault ?? "is not as many numbers as the operator takes." }, Laying.AsWritten);
        }

        var laid = OnnxLayouts.GatesToSlot(values[(block * wide)..((block + 1) * wide)], 1, hidden, Order);

        return new LayerNumber(slot, new HeldTensor(held.Value.Source, [wide]) { Values = laid }, Laying.AsWritten);
    }
}

/// <summary>Which of its outputs a recurrent node gives to something.</summary>
/// <param name="Steps">Whether the state at every step is taken.</param>
/// <param name="Last">Whether the last state is taken.</param>
internal readonly record struct Taking(bool Steps, bool Last);

/// <summary>What one block of a recurrent operator's weights reads, and how many numbers each of its gates makes.</summary>
/// <param name="Inputs">The numbers read.</param>
/// <param name="Hidden">The numbers each gate makes.</param>
internal readonly record struct Reads(int Inputs, int Hidden);

/// <summary>How a recurrent layer reads its series, and what it gives.</summary>
/// <param name="Both">Whether it reads each series both ways.</param>
/// <param name="Steps">Whether it gives the state at every step, rather than the last.</param>
internal readonly record struct Ways(bool Both, bool Steps);

/// <summary>ONNX's <c>LSTM</c>: input, output, forget and cell gates, in that order.</summary>
internal sealed class LstmKind : RecurrentOnnxKind
{
    /// <inheritdoc />
    public override string Name => "LSTM";

    /// <inheritdoc />
    protected override int Gates => 4;

    /// <inheritdoc />
    protected override int[] Order => [0, 3, 1, 2];

    /// <inheritdoc />
    protected override void Write(Sequential description, int hidden, Ways ways) => description.Lstm(hidden, ways.Both, 1, ways.Steps);
}

/// <summary>ONNX's <c>GRU</c>: update, reset and hidden gates, in that order, resetting after the product as PyTorch's does.</summary>
internal sealed class GruKind : RecurrentOnnxKind
{
    /// <inheritdoc />
    public override string Name => "GRU";

    /// <inheritdoc />
    protected override int Gates => 3;

    /// <inheritdoc />
    protected override int[] Order => [1, 0, 2];

    /// <inheritdoc />
    protected override void Write(Sequential description, int hidden, Ways ways) => description.Gru(hidden, ways.Both, 1, ways.Steps);

    /// <inheritdoc />
    protected override void Check(OnnxNode node)
    {
        if (node.Whole("linear_before_reset", 0) != 1)
        {
            node.Refuse("it resets before the product of the hidden state, linear_before_reset 0, and a recurrent layer here resets after it, as PyTorch's does.");
        }
    }
}
