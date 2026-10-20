// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A layer that reads a series a step at a time, written as ONNX's LSTM or GRU: one node for each layer of the stack, the two directions in
/// the one node, the weights turned round and their gates put in ONNX's order, and the series taken in step by batch by channel, which the
/// graph's channels-first series is turned into first. The last state or every step is taken out of the node's outputs, and the mask a
/// tokenizer's padding makes is written as the lengths of the series — padding stands at the end of a series, where a tokenizer puts it.
/// </summary>
/// <typeparam name="TLayer">The layer's class.</typeparam>
/// <param name="op">ONNX's name for the operator.</param>
/// <param name="order">For each gate in ONNX's order, which gate of the library's order it is.</param>
internal sealed class RecurrentKind<TLayer>(string op, int[] order) : ExportKind<TLayer>
    where TLayer : Recurrent
{
    /// <inheritdoc />
    protected override Value Write(TLayer layer, Writing at)
    {
        var graph = at.Graph;
        var build = layer.Recurrence;
        var both = build.BothWays;
        var current = graph.Node(at.Node("Transpose", 1, "Transpose.in") with { Attributes = node => node.States("perm", 2L, 0L, 1L) }, [at.Input.Name], $"{at.Path}/Transpose.in");
        var lengths = at.Inputs.Count == 2 ? Lengths(at, at.Inputs[1]) : string.Empty;
        string last = string.Empty;

        for (var depth = 0; depth < build.Layers; depth++)
        {
            var name = $"{at.Path}/{op}.l{depth}";
            string[] inputs = lengths.Length == 0 ? [current, Weights(at, layer, depth, "weight_ih"), Weights(at, layer, depth, "weight_hh"), Biases(at, layer, depth)]
                : [current, Weights(at, layer, depth, "weight_ih"), Weights(at, layer, depth, "weight_hh"), Biases(at, layer, depth), lengths];
            var made = graph.Nodes(
                at.Node(op, 7, $"{op}.l{depth}") with
                {
                    Attributes = node =>
                    {
                        node.State("hidden_size", (long)build.Hidden);

                        if (both)
                        {
                            node.State("direction", "bidirectional");
                        }

                        if (op == "GRU")
                        {
                            node.State("linear_before_reset", 1L);
                        }
                    },
                },
                inputs,
                [$"{name}/Y", $"{name}/Y_h"]);

            if (depth == build.Layers - 1 && !build.Sequences)
            {
                last = made[1];
            }
            else
            {
                current = Steps(at, made[0], depth);
            }
        }

        if (!build.Sequences)
        {
            return at.Giving(Final(at, last));
        }

        return at.Giving(graph.Node(at.Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", 1L, 2L, 0L) }, [current], at.Path));
    }

    // The lengths of the series: how many steps of each are steps, counted from the mask.
    private static string Lengths(Writing at, Value mask)
    {
        var counted = at.Graph.Node(
            at.Node("ReduceSum", 1, "ReduceSum.lengths") with
            {
                Attributes = node =>
                {
                    node.States("axes", 1L);
                    node.State("keepdims", 0L);
                },
            },
            [mask.Name],
            $"{at.Path}/ReduceSum.lengths");

        return at.Graph.Node(at.Node("Cast", 6, "Cast.lengths") with { Attributes = node => node.State("to", 6L) }, [counted], $"{at.Path}/Cast.lengths");
    }

    // What a layer made at every step, as the next layer takes it, or the answer is taken from it: step by batch by the states of every direction.
    private string Steps(Writing at, string made, int depth)
    {
        var graph = at.Graph;
        var layer = (Recurrent)at.Layer;

        if (!layer.Recurrence.BothWays)
        {
            return graph.Node(at.Node("Squeeze", 1, $"Squeeze.l{depth}") with { Attributes = node => node.States("axes", 1L) }, [made], $"{at.Path}/Squeeze.l{depth}");
        }

        var turned = graph.Node(at.Node("Transpose", 1, $"Transpose.l{depth}") with { Attributes = node => node.States("perm", 0L, 2L, 1L, 3L) }, [made], $"{at.Path}/Transpose.l{depth}");

        return graph.Node(at.Node("Reshape", 5, $"Reshape.l{depth}"), [turned, graph.Wholes(at.Number($"steps.l{depth}"), 0, 0, 2L * layer.Recurrence.Hidden)], $"{at.Path}/Reshape.l{depth}");
    }

    // The last state of every series: one row of the state of each direction, laid side by side.
    private string Final(Writing at, string states)
    {
        var graph = at.Graph;
        var layer = (Recurrent)at.Layer;

        if (!layer.Recurrence.BothWays)
        {
            return graph.Node(at.Node("Squeeze", 1) with { Attributes = node => node.States("axes", 0L) }, [states], at.Path);
        }

        var turned = graph.Node(at.Node("Transpose", 1, "Transpose.states") with { Attributes = node => node.States("perm", 1L, 0L, 2L) }, [states], $"{at.Path}/Transpose.states");

        return graph.Node(at.Node("Reshape", 5), [turned, graph.Wholes(at.Number("states"), 0, 2L * layer.Recurrence.Hidden)], at.Path);
    }

    // The input weights or the recurrent weights of one layer: for each direction, the gates one block of rows after another, in ONNX's order,
    // each row the weights of one number made, which the library keeps as a column.
    private string Weights(Writing at, Recurrent layer, int depth, string kind)
    {
        var directions = layer.Directions;
        var hidden = layer.Recurrence.Hidden;
        var wide = layer.Gates * hidden;
        var reads = depth == 0 ? layer.Recurrence.Inputs : hidden * directions;
        var inputs = kind == "weight_ih" ? reads : hidden;
        var values = new List<float>(directions * wide * inputs);

        for (var direction = 0; direction < directions; direction++)
        {
            values.AddRange(OnnxLayouts.SlotToGates(Slot(layer, $"{kind}_l{depth}{(direction == 1 ? "_reverse" : string.Empty)}"), inputs, hidden, order));
        }

        return at.Graph.Number(at.Number($"{kind}_l{depth}.onnx"), [directions, wide, inputs], [.. values]);
    }

    // The biases of one layer: for each direction, the input biases and then the recurrent ones, the gates in ONNX's order.
    private string Biases(Writing at, Recurrent layer, int depth)
    {
        var directions = layer.Directions;
        var wide = layer.Gates * layer.Recurrence.Hidden;
        var values = new List<float>(directions * 2 * wide);

        for (var direction = 0; direction < directions; direction++)
        {
            var suffix = $"_l{depth}{(direction == 1 ? "_reverse" : string.Empty)}";

            values.AddRange(OnnxLayouts.SlotToGates(Slot(layer, $"bias_ih{suffix}"), 1, layer.Recurrence.Hidden, order));
            values.AddRange(OnnxLayouts.SlotToGates(Slot(layer, $"bias_hh{suffix}"), 1, layer.Recurrence.Hidden, order));
        }

        return at.Graph.Number(at.Number($"bias_l{depth}.onnx"), [directions, 2 * wide], [.. values]);
    }

    private static float[] Slot(Recurrent layer, string name) => layer.Slots().Single(each => each.Path == name).Slot.Value.Values.ToArray();
}

/// <summary>
/// The mask of a row of ids: a one where an id is a word, a nought where it is padding. Ids are whole numbers from nought, so the mask is the
/// id held between nought and one, ONNX's Clip, from the eleventh operator set where its bounds are inputs.
/// </summary>
internal sealed class UnpaddedKind : ExportKind<Unpadded>
{
    /// <inheritdoc />
    protected override Value Write(Unpadded layer, Writing at) =>
        at.Giving(at.Graph.Node(
            at.Node("Clip", 11),
            [at.Input.Name, at.Graph.Number(at.Number("least"), [], [0f]), at.Graph.Number(at.Number("most"), [], [1f])],
            at.Path));
}

/// <summary>
/// A layer run over every step of a series: the series turned to step by channel, its steps laid out as rows, the layer written once over
/// the rows, and what it made put back as the series — the channels first again.
/// </summary>
internal sealed class EveryStepKind : ExportKind<EveryStep>
{
    /// <inheritdoc />
    protected override Value Write(EveryStep layer, Writing at)
    {
        var graph = at.Graph;
        var steps = at.Input.Each[0];
        var channels = at.Input.Each[at.Input.Each.Rank - 1];
        var turned = graph.Node(at.Node("Transpose", 1, "Transpose.in") with { Attributes = node => node.States("perm", 0L, 2L, 1L) }, [at.Input.Name], $"{at.Path}/Transpose.in");
        var rows = graph.Node(at.Node("Reshape", 5, "Reshape.in"), [turned, graph.Wholes(at.Number("rows"), -1, channels)], $"{at.Path}/Reshape.in");
        var each = new Shape(at.Made.Axes[1..].ToArray());

        if (each.Rank != 1)
        {
            throw new NotSupportedException($"A layer run over every step makes {each} of each, and the graph lays the steps back as rows of numbers: only a layer that makes a row of each step is written.");
        }

        var inside = $"{at.Path}.layer";
        var made = layer.Layer is LayerStack stack
            ? new StackWriting(graph).Write(stack, [new Value(rows, new Shape(channels))], inside)
            : ExportKinds.For(layer.Layer).Write(new Writing(graph, layer.Layer, inside, [new Value(rows, new Shape(channels))], each));
        var back = graph.Node(at.Node("Reshape", 5, "Reshape.out"), [made.Name, graph.Wholes(at.Number("series"), -1, steps, each[0])], $"{at.Path}/Reshape.out");

        return at.Giving(graph.Node(at.Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", 0L, 2L, 1L) }, [back], at.Path));
    }
}
