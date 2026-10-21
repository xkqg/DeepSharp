// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Import.Onnx;

/// <summary>The part of reading a graph that follows a recurrent operator: the state it starts from, and the nodes that put what it gave in order.</summary>
internal sealed partial class OnnxGraph
{
    // The operators that make each number from the number at its place, so that the layout of the value they work on does not matter.
    private static readonly HashSet<string> Pointwise = new(["Relu", "Tanh", "Sigmoid", "Gelu", "LeakyRelu", "Elu", "Selu", "Mish"], StringComparer.Ordinal);

    // The places of the nodes that work out the nought a recurrent node starts from, and the node that makes each value of the graph.
    private HashSet<int> _started = [];
    private readonly Dictionary<string, int> _makers = new(StringComparer.Ordinal);

    // Whether a value is laid out as ONNX's recurrent operators take and give a series, and not as the batch-first series here.
    private static bool Recurring(Flow flow) => flow is Flow.TimeMajor or Flow.Directed or Flow.DirectedByBatch or Flow.States or Flow.StatesByBatch;

    // Whether it is one of the layouts that only a recurrent operator's result is laid out in.
    private static bool Stepwise(Flow flow) => Recurring(flow) && flow != Flow.TimeMajor;

    // What a Transpose hands on of a series, or of what a recurrent operator gave: a series taken step by batch, as the first Transpose of the
    // graph's input declares when it is channels last; the series put back batch first; the steps with the batch before the directions; and
    // the last states likewise. Nothing for any other turn.
    private Flow? Sequenced(int[] turn, Reaching reaching, bool undeclared)
    {
        var flow = reaching.Flow;

        if (flow == Flow.Images && reaching.Axes == 1)
        {
            if (turn is [2, 0, 1])
            {
                return Flow.TimeMajor;
            }

            if (turn is [1, 0, 2] && undeclared)
            {
                _channelsLast = true;

                return Flow.TimeMajor;
            }

            return null;
        }

        if (flow == Flow.TimeMajor && turn is [1, 2, 0])
        {
            return Flow.Images;
        }

        if (flow == Flow.TimeMajor && turn is [1, 0, 2])
        {
            return Flow.ImagesLast;
        }

        if (flow == Flow.Directed && turn is [0, 2, 1, 3])
        {
            return Flow.DirectedByBatch;
        }

        return flow == Flow.States && turn is [1, 0, 2] ? Flow.StatesByBatch : null;
    }

    // The nodes that only work out the state a recurrent node starts from: its Expands of a constant nought, and what only they take. A node
    // that is also taken by something else, or given by the graph, is not among them.
    private HashSet<int> Starting()
    {
        var takers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var given = _graph.Output.Select(output => output.Name).ToHashSet(StringComparer.Ordinal);

        for (var place = 0; place < _graph.Node.Count; place++)
        {
            foreach (var output in _graph.Node[place].Output.Where(name => name.Length > 0))
            {
                _makers[output] = place;
            }

            foreach (var input in _graph.Node[place].Input.Where(name => name.Length > 0))
            {
                takers.TryAdd(input, []);
                takers[input].Add(place);
            }
        }

        var cone = new HashSet<int>();
        var work = new Stack<int>();

        var states = _graph.Node
            .Where(node => node.OpType is "LSTM" or "GRU")
            .SelectMany(node => node.Input.Skip(5).Take(2))
            .Where(name => name.Length > 0 && _makers.ContainsKey(name))
            .Select(name => _makers[name])
            .Where(maker => _graph.Node[maker].OpType == "Expand");

        foreach (var maker in states.Where(cone.Add))
        {
            work.Push(maker);
        }

        while (work.TryPop(out var place))
        {
            var only = _graph.Node[place].Input
                .Where(name => name.Length > 0 && _makers.ContainsKey(name))
                .Select(name => _makers[name])
                .Where(maker => !cone.Contains(maker) && _graph.Node[maker].OpType != "Constant"
                    && _graph.Node[maker].Output.All(output => !given.Contains(output) && takers.GetValueOrDefault(output, []).All(cone.Contains)));

            foreach (var maker in only.Where(cone.Add))
            {
                work.Push(maker);
            }
        }

        return cone;
    }

    // A recurrent node starts from nought: the states it is given are an Expand of a constant that holds nothing but noughts.
    private void Started(OnnxNode node)
    {
        for (var at = 5; at <= 6; at++)
        {
            var name = node.Input(at);

            if (name.Length > 0 && !(_makers.TryGetValue(name, out var maker) && _graph.Node[maker].OpType == "Expand" && Noughts(_graph.Node[maker].Input[0])))
            {
                node.Refuse($"its {(at == 5 ? "initial_h" : "initial_c")}, '{name.Quoted()}', is a state the graph works out or holds, and a recurrent layer here starts from nought: the state an Expand of a constant nought spells.");
            }
        }
    }

    private bool Noughts(string name) => _numbers.Holds(name) && _numbers.Floats(name).Values is { } values && values.All(value => value is 0f);

    // A node that works on a value laid out as a recurrent operator gives or takes a series: the Squeeze, the Gather and the Reshape that put
    // the result of one in order, which are followed without a layer of their own; a recurrent operator or an activation, which is read as
    // a layer; and any other, which is refused. False when the node is one to be read as a layer.
    private bool Stepped(OnnxNode node, Made made, string taken)
    {
        if (node.Operator is "LSTM" or "GRU" || Pointwise.Contains(node.Operator))
        {
            return false;
        }

        var flow = made.Reaching.Flow;

        if (node.Operator == "Squeeze" && flow is Flow.Directed or Flow.States)
        {
            Squeezed(node, made);
        }
        else if (node.Operator == "Gather" && flow == Flow.States && node.Input(0) == taken)
        {
            Gathered(node, made);
        }
        else if (node.Operator == "Reshape" && flow is Flow.DirectedByBatch or Flow.StatesByBatch)
        {
            Folded(node, made);
        }
        else
        {
            node.Refuse("it works on a series laid out as ONNX's recurrent operators take and give one, and a node there is read as a Transpose, a Squeeze, a Gather or a Reshape that puts what a recurrent layer gave in order, as an activation, or as another recurrent operator.");
            _values[node.Output] = made;
        }

        return true;
    }

    // The direction's axis of what a layer that reads one way gave dropped: the steps become a series laid out step by batch, the last states rows.
    private void Squeezed(OnnxNode node, Made made)
    {
        var reaching = made.Reaching;
        var steps = reaching.Flow == Flow.Directed;
        var axis = steps ? 1 : 0;
        var rank = steps ? 4 : 3;
        var axes = node.Axes(_numbers);

        if (axes is not null && !(axes is [var only] && (only == axis || only == axis - rank) && reaching.Directions == 1))
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"it drops axes [{string.Join(", ", axes)}] of what a recurrent layer gave, and a network here drops the axis of the direction, {axis}, of a layer that reads one way."));
        }

        _values[node.Output] = made with { Reaching = steps ? reaching with { Flow = Flow.TimeMajor } : new Reaching(Flow.Rows, reaching.Batch, 0) };
    }

    // The last state of a layer that reads one way: the one state along the first axis of the states, from the front or the back.
    private void Gathered(OnnxNode node, Made made)
    {
        var reaching = made.Reaching;
        var indices = _numbers.Wholes(node.Input(1));

        if (node.Whole("axis", 0) != 0 || reaching.Directions != 1 || indices.Values is not [0 or -1])
        {
            node.Refuse("it picks states out of those of a layer's directions, and a network here takes the last states of a layer that reads one way as they are: the one state of its one direction.");
        }

        _values[node.Output] = made with { Reaching = new Reaching(Flow.Rows, reaching.Batch, 0) };
    }

    // The states of the directions laid side by side by a Reshape that keeps the steps and the batch: a series, or rows of the last states.
    private void Folded(OnnxNode node, Made made)
    {
        var reaching = made.Reaching;
        var steps = reaching.Flow == Flow.DirectedByBatch;
        var width = made.Width is { } hidden ? hidden * reaching.Directions : (int?)null;
        var target = _numbers.Wholes(node.Input(1));
        var copies = node.Whole("allowzero", 0) == 0;

        if (target.Values is not { } lengths)
        {
            node.Refuse($"its target, '{node.Input(1).Quoted()}', {target.Fault}");
        }
        else if (!(steps ? FoldsSteps(lengths, reaching.Batch, copies, width) : FoldsStates(lengths, reaching.Batch, copies, width)))
        {
            node.Refuse($"it lays what a recurrent layer gave out as [{string.Join(", ", lengths)}], and a network here lays the states of the directions side by side and keeps the {(steps ? "steps and the batch: [0, 0, -1]" : "batch: [0, -1]")}.");
        }

        _values[node.Output] = made with
        {
            Reaching = steps ? reaching with { Flow = Flow.TimeMajor, Directions = 1 } : new Reaching(Flow.Rows, reaching.Batch, 0),
            Width = width,
        };
    }

    // [steps, batch, states] with the steps and the batch kept and the states' width stated or left to be worked out.
    private static bool FoldsSteps(long[] lengths, long? batch, bool copies, int? width) =>
        lengths is [var steps, var each, var states] && copies && steps == 0 && (each == 0 || each == batch) && (states == -1 || states == width);

    // [batch, states] likewise.
    private static bool FoldsStates(long[] lengths, long? batch, bool copies, int? width)
    {
        if (lengths is not [var each, var states])
        {
            return false;
        }

        var batchKept = (copies && each == 0) || each == batch || each == -1;
        var widthKept = states == -1 || states == width;

        return batchKept && widthKept && (each != -1 || states != -1);
    }
}
