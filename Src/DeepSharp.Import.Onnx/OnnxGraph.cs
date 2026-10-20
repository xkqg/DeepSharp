// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;
using static Onnx.TensorShapeProto.Types;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// An ONNX graph read as a stack of layers: the one batch it takes, each node written in the words it is, in order, each
/// taking the value the one before it made, and its numbers laid out and put into the network's slots — or every fault
/// that keeps it from being one, each where the graph says it.
/// </summary>
/// <remarks>
/// The graph is read in two sweeps, each naming every fault it finds at once: what the graph says — its input, its nodes and
/// its output — and then, once its words are lowered and an example of nothing has been sent through the layers to find
/// the shape reaching each, what can only be told with those shapes: a reshape's row, and the image each flattened row was.
/// </remarks>
internal sealed partial class OnnxGraph
{
    private readonly List<string> _faults = [];
    private readonly Sequential _description = new();
    private readonly GraphProto _graph;
    private readonly GraphNumbers _numbers;
    private readonly Loss _loss;
    private readonly string _where;
    private const int TheInput = -1;

    // The names of the values something takes: a node's input, or what the graph gives.
    private readonly HashSet<string> _used;

    // Whether the image the graph takes has its channels last, as the Transpose that first takes it declares.
    private bool _channelsLast;

    // The nodes read so far, and the value each of the graph's names stands for: the node that made it and how it is laid out.
    private readonly List<Built> _built = [];
    private readonly Dictionary<string, Made> _values = new(StringComparer.Ordinal);

    // How many nodes take each value, and the operator of the last node that is not an identity, a cast or a transpose.
    private readonly Dictionary<string, int> _consumers;
    private string _previous = string.Empty;

    /// <summary>A graph, where its numbers are, and the loss the network it becomes answers through.</summary>
    public OnnxGraph(GraphProto graph, GraphNumbers numbers, Loss loss)
    {
        _graph = graph;
        _numbers = numbers;
        _loss = loss;
        _where = $"graph '{graph.Name.Quoted()}'";
        _used = [.. graph.Node.SelectMany(node => node.Input), .. graph.Output.Select(output => output.Name)];
        _consumers = graph.Node.SelectMany(node => node.Input).Where(name => name.Length > 0).GroupBy(name => name, StringComparer.Ordinal).ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
    }

    /// <summary>The network the graph is, every slot holding the number the graph holds for it, and the loss handed in.</summary>
    /// <exception cref="FormatException">The graph says what no network here is built of: every such thing at once, each where it says it.</exception>
    /// <exception cref="SlotLoadException">The numbers do not fit the network: every fault at once, each at the initializer that holds the number.</exception>
    public SavedNetwork Loaded()
    {
        foreach (var initializer in _graph.Initializer.Where(initializer => !_numbers.Keep(initializer.Name, initializer, $"initializer '{initializer.Name}'")))
        {
            _faults.Add($"initializer '{initializer.Name.Quoted()}': the graph holds a value of that name already.");
        }

        var input = Input();
        var built = Walked(input);
        var placed = Written(built);

        if (_faults.Count > 0)
        {
            throw Refused();
        }

        var example = ExampleOf(input.Lengths!);
        var network = Lowered(example);
        var entries = new List<SlotEntry>();
        var unread = new List<SlotLoadFault>();
        var reaching = network.Shapes(example).Select(shapes => shapes.Reaching).ToList();
        ImageRow? flattened = null;

        foreach (var (place, node, layer) in placed)
        {
            var each = reaching[place];

            if (layer.Flattens)
            {
                flattened = new ImageRow(each);
            }

            if (layer.RowLength is { } length && length != each.Count)
            {
                node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it lays each example out as a row of {length} values, and each example reaching it holds {each.Count}."));
            }

            if (layer.Declared?.Mismatch(each) is { } fault)
            {
                node.Refuse(fault);
            }

            foreach (var number in layer.Numbers)
            {
                var path = string.Create(CultureInfo.InvariantCulture, $"{place}.{number.Slot}");

                if (number.Held.Values is null)
                {
                    unread.Add(new SlotLoadFault(number.Held.Source, path, $"'{path}' {number.Held.Fault}"));

                    continue;
                }

                var laid = number.Held.LaidOut(number.Laying, number.AlongFlattenedImage ? flattened : null);
                entries.Add(new SlotEntry(path, Tensor.From(new Shape(laid.Lengths), laid.Values!), laid.Source));
            }
        }

        if (_faults.Count > 0)
        {
            throw Refused();
        }

        return Loaded(network, entries, unread);
    }

    // The numbers put into the network, those the graph holds wrongly refused besides, each once: a number of another type
    // is refused as that, not also as missing.
    private SavedNetwork Loaded(LayerStack network, List<SlotEntry> entries, List<SlotLoadFault> unread)
    {
        var faults = new List<SlotLoadFault>(unread);

        try
        {
            network.Load(entries);
        }
        catch (SlotLoadException refused)
        {
            faults.AddRange(refused.Faults.Where(fault => fault.Source is not null || unread.All(other => other.Slot != fault.Slot)));
        }

        return faults.Count > 0 ? throw new SlotLoadException(faults) : new SavedNetwork(network, _loss);
    }

    // The one value the graph takes: a batch of rows, of series, of images or of volumes, each example of lengths the graph
    // states. Its fault is noted when there is not one, or it is none of those.
    private Taken Input()
    {
        var data = _graph.Input.Where(value => !_numbers.Holds(value.Name)).ToList();

        if (data.Count != 1)
        {
            _faults.Add(string.Create(CultureInfo.InvariantCulture, $"{_where}: it takes {data.Count} values, and a network here takes one, the batch of its examples."));

            return new Taken(data.FirstOrDefault()?.Name ?? string.Empty, new Reaching(Flow.Rows, null, 0), null);
        }

        var input = data[0];
        var tensor = input.Type?.TensorType;
        var type = tensor?.ElemType ?? 0;

        if (type != (int)TensorProto.Types.DataType.Float)
        {
            _faults.Add($"input '{input.Name.Quoted()}': it takes {GraphNumbers.TypeName(type)} values, and a network here takes single-precision numbers, FLOAT.");

            return new Taken(input.Name, new Reaching(Flow.Rows, null, 0), null);
        }

        var dims = tensor!.Shape?.Dim.ToArray() ?? [];
        var written = string.Join(", ", dims.Select(Written)).Quoted();

        if (dims.Length is < 2 or > 5)
        {
            _faults.Add(string.Create(
                CultureInfo.InvariantCulture,
                $"input '{input.Name.Quoted()}': it takes a batch of rank {dims.Length}, [{written}], and a network here takes rows (a batch of values), series (a batch of channels by steps), images (a batch of channels by rows by columns) or volumes (a batch of channels by planes by rows by columns)."));

            return new Taken(input.Name, new Reaching(Flow.Rows, null, 0), null);
        }

        if (dims.Skip(1).Any(dim => dim.ValueCase != Dimension.ValueOneofCase.DimValue || dim.DimValue is < 1 or > int.MaxValue))
        {
            _faults.Add($"input '{input.Name.Quoted()}': its shape, [{written}], does not state every length of an example, and a network here is built for examples of one shape.");

            return new Taken(input.Name, new Reaching(Flow.Rows, null, 0), null);
        }

        var lengths = dims.Skip(1).Select(dim => (int)dim.DimValue).ToArray();
        long? batch = dims[0].ValueCase == Dimension.ValueOneofCase.DimValue ? dims[0].DimValue : null;

        return new Taken(input.Name, new Reaching(lengths.Length == 1 ? Flow.Rows : Flow.Images, batch, lengths.Length - 1), lengths);
    }

    // The shape of one example here, from the lengths the graph states after the batch. Series, images and volumes here have
    // their channels last: one the graph takes channels first — as ONNX's Conv does — has its channels moved last, and an
    // image it takes channels last, as its first Transpose declares, is taken as it is.
    private Shape ExampleOf(int[] lengths) =>
        lengths.Length == 1 || _channelsLast ? new Shape(lengths) : new Shape([.. lengths[1..], lengths[0]]);

    // The graph's nodes in order, each read as the step it is: the layer it becomes, working on the value of the node that made what
    // it names — the one before it, or an earlier one. Constants kept, and identities, casts and transposes followed on the way; an
    // Add of a held value made the bias of the MatMul before it, an Add of two values the sum of two steps, a Concat of several
    // the steps laid side by side, and a Split or a Slice the columns it cuts. The graph's output is held to the last value made.
    private List<Built> Walked(Taken input)
    {
        _values[input.Value] = new Made(TheInput, input.Reaching, input.Reaching.Flow == Flow.Images, input.Lengths is [var width] ? width : null);
        _started = Starting();

        var flowing = input.Value;
        var last = _values[flowing];

        for (var place = 0; place < _graph.Node.Count; place++)
        {
            var node = new OnnxNode(_graph.Node[place], place, _faults, _used);

            if (node.Operator == "Constant")
            {
                Constant(node);

                continue;
            }

            if (_started.Contains(place))
            {
                continue;
            }

            // A recurrent node takes one value the graph works out, the series; its weights are held, and its states are read apart.
            var data = (node.Operator is "LSTM" or "GRU" ? node.Inputs.Take(1) : node.Inputs).Where(name => name.Length > 0 && !_numbers.Holds(name)).ToList();

            if (node.Operator == "Identity" && data.Count == 0 && _numbers.Holds(node.Input(0)))
            {
                _numbers.Alias(node.Output, node.Input(0));

                continue;
            }

            var from = data.Count == 1 && _values.ContainsKey(data[0]) ? data[0] : flowing;

            // A word an exporter spelled out of several operators, read as the one layer it is.
            if (OnnxSpellings.Read(_graph, _numbers, new SpellingAt(place, from, _values.GetValueOrDefault(from, last).Reaching)) is { } spelled)
            {
                foreach (var at in spelled.Constants)
                {
                    Constant(new OnnxNode(_graph.Node[at], at, _faults, _used));
                }

                var source = _values.GetValueOrDefault(from, last);

                _built.Add(new Built(node, spelled.Layer, false) { Sources = [source.Step] });
                _values[spelled.Output] = new Made(_built.Count - 1, source.Reaching with { Flow = spelled.Layer.After }, false, WidthOf(node, spelled.Layer, source));
                _previous = node.Operator;
                flowing = spelled.Output;
                last = _values[flowing];
                place = spelled.Last;

                continue;
            }

            var unmade = data.Where(name => !_values.ContainsKey(name)).ToList();

            if (unmade.Count > 0)
            {
                node.Refuse(NotMade(unmade));
                _previous = node.Operator;
            }
            else if (data.Count == 0)
            {
                node.Refuse("it takes none of the values the graph works out, which are its input and what a node before it made.");
                _previous = node.Operator;
            }
            else if (node.Operator == "Add" && data.Count == 2)
            {
                Joined(node, data, Joining.Add);
            }
            else if (node.Operator == "Concat")
            {
                Joined(node, data, Joining.Concatenate);
            }
            else if (node.Operator is "Split" or "Slice")
            {
                Cut(node, data);
            }
            else if (data.Count != 1)
            {
                node.Refuse(Several(node, data));
                _previous = node.Operator;
            }
            else
            {
                Single(node, place, data[0]);
            }

            // A node refused gives what the one before it gave, so that what takes it is not refused again for it.
            if (node.Output.Length > 0 && !_values.ContainsKey(node.Output))
            {
                _values[node.Output] = last;
            }

            flowing = node.Output;
            last = _values.GetValueOrDefault(flowing, last);
        }

        Output(flowing, last.Reaching.Flow);

        return _built;
    }

    // A node that works on the one value it takes: one the graph follows without a step, or the layer it is.
    private void Single(OnnxNode node, int place, string taken)
    {
        var made = _values[taken];
        var reaching = made.Reaching;

        if (node.Operator == "Transpose")
        {
            _values[node.Output] = made with { Reaching = reaching with { Flow = Turned(node, reaching, made.Undeclared) }, Undeclared = false };
        }
        else if (node.Operator == "Cast")
        {
            Cast(node);
            _values[node.Output] = made;
        }
        else if (node.Operator == "Unsqueeze" && reaching is { Flow: Flow.Images, Axes: 1 } && _graph.OpensLiftedPooling(place, _numbers))
        {
            _values[node.Output] = made with { Reaching = reaching with { Lifted = true } };
        }
        else if (node.Operator == "Squeeze" && reaching.Lifted)
        {
            _values[node.Output] = made with { Reaching = reaching with { Lifted = false } };
        }
        else if (node.Operator == "Identity")
        {
            _values[node.Output] = made;
        }
        else if (Recurring(reaching.Flow) && Stepped(node, made, taken))
        {
            _previous = node.Operator;
        }
        else
        {
            var before = _faults.Count;

            if (node.Operator is "LSTM" or "GRU")
            {
                Started(node);
            }

            var layer = OnnxKind.ByName.TryGetValue(node.Operator, out var kind) ? kind.Read(node, reaching, _numbers) : Unknown(node, reaching);

            if (layer.Adds is { } added)
            {
                Biased(node, added, made.Step, taken);
                _values[node.Output] = made with { Undeclared = false };
            }
            else
            {
                _built.Add(new Built(node, layer, _faults.Count > before) { Sources = [made.Step] });

                var given = layer.OutputAt < node.Outputs.Count ? node.Outputs[layer.OutputAt] : node.Output;
                _values[given] = new Made(_built.Count - 1, reaching with { Flow = layer.After, Directions = layer.Directions ?? reaching.Directions }, false, WidthOf(node, layer, made));
                _values.TryAdd(node.Output, _values[given]);
            }

            _previous = node.Operator;
        }
    }

    // Two values or more taken by one node, the steps that made them become one: an Add of two is their sum, a Concat of several
    // lays them side by side along the last axis. Values laid out otherwise are not joined.
    private void Joined(OnnxNode node, List<string> data, Joining join)
    {
        var parts = data.Select(name => _values[name]).ToList();
        var first = parts[0].Reaching;

        if (join == Joining.Concatenate && (data.Count < 2 || node.Inputs.Count(name => name.Length > 0) != data.Count))
        {
            node.Refuse("it lays side by side two values or more that nodes made, and no value the graph holds.");
        }
        else if (parts.Any(part => part.Reaching.Flow != first.Flow || part.Reaching.Axes != first.Axes || part.Reaching.Lifted || Stepwise(part.Reaching.Flow)))
        {
            node.Refuse("it takes values laid out otherwise, and a network here joins values laid out alike.");
        }
        else if (join == Joining.Concatenate && !IsLastAxis(first, node.Whole("axis", int.MinValue)))
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"it lays its values side by side along axis {node.Whole("axis", 0)}, and a network here lays them side by side along their last axis: a row's second, an image's channels."));
        }
        else
        {
            _built.Add(new Built(node, new OnnxLayer(null, first.Flow) { Join = join }, false) { Sources = [.. parts.Select(part => part.Step)] });
            _values[node.Output] = new Made(_built.Count - 1, first, false, join == Joining.Add ? parts[0].Width : Together(parts));
            _previous = node.Operator;
        }
    }

    // A Split or a Slice of a row's columns: each column block it gives is a step of its own, taking the value cut.
    private void Cut(OnnxNode node, List<string> data)
    {
        if (data.Count != 1)
        {
            node.Refuse(Several(node, data));

            return;
        }

        var made = _values[data[0]];
        var axis = node.Operator == "Split" ? node.Whole("axis", 0) : Axis(node);

        if (!IsLastAxis(made.Reaching, axis) || made.Reaching.Lifted || Stepwise(made.Reaching.Flow))
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"it cuts axis {axis}, and a network here cuts columns out of what it is handed: the last axis, a row's second or an image's channels."));

            return;
        }

        if (node.Operator == "Slice")
        {
            Sliced(node, made);

            return;
        }

        if (Sizes(node, made) is not { } sizes)
        {
            return;
        }

        var first = 0;

        for (var at = 0; at < sizes.Length; at++)
        {
            if (node.Taken(at) is { } name)
            {
                Columned(node, name, made, new Run(first, sizes[at]));
            }

            first += sizes[at];
        }
    }

    // The lengths a Split cuts into: the ones it states, as an input from ONNX 13 or an attribute before, or equal parts.
    private int[]? Sizes(OnnxNode node, Made made)
    {
        var count = node.Outputs.Count;
        int[] sizes;

        if (node.Input(1).Length > 0)
        {
            var held = _numbers.Wholes(node.Input(1));

            if (held.Values is not { } values || values.Any(value => value is < 1 or > int.MaxValue))
            {
                node.Refuse($"its sizes, '{node.Input(1).Quoted()}', {held.Fault ?? "are lengths of one column or more."}");

                return null;
            }

            sizes = [.. values.Select(value => (int)value)];
        }
        else if (node.Says("split"))
        {
            sizes = node.Wholes("split", []);
        }
        else if (made.Width is { } width && count > 0 && width % count == 0)
        {
            sizes = [.. Enumerable.Repeat(width / count, count)];
        }
        else
        {
            node.Refuse("it cuts into equal parts, and a network here needs the width of what it cuts to be stated by the graph's input or a dense layer before it, and to divide by the number of parts.");

            return null;
        }

        if (sizes.Length != count || sizes.Any(size => size < 1) || (made.Width is { } known && sizes.Sum() != known))
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"it cuts into [{string.Join(", ", sizes)}], and that is not {count} lengths of one column or more that make up what it cuts."));

            return null;
        }

        return sizes;
    }

    // A Slice of one run of columns: its start and end as the graph states them, an end past the last column the width of the value.
    private void Sliced(OnnxNode node, Made made)
    {
        var starts = Stated(node, 1, "starts");
        var ends = Stated(node, 2, "ends");
        var steps = Stated(node, 4, "steps");

        if (starts is not [var start] || ends is not [var end])
        {
            node.Refuse("it cuts one run of columns, and a network here reads a Slice that states one start and one end.");

            return;
        }

        if (steps is { Length: > 0 } && steps is not [1])
        {
            node.Refuse($"it steps over some columns, by [{string.Join(", ", steps)}], and a network here takes whole runs of columns.");

            return;
        }

        var width = made.Width;
        var from = start < 0 && width is { } known ? start + known : start;
        var to = end < 0 && width is { } wide ? end + wide : end;

        if (to >= int.MaxValue && width is { } whole)
        {
            to = whole;
        }

        if (from < 0 || to is < 0 or >= int.MaxValue)
        {
            node.Refuse("it cuts from or to a column counted from the end, or to the end of the row, and the width of what it cuts is not stated by the graph's input or a dense layer before it.");

            return;
        }

        to = Math.Min(to, width ?? to);

        if (to <= from)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it cuts columns {from} to {to}, and that is no column."));

            return;
        }

        Columned(node, node.Output, made, new Run((int)from, (int)(to - from)));
    }

    // One block of columns as a step of its own.
    private void Columned(OnnxNode node, string output, Made made, Run run)
    {
        _built.Add(new Built(node, new OnnxLayer(description => description.Columns(run.First, run.Count), made.Reaching.Flow), false) { Sources = [made.Step] });
        _values[output] = new Made(_built.Count - 1, made.Reaching, false, run.Count);
    }

    // What a Slice states at a place among its inputs, from ONNX 10, or as an attribute before; none when it states none.
    private long[]? Stated(OnnxNode node, int at, string attribute)
    {
        if (node.Input(at).Length == 0)
        {
            return at < 4 ? [.. node.Wholes(attribute, []).Select(value => (long)value)] : null;
        }

        var held = _numbers.Wholes(node.Input(at));

        if (held.Values is null)
        {
            node.Refuse($"its '{attribute}', '{node.Input(at).Quoted()}', {held.Fault}");
        }

        return held.Values;
    }

    // The one axis a Slice cuts: the first of its axes, where the batch's, nought, is what a Slice leaves out.
    private int Axis(OnnxNode node) => Stated(node, 3, "axes") is [var axis] && axis is >= int.MinValue and <= int.MaxValue ? (int)axis : 0;

    // Whether an ONNX axis is the one that is last here: a row's second, an image's channels, an image with its channels last's last.
    private static bool IsLastAxis(Reaching reaching, int axis)
    {
        var rank = reaching.Flow is Flow.Rows or Flow.FlattenedImages ? 2 : 2 + reaching.Axes;
        var at = axis < 0 ? axis + rank : axis;

        return reaching.Flow == Flow.Images ? at == 1 : at == rank - 1;
    }

    // How wide a row of what a layer made is: as it states, or as wide as what it took when it changes no length.
    private static int? WidthOf(OnnxNode node, OnnxLayer layer, Made taken) =>
        layer.Width ?? (Elementwise.Contains(node.Operator) ? taken.Width : null);

    private static int? Together(List<Made> parts) => parts.All(part => part.Width is not null) ? parts.Sum(part => part.Width!.Value) : null;

    // The operators that make each number of a row from the number at its place: a row as wide as the one they took.
    private static readonly HashSet<string> Elementwise =
        new(["Relu", "Tanh", "Sigmoid", "Gelu", "LeakyRelu", "PRelu", "Elu", "Selu", "Mish", "Softmax", "BatchNormalization", "LayerNormalization"], StringComparer.Ordinal);

    // What a Transpose hands on: an image with its channels moved last, [0, 2, 3, 1], or moved back after the batch,
    // [0, 3, 1, 2] — which, taking the image the graph takes before anything else does, declares that image's channels
    // last, as TensorFlow lays images out and as they are here. Any other turn is refused, and so is any turn of a series
    // or a volume, which have not the four axes of an image.
    private Flow Turned(OnnxNode node, Reaching reaching, bool undeclared)
    {
        var turn = node.Wholes("perm", []);
        var flow = reaching.Flow;

        if (turn is [0, 2, 3, 1] && flow == Flow.Images && reaching.Axes == 2)
        {
            return Flow.ImagesLast;
        }

        if (Sequenced(turn, reaching, undeclared) is { } sequenced)
        {
            return sequenced;
        }

        if (turn is [0, 3, 1, 2] && (flow == Flow.ImagesLast || (undeclared && reaching.Axes == 2)))
        {
            _channelsLast |= undeclared;

            return Flow.Images;
        }

        node.Refuse(
            $"it turns its value round by [{string.Join(", ", turn)}], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1].");

        return flow;
    }

    // A cast into single-precision numbers, which the network works in throughout, is nothing; one into any other is refused.
    private static void Cast(OnnxNode node)
    {
        var type = node.Whole("to", 0);

        if (type != (int)TensorProto.Types.DataType.Float)
        {
            node.Refuse($"it turns what the node before it made into {GraphNumbers.TypeName(type)}, and a network here works in single-precision numbers throughout.");
        }
    }

    // The bias an Add adds, made the bias of the MatMul it adds it to: a dense layer of the two. An Add after anything but a
    // MatMul is refused — after one refused already, it is that MatMul's bias and not refused again — and so is a bias of
    // another width than the layer's, and one added to a MatMul whose value something else takes too.
    private void Biased(OnnxNode node, AddedBias added, int step, string taken)
    {
        var mergeable = step == _built.Count - 1 && _consumers.GetValueOrDefault(taken) == 1;

        if (!mergeable || _built is not [.., { Layer.Unbiased: { } width } matmul])
        {
            if (!mergeable || _previous != "MatMul")
            {
                node.Refuse($"it adds '{added.Name.Quoted()}' to what the node before it made, and an Add is read here as the bias of the MatMul before it.");
            }

            return;
        }

        if (added.Held.Lengths is not [var length] || length != width)
        {
            node.Refuse(string.Create(
                CultureInfo.InvariantCulture,
                $"it adds '{added.Name.Quoted()}', written as {OnnxKind.Written(added.Held.Lengths)}, and a dense layer here adds a bias as long as the layer is wide, {width}."));
        }

        _built[^1] = matmul with { Layer = matmul.Layer with { Unbiased = null, Numbers = [.. matmul.Layer.Numbers, new LayerNumber("bias", added.Held, Laying.AsWritten)] } };
    }

    // A constant node's value, kept among the graph's values; its fault noted when it holds it as other than a tensor, or
    // gives it a name the graph holds a value under already.
    private void Constant(OnnxNode node)
    {
        if (node.Tensor("value") is { } value)
        {
            if (!_numbers.Keep(node.Output, value, node.Address))
            {
                node.Refuse($"it gives '{node.Output.Quoted()}', and the graph holds a value of that name already.");
            }
        }
        else if (!node.Says("value"))
        {
            node.Refuse($"it holds its value as {node.Attributes}, and a constant is read here when it holds a tensor, 'value'.");
        }
    }

    // The one value the graph gives, which is what its last node made — and not an image flattened as ONNX flattens one.
    private void Output(string flowing, Flow flow)
    {
        if (_graph.Output.Count != 1)
        {
            _faults.Add(string.Create(CultureInfo.InvariantCulture, $"{_where}: it gives {_graph.Output.Count} values, and a network here gives one."));
        }
        else if (_graph.Output[0].Name != flowing)
        {
            _faults.Add($"output '{_graph.Output[0].Name.Quoted()}': the graph gives '{_graph.Output[0].Name.Quoted()}', and a network here gives what its last layer makes, '{flowing.Quoted()}'.");
        }
        else if (flow == Flow.FlattenedImages)
        {
            _faults.Add(
                $"output '{flowing.Quoted()}': it is an image flattened as ONNX lays an image out, channel by channel, and a network here flattens an image place by place, so its values would come out in another order.");
        }
        else if (Recurring(flow))
        {
            _faults.Add($"output '{flowing.Quoted()}': it is a series laid out as ONNX's recurrent operators give one, with the steps before the batch, and a network here gives the batch first.");
        }
    }

    // Writes each layer read without a fault in its words, noting the place each stands at in the stack the description
    // lowers onto; the last, when the loss applies it itself, is left to the loss. A step that takes other than the step before
    // it says which, by the places the steps it takes stand at.
    private List<Placed> Written(List<Built> built)
    {
        var placed = new List<Placed>();
        var places = new Dictionary<int, int>();
        var lifted = built is [.., { Faulted: false, Layer.Owned: { } owned }] && owned == _loss.Activation;
        var before = TheInput;

        for (var at = 0; at < built.Count; at++)
        {
            var (node, layer, faulted) = built[at];
            var sources = built[at].Sources;

            if (faulted || (lifted && at == built.Count - 1))
            {
                continue;
            }

            // A MatMul no Add comes after is a dense layer that adds no bias.
            if (layer.Unbiased is not null && layer.Bare is { } bare)
            {
                layer = layer with { Words = bare, Unbiased = null };
            }

            try
            {
                // A step whose source was refused takes the step written last: the fault is noted where the source is.
                var taking = sources.Select(source => source == TheInput || (!places.ContainsKey(source) && placed.Count == 0)
                    ? Step.Input
                    : (places.TryGetValue(source, out var place) ? place : placed.Count - 1).ToString(CultureInfo.InvariantCulture)).ToArray();

                switch (layer.Join)
                {
                    case Joining.Add:
                        _description.Add(taking);
                        break;
                    case Joining.Concatenate:
                        _description.Concatenate(taking);
                        break;
                    default:
                        layer.Words!(_description);

                        if (sources is not [var only] || only != before)
                        {
                            _description.Takes(taking);
                        }

                        break;
                }

                places[at] = placed.Count;
                placed.Add(new Placed(placed.Count, node, layer));
                before = at;
            }
            catch (ArgumentException refused)
            {
                node.Refuse(refused.Message.ReplaceLineEndings(" "));
            }
        }

        if (placed.Count == 0)
        {
            _faults.Add($"{_where}: it holds no layer a network here is built of.");
        }

        return placed;
    }

    // The description lowered onto a stack of layers for examples of the input's shape; the starts it draws are replaced by
    // the graph's numbers.
    private LayerStack Lowered(Shape each)
    {
        try
        {
            return _description.Lower(each, new RandomStream(0));
        }
        catch (ArgumentException refused)
        {
            throw new FormatException($"{_where}: {refused.Message.ReplaceLineEndings(" ")}", refused);
        }
    }

    private FormatException Refused() => new(string.Join(Environment.NewLine, _faults));

    // What a node is refused with that takes values no node made, and the graph does not take.
    private static string NotMade(List<string> names) => names.Count == 1
        ? $"it takes '{names[0].Quoted()}', which is neither the input nor a value a node before it made."
        : $"it takes {Listed(names)}, which are neither the input nor values a node before it made.";

    // What a node is refused with that takes several values, and is none of the joins.
    private static string Several(OnnxNode node, List<string> data) =>
        $"it takes {Listed(data)}, {data.Count} values the graph works out, and a {node.Operator} here works on one: an Add takes two, to add them up, and a Concat any number, to lay them side by side.";

    private static string Listed(List<string> names) => names.Count == 1
        ? $"'{names[0].Quoted()}'"
        : $"{string.Join(", ", names.Take(names.Count - 1).Select(name => $"'{name.Quoted()}'"))} and '{names[^1].Quoted()}'";

    private static OnnxLayer Unknown(OnnxNode node, Reaching reaching)
    {
        node.Refuse($"'{node.Operator.Quoted()}' is no operator a network here is built of: an ONNX graph is read here when each of its nodes is a {OnnxKind.Listed}, or an Identity, a Cast, a Transpose or a Constant that hands a value on, or a Concat, a Split or a Slice that joins or cuts columns.");

        return new OnnxLayer(null, reaching.Flow);
    }

    // An axis's length as the graph writes it: a count, a name it leaves open, or nothing.
    private static string Written(Dimension dim) => dim.ValueCase switch
    {
        Dimension.ValueOneofCase.DimValue => dim.DimValue.ToString(CultureInfo.InvariantCulture),
        Dimension.ValueOneofCase.DimParam => dim.DimParam,
        _ => "?",
    };

    /// <summary>The value the graph takes, what reaches its first node, and the lengths of one example after the batch, when they can be read.</summary>
    private readonly record struct Taken(string Value, Reaching Reaching, int[]? Lengths);

    /// <summary>A node read as the layer it is, and whether reading it noted a fault; the steps it works on are the nodes that made the values it takes.</summary>
    private readonly record struct Built(OnnxNode Node, OnnxLayer Layer, bool Faulted)
    {
        /// <summary>The nodes whose work it takes, by their place among those read, or <see cref="TheInput"/> for the graph's input.</summary>
        public IReadOnlyList<int> Sources { get; init; } = [];
    }

    /// <summary>A value of the graph: the node that made it, how it is laid out, whether an image has declared no layout yet, and how wide a row of it is when that is known.</summary>
    private readonly record struct Made(int Step, Reaching Reaching, bool Undeclared, int? Width);

    /// <summary>A run of columns: where it starts and how many it holds.</summary>
    private readonly record struct Run(int First, int Count);

    /// <summary>A layer written into the description, and its place in the stack it lowers onto.</summary>
    private readonly record struct Placed(int Place, OnnxNode Node, OnnxLayer Layer);
}
