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
internal sealed class OnnxGraph
{
    private readonly List<string> _faults = [];
    private readonly Sequential _description = new();
    private readonly GraphProto _graph;
    private readonly GraphNumbers _numbers;
    private readonly Loss _loss;
    private readonly string _where;

    // The names of the values something takes: a node's input, or what the graph gives.
    private readonly HashSet<string> _used;

    // Whether the image the graph takes has its channels last, as the Transpose that first takes it declares.
    private bool _channelsLast;

    /// <summary>A graph, where its numbers are, and the loss the network it becomes answers through.</summary>
    public OnnxGraph(GraphProto graph, GraphNumbers numbers, Loss loss)
    {
        _graph = graph;
        _numbers = numbers;
        _loss = loss;
        _where = $"graph '{graph.Name.Quoted()}'";
        _used = [.. graph.Node.SelectMany(node => node.Input), .. graph.Output.Select(output => output.Name)];
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
        var reaching = ShapesReaching(network, example);
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

    // The graph's nodes in order, each read as the layer it is when it takes the value the one before it made; constants
    // kept, and identities, casts and transposes followed on the way; an Add made the bias of the MatMul before it. The
    // graph's output is held to the last value made.
    private List<Built> Walked(Taken input)
    {
        var built = new List<Built>();
        var flowing = input.Value;
        var reaching = input.Reaching;

        // Until a layer or a transpose takes it, an image the graph takes has declared no layout of its own.
        var undeclared = reaching.Flow == Flow.Images;

        // The operator of the last node on the chain that is not an identity, a cast or a transpose: an Add after a MatMul is
        // its bias, even when the MatMul itself is refused.
        var previous = string.Empty;

        for (var place = 0; place < _graph.Node.Count; place++)
        {
            var node = new OnnxNode(_graph.Node[place], place, _faults, _used);

            if (node.Operator == "Constant")
            {
                Constant(node);

                continue;
            }

            var data = node.Inputs.Where(name => name.Length > 0 && !_numbers.Holds(name)).ToList();

            if (node.Operator == "Identity" && data.Count == 0 && _numbers.Holds(node.Input(0)))
            {
                _numbers.Alias(node.Output, node.Input(0));

                continue;
            }

            if (data.Count != 1 || data[0] != flowing)
            {
                node.Refuse(Branch(data, flowing));
                previous = node.Operator;
            }
            else if (node.Operator == "Transpose")
            {
                reaching = reaching with { Flow = Turned(node, reaching, undeclared) };
                undeclared = false;
            }
            else if (node.Operator == "Cast")
            {
                Cast(node);
            }
            else if (node.Operator == "Unsqueeze" && reaching is { Flow: Flow.Images, Axes: 1 } && _graph.OpensLiftedPooling(place, _numbers))
            {
                reaching = reaching with { Lifted = true };
            }
            else if (node.Operator == "Squeeze" && reaching.Lifted)
            {
                reaching = reaching with { Lifted = false };
            }
            else if (node.Operator != "Identity")
            {
                var before = _faults.Count;
                var layer = OnnxKind.ByName.TryGetValue(node.Operator, out var kind) ? kind.Read(node, reaching, _numbers) : Unknown(node, reaching);

                undeclared = false;

                if (layer.Adds is { } added)
                {
                    Biased(built, node, added, previous == "MatMul");
                }
                else
                {
                    built.Add(new Built(node, layer, _faults.Count > before));
                    reaching = reaching with { Flow = layer.After };
                }

                previous = node.Operator;
            }

            flowing = node.Output;
        }

        Output(flowing, reaching.Flow);

        return built;
    }

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

    // The bias an Add adds, made the bias of the MatMul before it: a dense layer of the two. An Add after anything but a
    // MatMul is refused — after one refused already, it is that MatMul's bias and not refused again — and so is a bias of
    // another width than the layer's.
    private static void Biased(List<Built> built, OnnxNode node, AddedBias added, bool afterMatMul)
    {
        if (built is not [.., { Layer.Unbiased: { } width } matmul])
        {
            if (!afterMatMul)
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

        built[^1] = matmul with { Layer = matmul.Layer with { Unbiased = null, Numbers = [.. matmul.Layer.Numbers, new LayerNumber("bias", added.Held, Laying.AsWritten)] } };
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
    }

    // Writes each layer read without a fault in its words, noting the place each stands at in the stack the description
    // lowers onto; the last, when the loss applies it itself, is left to the loss.
    private List<Placed> Written(List<Built> built)
    {
        var placed = new List<Placed>();
        var lifted = built is [.., { Faulted: false, Layer.Owned: { } owned }] && owned == _loss.Activation;

        for (var at = 0; at < built.Count; at++)
        {
            var (node, layer, faulted) = built[at];

            if (faulted || (lifted && at == built.Count - 1))
            {
                continue;
            }

            // A MatMul no Add after it made a dense layer with adds no bias.
            if (layer.Unbiased is not null)
            {
                node.Refuse(OnnxKind.NoBias);

                continue;
            }

            // A layer written as no word here is a softmax, which only a cross-entropy that applies it takes over.
            if (layer.Words is not { } words)
            {
                node.Refuse("it is a softmax, which a network here leaves to its loss: it is read as the last of a graph whose cross-entropy applies it.");

                continue;
            }

            try
            {
                words(_description);
                placed.Add(new Placed(placed.Count, node, layer));
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

    // The shape of what reaches each layer of the stack, a batch of one example: an example of nothing sent through the
    // layers themselves, so the shape each makes is the one it makes, worked out nowhere else.
    private static List<Shape> ShapesReaching(LayerStack network, Shape each)
    {
        var pass = Pass.Evaluation(new CpuBackend());
        var shapes = new List<Shape>(network.Layers.Count);
        var example = Tensor.Zeros(new Shape([1, .. each.Axes]));

        foreach (var layer in network.Layers)
        {
            shapes.Add(example.Shape);
            example = layer.Forward(example, pass);
        }

        return shapes;
    }

    private FormatException Refused() => new(string.Join(Environment.NewLine, _faults));

    // What a node is refused with that does not take the value the node before it made — the one value of a stack.
    private static string Branch(List<string> data, string flowing)
    {
        var taken = data.Count switch
        {
            0 => "none of the values the graph works out",
            1 => $"'{data[0].Quoted()}'",
            _ => string.Create(
                CultureInfo.InvariantCulture,
                $"{string.Join(", ", data.Take(data.Count - 1).Select(name => $"'{name.Quoted()}'"))} and '{data[^1].Quoted()}', {data.Count} values the graph works out"),
        };

        return $"it takes {taken}, and a network here is a stack: each layer takes the value the layer before it made, '{flowing.Quoted()}'.";
    }

    private static OnnxLayer Unknown(OnnxNode node, Reaching reaching)
    {
        node.Refuse($"'{node.Operator.Quoted()}' is no operator a network here is built of: an ONNX graph is read here when each of its nodes is a {OnnxKind.Listed}, or an Identity, a Cast, a Transpose or a Constant that hands a value on.");

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

    /// <summary>A node read as the layer it is, and whether reading it noted a fault.</summary>
    private readonly record struct Built(OnnxNode Node, OnnxLayer Layer, bool Faulted);

    /// <summary>A layer written into the description, and its place in the stack it lowers onto.</summary>
    private readonly record struct Placed(int Place, OnnxNode Node, OnnxLayer Layer);
}
