// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// A Keras model as a network here: its description written in Keras's words, the loss it was compiled with, and where
/// each of its layers' numbers go — or every fault that keeps it from being one, at once.
/// </summary>
/// <remarks>
/// Each Keras layer is written as the words its kind writes, and the activation it carries as a word of its own after
/// them; the last activation the loss applies itself is not written, since the network ends where that loss takes over.
/// The description is lowered as any description in Keras's words is, and the file's numbers go in by path.
/// </remarks>
internal sealed class KerasModel
{
    private const string InputLayer = "InputLayer";
    private const string Sequential = "Sequential";
    private const string Functional = "Functional";

    // The activations read here, and the word each is written as: none for linear.
    private static readonly Dictionary<string, Action<Sequential>?> Activations = new(StringComparer.Ordinal)
    {
        ["linear"] = null,
        ["relu"] = description => description.Relu(),
        ["tanh"] = description => description.Tanh(),
        ["sigmoid"] = description => description.Sigmoid(),
        ["gelu"] = description => description.Gelu(),
        ["silu"] = description => description.Silu(),
        ["swish"] = description => description.Silu(),
        ["mish"] = description => description.Mish(),
        ["leaky_relu"] = description => description.LeakyRelu(0.2),
        ["elu"] = description => description.Elu(),
        ["selu"] = description => description.Selu(),
        ["softmax"] = description => description.Softmax(),
    };

    private readonly List<string> _faults = [];
    private readonly Dictionary<string, Placed> _placed = new(StringComparer.Ordinal);
    private readonly Sequential _description = new();
    private readonly string _where;
    private readonly Shape _input;
    private readonly Loss _loss;

    /// <summary>Reads what a Keras file says of its model.</summary>
    /// <exception cref="FormatException">It says what no network here is built of: every such thing, each where the file says it.</exception>
    public KerasModel(KerasSaved saved)
    {
        _where = saved.Where;

        var model = new KerasSettings(saved.Model, saved.Where, _faults);
        var loss = KerasLoss.Read(saved.Loss, saved.LossWhere, _faults);
        var className = model.Setting("class_name", string.Empty);

        if (className is not (Sequential or Functional))
        {
            model.Refuse(
                $"the model is a {className.Quoted()}, and a Keras model is read here when it is a Sequential — layers each handed what the one before it made — or a Functional one, whose layers each take the layers they name.");

            throw Refused();
        }

        var functional = className == Functional;
        var config = new KerasSettings(model.Setting("config", default(JsonElement)), saved.Where, _faults);
        var entries = config.Setting("layers", default(JsonElement));
        var described = entries.ValueKind == JsonValueKind.Array ? entries.EnumerateArray().Select(entry => Describe(entry, functional)).ToList() : [];
        var inputs = new List<ModelInput>();
        var input = functional ? FunctionalInputOf(described, config, inputs) : InputOf(described, config);
        var built = described.Select(Read).ToList();

        if (functional)
        {
            RequireOrder(described, inputs);
            RequireOutput(built, config);
        }

        input = Taken(built, input);
        Write(built, Lifted(built, loss), inputs);

        if (_faults.Count > 0)
        {
            throw Refused();
        }

        _input = input!.Value;
        _loss = loss!.Value.Loss;
    }

    /// <summary>
    /// The network the description builds, every slot holding the number the file holds for it, and the loss it answers
    /// through.
    /// </summary>
    /// <param name="numbers">The numbers of each layer the file holds numbers for.</param>
    /// <exception cref="FormatException">The layers the description names cannot be built one after another.</exception>
    /// <exception cref="SlotLoadException">
    /// The numbers do not fit the network: every fault at once, each where the file holds the number — a number of another
    /// precision among them.
    /// </exception>
    public SavedNetwork Loaded(IReadOnlyList<LayerNumbers> numbers)
    {
        var network = Lower();
        var entries = new List<SlotEntry>();
        var unread = new List<SlotLoadFault>();

        foreach (var held in numbers)
        {
            // A layer the model does not name has no slots, so each of its numbers is refused as a path the network lacks.
            var placed = _placed.TryGetValue(held.Layer, out var found) ? found : new Placed(-1, new KerasLayer(null, "linear"));

            for (var at = 0; at < held.Numbers.Count; at++)
            {
                var number = held.Numbers[at];

                // A number beyond the layer's slots goes nowhere, and is named by the layer and the name the file holds it
                // under; it is handed on in the shape the file writes, since no slot says another.
                var named = placed.Layer.SlotCalled(number.Name);
                var known = named is not null || at < placed.Layer.Slots.Count;
                var slot = named ?? (known ? placed.Layer.Slots[at] : new KerasSlot(number.Name, 0, []));
                var path = known
                    ? string.Create(CultureInfo.InvariantCulture, $"{placed.Place + slot.Away}.{slot.Named}")
                    : $"{held.Layer}.{number.Name}";

                if (number.Values is { } values)
                {
                    entries.Add(new SlotEntry(
                        path,
                        Tensor.From(placed.Layer.LaidOut(number.Lengths, slot), placed.Layer.Turned(values, number.Lengths, slot)),
                        number.Source));
                }
                else
                {
                    unread.Add(new SlotLoadFault(number.Source, path, $"'{path.Quoted()}' holds single-precision numbers here, and is written as {number.WrittenAs}."));
                }
            }
        }

        entries.AddRange(Fresh(network, numbers));

        var faults = new List<SlotLoadFault>(unread);

        try
        {
            network.Load(entries);
        }
        catch (SlotLoadException refused)
        {
            // A number of another precision is held all the same: it is refused as that, not as missing.
            faults.AddRange(refused.Faults.Where(fault => fault.Source is not null || unread.All(other => other.Slot != fault.Slot)));
        }

        return faults.Count > 0 ? throw new SlotLoadException(faults) : new SavedNetwork(network, _loss);
    }

    // The numbers of the slots a file leaves out because Keras merged them into another — a low-rank pair, in the archive Keras 3
    // saves — which the network begins afresh: the numbers it was lowered with go in, so the pair is there to be trained.
    private IEnumerable<SlotEntry> Fresh(LayerStack network, IReadOnlyList<LayerNumbers> numbers)
    {
        var slots = network.Slots().ToDictionary(named => named.Path, named => named.Slot.Value, StringComparer.Ordinal);

        foreach (var (name, placed) in _placed.Where(each => each.Value.Layer.Fresh.Count > 0))
        {
            var held = numbers.Where(each => each.Layer == name).Select(each => each.Numbers.Count).DefaultIfEmpty(0).Max();

            for (var at = Math.Max(held, 0); at < placed.Layer.Slots.Count; at++)
            {
                var slot = placed.Layer.Slots[at];

                if (!placed.Layer.Fresh.Contains(slot))
                {
                    continue;
                }

                var path = string.Create(CultureInfo.InvariantCulture, $"{placed.Place + slot.Away}.{slot.Named}");

                yield return new SlotEntry(path, slots[path], $"{_where}: the pair of layer '{name}' is saved merged into its kernel, so it begins afresh");
            }
        }
    }

    // What a Keras activation is owned by: the output activation of a loss here that applies it itself.
    private static string? Owned(Loss loss) => loss.Activation switch
    {
        OutputActivation.Sigmoid => "sigmoid",
        OutputActivation.Softmax => "softmax",
        _ => null,
    };

    // One of the description's layers, as the file lists it: its class, and its settings where its name places them.
    private Described Describe(JsonElement entry, bool functional)
    {
        var listed = new KerasSettings(entry, _where, _faults);
        var className = listed.Setting("class_name", string.Empty);
        var config = listed.Setting("config", default(JsonElement));
        var name = new KerasSettings(config, _where, _faults).Setting("name", string.Empty);
        var settings = new KerasSettings(config, $"{_where}, layer '{name}' ({className})", _faults);

        return functional && className != InputLayer ? SourcesOf(entry, new Described(name, className, settings)) : new Described(name, className, settings);
    }

    // The layers a layer of a functional model takes: the one call it is made in, its arguments the tensors of layers each at its
    // first node and tensor. A layer called twice, with anything but tensors, or in the form Keras 2 wrote, is refused.
    private Described SourcesOf(JsonElement entry, Described layer)
    {
        var at = layer.Settings;

        if (!entry.TryGetProperty("inbound_nodes", out var nodes) || nodes.ValueKind != JsonValueKind.Array || nodes.GetArrayLength() == 0)
        {
            at.Refuse("it takes no layer, and a layer of a functional model takes the layers it names.");

            return layer;
        }

        if (nodes.GetArrayLength() > 1)
        {
            at.Refuse(string.Create(CultureInfo.InvariantCulture, $"it is called {nodes.GetArrayLength()} times, and a layer is read here when it is called once."));

            return layer;
        }

        var node = nodes[0];

        if (node.ValueKind == JsonValueKind.Array)
        {
            at.Refuse("it takes its layers in the form Keras 2 wrote, and a Keras 3 model states them as arguments.");

            return layer;
        }

        if (node.ValueKind == JsonValueKind.Object && node.TryGetProperty("kwargs", out var kwargs) && kwargs.ValueKind == JsonValueKind.Object
            && kwargs.EnumerateObject().Select(each => each.Name).FirstOrDefault() is { } called)
        {
            at.Refuse($"it is called with '{called.Quoted()}', and a layer is read here when it is called with the tensors it takes alone.");

            return layer;
        }

        if (node.ValueKind != JsonValueKind.Object || !node.TryGetProperty("args", out var args) || args.ValueKind != JsonValueKind.Array || args.GetArrayLength() != 1)
        {
            at.Refuse("it is called with other than the tensors of layers, and a layer of a functional model is called with the tensors it takes.");

            return layer;
        }

        var tensors = args[0].ValueKind == JsonValueKind.Array ? args[0].EnumerateArray().ToList() : [args[0]];
        var sources = new List<string>();

        foreach (var tensor in tensors)
        {
            if (!TryHistory(tensor, out var source, out var nodeAt, out var tensorAt))
            {
                at.Refuse("it is called with other than the tensors of layers, and a layer of a functional model is called with the tensors it takes.");

                return layer;
            }

            if (nodeAt != 0 || tensorAt != 0)
            {
                at.Refuse(string.Create(
                    CultureInfo.InvariantCulture,
                    $"it takes the layer '{source.Quoted()}' at its node {nodeAt}, tensor {tensorAt}, and a layer is read here when it is called once, giving one tensor."));

                return layer;
            }

            sources.Add(source);
        }

        var rank = tensors[0].TryGetProperty("config", out var config) && config.TryGetProperty("shape", out var shape) && shape.ValueKind == JsonValueKind.Array ? shape.GetArrayLength() : 0;

        return layer with { Sources = sources, Rank = rank };
    }

    // Where a tensor of a functional model comes from: the layer, and the node and tensor of it, as Keras 3 writes them.
    private static bool TryHistory(JsonElement tensor, out string layer, out int node, out int output)
    {
        layer = string.Empty;
        node = output = 0;

        if (tensor.ValueKind != JsonValueKind.Object
            || !tensor.TryGetProperty("class_name", out var kind) || kind.ValueKind != JsonValueKind.String || kind.GetString() != "__keras_tensor__"
            || !tensor.TryGetProperty("config", out var config) || config.ValueKind != JsonValueKind.Object
            || !config.TryGetProperty("keras_history", out var history) || history.ValueKind != JsonValueKind.Array || history.GetArrayLength() != 3
            || history[0].ValueKind != JsonValueKind.String || !history[1].TryGetInt32(out node) || !history[2].TryGetInt32(out output))
        {
            return false;
        }

        layer = history[0].GetString()!;

        return true;
    }

    // The names of the layers a functional model lists under a key: one triple for one layer, a list of triples for several.
    private static List<string> Listed(KerasSettings config, string key)
    {
        var written = config.Setting(key, default(JsonElement));

        if (written.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        var items = written.EnumerateArray().ToList();

        return items.Count > 0 && items[0].ValueKind == JsonValueKind.String
            ? [items[0].GetString()!]
            : [.. items.Where(item => item.ValueKind == JsonValueKind.Array && item.GetArrayLength() > 0 && item[0].ValueKind == JsonValueKind.String).Select(item => item[0].GetString()!)];
    }

    // The shape of one example of a functional model, from its input layer or layers in the order the model lists them; several
    // inputs are handed to the network as the columns of one row, so each must be a row. The input layers are not layers built.
    private Shape? FunctionalInputOf(List<Described> described, KerasSettings config, List<ModelInput> inputs)
    {
        var names = Listed(config, "input_layers");
        var layers = described.Where(layer => layer.Class == InputLayer).ToList();

        described.RemoveAll(layer => layer.Class == InputLayer);

        if (names.Count == 0)
        {
            config.Refuse("the model lists no input layer under 'input_layers', and every width its layers take is worked out from the shape of its input.");

            return null;
        }

        var shapes = new List<int[]>();

        foreach (var name in names)
        {
            var layer = layers.FirstOrDefault(each => each.Name == name);

            if (layer.Settings is null)
            {
                config.Refuse($"the model lists the input '{name.Quoted()}', and no input layer of that name.");

                return null;
            }

            var lengths = layer.Settings.Setting<int?[]?>("batch_shape");

            if (lengths is null)
            {
                return null;
            }

            if (lengths.Skip(1).Any(length => length is not > 0))
            {
                layer.Settings.Refuse(
                    $"the input, [{string.Join(", ", lengths.Select(length => length?.ToString(CultureInfo.InvariantCulture) ?? "null")).Quoted()}], does not state every length of an example, and a network here is built for examples of one shape.");

                return null;
            }

            shapes.Add([.. lengths.Skip(1).Select(length => length!.Value)]);
        }

        if (names.Count == 1)
        {
            inputs.Add(new ModelInput(names[0], 0, 0));

            return new Shape(shapes[0]);
        }

        var first = 0;

        for (var at = 0; at < names.Count; at++)
        {
            if (shapes[at].Length != 1)
            {
                layers.First(each => each.Name == names[at]).Settings.Refuse(
                    $"it is an input of {string.Join('x', shapes[at])} values, and a model of several inputs is read here when each is a row of numbers, handed to the network side by side as the columns of one row.");

                return null;
            }

            inputs.Add(new ModelInput(names[at], first, shapes[at][0]));
            first += shapes[at][0];
        }

        return new Shape(first);
    }

    // A layer takes layers listed before it: the model's inputs, or a layer above it. The list of a functional model is in the order
    // the layers run in, so a layer that takes one below it, or one that is none, is refused where it says so.
    private static void RequireOrder(List<Described> described, List<ModelInput> inputs)
    {
        var seen = new HashSet<string>(inputs.Select(input => input.Name), StringComparer.Ordinal);

        foreach (var layer in described)
        {
            foreach (var source in (layer.Sources ?? []).Where(name => !seen.Contains(name)))
            {
                layer.Settings.Refuse($"it takes '{source.Quoted()}', which is no layer before it.");
            }

            seen.Add(layer.Name);
        }
    }

    // A functional model gives one tensor, and it is the one its last layer makes: the network ends where its last step does.
    private void RequireOutput(List<Built> built, KerasSettings config)
    {
        var outputs = Listed(config, "output_layers");

        if (outputs.Count != 1)
        {
            config.Refuse(string.Create(CultureInfo.InvariantCulture, $"the model gives {outputs.Count} tensors, and a network here gives one."));
        }
        else if (built.Count > 0 && built[^1].Described.Name != outputs[0])
        {
            config.Refuse($"the model gives the output of '{outputs[0].Quoted()}', and a network here gives what its last layer, '{built[^1].Described.Name.Quoted()}', makes.");
        }
    }

    // The shape of one example, from the input layer the description begins with or the shape it was built for, the batch's
    // length left off; nothing, with its fault noted, when neither states it. An input layer is not one of the layers built.
    private Shape? InputOf(List<Described> described, KerasSettings config)
    {
        int?[]? lengths;

        if (described.Count > 0 && described[0].Class == InputLayer)
        {
            lengths = described[0].Settings.Setting<int?[]?>("batch_shape");
            described.RemoveAt(0);
        }
        else if (config.Says("build_input_shape"))
        {
            lengths = config.Setting<int?[]?>("build_input_shape");
        }
        else
        {
            config.Refuse("the model states no shape for its input — it begins with no InputLayer and says no build_input_shape — and every width its layers take is worked out from that shape.");

            return null;
        }

        if (lengths is null)
        {
            return null;
        }

        var each = lengths.Skip(1).ToArray();

        if (each.Any(length => length is not > 0))
        {
            config.Refuse(
                $"the model's input, [{string.Join(", ", lengths.Select(length => length?.ToString(CultureInfo.InvariantCulture) ?? "null")).Quoted()}], does not state every length of an example, and a network here is built for examples of one shape.");

            return null;
        }

        return new Shape([.. each.Select(length => length!.Value)]);
    }

    // The shape of an example as the network takes it: the model's input, unless its first layer takes another in its place —
    // an embedding takes its category's ones and noughts where Keras takes one place. A layer that does is read first, over one
    // place per example, or refused.
    private static Shape? Taken(List<Built> built, Shape? input)
    {
        for (var at = 0; at < built.Count; at++)
        {
            if (built[at].Layer.Takes is not { } taken)
            {
                continue;
            }

            if (at > 0)
            {
                built[at].Described.Settings.Refuse("it looks a category up by its place, which only the model's input holds: an embedding is read as the first layer, over one category's place per example.");
            }
            else if (input is { } given && given != new Shape(1))
            {
                built[at].Described.Settings.Refuse($"it is handed examples of {given}, and an embedding is read over one category's place per example, which the network here takes as the category's ones and noughts.");
            }
            else
            {
                input = taken;
            }
        }

        return input;
    }

    // A layer read by its kind, and whether reading it found a fault; a kind not read here is its fault.
    private Built Read(Described layer)
    {
        var before = _faults.Count;
        KerasKind.RequireSinglePrecision(layer.Settings);

        var read = KerasKind.ByName.TryGetValue(layer.Class, out var kind) ? kind.Read(layer.Settings) : Merged(layer) ?? Unknown(layer);

        return new Built(layer, read, _faults.Count > before);
    }

    // An Add or a Concatenate of a functional model: the steps it joins are the layers it takes, and it has no numbers of its own.
    private static KerasLayer? Merged(Described layer)
    {
        if (layer.Sources is null || layer.Class is not ("Add" or "Concatenate"))
        {
            return null;
        }

        if (layer.Sources.Count < 2)
        {
            layer.Settings.Refuse($"it joins {layer.Sources.Count} tensor, and a layer that adds or lays side by side is read here when it takes two or more.");
        }

        if (layer.Class == "Concatenate")
        {
            var axis = layer.Settings.Setting("axis", -1);

            if (axis != -1 && axis != layer.Rank - 1)
            {
                layer.Settings.Refuse(string.Create(
                    CultureInfo.InvariantCulture,
                    $"it lays its tensors side by side along axis {axis}, and a network here lays them side by side along their last axis, -1."));
            }
        }

        return new KerasLayer(null, "linear");
    }

    private static KerasLayer Unknown(Described layer)
    {
        layer.Settings.Refuse($"'{layer.Class.Quoted()}' is no kind a network here is built of: a Keras model is read here when each of its layers is a {KerasKind.Listed}, or, in a Functional model, an Add or a Concatenate.");

        return new KerasLayer(null, "linear");
    }

    // Whether the model's last activation is lifted into its loss: when the loss applies one itself and was trained on the
    // chances it gives. A model whose end and loss disagree about it is refused at its last layer.
    private static bool Lifted(List<Built> built, KerasLoss? loss)
    {
        if (loss is not { } trained || Owned(trained.Loss) is not { } owned || built.Count == 0)
        {
            return false;
        }

        var last = built[^1];

        if ((last.Layer.Activation == owned) == trained.FromLogits)
        {
            last.Described.Settings.Refuse(trained.FromLogits
                ? $"the model ends in a {owned}, and its loss takes logits and applies the {owned} itself: every prediction would go through it twice."
                : $"its loss takes the chances a last {owned} gives, and the model ends in {last.Layer.Activation.Quoted()}: a network here ends before the {owned} its loss applies itself, so it is read when the model ends in one.");
        }

        return !trained.FromLogits;
    }

    // Writes each layer read without a fault in Keras's words, and its activation after it, noting the place each layer's
    // numbers go to: the layer's place in the stack the description lowers onto. In a functional model a layer that takes other than
    // the step before it says which, by the place of the step that made what it takes; several inputs are cut from the one row first.
    private void Write(List<Built> built, bool lifted, List<ModelInput> inputs)
    {
        var count = 0;
        var made = new Dictionary<string, int>(StringComparer.Ordinal);

        if (inputs.Count == 1)
        {
            made[inputs[0].Name] = -1;
        }
        else
        {
            foreach (var input in inputs)
            {
                _description.Columns(input.First, input.Width);

                if (count > 0)
                {
                    _description.Takes(Step.Input);
                }

                made[input.Name] = count++;
            }
        }

        for (var at = 0; at < built.Count; at++)
        {
            var (described, layer, faulted) = built[at];

            if (faulted)
            {
                continue;
            }

            var first = count;
            var sources = described.Sources?.Select(name => made.GetValueOrDefault(name, first - 1)).ToArray();

            _placed[described.Name] = new Placed(first, layer);

            if (described.Class is "Add" or "Concatenate" && sources is not null)
            {
                try
                {
                    _ = described.Class == "Add" ? _description.Add(Taking(sources)) : _description.Concatenate(Taking(sources));
                    count++;
                }
                catch (ArgumentException refused)
                {
                    described.Settings.Refuse(refused.Message.ReplaceLineEndings(" "));
                }
            }
            else if (layer.Words is { } words)
            {
                try
                {
                    words(_description);

                    // One Keras layer can be more than one layer here: its own words say how many it wrote.
                    count += layer.Places;

                    if (sources is not null && !(sources.Length == 1 && sources[0] == first - 1))
                    {
                        if (layer.Places > 1)
                        {
                            described.Settings.Refuse("it is more than one layer here, and a layer that is takes the layer before it.");
                        }
                        else
                        {
                            _description.Takes(Taking(sources));
                        }
                    }
                }
                catch (ArgumentException refused)
                {
                    described.Settings.Refuse(refused.Message.ReplaceLineEndings(" "));
                }
            }

            if (at == built.Count - 1 && lifted)
            {
                made[described.Name] = count - 1;

                continue;
            }

            if (!Activations.TryGetValue(layer.Activation, out var activation))
            {
                described.Settings.Refuse(
                    $"its activation, {layer.Activation.Quoted()}, is none of those read here: linear, relu, tanh, sigmoid, gelu, silu, swish, mish, leaky_relu, elu, selu and softmax.");
            }
            else if (activation is not null)
            {
                activation(_description);
                count++;
            }

            // What a layer makes is what its last step makes; a layer that wrote none hands on what it took.
            made[described.Name] = count > first ? count - 1 : sources is [var only] ? only : first - 1;
        }

        if (count == 0)
        {
            _faults.Add($"{_where}: the model holds no layer a network here is built of.");
        }
    }

    // The steps a layer takes, by the names they go by: the place they stand at, or the input.
    private static string[] Taking(int[] places) => [.. places.Select(place => place < 0 ? Step.Input : place.ToString(CultureInfo.InvariantCulture))];

    // The description lowered onto a stack of layers for examples of the input's shape; the starts it draws are replaced by
    // the file's numbers.
    private LayerStack Lower()
    {
        try
        {
            return _description.Lower(_input, new RandomStream(0));
        }
        catch (ArgumentException refused)
        {
            throw new FormatException($"{_where}: {refused.Message.ReplaceLineEndings(" ")}", refused);
        }
    }

    private FormatException Refused() => new(string.Join(Environment.NewLine, _faults));

    /// <summary>A layer as the description lists it: its name, its class, and its settings, placed where a fault names them.</summary>
    private readonly record struct Described(string Name, string Class, KerasSettings Settings)
    {
        /// <summary>The layers it takes, by name, when the model is a functional one; nothing when each layer takes the one before it.</summary>
        public IReadOnlyList<string>? Sources { get; init; }

        /// <summary>The rank of what it takes, the batch's axis included, as its first argument says.</summary>
        public int Rank { get; init; }
    }

    /// <summary>An input of a functional model, where its columns stand in the row the network is handed, and how wide it is.</summary>
    private readonly record struct ModelInput(string Name, int First, int Width);

    /// <summary>A layer read by its kind, and whether reading it noted a fault.</summary>
    private readonly record struct Built(Described Described, KerasLayer Layer, bool Faulted);

    /// <summary>A layer written into the description, and its place in the stack it lowers onto.</summary>
    private readonly record struct Placed(int Place, KerasLayer Layer);
}
