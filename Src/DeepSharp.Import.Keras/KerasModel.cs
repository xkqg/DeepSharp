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

    // The activations read here, and the word each is written as: none for linear.
    private static readonly Dictionary<string, Action<Sequential>?> Activations = new(StringComparer.Ordinal)
    {
        ["linear"] = null,
        ["relu"] = description => description.Relu(),
        ["tanh"] = description => description.Tanh(),
        ["sigmoid"] = description => description.Sigmoid(),
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

        if (className != "Sequential")
        {
            model.Refuse($"the model is a {className.Quoted()}, and a Keras model is read here when it is a Sequential: a stack of layers, each handed what the one before it made.");

            throw Refused();
        }

        var config = new KerasSettings(model.Setting("config", default(JsonElement)), saved.Where, _faults);
        var entries = config.Setting("layers", default(JsonElement));
        var described = entries.ValueKind == JsonValueKind.Array ? entries.EnumerateArray().Select(Describe).ToList() : [];
        var input = InputOf(described, config);
        var built = described.Select(Read).ToList();

        Write(built, Lifted(built, loss));

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
                var path = at < placed.Layer.Slots.Count
                    ? string.Create(CultureInfo.InvariantCulture, $"{placed.Place}.{placed.Layer.Slots[at]}")
                    : $"{held.Layer}.{number.Name}";

                if (number.Values is { } values)
                {
                    entries.Add(new SlotEntry(path, Tensor.From(placed.Layer.LaidOut(number.Lengths), values), number.Source));
                }
                else
                {
                    unread.Add(new SlotLoadFault(number.Source, path, $"'{path.Quoted()}' holds single-precision numbers here, and is written as {number.WrittenAs}."));
                }
            }
        }

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

    // What a Keras activation is owned by: the output activation of a loss here that applies it itself.
    private static string? Owned(Loss loss) => loss.Activation switch
    {
        OutputActivation.Sigmoid => "sigmoid",
        OutputActivation.Softmax => "softmax",
        _ => null,
    };

    // One of the description's layers, as the file lists it: its class, and its settings where its name places them.
    private Described Describe(JsonElement entry)
    {
        var listed = new KerasSettings(entry, _where, _faults);
        var className = listed.Setting("class_name", string.Empty);
        var config = listed.Setting("config", default(JsonElement));
        var name = new KerasSettings(config, _where, _faults).Setting("name", string.Empty);

        return new Described(name, className, new KerasSettings(config, $"{_where}, layer '{name}' ({className})", _faults));
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

    // A layer read by its kind, and whether reading it found a fault; a kind not read here is its fault.
    private Built Read(Described layer)
    {
        var before = _faults.Count;
        KerasKind.RequireSinglePrecision(layer.Settings);

        var read = KerasKind.ByName.TryGetValue(layer.Class, out var kind) ? kind.Read(layer.Settings) : Unknown(layer);

        return new Built(layer, read, _faults.Count > before);
    }

    private static KerasLayer Unknown(Described layer)
    {
        layer.Settings.Refuse($"'{layer.Class.Quoted()}' is no kind a network here is built of: a Keras model is read here when each of its layers is a {KerasKind.Listed}.");

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
    // numbers go to: the layer's place in the stack the description lowers onto.
    private void Write(List<Built> built, bool lifted)
    {
        var count = 0;

        for (var at = 0; at < built.Count; at++)
        {
            var (described, layer, faulted) = built[at];

            if (faulted)
            {
                continue;
            }

            _placed[described.Name] = new Placed(count, layer);

            if (layer.Words is { } words)
            {
                try
                {
                    words(_description);
                    count++;
                }
                catch (ArgumentException refused)
                {
                    described.Settings.Refuse(refused.Message.ReplaceLineEndings(" "));
                }
            }

            if (at == built.Count - 1 && lifted)
            {
                continue;
            }

            if (!Activations.TryGetValue(layer.Activation, out var activation))
            {
                described.Settings.Refuse(
                    $"its activation, {layer.Activation.Quoted()}, is none of those read here: linear, relu, tanh and sigmoid, and softmax as the last of a model a categorical cross-entropy trained.");
            }
            else if (activation is not null)
            {
                activation(_description);
                count++;
            }
        }

        if (count == 0)
        {
            _faults.Add($"{_where}: the model holds no layer a network here is built of.");
        }
    }

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
    private readonly record struct Described(string Name, string Class, KerasSettings Settings);

    /// <summary>A layer read by its kind, and whether reading it noted a fault.</summary>
    private readonly record struct Built(Described Described, KerasLayer Layer, bool Faulted);

    /// <summary>A layer written into the description, and its place in the stack it lowers onto.</summary>
    private readonly record struct Placed(int Place, KerasLayer Layer);
}
