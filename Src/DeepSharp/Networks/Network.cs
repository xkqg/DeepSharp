// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A whole network: what is compiled, trained, saved and served.
/// </summary>
/// <remarks>
/// Either door leads here. A stack of layers — written with <see cref="LayerStack"/>, or described in Keras's words and
/// built into one — is a network; a network written as code, the way PyTorch has it, derives from this class, adds its
/// layers in its constructor and writes its forward pass in <see cref="Layer.Compute"/>. Nothing downstream asks which door
/// a network came through.
/// </remarks>
public abstract class Network : Layer
{
    /// <summary>Makes the network ready to be fitted: what moves its parameters, what it is trained to bring down, and how the rate changes.</summary>
    /// <param name="optimizer">What moves its parameters.</param>
    /// <param name="loss">What it is trained to bring down: named, never assumed.</param>
    /// <param name="schedule">How the optimizer's rate changes from epoch to epoch; it stays as it is, unless said.</param>
    /// <returns>The compiled network, which alone is fitted and predicts.</returns>
    /// <exception cref="ArgumentException">
    /// The network is a stack whose last layer is the activation the loss applies to its outputs itself — a
    /// <see cref="Sigmoid"/> before a <see cref="BinaryCrossEntropy"/>, or a <see cref="Softmax"/> before a <see cref="CrossEntropy"/> —
    /// which would send every prediction through it twice.
    /// </exception>
    /// <remarks>
    /// Keras's <c>compile</c>, in its order: the optimizer first, then the loss. A stack says what its last layer is — a stack
    /// within it is read down to its own last — and a network written as code writes its own forward pass, which is not run
    /// or guessed at here, so it is compiled as it is written. A network read from a file keeps the loss it was written
    /// with, and is read and served as it was written.
    /// </remarks>
    public CompiledNetwork Compile(Optimizer optimizer, Loss loss, LearningRateSchedule? schedule = null)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(loss);

        var last = LastDeclared(this);

        if (loss.Applies(last))
        {
            var applied = last.GetType().Name;
            var named = applied.ToLowerInvariant();

            throw new ArgumentException(
                $"{loss.GetType().Name} applies the {named} to the network's outputs itself, and this network's last layer, '{last.Path}', is a {applied}: "
                + $"every prediction would go through the {named} twice. Leave that layer out; the predictions still come out through the loss's {named}.",
                nameof(loss));
        }

        return new CompiledNetwork(this, optimizer, loss, schedule);
    }

    /// <summary>What the network predicts for the given rows: one evaluation pass on the engine handed in, through a loss's output activation.</summary>
    /// <param name="features">The rows, as the network takes them.</param>
    /// <param name="loss">What it was trained to bring down, whose output activation every prediction goes through.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>A row of predictions for each row: numbers, shares or probabilities, as the loss has them.</returns>
    /// <remarks>
    /// The one way a network here answers rows. A compiled network predicts by it, and a network trained behind a pipeline
    /// is measured by the pipeline's report and serves rows by it, so each door answers alike on every engine.
    /// </remarks>
    public Tensor Predict(Tensor features, Loss loss, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(backend);

        return loss.Predictions(Forward(features, Pass.Evaluation(backend)), backend);
    }

    /// <summary>Puts numbers read from a file into the network's slots, each by the path of its slot: all of them, or none.</summary>
    /// <param name="entries">A number for every slot of the network, each with the path of its slot and where its file holds it.</param>
    /// <exception cref="SlotLoadException">
    /// The numbers do not fit the network — a slot left out, a path the network has no slot at, a path handed twice, a tensor
    /// of another shape than its slot, a value that is not a finite number — every fault at once, each where its file holds
    /// it; and no slot is changed.
    /// </exception>
    /// <exception cref="ArgumentException">An entry leaves out its path, its tensor or where its file holds it.</exception>
    /// <remarks>
    /// How numbers trained somewhere else go into a network here, the way PyTorch's <c>load_state_dict</c> puts a state into
    /// a module with <c>strict=True</c>: every <see cref="IImporter"/> ends here, and a network's own file is held to the same
    /// rule, in the same words. The numbers are put in as they are, so a reader turns them into the layout each slot keeps
    /// before it hands them over.
    /// </remarks>
    public void Load(IEnumerable<SlotEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);

        var load = new SlotLoad(Slots());

        foreach (var entry in entries)
        {
            load.Take(entry);
        }

        var faults = load.Faults;

        if (faults.Count > 0)
        {
            throw new SlotLoadException(faults);
        }

        load.Apply();
    }

    /// <summary>What every slot of the network holds now, to be brought back later.</summary>
    /// <returns>The snapshot: the tensor in each slot, parameters and running statistics alike.</returns>
    /// <remarks>
    /// It holds the tensors themselves, which never change, so it costs no copy — keeping the best epoch's weights costs a
    /// list of references.
    /// </remarks>
    public Snapshot Snapshot() => new(this, [.. Slots().Select(named => new SlotValue(named.Slot, named.Slot.Value))]);

    /// <summary>Puts back what every slot held when a snapshot was taken.</summary>
    /// <param name="snapshot">A snapshot of this network.</param>
    /// <exception cref="ArgumentException">The snapshot was taken of another network.</exception>
    public void Restore(Snapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        if (!ReferenceEquals(snapshot.Of, this))
        {
            throw new ArgumentException("A snapshot brings back the network it was taken of, and this is another.", nameof(snapshot));
        }

        foreach (var kept in snapshot.Values)
        {
            kept.Slot.Replace(kept.Value);
        }
    }

    // The last layer a network declares: a stack's last, and within a stack that is one, its own last. A network written as
    // code declares no order — its forward pass is its own code — and is its own last.
    private static Layer LastDeclared(Layer layer) => layer is LayerStack stack ? LastDeclared(stack.Layers[^1]) : layer;
}

/// <summary>What every slot of a network held at one moment: a best epoch kept, a state to go back to.</summary>
/// <remarks>Taken with <see cref="Network.Snapshot"/> and brought back with <see cref="Network.Restore"/>, on that network alone.</remarks>
public sealed class Snapshot
{
    internal Snapshot(Network of, IReadOnlyList<SlotValue> values)
    {
        Of = of;
        Values = values;
    }

    internal Network Of { get; }

    internal IReadOnlyList<SlotValue> Values { get; }
}

/// <summary>
/// Steps that run one after another, each handed what the one before it made — the stack Keras's words describe — or, where a step
/// says so, what an earlier step made, or what several made: the network a residual block, a block that reads everything made
/// before it, or one shaped like a U is.
/// </summary>
/// <remarks>
/// Each step is a layer under a name — its place in the stack, counted from nought, unless it is called something — and takes the
/// work of the step before it unless it names others. The last step is what the network gives; every other step must be taken by
/// a later one, or its work would be made and never used. A layer belongs to one step, so what a step made is handed on and the
/// layer is never run twice. A stack of layers alone is the straight stack it always was, written to its file as it always was.
/// </remarks>
public sealed class LayerStack : Network, ISaved<LayerStack>
{
    private const string NamesKey = "names";
    private const string TakesKey = "takes";

    private readonly PlacedStep[] _steps;
    private readonly bool _named;
    private readonly bool _branches;

    /// <summary>A stack of the given layers, in their order.</summary>
    /// <param name="layers">The layers; at least one, each held by no other network.</param>
    /// <exception cref="ArgumentException">There are no layers, or one is held by another network already.</exception>
    public LayerStack(params Layer[] layers)
        : this((IEnumerable<Layer>)layers)
    {
    }

    /// <summary>A stack of the given layers, in their order.</summary>
    /// <param name="layers">The layers; at least one, each held by no other network.</param>
    /// <exception cref="ArgumentException">There are no layers, or one is held by another network already.</exception>
    public LayerStack(IEnumerable<Layer> layers)
        : this(Straight(layers))
    {
    }

    /// <summary>A stack of the given steps, in their order.</summary>
    /// <param name="steps">The steps; at least one. Each takes the work of the step before it unless it names others, and each but the last is taken by a later one.</param>
    /// <exception cref="ArgumentException">
    /// There are no steps; a name is empty, holds a dot or a comma, is a whole number or is <see cref="Step.Input"/>, or is a name twice;
    /// a step takes one that is not before it; a layer takes more steps than it works on, or fewer than it joins; a step is taken by no
    /// later one; or a layer is held by another network already.
    /// </exception>
    public LayerStack(IEnumerable<Step> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        var asked = steps.ToArray();

        if (asked.Length == 0)
        {
            throw new ArgumentException("A stack holds at least one layer.", nameof(steps));
        }

        _steps = Placed(asked);
        _named = asked.Any(step => step.Name is not null);
        _branches = _steps.Select((step, place) => step.Takes.SequenceEqual([place == 0 ? Step.Input : _steps[place - 1].Name])).Any(straight => !straight);

        foreach (var step in _steps)
        {
            AddLayer(step.Name, step.Layer);
        }
    }

    /// <summary>The layers, in the order they run.</summary>
    /// <remarks>
    /// The stack's own layers alone; <see cref="Layer.HeldLayers()"/>, called, walks every layer below it with its path, as for
    /// any layer.
    /// </remarks>
    public IReadOnlyList<Layer> Layers => [.. _steps.Select(step => step.Layer)];

    /// <summary>The steps, in the order they run, each with the name it goes by and the names of the steps it takes.</summary>
    public IReadOnlyList<PlacedStep> Steps => _steps;

    /// <summary>The shape that reaches each step, a batch of one example: an example of nothing sent through the layers themselves.</summary>
    /// <param name="each">The shape of one example, without the batch's axis.</param>
    /// <returns>For each step in order, the shape of the work of the first step it takes — the one that reaches a layer that works on one, the first of those a join lays together.</returns>
    /// <remarks>Worked out nowhere else than in the layers, on the light engine, so the shape each makes is the one it makes.</remarks>
    public IReadOnlyList<Shape> ShapesReaching(Shape each)
    {
        var pass = Pass.Evaluation(new CpuBackend());
        var made = new Dictionary<string, Tensor>(StringComparer.Ordinal) { [Step.Input] = Tensor.Zeros(new Shape([1, .. each.Axes])) };
        var shapes = new List<Shape>(_steps.Length);

        foreach (var step in _steps)
        {
            var taken = step.Takes.Select(name => made[name]).ToList();

            shapes.Add(taken[0].Shape);
            made[step.Name] = step.Layer.Taking(taken, pass);
        }

        return shapes;
    }

    /// <inheritdoc />
    public static string Name => "stack";

    /// <inheritdoc />
    /// <remarks>The eighth when a step is named or takes the work of a step other than the one before it, which an older library cannot read; the second otherwise.</remarks>
    public int FileVersion => _named || _branches ? NetworkDocument.EighthVersion : NetworkDocument.SecondVersion;

    /// <inheritdoc />
    public static LayerStack Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var layers = rebuilding.Layers(settings, "layers");
        var names = rebuilding.Words(settings, NamesKey, []);
        var takes = rebuilding.WordLists(settings, TakesKey, []);

        if (names.Count > 0 && names.Count != layers.Count)
        {
            throw rebuilding.Refuse(NamesKey, $"one name for each of the {layers.Count} layers, and these are {names.Count}");
        }

        if (takes.Count > 0 && takes.Count != layers.Count)
        {
            throw rebuilding.Refuse(TakesKey, $"one list of steps for each of the {layers.Count} layers, and these are {takes.Count}");
        }

        try
        {
            return new(layers.Select((layer, place) => new Step(
                layer,
                names.Count > 0 && names[place] != place.ToString(CultureInfo.InvariantCulture) ? names[place] : null,
                takes.Count > 0 ? takes[place] : null)));
        }
        catch (ArgumentException refused)
        {
            throw rebuilding.Refuse(takes.Count > 0 ? TakesKey : NamesKey, refused.Message);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Its layers, each written as the kind it is, in their order — and, only where a step is named or takes another step than the
    /// one before it, the names of the steps and the names each takes, which a stack of layers alone has no need of.
    /// </remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartArray("layers");

        foreach (var step in _steps)
        {
            NetworkDocument.WriteKind(writer, step.Layer);
        }

        writer.WriteEndArray();

        if (_named)
        {
            writer.WriteStartArray(NamesKey);

            foreach (var step in _steps)
            {
                writer.WriteStringValue(step.Name);
            }

            writer.WriteEndArray();
        }

        if (_branches)
        {
            writer.WriteStartArray(TakesKey);

            foreach (var step in _steps)
            {
                writer.WriteStartArray();

                foreach (var taken in step.Takes)
                {
                    writer.WriteStringValue(taken);
                }

                writer.WriteEndArray();
            }

            writer.WriteEndArray();
        }
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        var made = new Dictionary<string, Tensor>(StringComparer.Ordinal) { [Step.Input] = input };
        var last = input;

        foreach (var step in _steps)
        {
            last = step.Layer.Taking([.. step.Takes.Select(taken => made[taken])], pass);
            made[step.Name] = last;
        }

        return last;
    }

    private static IEnumerable<Step> Straight(IEnumerable<Layer> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);

        return layers.Select(layer => new Step(layer));
    }

    // Each step with its name and what it takes worked out, and the rules that make a stack of them one network held to.
    private static PlacedStep[] Placed(Step[] asked)
    {
        var placed = new PlacedStep[asked.Length];
        var known = new HashSet<string>(StringComparer.Ordinal) { Step.Input };

        for (var place = 0; place < asked.Length; place++)
        {
            var step = asked[place];
            var name = step.Name is { } given ? RequireName(given) : place.ToString(CultureInfo.InvariantCulture);

            if (!known.Add(name))
            {
                throw new ArgumentException($"Two steps are called '{name}': a step is known by its name to the steps that take its work.", nameof(asked));
            }

            IReadOnlyList<string> takes = step.Takes is { } said ? [.. said] : [place == 0 ? Step.Input : placed[place - 1].Name];

            foreach (var taken in takes)
            {
                if (taken == name || !known.Contains(taken))
                {
                    throw new ArgumentException($"The step '{name}' takes '{taken}', which is not the input or a step before it.", nameof(asked));
                }
            }

            if (!(step.Layer ?? throw new ArgumentException($"The step '{name}' has no layer.", nameof(asked))).Accepts(takes.Count))
            {
                throw new ArgumentException(
                    $"The step '{name}' takes the work of {takes.Count} steps, and a {step.Layer.GetType().Name} works on {(step.Layer.Accepts(1) ? "one" : "two or more")}.",
                    nameof(asked));
            }

            placed[place] = new PlacedStep(name, step.Layer, takes);
        }

        var taken2 = placed.SelectMany(step => step.Takes).ToHashSet(StringComparer.Ordinal);

        foreach (var step in placed[..^1])
        {
            if (!taken2.Contains(step.Name))
            {
                throw new ArgumentException(
                    $"The step '{step.Name}' is taken by no later step, so its work would be made and never used: only the last step is what the network gives.",
                    nameof(asked));
            }
        }

        return placed;
    }

    internal static string RequireName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)
            || name.Contains('.', StringComparison.Ordinal)
            || name.Contains(',', StringComparison.Ordinal)
            || int.TryParse(name, NumberStyles.None, CultureInfo.InvariantCulture, out _)
            || name == Step.Input)
        {
            throw new ArgumentException(
                $"'{name}' cannot name a step: a name is some text with no dot or comma in it, which is not a whole number — those are the places of the steps nobody named — and is not '{Step.Input}'.",
                nameof(name));
        }

        return name;
    }
}

/// <summary>A slot and the tensor it held when a snapshot was taken.</summary>
internal readonly record struct SlotValue(Slot Slot, Tensor Value);
