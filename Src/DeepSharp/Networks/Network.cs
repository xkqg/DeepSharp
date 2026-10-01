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
    /// <see cref="Sigmoid"/> before a <see cref="BinaryCrossEntropy"/> — which would send every prediction through it twice.
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
            throw new ArgumentException(
                $"{loss.GetType().Name} applies the sigmoid to the network's outputs itself, and this network's last layer, '{last.Path}', is a Sigmoid: "
                + "every prediction would go through the sigmoid twice. Leave that layer out; the predictions still come out through the loss's sigmoid.",
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

/// <summary>Layers run one after another, each handed what the one before it made: the stack Keras's words describe.</summary>
/// <remarks>Each layer is named by its place in the stack, from nought, so its slots' paths start with it: <c>0.weight</c>.</remarks>
public sealed class LayerStack : Network, ISaved<LayerStack>
{
    private readonly Layer[] _layers;

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
    {
        ArgumentNullException.ThrowIfNull(layers);

        _layers = [.. layers];

        if (_layers.Length == 0)
        {
            throw new ArgumentException("A stack holds at least one layer.", nameof(layers));
        }

        for (var place = 0; place < _layers.Length; place++)
        {
            AddLayer(place.ToString(CultureInfo.InvariantCulture), _layers[place]);
        }
    }

    /// <summary>The layers, in the order they run.</summary>
    /// <remarks>
    /// The stack's own layers alone; <see cref="Layer.HeldLayers()"/>, called, walks every layer below it with its path, as for
    /// any layer.
    /// </remarks>
    public IReadOnlyList<Layer> Layers => _layers;

    /// <inheritdoc />
    public static string Name => "stack";

    /// <inheritdoc />
    public static LayerStack Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Layers(settings, "layers"));
    }

    /// <inheritdoc />
    /// <remarks>Its layers, each written as the kind it is, in their order.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartArray("layers");

        foreach (var layer in _layers)
        {
            NetworkDocument.WriteKind(writer, layer);
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => _layers.Aggregate(input, (handed, layer) => layer.Forward(handed, pass));
}

/// <summary>A slot and the tensor it held when a snapshot was taken.</summary>
internal readonly record struct SlotValue(Slot Slot, Tensor Value);
