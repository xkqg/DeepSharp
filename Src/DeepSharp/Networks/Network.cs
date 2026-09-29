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
    /// <remarks>Keras's <c>compile</c>, in its order: the optimizer first, then the loss.</remarks>
    public CompiledNetwork Compile(Optimizer optimizer, Loss loss, LearningRateSchedule? schedule = null)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(loss);

        return new CompiledNetwork(this, optimizer, loss, schedule);
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
