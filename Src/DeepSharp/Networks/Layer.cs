// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A piece of a network: a forward pass written in the backend's operations, the numbers it learns, and the layers it
/// holds.
/// </summary>
/// <remarks>
/// A layer says what it holds as it is built — <see cref="AddParameter"/>, <see cref="AddRunningStatistic"/>,
/// <see cref="AddLayer{TLayer}"/> — so every number a network learns has a path made of the names on the way to it:
/// <c>0.weight</c> in the first layer of a stack, <c>hidden.weight</c> in the layer a network written as code calls hidden.
/// Nothing is found by looking the class over, and nothing is registered anywhere else.
/// <para>
/// Its gradient is not written here: the pass runs on whatever backend it is handed, and a recording backend works the
/// gradient out from the operations the pass ran. Whether the pass trains or only uses the network is the pass's to say,
/// never the layer's, so no layer carries a mode that could be left switched on.
/// </para>
/// </remarks>
public abstract class Layer
{
    private readonly List<OwnSlot> _slots = [];
    private readonly List<HeldLayer> _layers = [];
    private Layer? _holder;
    private string _name = string.Empty;

    /// <summary>
    /// Where this layer stands in the network that holds it: the names on the way down, dotted; nothing for a layer no
    /// other holds.
    /// </summary>
    public string Path => _holder is null ? string.Empty : Joined(_holder.Path, _name);

    /// <summary>Runs the layer forward over a batch.</summary>
    /// <param name="input">What the layer is handed: a row for each example.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>What the layer makes of it.</returns>
    /// <exception cref="ArgumentNullException">The input or the pass is missing.</exception>
    public Tensor Forward(Tensor input, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(pass);

        return Compute(input, pass);
    }

    /// <summary>Every number this layer and the layers it holds learn, in the order they were added.</summary>
    /// <returns>The parameters, each once; running statistics are not among them, since no optimizer moves them.</returns>
    public IEnumerable<Parameter> Parameters() => Slots().Select(named => named.Slot).OfType<Parameter>();

    /// <summary>Every slot this layer and the layers it holds keep, with its path from this layer down.</summary>
    /// <returns>This layer's own slots in the order they were added, then each held layer's, in the order those were added.</returns>
    public IEnumerable<NamedSlot> Slots()
    {
        foreach (var own in _slots)
        {
            yield return new NamedSlot(own.Name, own.Slot);
        }

        foreach (var held in _layers)
        {
            foreach (var named in held.Layer.Slots())
            {
                yield return named with { Path = Joined(held.Name, named.Path) };
            }
        }
    }

    /// <summary>Every layer this layer holds, and every layer those hold, each with its path from this layer down.</summary>
    /// <returns>
    /// Each held layer in the order it was added, followed at once by the layers it holds: the order <see cref="Slots"/>
    /// walks them in, so the layer holding a slot is the one named by the slot's path without its last name.
    /// </returns>
    /// <remarks>
    /// What a layer holds is what it said it holds as it was built; nothing is found by looking the class over. A network
    /// written as code shows its layers here, and keeps to itself only the order its forward pass runs them in.
    /// </remarks>
    public IEnumerable<NamedLayer> HeldLayers()
    {
        foreach (var held in _layers)
        {
            yield return new NamedLayer(held.Name, held.Layer);

            foreach (var below in held.Layer.HeldLayers())
            {
                yield return below with { Path = Joined(held.Name, below.Path) };
            }
        }
    }

    /// <summary>What this layer computes of its input: the forward pass, written in the pass's backend.</summary>
    /// <param name="input">What the layer is handed.</param>
    /// <param name="pass">The backend the arithmetic runs on, and whether this pass trains.</param>
    /// <returns>What the layer makes of it.</returns>
    /// <remarks>A network written as code writes its forward pass here, handing the pass on to the layers it holds.</remarks>
    protected abstract Tensor Compute(Tensor input, Pass pass);

    /// <summary>Holds another layer under a name, as part of this one.</summary>
    /// <typeparam name="TLayer">What kind of layer it is.</typeparam>
    /// <param name="name">Its name here: part of the path of everything it holds.</param>
    /// <param name="layer">The layer; held by no other.</param>
    /// <returns>The layer, so it can be kept in a field as it is added.</returns>
    /// <exception cref="ArgumentException">
    /// The name is empty or holds a dot, this layer already holds something of that name, or the layer is held by another,
    /// is this one, or holds this one.
    /// </exception>
    protected TLayer AddLayer<TLayer>(string name, TLayer layer)
        where TLayer : Layer
    {
        ArgumentNullException.ThrowIfNull(layer);
        RequireName(name);

        if (layer._holder is not null)
        {
            throw new ArgumentException(
                $"The layer named '{name}' is held already, at '{layer.Path}', and a layer belongs to one network.", nameof(layer));
        }

        for (var above = this; above is not null; above = above._holder)
        {
            if (ReferenceEquals(above, layer))
            {
                throw new ArgumentException($"The layer named '{name}' holds the one it would be added to.", nameof(layer));
            }
        }

        layer._holder = this;
        layer._name = name;
        _layers.Add(new HeldLayer(name, layer));

        return layer;
    }

    /// <summary>Keeps a number this layer learns, under a name.</summary>
    /// <param name="name">Its name here: the last part of its path.</param>
    /// <param name="initial">What it starts at.</param>
    /// <returns>The slot, so it can be kept in a field as it is added.</returns>
    /// <exception cref="ArgumentException">The name is empty or holds a dot, or this layer already holds something of that name.</exception>
    protected Parameter AddParameter(string name, Tensor initial)
    {
        RequireName(name);

        return Kept(name, new Parameter(name, initial));
    }

    /// <summary>Keeps a number this layer measures while it trains and uses afterwards, under a name.</summary>
    /// <param name="name">Its name here: the last part of its path.</param>
    /// <param name="initial">What it starts at.</param>
    /// <returns>The slot, so it can be kept in a field as it is added.</returns>
    /// <exception cref="ArgumentException">The name is empty or holds a dot, or this layer already holds something of that name.</exception>
    /// <remarks>
    /// Measured on the training rows alone, a batch at a time, and used unchanged afterwards — as the pipeline does with
    /// everything it learns. No optimizer moves it, and only a training pass does.
    /// </remarks>
    protected RunningStatistic AddRunningStatistic(string name, Tensor initial)
    {
        RequireName(name);

        return Kept(name, new RunningStatistic(name, initial));
    }

    private static string Joined(string above, string name) => above.Length == 0 ? name : $"{above}.{name}";

    private TSlot Kept<TSlot>(string name, TSlot slot)
        where TSlot : Slot
    {
        _slots.Add(new OwnSlot(name, slot));

        return slot;
    }

    private void RequireName(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Contains('.', StringComparison.Ordinal))
        {
            throw new ArgumentException(
                $"'{name}' cannot stand in a path: a name is some text, and a dot is what separates the names in one.", nameof(name));
        }

        if (_slots.Any(own => own.Name == name) || _layers.Any(held => held.Name == name))
        {
            throw new ArgumentException($"This layer holds something named '{name}' already.", nameof(name));
        }
    }

    private readonly record struct OwnSlot(string Name, Slot Slot);

    private readonly record struct HeldLayer(string Name, Layer Layer);
}

/// <summary>A slot and its path from the layer that was asked, dotted: <c>0.weight</c>.</summary>
/// <param name="Path">The names on the way down to the slot.</param>
/// <param name="Slot">The slot.</param>
public readonly record struct NamedSlot(string Path, Slot Slot);

/// <summary>A layer another holds, and its path from the layer that was asked, dotted: <c>1.hidden</c>.</summary>
/// <param name="Path">The names on the way down to the layer.</param>
/// <param name="Layer">The layer.</param>
public readonly record struct NamedLayer(string Path, Layer Layer);
