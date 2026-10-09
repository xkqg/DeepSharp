// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// How PyTorch lays out every slot of one network, read off the network's own layers in the order they run, and the load of
/// a file's tensors into those slots: all of them, or none.
/// </summary>
/// <remarks>
/// The layout of a number is said by the layer that holds its slot, never guessed from the number's shape: a square
/// matrix turned round is as square as one that is not. A stack shows its layers and the order they run in; a network
/// written as code shows its layers too, and keeps to itself only that order, so whether one of its linear layers or
/// normalisations reads rows made of series, images or volumes is read only as the caller states it. A slot held by a layer
/// of a kind nobody here knows has no layout it could be read in, and is refused.
/// </remarks>
internal sealed class TorchLayout
{
    // What PyTorch keeps beside a batch normalisation's statistics: how many batches moved them, which no slot here keeps.
    private const string Counted = "num_batches_tracked";

    // What is handed for a tensor no slot is at: the load refuses such a path before it looks at the number.
    private static readonly Tensor Nothing = Tensor.Zeros(new Shape(0));

    private readonly Network _network;
    private readonly IReadOnlyDictionary<string, Shape> _stated;
    private readonly Dictionary<string, TorchSlot> _slots = new(StringComparer.Ordinal);
    private readonly HashSet<string> _counts = new(StringComparer.Ordinal);

    // The layers a statement can be made of: a stack's flattens, and the linear layers and normalisations code holds.
    private readonly HashSet<string> _statable = new(StringComparer.Ordinal);

    /// <summary>The layout of every slot of a network.</summary>
    /// <param name="network">The network the numbers go into.</param>
    /// <param name="said">What the caller says of it: an example it takes, and what layers are handed.</param>
    /// <exception cref="ArgumentException">What is said does not hold of the network.</exception>
    public TorchLayout(Network network, WhatIsSaid said)
    {
        _network = network;
        _stated = said.Flattened;

        var layers = PlacedLayer.Of(network).ToArray();

        Lay(layers, said.HandedTo(layers));

        if (said.Flattened.Keys.FirstOrDefault(path => !_statable.Contains(path)) is { } unknown)
        {
            throw new ArgumentException(
                $"'{unknown}' is neither a layer of a stack that makes rows of what it is handed nor a linear layer or normalisation of a network written as code, so nothing can be stated of it.",
                nameof(SafetensorsFile.Flattened));
        }
    }

    /// <summary>Puts a file's tensors into the network's slots, each turned into the layout its slot keeps.</summary>
    /// <param name="tensors">The file's tensors, in the order the file holds them.</param>
    /// <exception cref="SlotLoadException">
    /// Among the tensors: one that is no float, of another shape than PyTorch keeps its slot in, or of a layout not told, one
    /// for no slot, one not finite, or a slot no tensor is for — every fault at once, in the order the file holds them, and no
    /// slot changed.
    /// </exception>
    public void Load(IEnumerable<StoredTensor> tensors)
    {
        var entries = new List<SlotEntry>();
        var faults = new List<SlotLoadFault>();
        var places = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var tensor in tensors)
        {
            places[tensor.Name] = places.Count;

            if (_counts.Contains(tensor.Name))
            {
                continue;
            }

            if (!_slots.TryGetValue(tensor.Name, out var slot))
            {
                entries.Add(new SlotEntry(tensor.Name, Nothing, tensor.Name));

                continue;
            }

            var reading = slot.Read(tensor);

            if (reading.Fault is { } fault)
            {
                faults.Add(fault);
            }
            else
            {
                entries.Add(reading.Entry!.Value);
            }
        }

        // A fault found here names a slot whose tensor was kept back, so the load finds that slot missing and refuses too —
        // nothing goes in — and its "missing" gives way to the fault that kept the tensor back.
        try
        {
            _network.Load(entries);
        }
        catch (SlotLoadException refused) when (faults.Count > 0)
        {
            var kept = faults.Select(fault => fault.Slot).ToHashSet(StringComparer.Ordinal);

            throw new SlotLoadException(
            [
                .. faults.Concat(refused.Faults.Where(fault => fault.Source is not null)).OrderBy(fault => places[fault.Source!]),
                .. refused.Faults.Where(fault => fault.Source is null && !kept.Contains(fault.Slot)),
            ]);
        }
    }

    // Each slot's layout, the layers walked in the order they run: what a flatten was handed decides how the numbers read
    // after it are ordered, up to the linear layer that reads them; a reshape into images disowns the rows made before it.
    private void Lay(PlacedLayer[] layers, IReadOnlyDictionary<string, Shape> handed)
    {
        Flattening? flattened = null;
        string? untold = null;
        List<string> producers = [];

        // Whether the layer that walks axes nearest before the one being laid walks one axis, with no reshape or linear layer between:
        // what a flatten is handed in two axes is steps and channels only then.
        var afterASeries = false;

        foreach (var placed in layers)
        {
            var (path, layer) = placed;
            var slots = layer.Slots().Select(named => named with { Path = PlacedLayer.Joined(path, named.Path) }).ToArray();

            switch (layer)
            {
                case Dense dense:
                    var inputs = flattened?.OrderFor(path, dense.Inputs);

                    foreach (var named in slots)
                    {
                        Lay(ReferenceEquals(named.Slot, dense.Weight)
                            ? untold is null ? new LinearSlot(named.Path, dense, inputs) : new UntoldSlot(named.Path, untold)
                            : new VectorSlot(named.Path, dense.Outputs, null));
                    }

                    producers = [.. slots.Select(named => named.Path)];
                    (flattened, untold) = (null, null);
                    afterASeries = false;
                    break;

                case Convolution convolution:
                    foreach (var named in slots)
                    {
                        Lay(ReferenceEquals(named.Slot, convolution.Weight)
                            ? new KernelSlot(named.Path, convolution)
                            : new VectorSlot(named.Path, convolution.OutChannels, null));
                    }

                    producers.Clear();
                    afterASeries = convolution.WalksOneAxis;
                    break;

                case Normalisation norm:
                    var features = flattened?.OrderFor(path, norm.Features);

                    foreach (var named in slots)
                    {
                        Lay(untold is null ? new VectorSlot(named.Path, norm.Features, features) : new UntoldSlot(named.Path, untold));
                    }

                    producers.AddRange(slots.Select(named => named.Path));

                    if (norm is BatchNorm)
                    {
                        _counts.Add(PlacedLayer.Joined(path, Counted));
                    }

                    break;

                case var _ when placed.Flattens:
                    _statable.Add(path);
                    flattened = handed.TryGetValue(path, out var shape) && shape.IsHandableAfter(afterASeries) ? new Flattening(path, shape) : null;
                    untold = flattened is not null ? null
                        : handed.TryGetValue(path, out var unread)
                            ? unread.Rank == 2
                                ? $"reads the rows the layer at {path} makes of {unread}, which is no series, since no layer along one axis made it, and no image or volume: the channels of a series, an image or a volume are turned to the end, and a row is read as it is."
                                : $"reads the rows the layer at {path} makes of {unread}, which is neither a row, a series, an image nor a volume: the channels of a series, an image or a volume are turned to the end, and a row is read as it is."
                            : $"reads the rows the layer at {path} flattens, which PyTorch lays out channel by channel: hand the reader an example the network takes, or state what that layer is handed.";
                    afterASeries = false;
                    break;

                case Reshape reshape:
                    foreach (var producer in producers)
                    {
                        Lay(new UntoldSlot(producer, $"makes the rows the layer at {path} lays out as {reshape.Each}, which PyTorch lays out channel by channel: rows a flatten makes of images are turned, and images a reshape makes of rows are not."));
                    }

                    producers.Clear();
                    (flattened, untold) = (null, null);
                    afterASeries = false;
                    break;

                case Network code:
                    // Rows a flatten before it made of series, images or volumes, or of what nobody said, may reach any layer it holds.
                    LayCode(path, code, handedImages: untold is not null || flattened is { Handed.Rank: > 1 });
                    producers = [.. slots.Select(named => named.Path)];
                    (flattened, untold) = (null, null);
                    afterASeries = false;
                    break;

                default:
                    // Any other layer: one holding nothing — an activation, a pooling, a dropout of values or of whole channels, one of
                    // somebody's own — hands on what it is handed as it is handed it, channels last; how PyTorch lays out the numbers
                    // of one holding any is not told.
                    foreach (var named in slots)
                    {
                        Lay(new UntoldSlot(named.Path, OfAnUnknownKind(layer)));
                    }

                    afterASeries = layer.WalksAxes ? layer.WalksOneAxis : afterASeries;
                    break;
            }
        }
    }

    // Each slot of a network written as code, by the layer holding it: the layers show their kinds, and the code keeps to
    // itself the order it runs them in — so what a linear layer or a normalisation reads is read as stated, or, where rows
    // made of images may reach it, refused; where the code lays rows out as images, they are refused whatever is stated.
    private void LayCode(string path, Network network, bool handedImages)
    {
        var held = network.HeldLayers().Select(named => named with { Path = PlacedLayer.Joined(path, named.Path) }).ToArray();
        var reshapesIntoImages = held.Any(named => named.Layer is Reshape { Each.Rank: > 1 });
        var hidesImages = handedImages
            || held.Any(named => named.Layer is Convolution or Pooling or GlobalPooling or SpatialDropout or Flatten or Reshape);

        foreach (var named in Own(path, network))
        {
            Lay(new UntoldSlot(named.Path, OfAnUnknownKind(network)));
        }

        foreach (var (at, layer) in held)
        {
            var own = Own(at, layer);

            switch (layer)
            {
                case Dense or Normalisation when reshapesIntoImages:
                    _statable.Add(at);

                    foreach (var named in own)
                    {
                        Lay(new UntoldSlot(named.Path,
                            "is held by a network written as code that lays rows out as images with a reshape, which PyTorch reads channel by channel, and its forward pass keeps to itself which rows those are: rows a flatten makes of images are turned, and images a reshape makes of rows are not."));
                    }

                    break;

                case Dense or Normalisation:
                    _statable.Add(at);

                    var reads = layer is Dense linear ? linear.Inputs : ((Normalisation)layer).Features;
                    var said = _stated.TryGetValue(at, out var stated) ? new Flattening(at, stated) : (Flattening?)null;
                    var order = said?.OrderFor(at, reads);
                    var untold = said is null && hidesImages
                        ? $"is held by a network written as code, whose forward pass keeps to itself whether the layer at {at} reads rows made of images, which PyTorch lays out channel by channel: state what that layer reads in Flattened — the steps and channels of a series, the rows, columns and channels of an image, the planes, rows, columns and channels of a volume, or the length of a row."
                        : null;

                    foreach (var named in own)
                    {
                        // A linear layer's bias holds its outputs, which are what they are whatever the layer reads.
                        Lay(layer is Dense outputs && !ReferenceEquals(named.Slot, outputs.Weight) ? new VectorSlot(named.Path, outputs.Outputs, null)
                            : untold is not null ? new UntoldSlot(named.Path, untold)
                            : layer is Dense reader ? new LinearSlot(named.Path, reader, order)
                            : new VectorSlot(named.Path, reads, order));
                    }

                    if (layer is BatchNorm)
                    {
                        _counts.Add(PlacedLayer.Joined(at, Counted));
                    }

                    break;

                case Convolution convolution:
                    foreach (var named in own)
                    {
                        Lay(ReferenceEquals(named.Slot, convolution.Weight)
                            ? new KernelSlot(named.Path, convolution)
                            : new VectorSlot(named.Path, convolution.OutChannels, null));
                    }

                    break;

                default:
                    foreach (var named in own)
                    {
                        Lay(new UntoldSlot(named.Path, OfAnUnknownKind(layer)));
                    }

                    break;
            }
        }
    }

    // The slots a layer keeps of its own, not those of the layers it holds, at their paths in the network.
    private static IEnumerable<NamedSlot> Own(string path, Layer layer) =>
        layer.Slots().Where(named => !named.Path.Contains('.', StringComparison.Ordinal)).Select(named => named with { Path = PlacedLayer.Joined(path, named.Path) });

    private static string OfAnUnknownKind(Layer layer) => $"is held by a {layer.GetType().Name}, and how PyTorch lays out a layer of that kind is not known here.";

    private void Lay(TorchSlot slot) => _slots[slot.Path] = slot;
}

/// <summary>A layer of a network, at its path: a stack's layers each at their place, a stack within it read down to its own.</summary>
/// <param name="Path">Its path in the network: <c>3</c>, <c>0.2</c>; nothing for a network that is no stack.</param>
/// <param name="Layer">The layer.</param>
internal readonly record struct PlacedLayer(string Path, Layer Layer)
{
    /// <summary>Whether it makes each example handed to it one row: a flatten, or a reshape into rows.</summary>
    public bool Flattens => Layer is Flatten or Reshape { Each.Rank: 1 };

    /// <summary>The layers of a network in the order they run, a stack within it read down to its layers.</summary>
    public static IEnumerable<PlacedLayer> Of(Layer layer, string path = "") =>
        layer is LayerStack stack
            ? stack.Layers.SelectMany((held, place) => Of(held, Joined(path, place.ToString(System.Globalization.CultureInfo.InvariantCulture))))
            : [new PlacedLayer(path, layer)];

    /// <summary>A name under a path, dotted.</summary>
    public static string Joined(string path, string name) => path.Length == 0 ? name : $"{path}.{name}";
}

/// <summary>What a flatten is handed, per example, and where it stands.</summary>
/// <param name="Path">The flatten's path.</param>
/// <param name="Handed">A series — steps and channels — an image — rows, columns and channels — a volume — planes, rows, columns and channels — or a row.</param>
internal readonly record struct Flattening(string Path, Shape Handed)
{
    /// <summary>The order the numbers read after the flatten are in, for a layer reading so many: that of a series, an image or a volume, or none for a row.</summary>
    /// <exception cref="ArgumentException">The flatten is said to be handed another number of values than the layer reads.</exception>
    public ChannelOrder? OrderFor(string reader, int reads)
    {
        if (Handed.Count != reads)
        {
            throw new ArgumentException(
                $"What the layer at {Path} is handed is stated as {Handed}, which holds {Handed.Count} values, and the layer at {reader} reads {reads}.",
                nameof(SafetensorsFile.Flattened));
        }

        var channels = Handed[Handed.Rank - 1];

        return Handed.Rank > 1 ? new ChannelOrder(Handed.Count / channels, channels) : null;
    }
}

/// <summary>What a flatten, or a layer written as code, can be handed.</summary>
internal static class HandedShapeExtensions
{
    extension(Shape shape)
    {
        /// <summary>
        /// Whether it is a shape this reader reads rows of: a row, or a series, an image or a volume, whose channels come last —
        /// at least one axis and at most four.
        /// </summary>
        public bool IsHandable => shape.Rank is >= 1 and <= 4;

        /// <summary>
        /// Whether it is a shape a flatten can be handed at the place it stands: one of three or four axes, an image or a volume, or
        /// a row, anywhere; of two axes — steps and channels — only where a layer along one axis made it, since two axes a reshape
        /// made are rows laid out as steps by features, which PyTorch flattens as they stand.
        /// </summary>
        /// <param name="afterASeries">Whether the layer that walks axes nearest before the flatten walks one axis, with no reshape or linear layer between.</param>
        public bool IsHandableAfter(bool afterASeries) => shape.Rank is 1 or 3 or 4 || (shape.Rank == 2 && afterASeries);
    }
}

/// <summary>Which layers walk axes, and along how many, as far as a reader of PyTorch's state needs to know.</summary>
internal static class WalkedAxesExtensions
{
    extension(Layer layer)
    {
        /// <summary>Whether it walks a window along some axes: a convolution, a pooling, a global pooling or a dropout of whole channels.</summary>
        public bool WalksAxes => layer is Convolution or Pooling or GlobalPooling or SpatialDropout;

        /// <summary>Whether it walks one axis, along a series.</summary>
        public bool WalksOneAxis => layer is Conv1D or MaxPool1D or AvgPool1D or GlobalMaxPool1D or GlobalAvgPool1D or SpatialDropout1D;
    }

    extension(IReadOnlyList<PlacedLayer> layers)
    {
        /// <summary>
        /// Whether what the layer at a path is handed was made by a layer along one axis: the layer that walks axes nearest before it
        /// walks one, with no reshape, flatten or linear layer between. A layer a network written as code holds is handed what the
        /// code makes of it, which nobody here can see, so it is handed a series where the code holds a layer along one axis.
        /// </summary>
        public bool EndsInASeries(string path)
        {
            var at = -1;

            for (var place = 0; place < layers.Count && at < 0; place++)
            {
                at = layers[place].Path == path ? place : -1;
            }

            if (at < 0)
            {
                return layers.Any(placed => placed.Layer is Network code && code.HeldLayers().Any(held => held.Layer.WalksOneAxis));
            }

            for (var before = at - 1; before >= 0; before--)
            {
                var layer = layers[before].Layer;

                if (layer.WalksAxes)
                {
                    return layer.WalksOneAxis;
                }

                if (layer is Reshape or Flatten or Dense or Network)
                {
                    return false;
                }
            }

            return false;
        }
    }
}
