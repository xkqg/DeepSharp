// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// A file PyTorch saved a network's state in, read into the same network written here: every tensor into the slot its
/// name names — <c>0.weight</c>, <c>1.running_mean</c> — turned into the layout that slot keeps. Read one as the file it is:
/// <see cref="SafetensorsFile"/>, <see cref="TorchSaveFile"/>.
/// </summary>
/// <remarks>
/// <para>
/// The file holds numbers alone, so the reader is handed the network they belong to — a <see cref="LayerStack"/>, as the
/// Keras words build one or as it is written by hand, its layers numbered as PyTorch's <c>Sequential</c> numbers its modules,
/// or a network written as code, its layers named as the PyTorch module named them — and the loss it answers through, and
/// hands both back once the numbers are in. How PyTorch lays out each number is read off the layer that holds its slot,
/// never off the numbers' shape: a linear layer's weights are kept outputs by inputs there and inputs by outputs here, so
/// they are turned round; a convolution's kernel, along one, two or three axes, is kept channels out, channels in, then the
/// window's steps, rows and columns there, and here the window's places and then the channels in by the channels out;
/// biases and what a normalisation keeps are kept alike. PyTorch lays a series, an image or a volume out channel by channel
/// and this library place by place, so the rows a flatten makes of them are ordered otherwise there, and every number the next
/// linear layer and any normalisation before it keeps for them is put in its place here — which takes knowing what the
/// flatten is handed: an <see cref="Example"/> the network takes, run through it as zeros, or <see cref="Flattened"/> stating
/// it; told neither, those numbers are refused. The count of batches PyTorch keeps beside a batch normalisation's statistics,
/// <c>num_batches_tracked</c>, is left out by its name, since nothing here keeps it; everything else the file holds is put in
/// or refused.
/// </para>
/// <para>
/// A pooling, a global pooling and a dropout of whole channels hold no numbers, in PyTorch or here, so a file has no tensor
/// for one, and a tensor named for one is for no slot. What they do is not in the file: PyTorch's average pooling counts the
/// cells it pads with, so the network written here sets <see cref="AveragePooling.CountsPadding"/> for one that pads; its max
/// pooling never sees them. A convolution that PyTorch walks in groups keeps a kernel of fewer channels in than its layer does,
/// and is refused as a tensor of another shape. The strides, dilations and padding of a layer are not in the file at all, but
/// in the network written here, which says no dilation, a stride only as one for every axis, and a border only as one number on
/// every side or as 'same' or 'causal' work it out.
/// </para>
/// <para>
/// Numbers held as 16-bit floats or bfloat16 are widened to 32 bits exactly; 64-bit floats are rounded to the nearest
/// 32-bit float, as PyTorch rounds them, and one too large for 32 bits becomes an infinity, which is refused as a number
/// that is not finite. Whole numbers and truth values are refused.
/// </para>
/// <para>
/// The file is read whole into memory, from where it stands: at most as many bytes as one array holds,
/// <see cref="Array.MaxLength"/>. A longer file is refused by name — before a byte of it is read when the stream can say
/// its length, and where the reading passes that many when it cannot.
/// </para>
/// <para>
/// Every tensor goes into the network through <see cref="Network.Load"/>, so a tensor for no slot, a slot no tensor is for,
/// a shape other than the one PyTorch keeps the slot in, a kind of number that is not a float and a value that is not
/// finite are refused together, with a <see cref="SlotLoadException"/> naming each by its tensor, and the network keeps
/// what it held.
/// </para>
/// <para>
/// A network written as code shows the layers it holds, through <see cref="Layer.HeldLayers()"/>, and keeps to itself the
/// order its forward pass runs them in. So its linear layers' weights are turned round and its convolutions' kernels laid out
/// as for a stack; but whether one of its linear layers or normalisations reads rows made of series, images or volumes is
/// never guessed: where such rows may reach it — the network holds a convolution, a pooling, a global pooling, a dropout of
/// whole channels, a flatten or a reshape, or a flatten in the stack around it hands it rows of those or of what nobody said —
/// it is read only as <see cref="Flattened"/> states it, by that layer's path, and refused otherwise. A network written as
/// code that lays rows out as series, images or volumes with a reshape has its linear layers and normalisations refused
/// whatever is stated, as a stack's linear layer making such rows is: PyTorch reads them channel by channel. Every number of
/// a layer of a kind this reader does not know, and every number a network written as code keeps of its own, is refused. A
/// layer of your own that holds no numbers is taken to hand on what it is handed in the order it is handed it.
/// </para>
/// </remarks>
public abstract class PyTorchFile : IImporter
{
    private readonly Network _network;
    private readonly Loss _loss;
    private readonly IReadOnlyDictionary<string, Shape> _flattened = new Dictionary<string, Shape>(StringComparer.Ordinal);

    /// <summary>A reader of a file into the network its numbers belong to.</summary>
    /// <param name="network">The network the numbers belong to, its layers numbered as PyTorch numbered its modules.</param>
    /// <param name="loss">What it answers through, handed back with it.</param>
    /// <exception cref="ArgumentNullException">The network or the loss is missing.</exception>
    private protected PyTorchFile(Network network, Loss loss)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(loss);

        _network = network;
        _loss = loss;
    }

    /// <summary>
    /// The shape of one example the network takes, without the batch's axis — <c>new Shape(28, 28, 1)</c> for an image,
    /// <c>new Shape(100, 3)</c> for a series — run through the network as zeros to find what each flatten is handed; nothing,
    /// unless said.
    /// </summary>
    public Shape? Example { get; init; }

    /// <summary>
    /// What layers are handed, by the layer's path — steps and channels for rows a flatten made of a series, rows, columns and
    /// channels for an image, planes, rows, columns and channels for a volume, one length for a row — stated for a stack's
    /// flatten in place of an <see cref="Example"/>, and for a linear layer or normalisation of a network written as code,
    /// whose order no example shows. Nothing, unless said.
    /// </summary>
    /// <exception cref="ArgumentNullException">The statements are missing.</exception>
    public IReadOnlyDictionary<string, Shape> Flattened
    {
        get => _flattened;
        init => _flattened = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// What was said of the network does not hold: the example does not go through it; a statement names neither a stack's
    /// flatten nor a linear layer or normalisation of a network written as code, states none of a row, a series, an image
    /// and a volume, or states another number of values than the layer reading them reads; or an example is handed and a
    /// flatten's statement besides.
    /// </exception>
    public SavedNetwork Read(Stream file)
    {
        ArgumentNullException.ThrowIfNull(file);

        var layout = new TorchLayout(_network, new WhatIsSaid(Example, Flattened));

        layout.Load(Tensors(file));

        return new SavedNetwork(_network, _loss);
    }

    /// <summary>The file's tensors by name, in the order it holds them, each as the file holds it.</summary>
    /// <param name="file">The file.</param>
    /// <exception cref="FormatException">The file is none of the kind this reader reads.</exception>
    private protected abstract IReadOnlyList<StoredTensor> Tensors(Stream file);

    /// <summary>The file read whole into memory, from where it stands, and wound back to its start.</summary>
    /// <param name="file">The file.</param>
    /// <exception cref="FormatException">The file holds more bytes than one array does.</exception>
    private protected static MemoryStream Whole(Stream file)
    {
        if (file.CanSeek && file.Length - file.Position > Array.MaxLength)
        {
            throw TooLong();
        }

        var bytes = new MemoryStream();
        var buffer = new byte[81920];

        for (var read = file.Read(buffer); read > 0; read = file.Read(buffer))
        {
            if (bytes.Length + read > Array.MaxLength)
            {
                throw TooLong();
            }

            bytes.Write(buffer, 0, read);
        }

        bytes.Position = 0;

        return bytes;
    }

    private static FormatException TooLong() =>
        new($"The file holds more than {Array.MaxLength} bytes, the most a file read here whole into memory holds.");
}
