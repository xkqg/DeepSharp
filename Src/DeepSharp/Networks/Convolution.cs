// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A convolution over a batch of images: a window walked over each, every place it stands given a value for each channel
/// the layer makes — the patch it covers, times a kernel, plus a bias.
/// </summary>
/// <remarks>
/// Images are laid out with their channels last — image, row, column, channel — as Keras lays them out, so a batch is
/// the patches of every image one after another and nothing has to be turned round. The kernel is as many rows as the
/// window holds values — its rows, its columns and the channels in — by as many columns as channels out; PyTorch keeps the
/// same numbers channels first. Its window pads each image by the border it states for every side, or as TensorFlow's 'same'
/// does.
/// </remarks>
public sealed class Conv2D : Convolution, ISaved<Conv2D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each place of an image holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a side or a stride below one, or a border below nothing.</exception>
    public Conv2D(int inChannels, int outChannels, Window window, Draws draws)
        : this(new Channels { In = inChannels, Out = outChannels }, window, draws)
    {
    }

    /// <summary>A convolution of channels in groups, that starts as PyTorch's does.</summary>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Conv2D(Channels channels, Window window, Draws draws)
        : base(new PlaneWalk(window), channels, draws) => Window = window;

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's places times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv2D(Tensor kernel, Tensor bias, Window window)
        : base(new PlaneWalk(window), kernel, bias, null) => Window = window;

    /// <summary>A convolution of channels in groups, that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values times the channels in of one group, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="channels">How many channels it reads and makes, and how many groups they are split into.</param>
    /// <exception cref="ArgumentException">The kernel is not the shape the window and the channels ask for, or the window cannot stand anywhere.</exception>
    public Conv2D(Tensor kernel, Tensor bias, Window window, Channels channels)
        : base(new PlaneWalk(window), kernel, bias, channels) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window Window { get; }

    /// <inheritdoc />
    public static string Name => "conv2d";

    /// <inheritdoc />
    /// <remarks>Its padding is the number of rows and columns on every side, or the word <c>same</c> or <c>causal</c>.</remarks>
    public static Conv2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            new Channels
            {
                In = rebuilding.Whole(settings, "inChannels"),
                Out = rebuilding.Whole(settings, "outChannels"),
                Groups = rebuilding.Whole(settings, "groups", 1),
            },
            rebuilding.Window2DIn(settings),
            rebuilding.Draws);
    }

    /// <inheritdoc />
    public int FileVersion => Groups == 1 ? Window.FileVersion : NetworkDocument.FourthVersion;

    /// <inheritdoc />
    /// <remarks>
    /// Its padding is written as the number of rows and columns on every side, as 0.4.0 wrote it, as the word <c>same</c>
    /// for a window padded as Keras writes it, or as a pair of numbers for each axis where it states each end apart.
    /// </remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inChannels", InChannels);
        writer.WriteNumber("outChannels", OutChannels);

        if (Groups != 1)
        {
            writer.WriteNumber("groups", Groups);
        }

        writer.WriteWindow(Window);
    }
}

/// <summary>Keeps each example of a batch as one row: an image's rows, columns and channels one after another.</summary>
public sealed class Flatten : Layer, ISaved<Flatten>
{
    /// <inheritdoc />
    public static string Name => "flatten";

    /// <inheritdoc />
    public static Flatten Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input has no axis beside its examples.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank < 2)
        {
            throw new ArgumentException($"Flattening keeps each example of a batch as a row, and a {input.Shape} tensor is not a batch of any.", nameof(input));
        }

        return pass.Backend.Reshape(input, new Shape(input.Shape[0], new Shape([.. input.Shape.Axes[1..]]).Count));
    }
}

/// <summary>Lays each example of a batch out in another shape: a row of pixels as an image, say.</summary>
public sealed class Reshape : Layer, ISaved<Reshape>
{
    /// <summary>Lays each example out in the given shape.</summary>
    /// <param name="each">The shape of one example, without the batch's axis.</param>
    /// <exception cref="ArgumentException">The shape holds no value: it has no axis, or an axis of none.</exception>
    public Reshape(Shape each) => Each = RequireEach(each);

    /// <summary>The shape of one example.</summary>
    public Shape Each { get; }

    /// <inheritdoc />
    public static string Name => "reshape";

    /// <inheritdoc />
    public static Reshape Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(new Shape([.. rebuilding.Wholes(settings, "each")]));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartArray("each");

        foreach (var length in Each.Axes)
        {
            writer.WriteNumberValue(length);
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">An example holds another number of values than the shape does.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank == 0 || input.Shape.Count != (long)input.Shape[0] * Each.Count)
        {
            throw new ArgumentException(
                $"Each example of a {input.Shape} batch is to be laid out as {Each}, and does not hold as many values.", nameof(input));
        }

        return pass.Backend.Reshape(input, new Shape([input.Shape[0], .. Each.Axes]));
    }

    /// <summary>A shape one example can be laid out in: one axis at least, holding one value at least.</summary>
    /// <exception cref="ArgumentException">It holds no value.</exception>
    internal static Shape RequireEach(Shape each) =>
        each.Rank > 0 && each.Count > 0
            ? each
            : throw new ArgumentException($"An example is laid out along one axis at least, in one value at least, and a {each} shape holds none.", nameof(each));
}
