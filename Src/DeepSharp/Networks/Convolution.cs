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
/// same numbers channels first. It starts as PyTorch's does, within one over the root of the values a window holds.
/// </remarks>
public sealed class Conv2D : Layer, ISaved<Conv2D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each place of an image holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a side or a stride below one, or a border below nothing.</exception>
    public Conv2D(int inChannels, int outChannels, Window window, Draws draws)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inChannels, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outChannels, 1);
        ArgumentNullException.ThrowIfNull(draws);
        RequireWindow(window);

        var places = window.Height * window.Width;
        var fans = new Fans(places * inChannels, places * outChannels);

        Window = window;
        Weight = AddParameter("weight", new KaimingUniform().Draw(new Shape(places * inChannels, outChannels), fans, draws));
        Bias = AddParameter("bias", new FanInUniform().Draw(new Shape(outChannels), fans, draws));
    }

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's places times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv2D(Tensor kernel, Tensor bias, Window window)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(bias);
        RequireWindow(window);

        var places = window.Height * window.Width;

        if (kernel.Shape.Rank != 2 || kernel.Shape[0] % places != 0 || bias.Shape != new Shape(kernel.Shape[1]))
        {
            throw new ArgumentException(
                $"A convolution through a {window} takes a kernel of the window's {places} places times its channels in by its channels out, "
                + $"and a bias as long as its channels out; these are {kernel.Shape} and {bias.Shape}.",
                nameof(kernel));
        }

        Window = window;
        Weight = AddParameter("weight", kernel);
        Bias = AddParameter("bias", bias);
    }

    /// <summary>The kernel: the window's values, channels in included, by the channels out.</summary>
    public Parameter Weight { get; }

    /// <summary>The bias: one value for each channel out.</summary>
    public Parameter Bias { get; }

    /// <summary>The patch, and how it walks.</summary>
    public Window Window { get; }

    /// <summary>How many channels each place of an image holds.</summary>
    public int InChannels => Weight.Value.Shape[0] / (Window.Height * Window.Width);

    /// <summary>How many channels the layer makes.</summary>
    public int OutChannels => Weight.Value.Shape[1];

    /// <inheritdoc />
    public static string Name => "conv2d";

    /// <inheritdoc />
    public static Conv2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var window = new Window(rebuilding.Whole(settings, "height"), rebuilding.Whole(settings, "width"))
        {
            Stride = rebuilding.Whole(settings, "stride"),
            Padding = rebuilding.Whole(settings, "padding"),
        };

        return new(rebuilding.Whole(settings, "inChannels"), rebuilding.Whole(settings, "outChannels"), window, rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inChannels", InChannels);
        writer.WriteNumber("outChannels", OutChannels);
        writer.WriteNumber("height", Window.Height);
        writer.WriteNumber("width", Window.Width);
        writer.WriteNumber("stride", Window.Stride);
        writer.WriteNumber("padding", Window.Padding);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The images are not a batch of rows, columns and channels, or hold another number of channels.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 4 || input.Shape[3] != InChannels)
        {
            throw new ArgumentException(
                $"A convolution of {InChannels} channels takes a batch of images, image by row by column by channel, and was handed a {input.Shape} one.",
                nameof(input));
        }

        var backend = pass.Backend;
        var values = backend.AddRow(backend.MatMul(backend.Unfold(input, Window), Weight.Value), Bias.Value);

        return backend.Reshape(values, new Shape(input.Shape[0], Window.RowsOver(input.Shape[1]), Window.ColumnsOver(input.Shape[2]), OutChannels));
    }

    /// <summary>Refuses a window that cannot stand anywhere.</summary>
    /// <exception cref="ArgumentException">A side or the stride is below one, or the border below nothing.</exception>
    internal static void RequireWindow(Window window)
    {
        if (window.Height < 1 || window.Width < 1 || window.Stride < 1 || window.Padding < 0)
        {
            throw new ArgumentException(
                $"A {window} cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing.", nameof(window));
        }
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
