// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A convolution along one, two or three axes: a window walked over each example, every place it stands given a value for
/// each channel the layer makes — the patch it covers, times a kernel, plus a bias.
/// </summary>
/// <remarks>
/// What the three share is written once, here: the kernel — as many rows as the window holds values, channels in included,
/// by as many columns as channels out, starting as PyTorch's does, within one over the root of the values a window holds —
/// the bias, and the arithmetic. Each says only the window it walks and how that is written to a network file. Examples are
/// laid out with their channels last, as Keras lays them out, so a batch is the patches of every example one after another
/// and nothing has to be turned round; PyTorch keeps the same numbers channels first.
/// </remarks>
public abstract class Convolution : Layer
{
    private readonly ISpatialWalk _walk;

    /// <summary>A convolution that starts as PyTorch's does.</summary>
    private protected Convolution(ISpatialWalk walk, int inChannels, int outChannels, Draws draws)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inChannels, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outChannels, 1);
        ArgumentNullException.ThrowIfNull(draws);

        _walk = walk;

        var fans = new Fans(walk.Volume * inChannels, walk.Volume * outChannels);

        Weight = AddParameter("weight", new KaimingUniform().Draw(new Shape(walk.Volume * inChannels, outChannels), fans, draws));
        Bias = AddParameter("bias", new FanInUniform().Draw(new Shape(outChannels), fans, draws));
    }

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <exception cref="ArgumentException">The kernel is not a matrix whose rows are the window's values times a whole number of channels, or the bias is not as long as the kernel is wide.</exception>
    private protected Convolution(ISpatialWalk walk, Tensor kernel, Tensor bias)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        ArgumentNullException.ThrowIfNull(bias);

        if (kernel.Shape.Rank != 2 || kernel.Shape[0] % walk.Volume != 0 || bias.Shape != new Shape(kernel.Shape[1]))
        {
            throw new ArgumentException(
                $"A convolution through a {walk.Describe()} takes a kernel of the window's {walk.Volume} places times its channels in by its channels out, "
                + $"and a bias as long as its channels out; these are {kernel.Shape} and {bias.Shape}.",
                nameof(kernel));
        }

        _walk = walk;
        Weight = AddParameter("weight", kernel);
        Bias = AddParameter("bias", bias);
    }

    /// <summary>The kernel: the window's values, channels in included, by the channels out.</summary>
    public Parameter Weight { get; }

    /// <summary>The bias: one value for each channel out.</summary>
    public Parameter Bias { get; }

    /// <summary>How many channels each place of an example holds.</summary>
    public int InChannels => Weight.Value.Shape[0] / _walk.Volume;

    /// <summary>How many channels the layer makes.</summary>
    public int OutChannels => Weight.Value.Shape[1];

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of what the window walks, holds another number of channels, or is smaller than the window with its border.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        _walk.RequireBatch(input.Shape, InChannels, $"A convolution of {InChannels} channels");

        var backend = pass.Backend;
        var values = backend.AddRow(backend.MatMul(_walk.Unfold(backend, input), Weight.Value), Bias.Value);

        return backend.Reshape(values, new Shape([input.Shape[0], .. _walk.PlacesOver(_walk.ExtentsOf(input.Shape)), OutChannels]));
    }
}

/// <summary>
/// A convolution along a series: a window of so many steps slid over each, every place it stands given a value for each
/// channel the layer makes.
/// </summary>
/// <remarks>
/// A series is laid out step by channel, as Keras's <c>Conv1D</c> lays it out. Its window pads each series by the border it
/// states at each end, as TensorFlow's 'same' does, or — as Keras's 'causal' does — before the series alone, so no place sees
/// what comes after its own.
/// </remarks>
public sealed class Conv1D : Convolution, ISaved<Conv1D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each step of a series holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a length or a stride below one, or a border below nothing.</exception>
    public Conv1D(int inChannels, int outChannels, Window1D window, Draws draws)
        : base(new LineWalk(window), inChannels, outChannels, draws) => Window = window;

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's length times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv1D(Tensor kernel, Tensor bias, Window1D window)
        : base(new LineWalk(window), kernel, bias) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window1D Window { get; }

    /// <inheritdoc />
    public static string Name => "conv1d";

    /// <inheritdoc />
    /// <remarks>Its padding is the number of steps at each end, the word <c>same</c>, or the word <c>causal</c>.</remarks>
    public static Conv1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "inChannels"), rebuilding.Whole(settings, "outChannels"), rebuilding.Window1DIn(settings), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inChannels", InChannels);
        writer.WriteNumber("outChannels", OutChannels);
        writer.WriteWindow(Window);
    }
}

/// <summary>
/// A convolution through a volume: a block of planes, rows and columns slid through each, every place it stands given a
/// value for each channel the layer makes.
/// </summary>
/// <remarks>A volume is laid out plane by row by column by channel, as Keras's <c>Conv3D</c> lays it out.</remarks>
public sealed class Conv3D : Convolution, ISaved<Conv3D>
{
    /// <summary>A convolution that starts as PyTorch's does.</summary>
    /// <param name="inChannels">How many channels each place of a volume holds.</param>
    /// <param name="outChannels">How many channels the layer makes of each place its window stands.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <param name="draws">The draws its start is taken from: the kernel first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere: a side or a stride below one, or a border below nothing.</exception>
    public Conv3D(int inChannels, int outChannels, Window3D window, Draws draws)
        : base(new VolumeWalk(window), inChannels, outChannels, draws) => Window = window;

    /// <summary>A convolution that starts at the given kernel and bias.</summary>
    /// <param name="kernel">As many rows as the window holds values, channels in included, by as many columns as channels out.</param>
    /// <param name="bias">One value for each channel out.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <exception cref="ArgumentException">
    /// The kernel is not a matrix whose rows are the window's places times a whole number of channels, the bias is not as long
    /// as the kernel is wide, or the window cannot stand anywhere.
    /// </exception>
    public Conv3D(Tensor kernel, Tensor bias, Window3D window)
        : base(new VolumeWalk(window), kernel, bias) => Window = window;

    /// <summary>The block, and how it walks.</summary>
    public Window3D Window { get; }

    /// <inheritdoc />
    public static string Name => "conv3d";

    /// <inheritdoc />
    /// <remarks>Its padding is the number of planes, rows and columns on every side, the word <c>same</c>, or the word <c>causal</c>.</remarks>
    public static Conv3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "inChannels"), rebuilding.Whole(settings, "outChannels"), rebuilding.Window3DIn(settings), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inChannels", InChannels);
        writer.WriteNumber("outChannels", OutChannels);
        writer.WriteWindow(Window);
    }
}
