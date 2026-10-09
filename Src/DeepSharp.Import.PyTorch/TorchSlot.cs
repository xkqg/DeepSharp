// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnxify.Safetensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>How PyTorch keeps the numbers of one slot of a network, and what reading its tensor into the slot comes to.</summary>
/// <param name="path">The slot's path in the network.</param>
internal abstract class TorchSlot(string path)
{
    /// <summary>The slot's path in the network.</summary>
    public string Path => path;

    /// <summary>What the file's tensor for the slot comes to: the number to put into the slot, or why it cannot go in.</summary>
    public abstract Reading Read(StoredTensor tensor);

    /// <summary>The fault of the file's tensor for the slot, placed at its name.</summary>
    protected Reading Refused(StoredTensor tensor, string message) => new(null, new SlotLoadFault(tensor.Name, path, $"'{path}' {message}"));
}

/// <summary>What reading one tensor comes to: the number for its slot, or the fault that keeps it out.</summary>
/// <param name="Entry">The number, with the path of its slot; nothing when it is refused.</param>
/// <param name="Fault">Why it is refused; nothing when it goes in.</param>
internal readonly record struct Reading(SlotEntry? Entry, SlotLoadFault? Fault);

/// <summary>A slot whose numbers PyTorch keeps in a layout of its own, turned into the one the slot keeps.</summary>
internal abstract class TurnedSlot(string path) : TorchSlot(path)
{
    /// <summary>The shape PyTorch keeps the slot's tensor in.</summary>
    protected abstract Shape Kept { get; }

    /// <summary>The shape the slot keeps.</summary>
    protected abstract Shape Slot { get; }

    /// <inheritdoc />
    /// <remarks>Refused when its numbers are not floats, or it is not of the shape PyTorch keeps the slot in.</remarks>
    public sealed override Reading Read(StoredTensor tensor)
    {
        var numbers = tensor.Numbers();

        if (numbers.Singles() is not { } values)
        {
            return Refused(tensor,
                $"holds {numbers.DataType.ToWireName()} numbers, and a slot holds 32-bit floats: F16 and BF16 are widened to them and F64 is narrowed, and no other kind of number is read.");
        }

        if (!numbers.Shape.SequenceEqual(Kept.Axes.ToArray().Select(length => (ulong)length)))
        {
            var written = numbers.Shape.Count == 0 ? "scalar" : string.Join('x', numbers.Shape);

            return Refused(tensor, $"is a {Kept} tensor in PyTorch's layout here, and is written as {written}.");
        }

        return new Reading(new SlotEntry(Path, Tensor.From(Slot, Turned(values)), tensor.Name), null);
    }

    /// <summary>The numbers turned from the layout PyTorch keeps them in to the one the slot keeps.</summary>
    protected abstract float[] Turned(float[] kept);
}

/// <summary>
/// A vector — a bias, a normalisation's scale, shift or running statistics — kept alike, but for the features of a series, an
/// image or a volume a flatten made rows of, which PyTorch orders channel by channel.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="length">How many numbers it holds.</param>
/// <param name="order">The series, image or volume a flatten made its features of; nothing for features as they come.</param>
internal sealed class VectorSlot(string path, int length, ChannelOrder? order) : TurnedSlot(path)
{
    /// <inheritdoc />
    protected override Shape Kept => new(length);

    /// <inheritdoc />
    protected override Shape Slot => new(length);

    /// <inheritdoc />
    protected override float[] Turned(float[] kept) =>
        order is { } image ? [.. Enumerable.Range(0, length).Select(place => kept[image.TorchPlace(place)])] : kept;
}

/// <summary>
/// A linear layer's weights: PyTorch keeps them outputs by inputs, and here they are inputs by outputs — each input, when a
/// flatten made the inputs of a series, an image or a volume, found where PyTorch's channel by channel order puts it.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="dense">The layer.</param>
/// <param name="order">The series, image or volume a flatten made its inputs of; nothing for inputs as they come.</param>
internal sealed class LinearSlot(string path, Dense dense, ChannelOrder? order) : TurnedSlot(path)
{
    /// <inheritdoc />
    protected override Shape Kept => new(dense.Outputs, dense.Inputs);

    /// <inheritdoc />
    protected override Shape Slot => new(dense.Inputs, dense.Outputs);

    /// <inheritdoc />
    protected override float[] Turned(float[] kept)
    {
        var (inputs, outputs) = (dense.Inputs, dense.Outputs);
        var turned = new float[kept.Length];

        for (var input = 0; input < inputs; input++)
        {
            var place = order?.TorchPlace(input) ?? input;

            for (var output = 0; output < outputs; output++)
            {
                turned[(input * outputs) + output] = kept[(output * inputs) + place];
            }
        }

        return turned;
    }
}

/// <summary>
/// A convolution's kernel, along one, two or three axes: PyTorch keeps it channels out, channels in, then the window's steps,
/// or rows and columns, or planes, rows and columns; here it is the window's places and then the channels in, by the channels
/// out — the same numbers, nothing flipped.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="convolution">The layer.</param>
internal sealed class KernelSlot(string path, Convolution convolution) : TurnedSlot(path)
{
    /// <inheritdoc />
    protected override Shape Kept => new([convolution.OutChannels, convolution.InChannels, .. convolution.WindowSides()]);

    /// <inheritdoc />
    protected override Shape Slot => convolution.Weight.Value.Shape;

    /// <inheritdoc />
    protected override float[] Turned(float[] kept)
    {
        // The window's places are one run of numbers in both layouts, in the same order: first axis slowest, last axis fastest.
        var (outputs, inputs, places) = (convolution.OutChannels, convolution.InChannels, convolution.Weight.Value.Shape[0] / convolution.InChannels);
        var turned = new float[kept.Length];

        for (var output = 0; output < outputs; output++)
        {
            for (var input = 0; input < inputs; input++)
            {
                for (var place = 0; place < places; place++)
                {
                    turned[(((place * inputs) + input) * outputs) + output] = kept[(((output * inputs) + input) * places) + place];
                }
            }
        }

        return turned;
    }
}

/// <summary>What the reader asks of a convolution that the layers themselves do not say.</summary>
internal static class ConvolutionWindowExtensions
{
    extension(Convolution convolution)
    {
        /// <summary>The sides of its window in the order PyTorch keeps them: the steps; the rows and columns; the planes, rows and columns.</summary>
        public int[] WindowSides()
        {
            if (convolution is Conv1D series)
            {
                return [series.Window.Length];
            }

            if (convolution is Conv2D image)
            {
                return [image.Window.Height, image.Window.Width];
            }

            // A convolution is along one, two or three axes, and no other kind can be written.
            var volume = ((Conv3D)convolution).Window;

            return [volume.Depth, volume.Height, volume.Width];
        }
    }
}

/// <summary>A slot whose layout in PyTorch is not told: its tensor is refused, saying why.</summary>
/// <param name="path">The slot's path.</param>
/// <param name="why">Why, after the slot's name.</param>
internal sealed class UntoldSlot(string path, string why) : TorchSlot(path)
{
    /// <inheritdoc />
    public override Reading Read(StoredTensor tensor) => Refused(tensor, why);
}

/// <summary>
/// A series, an image or a volume a flatten made a row of: here place by place, each place's channels together; in PyTorch
/// channel by channel, each channel's places together in the order the axes run, the first slowest.
/// </summary>
/// <param name="Places">How many places each channel has: the steps of a series, the rows times the columns of an image, the planes times the rows times the columns of a volume.</param>
/// <param name="Channels">Its channels.</param>
internal readonly record struct ChannelOrder(int Places, int Channels)
{
    /// <summary>Where PyTorch's row keeps the value this row keeps at a place.</summary>
    /// <param name="place">The place in the row here.</param>
    public int TorchPlace(int place) => ((place % Channels) * Places) + (place / Channels);
}
