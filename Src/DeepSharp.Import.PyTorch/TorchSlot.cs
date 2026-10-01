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
/// A vector — a bias, a normalisation's scale, shift or running statistics — kept alike, but for the features of an image a
/// flatten made rows of, which PyTorch orders channel by channel.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="length">How many numbers it holds.</param>
/// <param name="order">The image a flatten made its features of; nothing for features as they come.</param>
internal sealed class VectorSlot(string path, int length, ImageOrder? order) : TurnedSlot(path)
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
/// flatten made the inputs of an image, found where PyTorch's channel by channel order puts it.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="dense">The layer.</param>
/// <param name="order">The image a flatten made its inputs of; nothing for inputs as they come.</param>
internal sealed class LinearSlot(string path, Dense dense, ImageOrder? order) : TurnedSlot(path)
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
/// A convolution's kernel: PyTorch keeps it channels out, channels in, rows, columns; here it is the window's rows, columns
/// and channels in, by the channels out — the same numbers, nothing flipped.
/// </summary>
/// <param name="path">The slot's path.</param>
/// <param name="convolution">The layer.</param>
internal sealed class KernelSlot(string path, Conv2D convolution) : TurnedSlot(path)
{
    /// <inheritdoc />
    protected override Shape Kept => new(convolution.OutChannels, convolution.InChannels, convolution.Window.Height, convolution.Window.Width);

    /// <inheritdoc />
    protected override Shape Slot => convolution.Weight.Value.Shape;

    /// <inheritdoc />
    protected override float[] Turned(float[] kept)
    {
        var (outputs, inputs, rows, columns) = (convolution.OutChannels, convolution.InChannels, convolution.Window.Height, convolution.Window.Width);
        var turned = new float[kept.Length];

        for (var output = 0; output < outputs; output++)
        {
            for (var input = 0; input < inputs; input++)
            {
                for (var row = 0; row < rows; row++)
                {
                    for (var column = 0; column < columns; column++)
                    {
                        turned[(((((row * columns) + column) * inputs) + input) * outputs) + output] = kept[(((((output * inputs) + input) * rows) + row) * columns) + column];
                    }
                }
            }
        }

        return turned;
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

/// <summary>An image a flatten made a row of: here row by row, each place's channels together; in PyTorch channel by channel.</summary>
/// <param name="Rows">Its rows.</param>
/// <param name="Columns">Its columns.</param>
/// <param name="Channels">Its channels.</param>
internal readonly record struct ImageOrder(int Rows, int Columns, int Channels)
{
    /// <summary>Where PyTorch's row keeps the value this row keeps at a place.</summary>
    /// <param name="place">The place in the row here.</param>
    public int TorchPlace(int place)
    {
        var channel = place % Channels;
        var column = place / Channels % Columns;
        var row = place / (Channels * Columns);

        return (((channel * Rows) + row) * Columns) + column;
    }
}
