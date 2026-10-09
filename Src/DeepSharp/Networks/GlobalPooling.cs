// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How many examples a batch holds and how many channels each place of an example holds.</summary>
/// <param name="Count">The examples.</param>
/// <param name="Channels">The channels.</param>
internal readonly record struct ChannelSplit(int Count, int Channels)
{
    /// <summary>The split of a batch laid out example, places, channel.</summary>
    public static ChannelSplit Of(Shape batch) => new(batch[0], batch[batch.Rank - 1]);
}

/// <summary>What a pooling does with the channels of an example: takes them apart so each is pooled on its own, and puts them back.</summary>
internal static class ChannelExtensions
{
    extension(ITensorBackend backend)
    {
        /// <summary>
        /// Every channel of every example turned into an example of its own, with one channel: <c>(example, places…, channel)</c>
        /// becomes <c>(channel × example, places…, 1)</c>, a channel's examples one after another.
        /// </summary>
        internal Tensor ChannelsApart(Tensor batch, int[] extents)
        {
            var split = ChannelSplit.Of(batch.Shape);

            return backend.Reshape(
                backend.Transpose(backend.Reshape(batch, new Shape(batch.Shape.Count / split.Channels, split.Channels))),
                new Shape([split.Channels * split.Count, .. extents, 1]));
        }

        /// <summary>
        /// What was pooled of each of those examples, a row of one column for every place, put back as the examples it came from:
        /// <c>(channel × example × places, 1)</c> becomes <c>(example, places…, channel)</c>.
        /// </summary>
        internal Tensor ChannelsTogether(Tensor pooled, ChannelSplit split, int[] places) =>
            backend.Reshape(
                backend.Transpose(backend.Reshape(pooled, new Shape(split.Channels, pooled.Shape.Count / split.Channels))),
                new Shape([split.Count, .. places, split.Channels]));
    }
}

/// <summary>A pooling that makes one value of every channel of an example: the largest, or the average, of all its places.</summary>
/// <remarks>
/// Its output is a row of channels for each example, as Keras's global poolings give them; <see cref="KeepsAxes"/> keeps the
/// axes it pooled as axes of one place, as Keras's <c>keepdims</c> and ONNX's <c>GlobalAveragePool</c> do. A global pooling
/// learns nothing.
/// </remarks>
public abstract class GlobalPooling : Layer
{
    private readonly int _rank;
    private readonly string _named;

    private protected GlobalPooling(int rank, string named)
    {
        _rank = rank;
        _named = named;
    }

    /// <summary>Whether the axes it pools are kept, each of one place, instead of dropped; dropped, unless said.</summary>
    public bool KeepsAxes { get; init; }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of what the pooling pools, or an example holds no place.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        input.Shape.RequireSpatial(_rank, null, _named);

        if (input.Shape.Axes[1..^1].Contains(0))
        {
            throw new ArgumentException($"{_named} pools the places of each channel, and a {input.Shape} batch holds none.", nameof(input));
        }

        var backend = pass.Backend;
        var split = ChannelSplit.Of(input.Shape);
        var places = input.Shape.Count / (split.Count * split.Channels);

        var series = backend.Reshape(backend.ChannelsApart(input, [.. input.Shape.Axes[1..^1]]), new Shape(split.Channels * split.Count, places));

        return backend.ChannelsTogether(Pool(backend, series), split, KeepsAxes ? [.. Enumerable.Repeat(1, _rank)] : []);
    }

    /// <summary>One value of each row of places: a row for every channel of every example.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="series">A row of the places of one channel of one example.</param>
    /// <returns>A matrix of one column.</returns>
    private protected abstract Tensor Pool(ITensorBackend backend, Tensor series);

    /// <summary>Writes whether the axes are kept, when they are: a pooling that drops them has nothing to say.</summary>
    private protected void WriteKeeping(Utf8JsonWriter writer)
    {
        if (KeepsAxes)
        {
            writer.WriteBoolean("keepsAxes", true);
        }
    }
}

/// <summary>A pooling that makes the largest value of every channel of an example, as the first largest, as PyTorch's does.</summary>
public abstract class GlobalMaxPooling : GlobalPooling
{
    private protected GlobalMaxPooling(int rank)
        : base(rank, "A global max pooling")
    {
    }

    /// <inheritdoc />
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor series) =>
        backend.MatMul(backend.Multiply(series, backend.FirstLargest(series)), backend.Fill(new Shape(series.Shape[1], 1), 1f));
}

/// <summary>A pooling that makes the average of every channel of an example.</summary>
public abstract class GlobalAveragePooling : GlobalPooling
{
    private protected GlobalAveragePooling(int rank)
        : base(rank, "A global average pooling")
    {
    }

    /// <inheritdoc />
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor series) =>
        backend.Scale(backend.MatMul(series, backend.Fill(new Shape(series.Shape[1], 1), 1f)), backend.Fill(new Shape(), 1f / series.Shape[1]));
}

/// <summary>The largest value of every channel of a series.</summary>
public sealed class GlobalMaxPool1D : GlobalMaxPooling, ISaved<GlobalMaxPool1D>
{
    /// <summary>A global max pooling of series.</summary>
    public GlobalMaxPool1D()
        : base(1)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalmaxpool1d";

    /// <inheritdoc />
    public static GlobalMaxPool1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}

/// <summary>The largest value of every channel of an image.</summary>
public sealed class GlobalMaxPool2D : GlobalMaxPooling, ISaved<GlobalMaxPool2D>
{
    /// <summary>A global max pooling of images.</summary>
    public GlobalMaxPool2D()
        : base(2)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalmaxpool2d";

    /// <inheritdoc />
    public static GlobalMaxPool2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}

/// <summary>The largest value of every channel of a volume.</summary>
public sealed class GlobalMaxPool3D : GlobalMaxPooling, ISaved<GlobalMaxPool3D>
{
    /// <summary>A global max pooling of volumes.</summary>
    public GlobalMaxPool3D()
        : base(3)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalmaxpool3d";

    /// <inheritdoc />
    public static GlobalMaxPool3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}

/// <summary>The average of every channel of a series.</summary>
public sealed class GlobalAvgPool1D : GlobalAveragePooling, ISaved<GlobalAvgPool1D>
{
    /// <summary>A global average pooling of series.</summary>
    public GlobalAvgPool1D()
        : base(1)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalavgpool1d";

    /// <inheritdoc />
    public static GlobalAvgPool1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}

/// <summary>The average of every channel of an image.</summary>
public sealed class GlobalAvgPool2D : GlobalAveragePooling, ISaved<GlobalAvgPool2D>
{
    /// <summary>A global average pooling of images.</summary>
    public GlobalAvgPool2D()
        : base(2)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalavgpool2d";

    /// <inheritdoc />
    public static GlobalAvgPool2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}

/// <summary>The average of every channel of a volume.</summary>
public sealed class GlobalAvgPool3D : GlobalAveragePooling, ISaved<GlobalAvgPool3D>
{
    /// <summary>A global average pooling of volumes.</summary>
    public GlobalAvgPool3D()
        : base(3)
    {
    }

    /// <inheritdoc />
    public static string Name => "globalavgpool3d";

    /// <inheritdoc />
    public static GlobalAvgPool3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { KeepsAxes = rebuilding.Holds(settings, "keepsAxes") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteKeeping(writer);
    }
}
