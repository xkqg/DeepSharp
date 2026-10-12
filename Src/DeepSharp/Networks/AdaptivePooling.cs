// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Which places of an axis one bin of a pooling to another size covers: from the first to the last, the last not among them.
/// </summary>
/// <param name="Start">The first place it covers.</param>
/// <param name="End">The place after the last it covers.</param>
internal readonly record struct Bin(int Start, int End)
{
    /// <summary>How many places it covers.</summary>
    public int Count => End - Start;
}

/// <summary>
/// A pooling to a size of its own rather than to one place: PyTorch's <c>AdaptiveAvgPool</c> and <c>AdaptiveMaxPool</c>,
/// which say what shape to come out as and work out for themselves which places of the example each value is made of.
/// </summary>
/// <remarks>
/// Bin <c>i</c> of an axis of <c>L</c> places pooled to <c>O</c> covers the places from <c>floor(i·L/O)</c> up to
/// <c>ceil((i+1)·L/O)</c>, which is PyTorch's own rule. Where <c>O</c> divides <c>L</c> those bins neither overlap nor
/// differ in size and the pooling IS the ordinary one whose window is as wide as its stride, which is what it lowers to
/// then — a pooling of a two-hundred-and-twenty-four-place axis to seven costs what a pooling costs. Where <c>O</c> does not
/// divide <c>L</c> the bins genuinely overlap and differ, so no one window covers them, and each axis is pooled by a matrix
/// worked out from the axis's own length: one pass an axis, which is exact because a bin is the product of the bins of each
/// axis and every place is counted once.
/// <para>
/// Each channel of each example is taken apart as a pooling takes it apart, so a channel is pooled on its own. A pooling to
/// a size learns nothing, and the matrices it pools by are worked out from the lengths it is handed, never from the values.
/// </para>
/// </remarks>
public abstract class AdaptivePooling : Layer
{
    private readonly int _rank;
    private readonly string _named;

    private protected AdaptivePooling(Shape sizes, int rank, string named)
    {
        _rank = rank;
        _named = named;
        Sizes = RequireSizes(sizes, rank, named);
    }

    /// <summary>How many places each axis is pooled to.</summary>
    public Shape Sizes { get; }

    /// <summary>Which places of an axis so long each bin covers, pooled to so many.</summary>
    /// <param name="length">How many places the axis has.</param>
    /// <param name="size">How many places it is pooled to.</param>
    /// <returns>One bin for each place pooled to, in order.</returns>
    internal static Bin[] BinsOver(int length, int size) =>
        [.. Enumerable.Range(0, size).Select(at => new Bin(at * length / size, (((at + 1) * length) + size - 1) / size))];

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of what the pooling pools, or an example holds no place along an axis.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        input.Shape.RequireSpatial(_rank, null, _named);

        var extents = input.Shape.Axes[1..^1].ToArray();

        if (extents.Contains(0))
        {
            throw new ArgumentException($"{_named} pools the places of each channel, and a {input.Shape} batch holds none.", nameof(input));
        }

        if (extents.Select((length, axis) => length % Sizes[axis] == 0).All(divides => divides))
        {
            return Ordinary(new Shape([.. extents.Select((length, axis) => length / Sizes[axis])])).Forward(input, pass);
        }

        var backend = pass.Backend;
        var split = ChannelSplit.Of(input.Shape);
        var apart = backend.ChannelsApart(input, extents);
        var rows = split.Channels * split.Count;
        var values = backend.Reshape(apart, new Shape(apart.Shape.Count / extents[_rank - 1], extents[_rank - 1]));

        // One axis at a time, the last first: pool it, turn the matrix so the axis just pooled leads, and lay the next axis
        // out last. After the last axis the places pooled to lead and the examples follow, so one more turn puts them back.
        for (var axis = _rank - 1; axis >= 0; axis--)
        {
            var turned = backend.Transpose(Pool(backend, values, BinsOver(extents[axis], Sizes[axis])));

            values = axis == 0 ? turned : backend.Reshape(turned, new Shape(turned.Shape.Count / extents[axis - 1], extents[axis - 1]));
        }

        return backend.ChannelsTogether(
            backend.Transpose(backend.Reshape(values, new Shape(Sizes.Count, rows))),
            split,
            [.. Sizes.Axes]);
    }

    /// <summary>One value of each bin, along the last axis of a matrix of places.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="values">A row for every other axis, holding the places of the axis being pooled.</param>
    /// <param name="bins">Which places of that axis each bin covers.</param>
    /// <returns>The same rows, holding one value for each bin.</returns>
    private protected abstract Tensor Pool(ITensorBackend backend, Tensor values, Bin[] bins);

    /// <summary>The ordinary pooling this one is, where every axis is pooled to a size that divides it: its window as wide as its stride.</summary>
    /// <param name="kernels">How many places of each axis go into one.</param>
    private protected abstract Layer Ordinary(Shape kernels);

    /// <summary>Writes the sizes: the one setting every pooling to a size has.</summary>
    private protected void WriteSizes(Utf8JsonWriter writer)
    {
        writer.WriteStartArray("sizes");

        foreach (var size in Sizes.Axes)
        {
            writer.WriteNumberValue(size);
        }

        writer.WriteEndArray();
    }

    /// <summary>Reads the sizes a pooling to a size was written with.</summary>
    private protected static Shape SizesIn(JsonElement settings, Rebuilding rebuilding) =>
        new([.. rebuilding.Wholes(settings, "sizes")]);

    // A size for each axis the pooling pools, each of one place at least: what a pooling cannot work out for itself.
    private static Shape RequireSizes(Shape sizes, int rank, string named)
    {
        if (sizes.Rank != rank || sizes.Axes.ToArray().Any(size => size < 1))
        {
            throw new ArgumentException(
                $"{named} pools {rank} {(rank == 1 ? "axis" : "axes")} to one place at least each, and was asked for {sizes}.",
                nameof(sizes));
        }

        return sizes;
    }
}

/// <summary>A pooling that makes the average of the places of each bin, for each channel.</summary>
public abstract class AdaptiveAveragePooling : AdaptivePooling
{
    private protected AdaptiveAveragePooling(Shape sizes, int rank)
        : base(sizes, rank, "A pooling to a size")
    {
    }

    /// <inheritdoc />
    /// <remarks>
    /// One matrix product: a column for each bin, holding one over the places it covers where it covers them and nothing
    /// where it does not. The matrix is worked out from the axis's length and nothing else, so the gradient of a place
    /// reaches every bin that covers it, as many times as it is covered.
    /// </remarks>
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor values, Bin[] bins)
    {
        var length = values.Shape[1];
        var shares = new float[length * bins.Length];

        for (var bin = 0; bin < bins.Length; bin++)
        {
            var share = 1f / bins[bin].Count;

            for (var place = bins[bin].Start; place < bins[bin].End; place++)
            {
                shares[(place * bins.Length) + bin] = share;
            }
        }

        return backend.MatMul(values, Tensor.From(new Shape(length, bins.Length), shares));
    }
}

/// <summary>A pooling that makes the largest of the places of each bin, for each channel.</summary>
public abstract class AdaptiveMaxPooling : AdaptivePooling
{
    private protected AdaptiveMaxPooling(Shape sizes, int rank)
        : base(sizes, rank, "A pooling to a size")
    {
    }

    /// <inheritdoc />
    /// <remarks>
    /// The bins are gathered side by side into rows as wide as the widest of them — one matrix product — and the places a
    /// shorter bin does not have are marked as no values of the example, so the largest of a bin is picked from its own
    /// places alone, however negative they are. The largest is then the one the whole family picks.
    /// </remarks>
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor values, Bin[] bins)
    {
        var length = values.Shape[1];
        var widest = bins.Max(bin => bin.Count);
        var gather = new float[length * bins.Length * widest];
        var real = new float[bins.Length * widest];

        for (var bin = 0; bin < bins.Length; bin++)
        {
            for (var at = 0; at < widest; at++)
            {
                if (bins[bin].Start + at >= bins[bin].End)
                {
                    continue;
                }

                gather[((bins[bin].Start + at) * bins.Length * widest) + (bin * widest) + at] = 1f;
                real[(bin * widest) + at] = 1f;
            }
        }

        var spread = backend.MatMul(values, Tensor.From(new Shape(length, bins.Length * widest), gather));
        var rows = new Shape(values.Shape[0] * bins.Length, widest);

        return backend.Reshape(
            backend.Largest(
                backend.Reshape(spread, rows),
                backend.Reshape(backend.MatMul(backend.Fill(new Shape(values.Shape[0], 1), 1f), Tensor.From(new Shape(1, bins.Length * widest), real)), rows)),
            new Shape(values.Shape[0], bins.Length));
    }
}

/// <summary>The average of the places of each bin along a series, for each channel.</summary>
public sealed class AdaptiveAvgPool1D : AdaptiveAveragePooling, ISaved<AdaptiveAvgPool1D>
{
    /// <summary>A pooling of each series to so many places.</summary>
    /// <param name="sizes">How many places the series is pooled to: one number.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along the one axis.</exception>
    public AdaptiveAvgPool1D(Shape sizes)
        : base(sizes, 1)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptiveavgpool1d";

    /// <inheritdoc />
    public static AdaptiveAvgPool1D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) => new AvgPool1D(new Window1D(kernels[0]) { Stride = kernels[0] });
}

/// <summary>The average of the places of each bin over an image, for each channel.</summary>
public sealed class AdaptiveAvgPool2D : AdaptiveAveragePooling, ISaved<AdaptiveAvgPool2D>
{
    /// <summary>A pooling of each image to so many rows and columns.</summary>
    /// <param name="sizes">How many rows and columns the image is pooled to.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along each of the two axes.</exception>
    public AdaptiveAvgPool2D(Shape sizes)
        : base(sizes, 2)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptiveavgpool2d";

    /// <inheritdoc />
    public static AdaptiveAvgPool2D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) =>
        new AvgPool2D(new Window(kernels[0], kernels[1]) { Stride = new Steps(kernels[0], kernels[1]) });
}

/// <summary>The average of the places of each bin through a volume, for each channel.</summary>
public sealed class AdaptiveAvgPool3D : AdaptiveAveragePooling, ISaved<AdaptiveAvgPool3D>
{
    /// <summary>A pooling of each volume to so many planes, rows and columns.</summary>
    /// <param name="sizes">How many planes, rows and columns the volume is pooled to.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along each of the three axes.</exception>
    public AdaptiveAvgPool3D(Shape sizes)
        : base(sizes, 3)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptiveavgpool3d";

    /// <inheritdoc />
    public static AdaptiveAvgPool3D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) =>
        new AvgPool3D(new Window3D(kernels[0], kernels[1], kernels[2]) { Stride = new Steps3D(kernels[0], kernels[1], kernels[2]) });
}

/// <summary>The largest of the places of each bin along a series, for each channel.</summary>
public sealed class AdaptiveMaxPool1D : AdaptiveMaxPooling, ISaved<AdaptiveMaxPool1D>
{
    /// <summary>A pooling of each series to so many places.</summary>
    /// <param name="sizes">How many places the series is pooled to: one number.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along the one axis.</exception>
    public AdaptiveMaxPool1D(Shape sizes)
        : base(sizes, 1)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptivemaxpool1d";

    /// <inheritdoc />
    public static AdaptiveMaxPool1D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) => new MaxPool1D(new Window1D(kernels[0]) { Stride = kernels[0] });
}

/// <summary>The largest of the places of each bin over an image, for each channel.</summary>
public sealed class AdaptiveMaxPool2D : AdaptiveMaxPooling, ISaved<AdaptiveMaxPool2D>
{
    /// <summary>A pooling of each image to so many rows and columns.</summary>
    /// <param name="sizes">How many rows and columns the image is pooled to.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along each of the two axes.</exception>
    public AdaptiveMaxPool2D(Shape sizes)
        : base(sizes, 2)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptivemaxpool2d";

    /// <inheritdoc />
    public static AdaptiveMaxPool2D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) =>
        new MaxPool2D(new Window(kernels[0], kernels[1]) { Stride = new Steps(kernels[0], kernels[1]) });
}

/// <summary>The largest of the places of each bin through a volume, for each channel.</summary>
public sealed class AdaptiveMaxPool3D : AdaptiveMaxPooling, ISaved<AdaptiveMaxPool3D>
{
    /// <summary>A pooling of each volume to so many planes, rows and columns.</summary>
    /// <param name="sizes">How many planes, rows and columns the volume is pooled to.</param>
    /// <exception cref="ArgumentException">The sizes are not one place at least along each of the three axes.</exception>
    public AdaptiveMaxPool3D(Shape sizes)
        : base(sizes, 3)
    {
    }

    /// <inheritdoc />
    public static string Name => "adaptivemaxpool3d";

    /// <inheritdoc />
    public static AdaptiveMaxPool3D Rebuild(JsonElement settings, Rebuilding rebuilding) => new(SizesIn(settings, rebuilding));

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteSizes(writer);
    }

    /// <inheritdoc />
    private protected override Layer Ordinary(Shape kernels) =>
        new MaxPool3D(new Window3D(kernels[0], kernels[1], kernels[2]) { Stride = new Steps3D(kernels[0], kernels[1], kernels[2]) });
}
