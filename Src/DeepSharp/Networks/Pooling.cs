// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A window walked over a series, an image or a volume, making one value of what it covers, for each channel on its own: the
/// largest, or the average.
/// </summary>
/// <remarks>
/// What the pooling layers share is written once, here: each channel is taken apart from the rest — turned into an example of
/// its own with one channel — so the engine's unfolding lays a window's values of one channel side by side, one row for every
/// place; a subclass says what one row comes to, and the channels are put back together. A pooling learns nothing. Examples
/// are laid out with their channels last, as Keras lays them out; the border a window pads with is never a value of the
/// example, whatever it is padded as, so the largest is not a nought a border supplies.
/// </remarks>
public abstract class Pooling : Layer
{
    private readonly ISpatialWalk _walk;
    private readonly string _named;

    private protected Pooling(ISpatialWalk walk, string named)
    {
        _walk = walk;
        _named = named;
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of what the window walks, or is smaller than the window with its border.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        _walk.RequireBatch(input.Shape, null, _named);

        var backend = pass.Backend;
        var extents = _walk.ExtentsOf(input.Shape);
        var places = _walk.PlacesOver(extents);

        var singles = backend.ChannelsApart(input, extents);
        var patches = _walk.Unfold(backend, singles);

        // Which cells of a patch are values of the example rather than its border, when a window pads at all.
        var real = _walk.PadsOver(extents) ? _walk.Unfold(backend, backend.Fill(singles.Shape, 1f)) : null;
        var pooled = Pool(backend, patches, real);

        return backend.ChannelsTogether(pooled, ChannelSplit.Of(input.Shape), places);
    }

    /// <summary>One value of each row of patches: what the window covers there, a row for every place of every channel of every example.</summary>
    /// <param name="backend">The engine the arithmetic runs on.</param>
    /// <param name="patches">A row of the window's values for each place; a border cell is a nought.</param>
    /// <param name="real">One where a cell of a patch is a value of the example, nought where it is border; nothing where the window pads nowhere.</param>
    /// <returns>A matrix of one column.</returns>
    private protected abstract Tensor Pool(ITensorBackend backend, Tensor patches, Tensor? real);
}

/// <summary>A pooling that makes the largest value a window covers of each channel.</summary>
/// <remarks>
/// The value picked is the first largest of the window's values in the order the window reads them, as PyTorch's max pooling
/// picks it, and the gradient of a loss goes to that value alone. A window over the border sees the values of the example
/// only: the border is never the largest, whatever the values are.
/// </remarks>
public abstract class MaxPooling : Pooling
{
    // Far enough below any value a network holds that no border cell is ever the largest, and far enough from the limits of a
    // single-precision number that adding it to one cannot overflow.
    private const float Far = 1e30f;

    private protected MaxPooling(ISpatialWalk walk)
        : base(walk, "A max pooling")
    {
    }

    /// <inheritdoc />
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor patches, Tensor? real)
    {
        var scores = real is null
            ? patches
            : backend.Add(patches, backend.Scale(backend.Subtract(real, backend.Fill(real.Shape, 1f)), backend.Fill(new Shape(), Far)));

        return backend.MatMul(backend.Multiply(patches, backend.FirstLargest(scores)), backend.Fill(new Shape(patches.Shape[1], 1), 1f));
    }
}

/// <summary>A pooling that makes the average of the values a window covers of each channel.</summary>
/// <remarks>
/// The border a window pads with is left out of the average, as Keras's and ONNX's are; <see cref="CountsPadding"/> counts a
/// border that is stated as PyTorch's <c>count_include_pad</c> does.
/// </remarks>
public abstract class AveragePooling : Pooling
{
    private protected AveragePooling(ISpatialWalk walk)
        : base(walk, "An average pooling")
    {
    }

    /// <summary>
    /// Whether the border counts as values of nought in each average, as PyTorch's <c>count_include_pad</c> counts it; the
    /// border is left out, unless said.
    /// </summary>
    public bool CountsPadding { get; init; }

    /// <inheritdoc />
    private protected sealed override Tensor Pool(ITensorBackend backend, Tensor patches, Tensor? real)
    {
        var ones = backend.Fill(new Shape(patches.Shape[1], 1), 1f);
        var sums = backend.MatMul(patches, ones);

        return real is null || CountsPadding
            ? backend.Scale(sums, backend.Fill(new Shape(), 1f / patches.Shape[1]))
            : backend.Divide(sums, backend.MatMul(real, ones));
    }

    /// <summary>Writes whether the border counts, when it does: a pooling that leaves it out has nothing to say.</summary>
    private protected void WriteCounting(Utf8JsonWriter writer)
    {
        if (CountsPadding)
        {
            writer.WriteBoolean("countsPadding", true);
        }
    }
}

/// <summary>The largest of the values a window covers along a series, for each channel.</summary>
public sealed class MaxPool1D : MaxPooling, ISaved<MaxPool1D>
{
    /// <summary>A max pooling along a series.</summary>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public MaxPool1D(Window1D window)
        : base(new LineWalk(window)) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window1D Window { get; }

    /// <inheritdoc />
    public static string Name => "maxpool1d";

    /// <inheritdoc />
    public static MaxPool1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window1DIn(settings));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
    }
}

/// <summary>The largest of the values a window covers over an image, for each channel.</summary>
public sealed class MaxPool2D : MaxPooling, ISaved<MaxPool2D>
{
    /// <summary>A max pooling over an image.</summary>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public MaxPool2D(Window window)
        : base(new PlaneWalk(window)) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window Window { get; }

    /// <inheritdoc />
    public static string Name => "maxpool2d";

    /// <inheritdoc />
    public static MaxPool2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window2DIn(settings));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
    }
}

/// <summary>The largest of the values a window covers through a volume, for each channel.</summary>
public sealed class MaxPool3D : MaxPooling, ISaved<MaxPool3D>
{
    /// <summary>A max pooling through a volume.</summary>
    /// <param name="window">The block, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public MaxPool3D(Window3D window)
        : base(new VolumeWalk(window)) => Window = window;

    /// <summary>The block, and how it walks.</summary>
    public Window3D Window { get; }

    /// <inheritdoc />
    public static string Name => "maxpool3d";

    /// <inheritdoc />
    public static MaxPool3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window3DIn(settings));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
    }
}

/// <summary>The average of the values a window covers along a series, for each channel.</summary>
public sealed class AvgPool1D : AveragePooling, ISaved<AvgPool1D>
{
    /// <summary>An average pooling along a series.</summary>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public AvgPool1D(Window1D window)
        : base(new LineWalk(window)) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window1D Window { get; }

    /// <inheritdoc />
    public static string Name => "avgpool1d";

    /// <inheritdoc />
    public static AvgPool1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window1DIn(settings)) { CountsPadding = rebuilding.Holds(settings, "countsPadding") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
        WriteCounting(writer);
    }
}

/// <summary>The average of the values a window covers over an image, for each channel.</summary>
public sealed class AvgPool2D : AveragePooling, ISaved<AvgPool2D>
{
    /// <summary>An average pooling over an image.</summary>
    /// <param name="window">The patch, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public AvgPool2D(Window window)
        : base(new PlaneWalk(window)) => Window = window;

    /// <summary>The patch, and how it walks.</summary>
    public Window Window { get; }

    /// <inheritdoc />
    public static string Name => "avgpool2d";

    /// <inheritdoc />
    public static AvgPool2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window2DIn(settings)) { CountsPadding = rebuilding.Holds(settings, "countsPadding") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
        WriteCounting(writer);
    }
}

/// <summary>The average of the values a window covers through a volume, for each channel.</summary>
public sealed class AvgPool3D : AveragePooling, ISaved<AvgPool3D>
{
    /// <summary>An average pooling through a volume.</summary>
    /// <param name="window">The block, and how it walks.</param>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public AvgPool3D(Window3D window)
        : base(new VolumeWalk(window)) => Window = window;

    /// <summary>The block, and how it walks.</summary>
    public Window3D Window { get; }

    /// <inheritdoc />
    public static string Name => "avgpool3d";

    /// <inheritdoc />
    public static AvgPool3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Window3DIn(settings)) { CountsPadding = rebuilding.Holds(settings, "countsPadding") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteWindow(Window);
        WriteCounting(writer);
    }
}
