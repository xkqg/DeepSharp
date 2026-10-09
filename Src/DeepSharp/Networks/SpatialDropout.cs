// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Leaves whole channels out at random while a network trains, instead of single values, and scales the ones it keeps by one
/// over the share it keeps: Keras's <c>SpatialDropout</c> and PyTorch's <c>Dropout1d</c>, <c>Dropout2d</c> and <c>Dropout3d</c>.
/// </summary>
/// <remarks>
/// Neighbouring values of a series, an image or a volume are so alike that leaving one out changes nothing the next does not
/// give back; leaving a channel out of an example in every place at once is what regularises. Which channels are left out is
/// drawn from the run's stream for this layer's place in the network, the epoch and the step, so the same run leaves out the
/// same channels and two layers never the same ones. An evaluation pass leaves nothing out and scales nothing. What is left out
/// is a tensor of noughts and scales multiplied in, so the gradient passes where a channel was kept, scaled as it was.
/// </remarks>
public abstract class SpatialDropout : Layer
{
    private readonly int _rank;

    private protected SpatialDropout(int rank, double rate)
    {
        _rank = rank;
        Rate = Dropout.RequireRate(rate);
    }

    /// <summary>The share of the channels left out.</summary>
    public double Rate { get; }

    /// <summary>Writes the rate: the one setting every spatial dropout has.</summary>
    private protected void WriteRate(Utf8JsonWriter writer) => writer.WriteNumber("rate", Rate);

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of series, images or volumes of the rank this dropout walks.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        input.Shape.RequireSpatial(_rank, null, "A spatial dropout");

        if (pass.Mode == PassMode.Evaluation || Rate <= 0)
        {
            return input;
        }

        var draws = pass.Draws($"spatialdropout:{Path}");
        var kept = 1 - Rate;
        var scale = (float)(1 / kept);
        var channels = input.Shape[input.Shape.Rank - 1];
        var count = input.Shape[0];
        var places = input.Shape.Count / (count * channels);
        var chosen = new float[count * channels];

        for (var at = 0; at < chosen.Length; at++)
        {
            chosen[at] = draws.NextDouble() < kept ? scale : 0f;
        }

        // A channel's choice for an example stands in every place of it.
        var mask = new float[input.Shape.Count];

        for (var example = 0; example < count; example++)
        {
            for (var place = 0; place < places; place++)
            {
                chosen.AsSpan(example * channels, channels).CopyTo(mask.AsSpan(((example * places) + place) * channels, channels));
            }
        }

        return pass.Backend.Multiply(input, Tensor.From(input.Shape, mask));
    }
}

/// <summary>Leaves whole channels of a series out at random while a network trains.</summary>
public sealed class SpatialDropout1D : SpatialDropout, ISaved<SpatialDropout1D>
{
    /// <summary>A dropout that leaves out the given share of the channels of each series.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public SpatialDropout1D(double rate)
        : base(1, rate)
    {
    }

    /// <inheritdoc />
    public static string Name => "spatialdropout1d";

    /// <inheritdoc />
    public static SpatialDropout1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteRate(writer);
    }
}

/// <summary>Leaves whole channels of an image out at random while a network trains.</summary>
public sealed class SpatialDropout2D : SpatialDropout, ISaved<SpatialDropout2D>
{
    /// <summary>A dropout that leaves out the given share of the channels of each image.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public SpatialDropout2D(double rate)
        : base(2, rate)
    {
    }

    /// <inheritdoc />
    public static string Name => "spatialdropout2d";

    /// <inheritdoc />
    public static SpatialDropout2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteRate(writer);
    }
}

/// <summary>Leaves whole channels of a volume out at random while a network trains.</summary>
public sealed class SpatialDropout3D : SpatialDropout, ISaved<SpatialDropout3D>
{
    /// <summary>A dropout that leaves out the given share of the channels of each volume.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public SpatialDropout3D(double rate)
        : base(3, rate)
    {
    }

    /// <inheritdoc />
    public static string Name => "spatialdropout3d";

    /// <inheritdoc />
    public static SpatialDropout3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        WriteRate(writer);
    }
}
