// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Leaves values out at random while a network trains, and scales the ones it keeps by one over the share it keeps, so
/// the sum the next layer sees stays what it would have been: PyTorch's inverted dropout.
/// </summary>
/// <remarks>
/// Which values are left out is drawn from the run's stream for this layer's place in the network, the epoch and the step,
/// so the same run leaves out the same values and two dropout layers never the same ones. An evaluation pass leaves
/// nothing out and scales nothing. What is left out is a tensor of noughts and scales multiplied in, so its gradient
/// passes where a value was kept, scaled as it was.
/// </remarks>
public sealed class Dropout : Layer, ISaved<Dropout>
{
    /// <summary>A dropout that leaves out the given share of the values.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one: a layer that left out everything would pass nothing on.</exception>
    public Dropout(double rate)
    {
        Rate = RequireRate(rate);
    }

    /// <summary>The share of the values left out.</summary>
    public double Rate { get; }

    /// <inheritdoc />
    public static string Name => "dropout";

    /// <inheritdoc />
    public static Dropout Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "rate"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("rate", Rate);
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (pass.Mode == PassMode.Evaluation || Rate <= 0)
        {
            return input;
        }

        var draws = pass.Draws($"dropout:{Path}");
        var kept = 1 - Rate;
        var scale = (float)(1 / kept);
        var mask = new float[input.Shape.Count];

        for (var at = 0; at < mask.Length; at++)
        {
            mask[at] = draws.NextDouble() < kept ? scale : 0f;
        }

        return pass.Backend.Multiply(input, Tensor.From(input.Shape, mask));
    }

    /// <summary>A share a dropout can leave out: from nothing to below one.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not such a share.</exception>
    internal static double RequireRate(double rate) =>
        rate is >= 0 and < 1 ? rate : throw new ArgumentOutOfRangeException(nameof(rate), rate, "A dropout leaves out a share of the values, from nothing to below one.");
}
