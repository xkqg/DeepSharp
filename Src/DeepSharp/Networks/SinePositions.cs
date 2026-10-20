// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Adds to each step of a series the place it stands at, as the sine and cosine waves of 'Attention Is All You Need' write it: the even channels
/// the sine of the step over ten thousand to the power of the channel over the width, the odd channels the cosine of the same.
/// </summary>
/// <remarks>
/// It learns nothing and keeps no slot: the positions are worked out from the step and the width, the same for every series, so attention,
/// which does not see the order of the steps it looks at, can tell one step from another by where it stands.
/// </remarks>
public sealed class SinePositions : Layer, ISaved<SinePositions>
{
    /// <inheritdoc />
    public static string Name => "sinePositions";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static SinePositions Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 3)
        {
            throw new ArgumentException($"Positions are added to a batch of series, series by step by channel, and this was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var steps = input.Shape[1];
        var width = input.Shape[2];
        var positions = new float[steps * width];

        for (var step = 0; step < steps; step++)
        {
            for (var channel = 0; channel < width; channel++)
            {
                var angle = step / Math.Pow(10000, (channel - (channel % 2)) / (double)width);

                positions[(step * width) + channel] = (float)(channel % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }

        var rows = backend.Reshape(input, new Shape(input.Shape[0], steps * width));

        return backend.Reshape(backend.AddRow(rows, Tensor.From(new Shape(steps * width), positions)), input.Shape);
    }
}
