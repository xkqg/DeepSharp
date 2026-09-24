// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Places a correlation on the whole of its scale, from minus one to one, whatever the coefficients drawn happen
/// to span.
/// </summary>
/// <remarks>
/// A heatmap otherwise stretches its colours over the smallest and largest value it holds, so a table of weak
/// correlations would be drawn as if one of them were perfect. Nought is the middle of the map, always.
/// </remarks>
internal sealed class FromMinusOneToOne : INormalizer
{
    /// <summary>The one normalizer a correlation needs.</summary>
    public static FromMinusOneToOne Instance { get; } = new();

    private FromMinusOneToOne()
    {
    }

    /// <inheritdoc />
    public double Normalize(double value, double min, double max) => Math.Clamp((value + 1) / 2, 0, 1);
}
