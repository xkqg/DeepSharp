// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using MatPlotLibNet.Styling;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// A colour map that also says what colour lies below its range, above it, and where there is no value at all.
/// </summary>
/// <remarks>
/// A decorator over a map MatPlotLibNet already has, the way its own reversed map is: the stops of the map are
/// borrowed, never copied. A grid needs the three extra colours because a range is learned on the training rows —
/// a validation row beyond it is not the reddest training row, and a gap is not the bluest; coloured as either,
/// the grid would claim something about a value that is not true.
/// </remarks>
/// <param name="inner">The map inside the range.</param>
/// <param name="under">Below the range.</param>
/// <param name="over">Above the range.</param>
/// <param name="bad">No value: a gap, or a value that is not a number.</param>
internal sealed class EdgesColorMap(IColorMap inner, Color under, Color over, Color bad) : IColorMap
{
    /// <summary>Coolwarm, from blue at the smallest training value to red at the largest, with darker ends beyond them.</summary>
    public static EdgesColorMap Coolwarm { get; } = new(
        ColorMaps.Coolwarm, Color.FromHex("#1b2a6b"), Color.FromHex("#5c0014"), Color.FromHex("#d9d9d9"));

    /// <inheritdoc />
    public string Name => inner.Name;

    /// <inheritdoc />
    public Color GetColor(double value) => inner.GetColor(value);

    /// <inheritdoc />
    public Color? GetUnderColor() => under;

    /// <inheritdoc />
    public Color? GetOverColor() => over;

    /// <inheritdoc />
    public Color? GetBadColor() => bad;
}
