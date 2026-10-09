// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using MatPlotLibNet.Styling;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Charts;

/// <summary>
/// A colour map that leaves a cell with no number in it without a colour, and colours every other one as the map it wraps does.
/// </summary>
/// <param name="colours">The map that colours the cells that have a number.</param>
/// <remarks>
/// A correlation between a column that never changes and another has no coefficient, and a coefficient nobody could work out
/// is not nought: a cell coloured as nought would read as a measurement. The drawing library takes such a cell through its
/// colour map like any other, and on .NET 8 asks its map for the colour of a number that is not one and fails, so the cell is
/// answered here, the same way on every runtime: with no colour at all.
/// </remarks>
internal sealed class BlankWhereUndefined(IColorMap colours) : IColorMap
{
    /// <inheritdoc />
    public string Name => colours.Name;

    /// <inheritdoc />
    public Color GetColor(double value) => double.IsNaN(value) ? new Color(0, 0, 0, 0) : colours.GetColor(value);

    /// <inheritdoc />
    public Color? GetUnderColor() => colours.GetUnderColor();

    /// <inheritdoc />
    public Color? GetOverColor() => colours.GetOverColor();

    /// <inheritdoc />
    public Color? GetBadColor() => colours.GetBadColor();
}
