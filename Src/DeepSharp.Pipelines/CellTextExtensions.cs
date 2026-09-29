// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>What one cell says, read as one kind: a value, a gap, or why its text is not that kind.</summary>
/// <typeparam name="T">What the kind holds.</typeparam>
/// <param name="Value">The value; nothing for a gap, or for text that is not the kind.</param>
/// <param name="Refusal">
/// Why the text is not the kind, as the end of a sentence about the cell — "is not a number"; nothing when the cell was
/// read or is a gap.
/// </param>
internal readonly record struct CellRead<T>(T? Value, string? Refusal)
    where T : struct;

/// <summary>
/// Reading the text of one cell as each kind a column can hold.
/// </summary>
/// <remarks>
/// The one reading of what a cell says: the schema binds a column by it, and whatever else asks what a column holds asks
/// it too, so two parts of the library never disagree about whether a cell is a number. Every form is read under the
/// invariant culture, which settles the decimal point and the order of a date before a machine can settle them
/// differently. A cell that is nothing, or nothing but spaces, is a gap for every kind but words.
/// </remarks>
internal static class CellTextExtensions
{
    private static readonly HashSet<string> SpellingsOfTrue = new(["true", "1", "yes", "y", "t"], StringComparer.OrdinalIgnoreCase);

    private static readonly HashSet<string> SpellingsOfFalse = new(["false", "0", "no", "n", "f"], StringComparer.OrdinalIgnoreCase);

    /// <summary>Whether the cell is a gap for a kind that is not words: nothing, or nothing but spaces.</summary>
    /// <param name="cell">The cell's text.</param>
    /// <returns><see langword="true"/> for a gap.</returns>
    internal static bool IsGap(this string? cell) => string.IsNullOrWhiteSpace(cell);

    /// <summary>The cell read as a number that can have a fraction.</summary>
    /// <param name="cell">The cell's text.</param>
    /// <returns>The number, a gap, or why the text is not one.</returns>
    /// <remarks>
    /// A finite number, or one of the invariant spellings of a value that is not one — NaN, Infinity, -Infinity — which
    /// is read as what it says, for fill.nan to deal with. Digits that make an infinity are a number too large to hold,
    /// and any other spelling of those words is not a number at all.
    /// </remarks>
    internal static CellRead<double> AsNumber(this string? cell)
    {
        if (cell.IsGap())
        {
            return default;
        }

        if (!double.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
        {
            return new(null, "is not a number");
        }

        return double.IsFinite(value) || cell!.Trim() is "NaN" or "Infinity" or "-Infinity" ? new(value, null)
            : cell!.Any(char.IsDigit) ? new(null, "is a number too large to hold")
            : new(null, "is not a number");
    }

    /// <summary>The cell read as a whole number.</summary>
    /// <param name="cell">The cell's text.</param>
    /// <returns>The number, a gap, or why the text is not one.</returns>
    internal static CellRead<long> AsWholeNumber(this string? cell) =>
        cell.IsGap() ? default
        : long.TryParse(cell, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) ? new(value, null)
        : new(null, "is not a whole number");

    /// <summary>The cell read as true or false, in whichever of the usual spellings it is written.</summary>
    /// <param name="cell">The cell's text.</param>
    /// <returns>The answer, a gap, or why the text is neither.</returns>
    /// <remarks>
    /// One tool writes True, another true, a third 1, and a database export says Y. A parser that knows only one of those
    /// does not fail on the others: it reads text, and the column quietly becomes a category nobody meant to make.
    /// </remarks>
    internal static CellRead<bool> AsTrueOrFalse(this string? cell) =>
        cell.IsGap() ? default
        : SpellingsOfTrue.Contains(cell!.Trim()) ? new(true, null)
        : SpellingsOfFalse.Contains(cell!.Trim()) ? new(false, null)
        : new(null, "is not true or false");

    /// <summary>The cell read as a moment in time.</summary>
    /// <param name="cell">The cell's text.</param>
    /// <returns>The moment, a gap, or why the text is not one.</returns>
    /// <remarks>
    /// Invariant and universal on purpose: the same file read on two machines has to produce the same moment, and a date
    /// order that follows whoever is logged in is how that stops being true.
    /// </remarks>
    internal static CellRead<DateTime> AsMoment(this string? cell) =>
        cell.IsGap() ? default
        : DateTime.TryParse(cell, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var value)
            ? new(value, null)
            : new(null, "is not a moment in time");
}
