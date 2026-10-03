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

    // The shapes ISO 8601 writes a date and a time in: a date alone, a time to the minute, the second or a fraction of it,
    // with a T or a space between, and with a zone or an offset or none. Each reads the moment the earlier reading did.
    private static readonly string[] Iso8601 =
    [
        "yyyy-MM-dd", "yyyy-MM-dd HH:mm", "yyyy-MM-dd HH:mm:ss", "yyyy-MM-ddTHH:mm", "yyyy-MM-ddTHH:mm:ss",
        "yyyy-MM-ddTHH:mm:ss.FFFFFFF", "yyyy-MM-ddTHH:mm:ssK", "yyyy-MM-ddTHH:mm:ss.FFFFFFFK", "yyyy-MM-dd HH:mm:ssK",
        "yyyy-MM-dd HH:mm:ss.FFFFFFF",
    ];

    // The form a source that holds moments as moments — a database, a frame already typed — hands each one over in. A
    // format says how a file writes its moments, and none mistakes this form for another moment, so it is read whatever
    // the format says.
    private const string RoundTrip = "O";

    // In universal time, assumed when the text names no zone, with the spaces around a cell allowed as they are for a number.
    private const DateTimeStyles MomentStyles = DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal | DateTimeStyles.AllowWhiteSpaces;

    extension(string? cell)
    {
        /// <summary>Whether the cell is a gap for a kind that is not words: nothing, or nothing but spaces.</summary>
        /// <returns><see langword="true"/> for a gap.</returns>
        internal bool IsGap() => string.IsNullOrWhiteSpace(cell);

        /// <summary>The cell read as a number that can have a fraction.</summary>
        /// <returns>The number, a gap, or why the text is not one.</returns>
        /// <remarks>
        /// A finite number, or one of the invariant spellings of a value that is not one — NaN, Infinity, -Infinity — which
        /// is read as what it says, for fill.nan to deal with. Digits that make an infinity are a number too large to hold,
        /// and any other spelling of those words is not a number at all.
        /// </remarks>
        internal CellRead<double> AsNumber()
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
        /// <returns>The number, a gap, or why the text is not one.</returns>
        internal CellRead<long> AsWholeNumber() =>
            cell.IsGap() ? default
            : long.TryParse(cell, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) ? new(value, null)
            : new(null, "is not a whole number");

        /// <summary>The cell read as true or false, in whichever of the usual spellings it is written.</summary>
        /// <returns>The answer, a gap, or why the text is neither.</returns>
        /// <remarks>
        /// One tool writes True, another true, a third 1, and a database export says Y. A parser that knows only one of those
        /// does not fail on the others: it reads text, and the column quietly becomes a category nobody meant to make.
        /// </remarks>
        internal CellRead<bool> AsTrueOrFalse() =>
            cell.IsGap() ? default
            : SpellingsOfTrue.Contains(cell!.Trim()) ? new(true, null)
            : SpellingsOfFalse.Contains(cell!.Trim()) ? new(false, null)
            : new(null, "is not true or false");

        /// <summary>The cell read as a moment in time, written in a format or as ISO 8601 writes one.</summary>
        /// <param name="format">How the column's moments are written, as .NET writes a date format; nothing for ISO 8601.</param>
        /// <returns>The moment, a gap, or why the text is not one.</returns>
        /// <remarks>
        /// Read exactly as written, and in universal time, so the same file gives the same moment on every machine and on every
        /// day: a reading that filled in what the text leaves out read 7.25 as the twenty-fifth of July of whichever year it
        /// ran in, 12:30 as that time today, and 02/03/2015 as the third of February everywhere. A column with a format also
        /// reads the round-trip form a database or a typed frame hands its moments over in, which is ISO 8601 as well.
        /// </remarks>
        internal CellRead<DateTime> AsMoment(string? format)
        {
            if (cell.IsGap())
            {
                return default;
            }

            if (format is null)
            {
                return DateTime.TryParseExact(cell, Iso8601, CultureInfo.InvariantCulture, MomentStyles, out var iso)
                    ? new(iso, null)
                    : new(null, "is not a moment in time as ISO 8601 writes one, such as 2015-02-18 or 2015-02-18T09:30:15; a column of moments written another way declares its format");
            }

            return DateTime.TryParseExact(cell, [format, RoundTrip], CultureInfo.InvariantCulture, MomentStyles, out var written)
                ? new(written, null)
                : new(null, $"is not a moment in time written as {format}");
        }
    }
}
