// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// A value a typed source holds — a database's, a Parquet file's, a spreadsheet's — as the cell the pipeline reads, and
/// the kind a column of such values says it holds.
/// </summary>
/// <remarks>
/// A typed source holds numbers and moments rather than the text a file writes, and every such source hands its values
/// over by this one rule, so a value is spelled one way whichever of them held it, and a row read from one is known by
/// the same text as the same row read from another. The spelling is the invariant culture's, so no machine settles a
/// decimal point or the order of a date differently; a number is its shortest exact form, so 2.0 is handed over as 2.
/// </remarks>
public static class TypedValueExtensions
{
    // What each kind of value a typed source holds is, as a column's kind.
    private static readonly Dictionary<Type, ColumnKind> Kinds = new()
    {
        [typeof(string)] = ColumnKind.Text,
        [typeof(char)] = ColumnKind.Text,
        [typeof(bool)] = ColumnKind.Boolean,
        [typeof(DateTime)] = ColumnKind.Timestamp,
        [typeof(DateOnly)] = ColumnKind.Timestamp,
        [typeof(byte)] = ColumnKind.Integer,
        [typeof(sbyte)] = ColumnKind.Integer,
        [typeof(short)] = ColumnKind.Integer,
        [typeof(ushort)] = ColumnKind.Integer,
        [typeof(int)] = ColumnKind.Integer,
        [typeof(uint)] = ColumnKind.Integer,
        [typeof(long)] = ColumnKind.Integer,
        [typeof(ulong)] = ColumnKind.Integer,
        [typeof(float)] = ColumnKind.Number,
        [typeof(double)] = ColumnKind.Number,
        [typeof(decimal)] = ColumnKind.Number,
    };

    // The form a moment is handed over in: ISO 8601, which a schema reads whatever format it says its moments are
    // written in.
    private const string RoundTrip = "O";

    /// <summary>The value as the cell the pipeline reads.</summary>
    /// <param name="value">The value, as the source holds it; nothing for a gap.</param>
    /// <returns>
    /// Nothing for nothing; a moment or a day in the form ISO 8601 writes it, <c>2015-02-18T09:30:00.0000000</c> or
    /// <c>2015-02-18</c>; anything else as the invariant culture writes it — a number in its shortest exact form, true
    /// and false as <c>True</c> and <c>False</c>.
    /// </returns>
    public static string? AsCell(this object? value) => value switch
    {
        null => null,
        DateTime moment => moment.ToString(RoundTrip, CultureInfo.InvariantCulture),
        DateOnly day => day.ToString(RoundTrip, CultureInfo.InvariantCulture),
        _ => Convert.ToString(value, CultureInfo.InvariantCulture),
    };

    /// <summary>The kind a column of values of this type says it holds.</summary>
    /// <param name="type">The type of the column's values.</param>
    /// <returns>
    /// Words, true or false, a moment, a whole number or a number; nothing for a type that says none of those, whose
    /// values are then proposed from what their cells look like.
    /// </returns>
    public static ColumnKind? AsColumnKind(this Type type)
    {
        ArgumentNullException.ThrowIfNull(type);

        return Kinds.TryGetValue(type, out var kind) ? kind : null;
    }
}
