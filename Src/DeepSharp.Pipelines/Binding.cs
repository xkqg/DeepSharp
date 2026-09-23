// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// Turning the text in a source into the columns a schema declared.
/// </summary>
/// <remarks>
/// This is where a declaration meets real data, and where the two halves of checking a pipeline part
/// company: the shape of it needs no data at all, while this needs the source open. A cell that is empty is
/// a gap; a cell that is there and cannot be read as what it was declared to be is a fault, and it is
/// reported with the row, the column and the value rather than as a number that quietly went wrong.
/// </remarks>
public static class SchemaBinding
{
    /// <summary>Reads the rows of a source into the columns a schema declares.</summary>
    /// <param name="schema">The declared columns and the policy for the rest.</param>
    /// <param name="source">Where the rows come from.</param>
    /// <returns>The table the pipeline carries from here on.</returns>
    /// <exception cref="InvalidOperationException">A declared column is not in the source at all.</exception>
    /// <exception cref="FormatException">A cell cannot be read as the kind its column was declared to be.</exception>
    public static Table Bind(DeclareStep schema, IRowSource source)
    {
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(source);

        var rows = source.Rows.ToArray();
        var positions = Positions(schema, source);
        var columns = new List<IColumn>();

        foreach (var declared in schema.Columns)
        {
            if (!positions.TryGetValue(declared.Name, out var at))
            {
                continue;
            }

            columns.Add(Read(declared, rows, at));
        }

        if (schema.Remainder == Remainder.Keep)
        {
            for (var at = 0; at < source.ColumnNames.Count; at++)
            {
                var name = source.ColumnNames[at];

                if (schema.Columns.All(column => column.Name != name))
                {
                    columns.Add(Read(new ColumnDeclaration(name, ColumnKind.Text, Optional: true), rows, at));
                }
            }
        }

        // Each row is keyed by the record it was read from, every column of it, so leaving a column out of the
        // schema moves no row to another part of a split.
        var digest = new RecordDigest(source.ColumnNames);
        var identities = new RowIdentity[rows.Length];

        for (var at = 0; at < rows.Length; at++)
        {
            identities[at] = new RowIdentity(at, digest.Of(rows[at]));
        }

        return Table.Owning(columns, identities);
    }

    private static Dictionary<string, int> Positions(DeclareStep schema, IRowSource source)
    {
        var positions = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var at = 0; at < source.ColumnNames.Count; at++)
        {
            positions[source.ColumnNames[at]] = at;
        }

        var absent = schema.Columns
            .Where(column => !column.Optional && !positions.ContainsKey(column.Name))
            .Select(column => column.Name)
            .ToArray();

        if (absent.Length > 0)
        {
            throw new InvalidOperationException(
                $"The source has no column called {string.Join(" or ", absent.Select(name => $"'{name}'"))}. "
                + $"It offers: {string.Join(", ", source.ColumnNames)}.");
        }

        var unexpected = source.ColumnNames
            .Where(name => schema.Columns.All(column => column.Name != name))
            .ToArray();

        if (schema.Remainder == Remainder.Refuse && unexpected.Length > 0)
        {
            throw new InvalidOperationException(
                $"The source carries columns the schema does not name: {string.Join(", ", unexpected)}.");
        }

        return positions;
    }

    private static IColumn Read(ColumnDeclaration declared, IReadOnlyList<string?>[] rows, int at)
    {
        var cells = rows.Select(row => at < row.Count ? row[at] : null).ToArray();

        return declared.Kind switch
        {
            ColumnKind.Text or ColumnKind.Category => new TextColumn(
                declared.Name, declared.Kind, cells.Select(cell => string.IsNullOrEmpty(cell) ? null : cell)),
            ColumnKind.Number => new Column<double>(
                declared.Name, ColumnKind.Number, cells.Select((cell, row) => AsNumber(declared, cell, row))),
            ColumnKind.Integer => new Column<long>(
                declared.Name, ColumnKind.Integer, cells.Select((cell, row) => AsInteger(declared, cell, row))),
            ColumnKind.Boolean => new Column<bool>(
                declared.Name, ColumnKind.Boolean, cells.Select((cell, row) => AsBoolean(declared, cell, row))),
            _ => new Column<DateTime>(
                declared.Name, ColumnKind.Timestamp, cells.Select((cell, row) => AsTimestamp(declared, cell, row))),
        };
    }

    private static double? AsNumber(ColumnDeclaration declared, string? cell, int row)
    {
        if (IsGap(cell))
        {
            return null;
        }

        if (!double.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
        {
            throw Unreadable(declared, cell, row, "a number");
        }

        // A finite number, or one of the invariant spellings of a value that is not one: those are read as
        // what they say, for fill.nan to deal with. Digits that make an infinity are a number too large to
        // hold, and any other spelling of the words is not a number at all.
        return double.IsFinite(value) || cell!.Trim() is "NaN" or "Infinity" or "-Infinity"
            ? value
            : throw (cell!.Any(char.IsDigit)
                ? new FormatException($"Row {row + 1}, column '{declared.Name}': '{cell}' is a number too large to hold.")
                : Unreadable(declared, cell, row, "a number"));
    }

    private static long? AsInteger(ColumnDeclaration declared, string? cell, int row)
    {
        if (IsGap(cell))
        {
            return null;
        }

        return long.TryParse(cell, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw Unreadable(declared, cell, row, "a whole number");
    }

    private static bool? AsBoolean(ColumnDeclaration declared, string? cell, int row)
    {
        if (IsGap(cell))
        {
            return null;
        }

        // One tool writes True, another true, a third 1, and a database export says Y. A parser that knows
        // only one of those does not fail on the others: it reads text, and the column quietly becomes a
        // category nobody meant to make.
        return cell!.Trim().ToLowerInvariant() switch
        {
            "true" or "1" or "yes" or "y" or "t" => true,
            "false" or "0" or "no" or "n" or "f" => false,
            _ => throw Unreadable(declared, cell, row, "true or false"),
        };
    }

    private static DateTime? AsTimestamp(ColumnDeclaration declared, string? cell, int row)
    {
        if (IsGap(cell))
        {
            return null;
        }

        // Invariant and universal on purpose: the same file read on two machines has to produce the same
        // moment, and a date order that follows whoever is logged in is how that stops being true.
        return DateTime.TryParse(
            cell, CultureInfo.InvariantCulture,
            DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var value)
            ? value
            : throw Unreadable(declared, cell, row, "a moment in time");
    }

    private static bool IsGap(string? cell) => string.IsNullOrWhiteSpace(cell);

    private static FormatException Unreadable(ColumnDeclaration declared, string? cell, int row, string wanted) =>
        new($"Row {row + 1}, column '{declared.Name}': '{cell}' is not {wanted}.");
}
