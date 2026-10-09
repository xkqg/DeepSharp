// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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
    // How many of a column's unreadable cells its sentence names.
    private const int NamedCells = 3;

    /// <summary>Reads the rows of a source into the columns a schema declares.</summary>
    /// <param name="schema">The declared columns and the policy for the rest.</param>
    /// <param name="source">Where the rows come from.</param>
    /// <returns>The table the pipeline carries from here on.</returns>
    /// <exception cref="InvalidOperationException">A declared column is not in the source at all.</exception>
    /// <exception cref="FormatException">
    /// Cells cannot be read as the kind their column was declared to be: every such column is named, each with its row and
    /// value, or with how many cells and the first three.
    /// </exception>
    public static Table Bind(DeclareStep schema, IRowSource source)
    {
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(source);

        var rows = source.Rows.ToArray();
        var positions = Positions(schema, source);
        var columns = new List<IColumn>();
        var faults = new List<string>();

        foreach (var declared in schema.Taking)
        {
            if (!positions.TryGetValue(declared.Name, out var at))
            {
                continue;
            }

            columns.Add(Read(declared, rows, at, faults));
        }

        if (faults.Count > 0)
        {
            throw new FormatException(string.Join(Environment.NewLine, faults));
        }

        if (schema.Remainder == Remainder.Keep)
        {
            for (var at = 0; at < source.ColumnNames.Count; at++)
            {
                var name = source.ColumnNames[at];

                // A column the schema names, one it excludes too, is not the rest of the file.
                if (schema.Columns.All(column => column.Name != name))
                {
                    columns.Add(Read(new ColumnDeclaration(name, ColumnKind.Text, Optional: true), rows, at, faults));
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

        // The column that names each row, when the schema says one does and the rows have it.
        var id = schema.IdColumn is { } named && columns.Any(column => column.Name == named) ? named : null;

        return Table.Owning(columns, identities, id);
    }

    private static Dictionary<string, int> Positions(DeclareStep schema, IRowSource source)
    {
        var positions = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var at = 0; at < source.ColumnNames.Count; at++)
        {
            positions[source.ColumnNames[at]] = at;
        }

        // A column the schema excludes is named and not taken: not asked of the source, and not the rest of the file.
        var absent = schema.Taking
            .Where(column => !column.Optional && !positions.ContainsKey(column.Name))
            .Select(column => column.Name)
            .ToArray();

        if (absent.Length > 0)
        {
            // Rows served to a model carry the answers they await as gaps, and offer only what they were handed in with.
            var offered = source is RowsAwaitingAnAnswer awaiting ? awaiting.HandedIn : source.ColumnNames;

            throw new InvalidOperationException(
                $"The source has no column called {string.Join(" or ", absent.Select(name => $"'{name}'"))}. "
                + $"It offers: {string.Join(", ", offered)}.");
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

    // Reads one declared column. A cell that cannot be read as the column's kind is written down rather than thrown, so
    // every such column can be named at once. A cell holding the value the column says stands for a gap is a gap before
    // anything reads it.
    private static IColumn Read(ColumnDeclaration declared, IReadOnlyList<string?>[] rows, int at, List<string> faults)
    {
        var gap = StandsForAGap(declared);
        var cells = rows.Select(row => at < row.Count && !gap(row[at]) ? row[at] : null).ToArray();

        return declared.Kind switch
        {
            // Words keep a cell of spaces as the words it is; only an empty cell is a gap.
            ColumnKind.Text or ColumnKind.Category => new TextColumn(
                declared.Name, declared.Kind, cells.Select(cell => string.IsNullOrEmpty(cell) ? null : cell)),
            ColumnKind.Number => new Column<double>(declared.Name, ColumnKind.Number, Values(declared, cells, CellTextExtensions.AsNumber, faults)),
            ColumnKind.Integer => new Column<long>(declared.Name, ColumnKind.Integer, Values(declared, cells, CellTextExtensions.AsWholeNumber, faults)),
            ColumnKind.Boolean => new Column<bool>(declared.Name, ColumnKind.Boolean, Values(declared, cells, CellTextExtensions.AsTrueOrFalse, faults)),
            _ => new Column<DateTime>(declared.Name, ColumnKind.Timestamp, Values(declared, cells, cell => cell.AsMoment(declared.Format), faults)),
        };
    }

    private static T?[] Values<T>(ColumnDeclaration declared, string?[] cells, Func<string?, CellRead<T>> read, List<string> faults)
        where T : struct
    {
        var values = new T?[cells.Length];
        var unreadable = new List<UnreadableCell>();

        for (var row = 0; row < cells.Length; row++)
        {
            var reading = read(cells[row]);

            if (reading.Refusal is { } refusal)
            {
                unreadable.Add(new UnreadableCell(row, cells[row], refusal));
            }

            values[row] = reading.Value;
        }

        if (unreadable.Count > 0)
        {
            faults.Add(Unreadable(declared, unreadable));
        }

        return values;
    }

    // Whether a cell holds the value the column says stands for a gap: the same value as the column's kind reads it, when
    // the value reads as that kind; the same text otherwise, and always among words.
    private static Func<string?, bool> StandsForAGap(ColumnDeclaration declared) => declared.Missing is not { } missing
        ? _ => false
        : declared.Kind switch
        {
            ColumnKind.Number => Same(missing, CellTextExtensions.AsNumber),
            ColumnKind.Integer => Same(missing, CellTextExtensions.AsWholeNumber),
            ColumnKind.Boolean => Same(missing, CellTextExtensions.AsTrueOrFalse),
            ColumnKind.Timestamp => Same(missing, cell => cell.AsMoment(declared.Format)),
            _ => cell => cell == missing,
        };

    private static Func<string?, bool> Same<T>(string missing, Func<string?, CellRead<T>> read)
        where T : struct => read(missing).Value is { } value
            ? cell => read(cell).Value is { } held && EqualityComparer<T>.Default.Equals(held, value)
            : cell => cell == missing;

    // One sentence for a column: the one cell as it always was, or how many and the first few of them.
    private static string Unreadable(ColumnDeclaration declared, List<UnreadableCell> cells)
    {
        if (cells.Count == 1)
        {
            return $"Row {cells[0].Row + 1}, column '{declared.Name}': '{cells[0].Cell}' {cells[0].Refusal}.";
        }

        var named = string.Join(", ", cells.Take(NamedCells).Select(cell => $"row {cell.Row + 1} '{cell.Cell}' {cell.Refusal}"));
        var more = cells.Count > NamedCells ? $", and {cells.Count - NamedCells} more" : string.Empty;

        return $"Column '{declared.Name}': {cells.Count} cells cannot be read — {named}{more}.";
    }

    // A cell that could not be read as its column's kind, where it stood and why.
    private readonly record struct UnreadableCell(int Row, string? Cell, string Refusal);
}
