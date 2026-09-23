// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// Rows of text, with a name for each column.
/// </summary>
/// <remarks>
/// The one door data comes through. Everything a source has to offer is a list of column names and rows of
/// cells that are still text — deciding what a cell means is the schema's job, so a format nobody shipped
/// is a few lines away rather than a fork. An absent cell is <see langword="null"/>, which is not the same
/// as an empty one: the first says the row was short, the second says the field was there and blank.
/// </remarks>
public interface IRowSource
{
    /// <summary>The column names, in the order the cells arrive.</summary>
    IReadOnlyList<string> ColumnNames { get; }

    /// <summary>The rows, read one at a time.</summary>
    IEnumerable<IReadOnlyList<string?>> Rows { get; }
}

/// <summary>
/// Rows held in memory, for a caller who already has them.
/// </summary>
public sealed class InMemoryRowSource : IRowSource
{
    /// <summary>A source of exactly these rows.</summary>
    /// <param name="columnNames">The column names, in the order the cells arrive.</param>
    /// <param name="rows">The rows.</param>
    public InMemoryRowSource(IEnumerable<string> columnNames, IEnumerable<IReadOnlyList<string?>> rows)
    {
        ArgumentNullException.ThrowIfNull(columnNames);
        ArgumentNullException.ThrowIfNull(rows);

        ColumnNames = [.. columnNames];
        Rows = [.. rows];
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows { get; }
}

/// <summary>
/// Rows read from a comma-separated file.
/// </summary>
/// <remarks>
/// Small on purpose. A quoted field may hold commas, line breaks and doubled quotes; everything else about
/// the dialect — other separators, comment lines, a missing header — belongs to a reader that declares it
/// rather than to guesswork here. A row with the wrong number of cells is refused, with its line number,
/// because a file that has silently shifted by one column is the worst kind of correct-looking data.
/// </remarks>
public sealed class CsvRowSource : IRowSource
{
    private readonly List<IReadOnlyList<string?>> _rows = [];

    /// <summary>Reads a comma-separated file.</summary>
    /// <param name="path">The file to read.</param>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">The file has no header, or a row has the wrong number of cells.</exception>
    public CsvRowSource(string path)
        : this(File.ReadAllText(path), path)
    {
    }

    private CsvRowSource(string text, string what)
    {
        var lines = Split(text);

        if (lines.Count == 0)
        {
            throw new FormatException($"{what} has no header row, so its columns have no names.");
        }

        ColumnNames = [.. lines[0].Select(cell => cell ?? string.Empty)];

        for (var line = 1; line < lines.Count; line++)
        {
            if (lines[line].Count != ColumnNames.Count)
            {
                throw new FormatException(
                    $"{what} line {line + 1} has {lines[line].Count} cells where the header names {ColumnNames.Count}.");
            }

            _rows.Add(lines[line]);
        }
    }

    /// <summary>Reads comma-separated text that is already in hand.</summary>
    /// <param name="text">The text, header row first.</param>
    /// <returns>A source of the rows it holds.</returns>
    /// <exception cref="FormatException">The text has no header, or a row has the wrong number of cells.</exception>
    public static CsvRowSource FromText(string text) => new(text, "The text");

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => _rows;

    private static List<IReadOnlyList<string?>> Split(string text)
    {
        var lines = new List<IReadOnlyList<string?>>();
        var cells = new List<string?>();
        var cell = new StringBuilder();
        var quoted = false;
        var anything = false;

        for (var at = 0; at < text.Length; at++)
        {
            var character = text[at];

            if (quoted)
            {
                if (character != '"')
                {
                    cell.Append(character);
                }
                else if (at + 1 < text.Length && text[at + 1] == '"')
                {
                    cell.Append('"');
                    at++;
                }
                else
                {
                    quoted = false;
                }

                continue;
            }

            switch (character)
            {
                case '"':
                    quoted = true;
                    anything = true;
                    break;

                case ',':
                    cells.Add(cell.ToString());
                    cell.Clear();
                    anything = true;
                    break;

                case '\r':
                    break;

                case '\n':
                    cells.Add(cell.ToString());
                    cell.Clear();
                    lines.Add(cells);
                    cells = [];
                    anything = false;
                    break;

                default:
                    cell.Append(character);
                    anything = true;
                    break;
            }
        }

        // A file usually ends with a line break, and the row before it is already closed. One that does not
        // still has a last row, and a file that ends with a break has no empty row after it.
        if (anything || cell.Length > 0 || cells.Count > 0)
        {
            cells.Add(cell.ToString());
            lines.Add(cells);
        }

        return lines;
    }
}
