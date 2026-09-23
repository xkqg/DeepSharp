// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// One column of a table: its name, what it holds, and which of its cells were never there.
/// </summary>
/// <remarks>
/// Absent is a state of its own, beside every value the column can hold. Collapsing it into a zero or an
/// empty word destroys the difference permanently, and the difference is often the most informative thing
/// in the column.
/// </remarks>
public interface IColumn
{
    /// <summary>The column's name.</summary>
    string Name { get; }

    /// <summary>What the column holds.</summary>
    ColumnKind Kind { get; }

    /// <summary>How many rows it has.</summary>
    int Count { get; }

    /// <summary>Whether the cell in this row was never there.</summary>
    /// <param name="row">The row to look at.</param>
    /// <returns><see langword="true"/> when the cell is a gap.</returns>
    bool IsMissing(int row);

    /// <summary>The cell as it would be written back out, or nothing when it is a gap.</summary>
    /// <param name="row">The row to look at.</param>
    /// <returns>The value as text, or <see langword="null"/>.</returns>
    string? TextAt(int row);

    /// <summary>The same column, from this row onwards.</summary>
    /// <param name="first">The first row to keep.</param>
    /// <returns>A column of the rows that were kept.</returns>
    IColumn From(int first);

    /// <summary>How many rows at the start of this column are gaps.</summary>
    /// <returns>The number of leading gaps, which for an indicator is its warm-up.</returns>
    int LeadingGaps();
}

/// <summary>
/// A column holding values of one kind.
/// </summary>
/// <typeparam name="T">What the column holds.</typeparam>
public sealed class Column<T> : IColumn
    where T : struct
{
    private readonly T?[] _values;

    /// <summary>A column of these values, in this order.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="kind">What it holds.</param>
    /// <param name="values">The values; nothing where a cell is a gap.</param>
    public Column(string name, ColumnKind kind, IEnumerable<T?> values)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(values);

        Name = name;
        Kind = kind;
        _values = [.. values];
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public ColumnKind Kind { get; }

    /// <inheritdoc />
    public int Count => _values.Length;

    /// <summary>The value in a row, or nothing when the cell is a gap.</summary>
    /// <param name="row">The row to look at.</param>
    /// <returns>The value, or <see langword="null"/>.</returns>
    public T? this[int row]
    {
        get => _values[row];
        set => _values[row] = value;
    }

    /// <inheritdoc />
    public bool IsMissing(int row) => _values[row] is null;

    /// <inheritdoc />
    public string? TextAt(int row) =>
        _values[row] is { } value
            ? Convert.ToString(value, CultureInfo.InvariantCulture)
            : null;

    /// <inheritdoc />
    public IColumn From(int first) => new Column<T>(Name, Kind, _values.Skip(first));

    /// <inheritdoc />
    public int LeadingGaps() => _values.TakeWhile(value => value is null).Count();
}

/// <summary>
/// A column of words.
/// </summary>
/// <remarks>
/// Text needs its own column because a missing word and an empty word are different answers, and one
/// nullable string cannot tell them apart from the outside.
/// </remarks>
public sealed class TextColumn : IColumn
{
    private readonly string?[] _values;

    /// <summary>A column of these words, in this order.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="values">The words; nothing where a cell is a gap.</param>
    public TextColumn(string name, IEnumerable<string?> values)
        : this(name, ColumnKind.Text, values)
    {
    }

    /// <summary>A column of these words, said to be plain text or a category.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="kind">Text, or a category.</param>
    /// <param name="values">The words; nothing where a cell is a gap.</param>
    /// <exception cref="ArgumentException">The kind is not one a column of words can be.</exception>
    public TextColumn(string name, ColumnKind kind, IEnumerable<string?> values)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(values);

        if (kind is not (ColumnKind.Text or ColumnKind.Category))
        {
            throw new ArgumentException($"A column of words is text or a category, not {kind}.", nameof(kind));
        }

        Name = name;
        Kind = kind;
        _values = [.. values];
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public ColumnKind Kind { get; }

    /// <inheritdoc />
    public int Count => _values.Length;

    /// <summary>The word in a row, or nothing when the cell is a gap.</summary>
    /// <param name="row">The row to look at.</param>
    /// <returns>The word, or <see langword="null"/>.</returns>
    public string? this[int row] => _values[row];

    /// <inheritdoc />
    public bool IsMissing(int row) => _values[row] is null;

    /// <inheritdoc />
    public string? TextAt(int row) => _values[row];

    /// <inheritdoc />
    public IColumn From(int first) => new TextColumn(Name, Kind, _values.Skip(first));

    /// <inheritdoc />
    public int LeadingGaps() => _values.TakeWhile(value => value is null).Count();
}

/// <summary>
/// The data as the pipeline carries it: named columns of equal length.
/// </summary>
/// <remarks>
/// Columns are kept in the order they were declared, because that order is what a model eventually sees and
/// a model fed the same numbers in another order is quietly a different model.
/// </remarks>
public sealed class Table
{
    private readonly List<IColumn> _columns;

    /// <summary>A table of exactly these columns.</summary>
    /// <param name="columns">The columns, in the order they should be seen.</param>
    /// <exception cref="ArgumentException">The columns are not all the same length, or a name repeats.</exception>
    public Table(IEnumerable<IColumn> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        _columns = [.. columns];

        var lengths = _columns.Select(column => column.Count).Distinct().ToArray();

        if (lengths.Length > 1)
        {
            throw new ArgumentException(
                $"The columns are of different lengths ({string.Join(", ", lengths)}), so there is no row to speak of.",
                nameof(columns));
        }

        var duplicate = _columns.GroupBy(column => column.Name).FirstOrDefault(group => group.Count() > 1);

        if (duplicate is not null)
        {
            throw new ArgumentException($"There are two columns named '{duplicate.Key}'.", nameof(columns));
        }

        RowCount = lengths.Length == 0 ? 0 : lengths[0];
    }

    /// <summary>The columns, in the order they should be seen.</summary>
    public IReadOnlyList<IColumn> Columns => _columns;

    /// <summary>How many rows the table has.</summary>
    public int RowCount { get; }

    /// <summary>The column with this name.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns>The column.</returns>
    /// <exception cref="KeyNotFoundException">There is no column with that name.</exception>
    public IColumn this[string name] =>
        _columns.FirstOrDefault(column => column.Name == name)
        ?? throw new KeyNotFoundException(
            $"There is no column called '{name}'. The table holds: {string.Join(", ", _columns.Select(column => column.Name))}.");

    /// <summary>Whether a column of that name is here at all.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns><see langword="true"/> when the table has one.</returns>
    public bool Has(string name) => _columns.Any(column => column.Name == name);

    /// <summary>Puts a column at the end, or replaces the one that has its name.</summary>
    /// <param name="column">The column to add.</param>
    /// <exception cref="ArgumentException">The column is not as long as the others.</exception>
    public void Put(IColumn column)
    {
        ArgumentNullException.ThrowIfNull(column);

        if (_columns.Count > 0 && column.Count != RowCount)
        {
            throw new ArgumentException(
                $"The column '{column.Name}' has {column.Count} rows where the table has {RowCount}.",
                nameof(column));
        }

        var at = _columns.FindIndex(existing => existing.Name == column.Name);

        if (at < 0)
        {
            _columns.Add(column);
        }
        else
        {
            _columns[at] = column;
        }
    }

    /// <summary>Keeps the rows from this one onwards, dropping everything before it.</summary>
    /// <param name="first">The first row to keep.</param>
    /// <returns>A table of the rows that were kept, with the same columns in the same order.</returns>
    /// <exception cref="ArgumentOutOfRangeException">That row is not in the table.</exception>
    /// <remarks>
    /// A new table rather than a change in place: rows are what a split divides and what a fit counts, so a
    /// step that drops some of them is making a different dataset and had better say so.
    /// </remarks>
    public Table From(int first)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(first, RowCount);

        return new Table(_columns.Select(column => column.From(first)));
    }

    /// <summary>Removes the column with this name, if it is there.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns><see langword="true"/> when a column was removed.</returns>
    public bool Remove(string name) => _columns.RemoveAll(column => column.Name == name) > 0;
}
