// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Security.Cryptography;

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

    /// <summary>The same column, holding these rows in this order.</summary>
    /// <param name="rows">The rows to hold, by where they are now.</param>
    /// <returns>A column of those rows.</returns>
    IColumn Rows(IReadOnlyList<int> rows);

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
    public IColumn Rows(IReadOnlyList<int> rows) => new Column<T>(Name, Kind, rows.Select(row => _values[row]));

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
    public IColumn Rows(IReadOnlyList<int> rows) => new TextColumn(Name, Kind, rows.Select(row => _values[row]));

    /// <inheritdoc />
    public int LeadingGaps() => _values.TakeWhile(value => value is null).Count();
}

/// <summary>
/// The data as the pipeline carries it: named columns of equal length.
/// </summary>
/// <remarks>
/// Columns are kept in the order they were declared, because that order is what a model eventually sees and
/// a model fed the same numbers in another order is quietly a different model.
/// <para>
/// Every row carries its identity — where it was read, and a key made from what it says — and the table is the
/// one owner of it: keeping some rows and putting rows in order both move each identity with its row.
/// </para>
/// </remarks>
public sealed class Table
{
    private readonly List<IColumn> _columns;
    private readonly RowIdentity[] _identities;

    /// <summary>A table of exactly these columns, each row known by its own cells and its place.</summary>
    /// <param name="columns">The columns, in the order they should be seen.</param>
    /// <exception cref="ArgumentException">The columns are not all the same length, or a name repeats.</exception>
    /// <remarks>
    /// A table built by hand has no record it was read from, so each row's key is made from the table's own
    /// cells by the rule every key is made by, and its place is where it stands.
    /// </remarks>
    public Table(IEnumerable<IColumn> columns)
        : this(columns, null)
    {
    }

    /// <summary>A table of exactly these columns, whose rows carry the identities they were read with.</summary>
    /// <param name="columns">The columns, in the order they should be seen.</param>
    /// <param name="identities">One identity per row, in row order; nothing to make them from the cells.</param>
    /// <exception cref="ArgumentException">
    /// The columns are not all the same length, a name repeats, or there is not one identity per row.
    /// </exception>
    public Table(IEnumerable<IColumn> columns, IReadOnlyList<RowIdentity>? identities)
        : this([.. columns ?? throw new ArgumentNullException(nameof(columns))], identities is null ? null : [.. identities])
    {
    }

    // The one place a table is made. Both lists belong to it from here on, so nothing outside can change them.
    private Table(List<IColumn> columns, RowIdentity[]? identities)
    {
        _columns = columns;

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

        if (identities is not null && identities.Length != RowCount)
        {
            throw new ArgumentException(
                $"There are {identities.Length} identities for {RowCount} rows, and every row has exactly one.", nameof(identities));
        }

        _identities = identities ?? [.. OwnIdentities()];
    }

    /// <summary>A table of columns and identities made for it and handed over, which nothing else holds.</summary>
    /// <param name="columns">The columns, in the order they should be seen.</param>
    /// <param name="identities">One identity per row, in row order.</param>
    /// <returns>The table, owning both.</returns>
    internal static Table Owning(List<IColumn> columns, RowIdentity[] identities) => new(columns, identities);

    /// <summary>Who each row is: where it was read, and the key of what it says, in row order.</summary>
    public IReadOnlyList<RowIdentity> Identities => _identities;

    /// <summary>A digest of the rows this table holds, the same whatever order they stand in.</summary>
    /// <returns>Sixty-four hexadecimal digits: SHA-256 over the rows' keys, in the order of the keys.</returns>
    /// <remarks>
    /// What a split writes down about the rows it divided: two runs that divided the same rows have the same
    /// digest, whichever order the file gave them in, and a row that changed changes it.
    /// </remarks>
    public string Digest()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> bytes = stackalloc byte[32];

        foreach (var key in _identities.Select(identity => identity.Key).Order())
        {
            key.WriteTo(bytes);
            hash.AppendData(bytes);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    private IEnumerable<RowIdentity> OwnIdentities()
    {
        var digest = new RecordDigest([.. _columns.Select(column => column.Name)]);

        for (var row = 0; row < RowCount; row++)
        {
            yield return new RowIdentity(row, digest.Of([.. _columns.Select(column => column.TextAt(row))]));
        }
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

    /// <summary>Keeps the rows a mask says to keep, each with its identity.</summary>
    /// <param name="keep">One answer per row: whether to keep it.</param>
    /// <returns>A table of the rows that were kept, in the order they stood, with the same columns.</returns>
    /// <exception cref="ArgumentException">The mask does not have one answer per row.</exception>
    /// <remarks>
    /// A new table rather than a change in place: rows are what a split divides and what a fit counts, so a
    /// step that drops some of them is making a different dataset and had better say so.
    /// </remarks>
    public Table Keep(IReadOnlyList<bool> keep)
    {
        ArgumentNullException.ThrowIfNull(keep);

        if (keep.Count != RowCount)
        {
            throw new ArgumentException(
                $"There are {keep.Count} answers for {RowCount} rows, and a mask says of every row whether it stays.", nameof(keep));
        }

        return Picked([.. Enumerable.Range(0, RowCount).Where(row => keep[row])]);
    }

    /// <summary>Puts the rows in a given order, each with its identity.</summary>
    /// <param name="order">The rows as they are now, in the order they are to stand: every row, once.</param>
    /// <returns>A table of the same rows in that order, with the same columns.</returns>
    /// <exception cref="ArgumentException">The order does not name every row exactly once.</exception>
    public Table Ordered(IReadOnlyList<int> order)
    {
        ArgumentNullException.ThrowIfNull(order);

        if (order.Count != RowCount || order.Any(row => row < 0 || row >= RowCount) || order.Distinct().Count() != RowCount)
        {
            throw new ArgumentException(
                $"An order of {RowCount} rows names each of them once; this one names {order.Count} places.", nameof(order));
        }

        return Picked(order);
    }

    /// <summary>A copy of this table as it stands, which later steps changing this one do not touch.</summary>
    /// <returns>The copy.</returns>
    internal Table Snapshot() => Picked([.. Enumerable.Range(0, RowCount)]);

    private Table Picked(IReadOnlyList<int> rows)
    {
        var identities = new RowIdentity[rows.Count];

        for (var at = 0; at < identities.Length; at++)
        {
            identities[at] = _identities[rows[at]];
        }

        return Owning([.. _columns.Select(column => column.Rows(rows))], identities);
    }

    /// <summary>Removes the column with this name, if it is there.</summary>
    /// <param name="name">The column's name.</param>
    /// <returns><see langword="true"/> when a column was removed.</returns>
    public bool Remove(string name) => _columns.RemoveAll(column => column.Name == name) > 0;

    /// <summary>How many rows are the same row more than once, and how many of those a split has pulled apart.</summary>
    /// <param name="parts">Which part each row belongs to, when the rows have been divided.</param>
    /// <returns>The groups of identical rows, the rows in them, the copies beyond the first, and the groups whose copies landed in more than one part.</returns>
    /// <exception cref="ArgumentException">The parts are not one per row.</exception>
    /// <remarks>
    /// Two rows are the same when every column says the same thing about them, a gap included: absent is a
    /// value of its own here as everywhere else. A group whose copies sit in two parts is a row a model can
    /// meet in test after learning it in training, which reads as skill and is not.
    /// </remarks>
    public DuplicateRows Duplicates(IReadOnlyList<Part>? parts = null)
    {
        if (parts is not null && parts.Count != RowCount)
        {
            throw new ArgumentException(
                $"There are {parts.Count} parts for {RowCount} rows, and a row belongs to exactly one part.", nameof(parts));
        }

        var groups = new Dictionary<string, List<int>>(StringComparer.Ordinal);

        for (var row = 0; row < RowCount; row++)
        {
            var key = string.Join('\u001F', _columns.Select(column => column.TextAt(row) is { } text ? $"\u0001{text}" : "\u0000"));

            if (!groups.TryGetValue(key, out var rows))
            {
                groups[key] = rows = [];
            }

            rows.Add(row);
        }

        var repeated = groups.Values.Where(rows => rows.Count > 1).ToArray();

        return new DuplicateRows(
            repeated.Length,
            repeated.Sum(rows => rows.Count),
            repeated.Sum(rows => rows.Count - 1),
            parts is null ? 0 : repeated.Count(rows => rows.Select(row => parts[row]).Distinct().Count() > 1));
    }
}

/// <summary>
/// The order a column puts rows in.
/// </summary>
internal static class RowOrdering
{
    /// <summary>How a column orders two rows, each value compared as what it is.</summary>
    /// <param name="table">The rows.</param>
    /// <param name="name">The column that orders them: a moment, a whole number or a number.</param>
    /// <returns>A comparison of two rows by their places in the table.</returns>
    /// <exception cref="InvalidOperationException">
    /// A row has no value in the column, a number that is not a number, or the column holds something with no order.
    /// </exception>
    /// <remarks>
    /// A moment as a moment and a whole number as a whole number: turning them all into one kind of number
    /// would call two different moments the same once they were close enough together.
    /// </remarks>
    internal static Comparison<int> RowOrderBy(this Table table, string name)
    {
        var column = table[name];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (column.IsMissing(row))
            {
                throw new InvalidOperationException(
                    $"Row {table.Identities[row].ReadAt + 1} has no '{name}', so it has no place in the order.");
            }
        }

        return column switch
        {
            Column<DateTime> moments => (one, other) => moments[one]!.Value.CompareTo(moments[other]!.Value),
            Column<long> whole => (one, other) => whole[one]!.Value.CompareTo(whole[other]!.Value),
            Column<double> numbers => Numbers(table, name, numbers),
            _ => throw new InvalidOperationException(
                $"'{name}' holds {column.Kind.ToString().ToLowerInvariant()}, which has no order to put rows in."),
        };
    }

    private static Comparison<int> Numbers(Table table, string name, Column<double> numbers)
    {
        for (var row = 0; row < numbers.Count; row++)
        {
            if (double.IsNaN(numbers[row]!.Value))
            {
                throw new InvalidOperationException(
                    $"Row {table.Identities[row].ReadAt + 1} of '{name}' is not a number, so it has no place in the order.");
            }
        }

        return (one, other) => numbers[one]!.Value.CompareTo(numbers[other]!.Value);
    }
}

/// <summary>
/// Rows that are the same row more than once.
/// </summary>
/// <param name="Groups">How many distinct rows appear more than once.</param>
/// <param name="Rows">How many rows belong to one of those groups.</param>
/// <param name="ExtraCopies">The copies beyond the first of each group: what dropping them would remove.</param>
/// <param name="GroupsAcrossParts">How many groups have copies in more than one part of the data.</param>
public readonly record struct DuplicateRows(int Groups, int Rows, int ExtraCopies, int GroupsAcrossParts);
