// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that puts the rows in an order, learning nothing.
/// </summary>
/// <remarks>
/// The order a file or a query hands rows over in is an accident: a database returns them in whatever order it
/// likes unless it is told, and a file exported the other way round is the same data. A step that reads rows
/// in their order — a moving average, a warm-up at the start, a gap filled with the value before it — has to
/// read an order somebody declared, so declaring it is a step of its own, above the split, at most once.
/// </remarks>
public interface IOrdersRows : IActsInAWalk
{
    /// <summary>The order the rows are to stand in.</summary>
    /// <param name="table">The data as it stands.</param>
    /// <returns>The rows as they are now, in the order they are to stand: every row, once.</returns>
    IReadOnlyList<int> RowOrder(Table table);

    /// <summary>The columns the rows are put in order by, the one that decides first first, when the step says.</summary>
    /// <remarks>Nothing, unless the step orders by columns; a step that reads rows ahead needs to know which.</remarks>
    IReadOnlyList<string> OrderedBy => [];

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Reorder(RowOrder(walk.Table));
}

/// <summary>
/// A step whose value for a row is read from rows after it, in the declared order.
/// </summary>
/// <remarks>
/// Only an output may: a feature that knows the future is a leak in mathematical dress. And an output may only
/// across a split in time that keeps a gap at least as wide as how far it reads, the rows ordered by the column the
/// split divides by, or the last rows a model learns from read their answers from the rows it is measured on.
/// </remarks>
public interface IReadsRowsAhead : IReadsRowOrder
{
    /// <summary>How many rows later the value is read.</summary>
    int Ahead { get; }
}

/// <summary>
/// A step whose value for a row depends on the rows before it.
/// </summary>
/// <remarks>
/// It says so, and a declaration then refuses it unless the rows were put in order above it: without a
/// declared order, the rows before are whichever the file happened to put there.
/// </remarks>
public interface IReadsRowOrder : IPipelineStep
{
}

/// <summary>
/// A split that divides the rows by when they happened, and keeps a gap between its parts.
/// </summary>
/// <remarks>
/// What a step reading rows ahead needs to know of the split above it: the column it divides by, which the rows are
/// to be ordered by alone, and how many moments it keeps apart at the end of every part.
/// </remarks>
public interface IDividesInTime : IPipelineStep
{
    /// <summary>The column that says when a row happened.</summary>
    string Column { get; }

    /// <summary>How many of the last moments of every part are kept apart.</summary>
    int Gap { get; }
}

/// <summary>
/// Puts the rows in order by one or more columns, smallest first.
/// </summary>
/// <remarks>
/// Smallest first only, because the steps that read an order look backwards: the rows before a row are the
/// ones that came earlier. The first column decides, the next one decides between rows the first calls equal,
/// and so on; a row with no value in a key, or a key that is not a number, has no place in the order and is
/// refused, and so are two rows every key calls equal — nothing then says which came first, and taking
/// whichever the file put first would be the accident this step exists to end.
/// </remarks>
public sealed record OrderByStep : IPipelineStep<OrderByStep>, IOrdersRows, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns the rows are put in order by: the first decides, each next one decides between rows the ones before call equal.",
        ["when"], ColumnKinds.Ordered);

    /// <summary>Declares the order the rows stand in.</summary>
    /// <param name="columns">The columns to order by, the one that decides first first.</param>
    /// <exception cref="ArgumentException">There are none, one has no name, or one is named twice.</exception>
    public OrderByStep(IEnumerable<string> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
    }

    /// <summary>The columns the rows are put in order by.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> OrderedBy => Columns;

    /// <inheritdoc />
    public static string Name => "order.by";

    /// <inheritdoc />
    public static string Purpose => "Puts the rows in order by one or more columns, smallest first, for the steps that read the rows before a row.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<OrderByStep> Parameters { get; } =
        new StepParameters<OrderByStep>().With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(OrderByStep? other) => other is not null && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A key is a gap or not a number, or two rows are equal on every key.</exception>
    public IReadOnlyList<int> RowOrder(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var keys = Columns.Select(table.RowOrderBy).ToArray();
        var order = Enumerable.Range(0, table.RowCount).ToArray();

        Array.Sort(order, (one, other) => Compare(keys, one, other));

        for (var at = 1; at < order.Length; at++)
        {
            if (Compare(keys, order[at - 1], order[at]) == 0)
            {
                var first = Math.Min(order[at - 1], order[at]);
                var second = Math.Max(order[at - 1], order[at]);

                throw new InvalidOperationException(
                    $"Rows {table.Identities[first].ReadAt + 1} and {table.Identities[second].ReadAt + 1} are the same in "
                    + $"{string.Join(" and ", Columns.Select(column => $"'{column}'"))}, so nothing says which came first. "
                    + "Order by another column as well, one that tells them apart.");
            }
        }

        return order;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static OrderByStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));

    private static int Compare(Comparison<int>[] keys, int one, int other)
    {
        foreach (var key in keys)
        {
            var compared = key(one, other);

            if (compared != 0)
            {
                return compared;
            }
        }

        return 0;
    }
}
