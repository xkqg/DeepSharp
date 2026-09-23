// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that drops rows, changing what there is to divide.
/// </summary>
/// <remarks>
/// Rows are what a split divides and what a fit counts, so dropping some of them is making a different
/// dataset. It happens before the split for exactly that reason: a row nobody can use should never land in
/// one, and certainly not in the part a model is measured on.
/// </remarks>
public interface IDropsRows : IActsInAWalk
{
    /// <summary>Which rows stay.</summary>
    /// <param name="table">The data as it stands.</param>
    /// <returns>One answer per row: whether it stays.</returns>
    IReadOnlyList<bool> RowsToKeep(Table table);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Keep(RowsToKeep(walk.Table));
}

/// <summary>
/// Drops every row that has a gap in any of the named columns.
/// </summary>
/// <remarks>
/// The other answer to a gap besides filling it, for a row that is no use without the value. Above the split,
/// where every part loses the row alike; the split is what makes a part's rows the part's.
/// </remarks>
public sealed record DropGapsStep : IPipelineStep<DropGapsStep>, IDropsRows, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns a row may not have a gap in.", ["column"], ColumnKinds.Any);

    /// <summary>Declares that every row with a gap in any of these columns is dropped.</summary>
    /// <param name="columns">The columns a row may not have a gap in.</param>
    /// <exception cref="ArgumentException">There are none, one has no name, or one is named twice.</exception>
    public DropGapsStep(IEnumerable<string> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
    }

    /// <summary>The columns a row may not have a gap in.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "drop.gaps";

    /// <inheritdoc />
    public static string Purpose => "Drops every row that has a gap in any of the named columns, before the rows are divided.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<DropGapsStep> Parameters { get; } =
        new StepParameters<DropGapsStep>().With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(DropGapsStep? other) => other is not null && Columns.SequenceEqual(other.Columns);

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
    public IReadOnlyList<bool> RowsToKeep(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var columns = Columns.Select(column => table[column]).ToArray();

        return [.. Enumerable.Range(0, table.RowCount).Select(row => !columns.Any(column => column.IsMissing(row)))];
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DropGapsStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));
}

/// <summary>
/// Drops the rows at the start that no column can speak for yet.
/// </summary>
/// <remarks>
/// An indicator of period N says nothing about the first N rows — not "nothing happened" but "there is not
/// enough history yet to say". Filling those with a number learned from the training rows would invent a
/// measurement nobody took, which is the one thing this library refuses everywhere else. So they go, and
/// the honest row count of a dataset with indicators on it is <c>rows − the longest warm-up</c>: 506 rows
/// with a twenty-period average on them are 487 rows of data and nineteen rows of not-yet.
/// <para>
/// One number for the whole table, not one per column, because the columns have to stay the same length.
/// The longest warm-up wins, and every column starts where the slowest of them can first speak.
/// </para>
/// </remarks>
public sealed record DropWarmUpStep : IPipelineStep<DropWarmUpStep>, IDropsRows, IReadsRowOrder, IDescribesColumns
{
    private static readonly WholeNumberParameter AtMostKey = new(
        "atMost", "The most rows this may drop from the start; beyond it the run stops rather than shrink the data to nothing.", 1000, atLeast: 0);

    /// <summary>Drops the rows at the start that any column is still silent about.</summary>
    /// <param name="atMost">The most rows this is allowed to drop; beyond it the run stops.</param>
    /// <exception cref="ArgumentOutOfRangeException">The limit is below nothing.</exception>
    public DropWarmUpStep(int atMost = 1000) => AtMost = AtMostKey.Require(atMost);

    /// <inheritdoc />
    public static StepParameters<DropWarmUpStep> Parameters { get; } =
        new StepParameters<DropWarmUpStep>().With(AtMostKey, step => step.AtMost);

    /// <summary>The most rows this is allowed to drop.</summary>
    /// <remarks>
    /// A guard rather than a setting. A column that is empty from the top for another reason entirely — a
    /// source that starts late, a join that missed — would otherwise quietly take the whole dataset with
    /// it, and a dataset that shrank to nothing is a run that should stop rather than succeed.
    /// </remarks>
    public int AtMost { get; }

    /// <inheritdoc />
    public static string Name => "drop.warmup";

    /// <inheritdoc />
    public static string Purpose => "Drops the rows at the start that an indicator cannot yet speak for.";

    /// <inheritdoc />
    /// <remarks>The second version drops the start of the declared order, where the first dropped the start of the file.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>The leading run and nothing else: a gap further down is a gap in the data, for the steps that deal with gaps.</remarks>
    public IReadOnlyList<bool> RowsToKeep(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var warmUp = WarmUp(table);

        return [.. Enumerable.Range(0, table.RowCount).Select(row => row >= warmUp)];
    }

    private int WarmUp(Table table)
    {
        var warmUp = table.Columns.Count == 0 ? 0 : table.Columns.Max(column => column.LeadingGaps());

        if (warmUp == 0)
        {
            return 0;
        }

        var slowest = table.Columns.OrderByDescending(column => column.LeadingGaps()).First();

        // A table with no rows left is a run that should stop rather than succeed, whatever the limit says:
        // a handful of rows served to a long average are all warm-up, and the limit is far above a handful.
        if (warmUp >= table.RowCount)
        {
            throw new InvalidOperationException(
                $"'{slowest.Name}' says nothing about every one of its {table.RowCount} rows, so dropping its "
                + "warm-up would leave no rows at all. Hand in enough history for it, or leave it out.");
        }

        if (warmUp > AtMost)
        {
            throw new InvalidOperationException(
                $"'{slowest.Name}' says nothing about its first {warmUp} rows, and this pipeline drops at "
                + $"most {AtMost}. Either that column starts later than the data does, or the warm-up is "
                + "longer than the dataset can afford.");
        }

        return warmUp;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DropWarmUpStep ReadFrom(JsonElement element) => new(AtMostKey.Read(element));
}
