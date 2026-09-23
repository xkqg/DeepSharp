// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that takes columns off the table, learning nothing.
/// </summary>
/// <remarks>
/// The column's counterpart of <see cref="IDropsRows"/>. It may stand anywhere after the columns are read,
/// above the split or below it, because leaving a column out changes what a model is shown and not which
/// rows there are.
/// </remarks>
public interface IDropsColumns : IActsInAWalk
{
    /// <summary>Takes the columns off the table.</summary>
    /// <param name="table">The data, changed in place.</param>
    void DropFrom(Table table);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => DropFrom(walk.Table);
}

/// <summary>
/// Leaves columns out from here on.
/// </summary>
/// <remarks>
/// Leaving a column out of the schema is the first way, and the right one for a column nobody needs at all.
/// This is the way for the rest: a column a step reads before it goes, and a column a step made — the one
/// that says where a gap was, which is written always and dropped by whoever does not want it.
/// </remarks>
public sealed record DropColumnsStep : IPipelineStep<DropColumnsStep>, IDropsColumns, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to leave out from here on.", ["column"], ColumnKinds.Any);

    /// <summary>Declares that these columns are left out from here on.</summary>
    /// <param name="columns">The columns to leave out.</param>
    /// <exception cref="ArgumentException">There are none, one has no name, or one is named twice.</exception>
    public DropColumnsStep(IEnumerable<string> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
    }

    /// <summary>The columns left out.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "drop.columns";

    /// <inheritdoc />
    public static string Purpose => "Leaves columns out from here on.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<DropColumnsStep> Parameters { get; } =
        new StepParameters<DropColumnsStep>().With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Columns.Aggregate(before, (state, column) => state.Without(column));
    }

    /// <inheritdoc />
    public bool Equals(DropColumnsStep? other) => other is not null && Columns.SequenceEqual(other.Columns);

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
    /// <exception cref="InvalidOperationException">A column is not on the table at this point.</exception>
    public void DropFrom(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        foreach (var column in Columns)
        {
            if (!table.Remove(column))
            {
                throw new InvalidOperationException(
                    $"'{column}' is not here to be left out. The table holds: "
                    + $"{string.Join(", ", table.Columns.Select(each => each.Name))}.");
            }
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DropColumnsStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));
}
