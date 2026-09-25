// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// One row as it was read, before any step changed it: what a way back reads when the number alone cannot say
/// where it came from.
/// </summary>
/// <remarks>
/// A share per bird comes back as a count only by the birds of its own row, and a return comes back as a price only
/// by the price it was made from. Only what a way back names is kept as it was read: the column each answer comes
/// back to, and every column a step on the way reads.
/// </remarks>
public readonly record struct RowAsRead
{
    private readonly ColumnsAsRead? _columns;

    /// <summary>A row of the columns kept as they were read.</summary>
    /// <param name="columns">The columns.</param>
    /// <param name="readAt">The row's place among the rows as they were read.</param>
    internal RowAsRead(ColumnsAsRead columns, int readAt)
    {
        _columns = columns;
        ReadAt = readAt;
    }

    /// <summary>The row's place among the rows as they were read, counting from nought.</summary>
    public int ReadAt { get; }

    /// <summary>A column's value in this row as it was read.</summary>
    /// <param name="column">The column's name.</param>
    /// <returns>The value, or nothing where the row held a gap.</returns>
    /// <exception cref="InvalidOperationException">The column is not kept as it was read: no way back names it.</exception>
    public double? this[string column] => _columns is { } columns && columns.Holds(column)
        ? columns.At(column, ReadAt)
        : throw new InvalidOperationException(
            $"'{column}' is not kept as it was read. A way back reads from the row only the columns its steps name, "
            + "and the column its answer comes back to.");
}

/// <summary>
/// The columns the ways back of a declaration need, as the rows were read into them.
/// </summary>
/// <remarks>
/// Kept once, where the rows are read, for every answer at once: the run's check, predictions put back for a part,
/// and predictions put back for served rows all read the same columns the same way.
/// </remarks>
internal sealed class ColumnsAsRead
{
    private readonly Dictionary<string, double?[]> _columns;

    private ColumnsAsRead(Dictionary<string, double?[]> columns) => _columns = columns;

    /// <summary>Nothing kept: before the rows are read, or for a pipeline no run left anything with.</summary>
    public static ColumnsAsRead None { get; } = new([]);

    /// <summary>Keeps, from the rows as they were read, the columns the declaration's ways back need.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="table">The rows as they were read into columns, before any step changed them.</param>
    /// <returns>Those of them that hold numbers: words and moments have no arithmetic to come back through.</returns>
    public static ColumnsAsRead Of(PipelineDeclaration declaration, Table table)
    {
        var columns = new Dictionary<string, double?[]>(StringComparer.Ordinal);

        foreach (var name in UndoChain.ReadBy(declaration))
        {
            if (table.Has(name) && ColumnKinds.Numbers.Contains(table[name].Kind))
            {
                columns[name] = table.NumbersOf(name);
            }
        }

        return new ColumnsAsRead(columns);
    }

    /// <summary>Whether a column is kept.</summary>
    /// <param name="column">The column's name.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    public bool Holds(string column) => _columns.ContainsKey(column);

    /// <summary>A kept column's value in one row as it was read.</summary>
    /// <param name="column">The column's name.</param>
    /// <param name="readAt">The row's place among the rows as they were read.</param>
    /// <returns>The value, or nothing where the row held a gap.</returns>
    public double? At(string column, int readAt) => _columns[column][readAt];
}

/// <summary>A step on a way back, and what it learned.</summary>
/// <param name="Step">The step.</param>
/// <param name="Fitted">What it learned, when it learned anything.</param>
internal readonly record struct UndoLink(IUndoesItself Step, FittedStepValues? Fitted);

/// <summary>
/// The way back for one answer: every step that changed it, the last one first, and the column as read it comes
/// back to.
/// </summary>
/// <remarks>
/// Walked from the last step up. A step that undoes the column being followed is undone, and the way back goes on
/// with the column that step made it from, so an answer made into a new column comes back through whatever was done
/// to the column it was made from. Worked out here and nowhere else: putting predictions back, the run's check, and
/// what is kept as it was read all ask it.
/// </remarks>
internal sealed class UndoChain
{
    private static readonly IReadOnlyDictionary<int, FittedStepValues> NothingLearned = new Dictionary<int, FittedStepValues>();

    private UndoChain(string end, IReadOnlyList<UndoLink> links)
    {
        End = end;
        Links = links;
    }

    /// <summary>The column as read that the way back comes back to.</summary>
    public string End { get; }

    /// <summary>The steps to undo, the last one first.</summary>
    public IReadOnlyList<UndoLink> Links { get; }

    /// <summary>The way back for one answer.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="fitted">What each step learned, by its place.</param>
    /// <param name="answer">The answer column.</param>
    /// <returns>Its way back.</returns>
    public static UndoChain For(PipelineDeclaration declaration, IReadOnlyDictionary<int, FittedStepValues> fitted, string answer)
    {
        var followed = answer;
        var links = new List<UndoLink>();

        for (var at = declaration.Steps.Count - 1; at >= 0; at--)
        {
            if (declaration.Steps[at] is IUndoesItself step && step.Undoes(followed))
            {
                links.Add(new UndoLink(step, fitted.GetValueOrDefault(at)));
                followed = step.From(followed);
            }
        }

        return new UndoChain(followed, links);
    }

    /// <summary>Every column the ways back of a declaration's answers need as it was read.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <returns>The column each answer comes back to, and every column a step on the way reads, each once.</returns>
    public static IEnumerable<string> ReadBy(PipelineDeclaration declaration) =>
        (declaration.Output?.Answers ?? [])
            .Select(answer => For(declaration, NothingLearned, answer))
            .SelectMany(chain => chain.Links.SelectMany(link => link.Step.ColumnsRead.Select(read => read.Column)).Prepend(chain.End))
            .Distinct(StringComparer.Ordinal);

    /// <summary>One value back, by the steps alone.</summary>
    /// <param name="value">The value as the last step left it.</param>
    /// <returns>The value in the units of the column the way back comes back to.</returns>
    public double Back(double value) => Links.Aggregate(value, (each, link) => link.Step.Undo(each, link.Fitted));

    /// <summary>One value back, with the row it belongs to as it was read.</summary>
    /// <param name="value">The value as the last step left it.</param>
    /// <param name="row">The row it belongs to.</param>
    /// <returns>The value in the units of the column the way back comes back to.</returns>
    public double Back(double value, RowAsRead row) => Links.Aggregate(value, (each, link) => link.Step.Undo(each, link.Fitted, row));
}
