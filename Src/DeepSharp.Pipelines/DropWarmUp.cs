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
public interface IDropsRows : IPipelineStep
{
    /// <summary>The first row worth keeping.</summary>
    /// <param name="table">The data as it stands.</param>
    /// <returns>The index of the first row to keep; nought keeps everything.</returns>
    int FirstUsableRow(Table table);
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
public sealed record DropWarmUpStep : IPipelineStep<DropWarmUpStep>, IDropsRows
{
    /// <summary>Drops the rows at the start that any column is still silent about.</summary>
    /// <param name="atMost">The most rows this is allowed to drop; beyond it the run stops.</param>
    /// <exception cref="ArgumentOutOfRangeException">The limit is below nothing.</exception>
    public DropWarmUpStep(int atMost = 1000)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(atMost);

        AtMost = atMost;
    }

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
    public string Verb => Name;

    /// <inheritdoc />
    public int FirstUsableRow(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var warmUp = table.Columns.Count == 0 ? 0 : table.Columns.Max(column => column.LeadingGaps());

        if (warmUp > AtMost)
        {
            var slowest = table.Columns.OrderByDescending(column => column.LeadingGaps()).First();

            throw new InvalidOperationException(
                $"'{slowest.Name}' says nothing about its first {warmUp} rows, and this pipeline drops at "
                + $"most {AtMost}. Either that column starts later than the data does, or the warm-up is "
                + "longer than the dataset can afford.");
        }

        return warmUp;
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteNumber("atMost", AtMost);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DropWarmUpStep ReadFrom(JsonElement element) => new((int)element.RequiredNumber("atMost"));
}
