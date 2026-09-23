// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>Where a row stands at some place in a pipeline.</summary>
/// <remarks>
/// The part a split puts it in — the split declared below as much as one above — or that it is dropped before
/// the split reaches it, or that nothing divides it at all. Never "training" for a row no split divides: a
/// statistic over every row that calls itself one over the training rows is how the rows a model is measured
/// on come to shape what it is shown.
/// </remarks>
public enum Standing
{
    /// <summary>A row the split puts in training: the rows anything is measured and fitted on.</summary>
    Train,

    /// <summary>A row the split puts in validation.</summary>
    Validation,

    /// <summary>A row the split puts in test.</summary>
    Test,

    /// <summary>A row the split holds back to predict on.</summary>
    Predict,

    /// <summary>A row dropped between here and the split, which lands in no part.</summary>
    Dropped,

    /// <summary>A row nothing divides, because the pipeline has no split.</summary>
    Undivided,
}

/// <summary>
/// The data as it stands after some of a pipeline's steps, with where each row stands.
/// </summary>
/// <remarks>
/// What the grid under a block of a notebook shows, and what evidence at that place measures. The rows are
/// joined to the split wherever it is declared — above this place or below it — by where each row was read,
/// so a range drawn here, or a profile, is drawn over the rows the split trains on and no others.
/// </remarks>
public sealed class PipelineView
{
    internal PipelineView(Table table, IReadOnlyList<Standing> standings, Standing measured, IReadOnlyDictionary<int, Evidence> evidence)
    {
        Table = table;
        Standings = standings;
        Measured = measured;
        Evidence = evidence;
    }

    /// <summary>The data as it stands here.</summary>
    public Table Table { get; }

    /// <summary>Where each row stands, in row order.</summary>
    public IReadOnlyList<Standing> Standings { get; }

    /// <summary>
    /// The rows everything here is measured on: the training rows when the pipeline divides its rows, and the
    /// undivided ones when it does not.
    /// </summary>
    public Standing Measured { get; }

    /// <summary>What each step that produces evidence produced on the way here, by its place.</summary>
    /// <remarks>
    /// The walk to a view goes on to the split when the split is below it, so evidence written below this place and
    /// above the split is here too: each measured over the training rows the split will train on.
    /// </remarks>
    public IReadOnlyDictionary<int, Evidence> Evidence { get; }

    /// <summary>What the measured rows of a column hold, by the one rule every fit reads them by.</summary>
    /// <param name="column">The column's name.</param>
    /// <returns>The values, their gaps and the values that are not numbers, over the measured rows alone.</returns>
    /// <exception cref="InvalidOperationException">The column holds something that is not a number.</exception>
    public TrainingValues MeasuredValues(string column) =>
        Table.ValuesOf(column, row => Standings[row] == Measured);
}

/// <summary>
/// Joining the rows at one place to the parts the split gave them, by where each was read.
/// </summary>
internal static class Standings
{
    /// <summary>Where each row of a table stands, given the rows the split divided and the part each landed in.</summary>
    /// <param name="rows">The rows to place.</param>
    /// <param name="divided">The table the split divided, or nothing when no split was reached.</param>
    /// <param name="parts">The part each divided row landed in.</param>
    /// <returns>One standing per row.</returns>
    internal static Standing[] Of(Table rows, Table? divided, IReadOnlyList<Part>? parts)
    {
        if (divided is null || parts is null)
        {
            return [.. Enumerable.Repeat(Standing.Undivided, rows.RowCount)];
        }

        var landed = new Dictionary<int, Standing>(divided.RowCount);

        for (var row = 0; row < divided.RowCount; row++)
        {
            landed[divided.Identities[row].ReadAt] = parts[row] switch
            {
                Part.Train => Standing.Train,
                Part.Validation => Standing.Validation,
                Part.Test => Standing.Test,
                _ => Standing.Predict,
            };
        }

        return [.. rows.Identities.Select(identity => landed.GetValueOrDefault(identity.ReadAt, Standing.Dropped))];
    }
}
