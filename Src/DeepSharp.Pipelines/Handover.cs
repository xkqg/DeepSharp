// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// The numbers of one split, in the shape anything that learns can take them.
/// </summary>
/// <param name="FeatureNames">The columns, in the order every row lists them.</param>
/// <param name="Features">One row of numbers per row of data.</param>
/// <param name="Labels">The answer for each row, when the pipeline named one.</param>
/// <remarks>
/// The column order is part of the handover, not an accident of iteration: a model fed the same numbers in
/// a different order is quietly a different model, and nothing about the numbers themselves would say so.
/// </remarks>
public readonly record struct Batch(
    IReadOnlyList<string> FeatureNames,
    IReadOnlyList<double[]> Features,
    IReadOnlyList<double>? Labels)
{
    /// <summary>How many rows this batch holds.</summary>
    public int RowCount => Features.Count;

    /// <summary>How many numbers each row holds.</summary>
    public int Width => FeatureNames.Count;
}

/// <summary>
/// Rows served through a trained pipeline, in the shape the training rows were handed over in.
/// </summary>
/// <param name="FeatureNames">The columns, in the order every row lists them — the training order.</param>
/// <param name="Features">One row of numbers per served row.</param>
/// <param name="HandedInAt">For each served row, which of the handed-in rows it is, counting from nought.</param>
/// <remarks>
/// No answers: a served row is asked for one. Which handed-in row each is, because a replay can drop rows
/// and put them in order, and a prediction has to find its way back to the row it was made for.
/// </remarks>
public readonly record struct ServedBatch(
    IReadOnlyList<string> FeatureNames,
    IReadOnlyList<double[]> Features,
    IReadOnlyList<int> HandedInAt)
{
    /// <summary>How many rows were served.</summary>
    public int RowCount => Features.Count;
}

/// <summary>
/// Handing prepared data over to whatever learns from it.
/// </summary>
/// <remarks>
/// This is where the pipeline stops. Everything up to here is the same whatever is going to learn from the
/// result, so what comes out is the same too: rows of numbers, their column names, and the answer when one
/// was named. A network written here, a trainer from an established .NET library and something a caller
/// wrote all take the same handover — which is the only reason two of them can honestly be compared.
/// </remarks>
public static class Handover
{
    /// <summary>The numbers of one part, ready for something that learns.</summary>
    /// <param name="prepared">The data as the pipeline left it.</param>
    /// <param name="part">Which part of it to hand over.</param>
    /// <returns>The rows of that split, and their answers when the pipeline named a target.</returns>
    /// <exception cref="InvalidOperationException">
    /// A column still holds words, or the target column is not there.
    /// </exception>
    public static Batch Batch(this PreparedData prepared, Part part)
    {
        ArgumentNullException.ThrowIfNull(prepared);

        var rows = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => prepared.Parts[row] == part)
            .ToArray();

        return HandedOver(prepared, prepared.Table, rows, withAnswers: true);
    }

    /// <summary>Rows that arrived after training, replayed and handed over in the shape the training rows were.</summary>
    /// <param name="prepared">The trained pipeline.</param>
    /// <param name="rows">The rows to serve, without the answer — it is what is being asked.</param>
    /// <returns>Their numbers in the training order, and which handed-in row each is.</returns>
    /// <exception cref="InvalidOperationException">
    /// A served row is refused for what a training row would be: a column still holding words, a gap, a value
    /// that is not a finite number.
    /// </exception>
    /// <remarks>
    /// Nothing is fitted: the rows are replayed with the numbers the training rows produced, which is the
    /// only way a model sees tomorrow's rows the way it saw the ones it learned from.
    /// </remarks>
    public static ServedBatch Served(this PreparedData prepared, IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(prepared);
        ArgumentNullException.ThrowIfNull(rows);

        var table = prepared.Replay(rows);
        var all = Enumerable.Range(0, table.RowCount).ToArray();
        var batch = HandedOver(prepared, table, all, withAnswers: false);

        return new ServedBatch(batch.FeatureNames, batch.Features, [.. all.Select(row => table.Identities[row].ReadAt)]);
    }

    private static Batch HandedOver(PreparedData prepared, Table table, int[] rows, bool withAnswers)
    {
        var answers = prepared.Declaration.Output?.Answers ?? [];

        if (answers.FirstOrDefault(answer => !table.Has(answer)) is { } vanished)
        {
            throw new InvalidOperationException(
                $"This pipeline predicts '{vanished}', and no column of that name reached the end of it.");
        }

        // Every answer the output names is left out of what a model is shown, not only the first: the others would
        // reach it as features.
        var features = table.Columns
            .Where(column => !answers.Contains(column.Name, StringComparer.Ordinal))
            .ToArray();

        var words = features.FirstOrDefault(column => column is TextColumn);

        if (words is not null)
        {
            // Words are not numbers, and turning them into one quietly is how a category becomes an order
            // nobody meant. Encode it, or leave it out of the schema.
            throw new InvalidOperationException(
                $"'{words.Name}' still holds words. Encode it, or do not declare it.");
        }

        var values = features.Select(column => table.NumbersOf(column.Name)).ToArray();
        // The labels are one number per row, so they carry an output of one answer.
        var target = withAnswers && answers is [var only] ? only : null;
        var known = target is null ? null : table.NumbersOf(target);
        var batch = new List<double[]>(rows.Length);
        var labels = known is null ? null : new List<double>(rows.Length);

        foreach (var row in rows)
        {
            // A refusal names the row as it was read, which is the row a person can find in their file.
            var readAt = table.Identities[row].ReadAt;
            var line = new double[features.Length];

            for (var at = 0; at < features.Length; at++)
            {
                line[at] = Handed(values[at][row], features[at].Name, readAt);
            }

            batch.Add(line);
            labels?.Add(Handed(known![row], target!, readAt));
        }

        return new Batch([.. features.Select(column => column.Name)], batch, labels);
    }

    /// <summary>A value as it may be handed to something that learns, or the reason it may not.</summary>
    private static double Handed(double? value, string column, int readAt) => value switch
    {
        null => throw new InvalidOperationException(
            $"Row {readAt + 1} of '{column}' is still a gap. Fill it, drop it, or leave the column out."),

        // A model handed an infinity or a not-a-number learns nothing from that row and says nothing about
        // it. Either is a fault upstream, and the verb for it is fill.nan.
        { } number when !double.IsFinite(number) => throw new InvalidOperationException(string.Create(
            CultureInfo.InvariantCulture,
            $"Row {readAt + 1} of '{column}' is not a finite number ({number}). Declare fill.nan for it, or leave the column out.")),

        { } number => number,
    };
}
