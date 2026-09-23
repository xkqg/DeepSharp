// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Names the column a model is being asked to predict.
/// </summary>
/// <remarks>
/// The answer is not a feature, so it is taken out of what a model is shown and handed over separately. A
/// pipeline that left it among the inputs would produce a model that scores perfectly and knows nothing —
/// and the same mistake wears a quieter costume when a column merely restates the answer, which is what
/// declaring the columns is for.
/// </remarks>
public sealed record TargetStep : IPipelineStep<TargetStep>
{
    /// <summary>Declares which column holds the answer.</summary>
    /// <param name="column">The column being predicted.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public TargetStep(string column)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        Column = column;
    }

    /// <summary>The column being predicted.</summary>
    public string Column { get; }

    /// <inheritdoc />
    public static string Name => "target";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static TargetStep ReadFrom(JsonElement element) => new(element.RequiredString("column"));
}

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
    /// <summary>The numbers of one split, ready for something that learns.</summary>
    /// <param name="prepared">The data as the pipeline left it.</param>
    /// <param name="split">Which part of it to hand over.</param>
    /// <returns>The rows of that split, and their answers when the pipeline named a target.</returns>
    /// <exception cref="InvalidOperationException">
    /// A column still holds words, or the target column is not there.
    /// </exception>
    public static Batch Batch(this PreparedData prepared, Split split)
    {
        ArgumentNullException.ThrowIfNull(prepared);

        var target = prepared.Declaration.Steps.OfType<TargetStep>().LastOrDefault()?.Column;

        if (target is not null && !prepared.Table.Has(target))
        {
            throw new InvalidOperationException(
                $"This pipeline predicts '{target}', and no column of that name reached the end of it.");
        }

        var features = prepared.Table.Columns
            .Where(column => column.Name != target)
            .ToArray();

        var words = features.FirstOrDefault(column => column is TextColumn);

        if (words is not null)
        {
            // Words are not numbers, and turning them into one quietly is how a category becomes an order
            // nobody meant. Encode it, or leave it out of the schema.
            throw new InvalidOperationException(
                $"'{words.Name}' still holds words. Encode it, or do not declare it.");
        }

        var rows = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => prepared.Splits[row] == split)
            .ToArray();

        var values = features.Select(column => Numbers.Of(prepared.Table, column.Name)).ToArray();
        var answers = target is null ? null : Numbers.Of(prepared.Table, target);
        var batch = new List<double[]>(rows.Length);
        var labels = answers is null ? null : new List<double>(rows.Length);

        foreach (var row in rows)
        {
            var line = new double[features.Length];

            for (var at = 0; at < features.Length; at++)
            {
                line[at] = values[at][row] ?? throw Unfilled(features[at].Name, row);
            }

            batch.Add(line);
            labels?.Add(answers![row] ?? throw Unfilled(target!, row));
        }

        return new Batch([.. features.Select(column => column.Name)], batch, labels);
    }

    private static InvalidOperationException Unfilled(string column, int row) =>
        new($"Row {row + 1} of '{column}' is still a gap. Fill it, drop it, or leave the column out.");
}
