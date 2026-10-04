// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// One row as ML.NET takes it: the features as one vector, and the answer beside them.
/// </summary>
/// <remarks>
/// The features are <see langword="float" /> because every ML.NET trainer refuses a vector of doubles by name, and
/// because the network path narrows in exactly the same place — so both learners see the same numbers and a comparison
/// between them is a comparison of the learners rather than of two precisions.
/// </remarks>
internal sealed class HandedRow
{
    /// <summary>The features of one row, in the order the handover names them.</summary>
    public float[] Features { get; init; } = [];

    /// <summary>The answer the row is labelled with, where the answer is a number.</summary>
    public float Label { get; init; }

    /// <summary>The answer the row is labelled with, where the answer is one of two classes.</summary>
    /// <remarks>ML.NET's binary trainers take a boolean and refuse a number, so a nought-or-one answer is handed over as one.</remarks>
    public bool Class { get; init; }
}

/// <summary>
/// The rows a pipeline hands over, as a view ML.NET can read.
/// </summary>
/// <remarks>
/// The order is the handover's order and nothing may change it: a view that let a trainer shuffle would put the row
/// order outside the declaration, and two runs of one declaration could then differ. The column names ride along as the
/// vector's slot names, so a model read back a year later still says which feature each number was.
/// </remarks>
internal static class HandedRows
{
    /// <summary>The name the features are handed over under.</summary>
    public const string FeaturesColumn = "Features";

    /// <summary>The name the answer is handed over under.</summary>
    public const string LabelColumn = "Label";

    /// <summary>A view of one part's rows, as ML.NET reads them.</summary>
    /// <param name="context">The context the view belongs to.</param>
    /// <param name="batch">The rows the pipeline hands over.</param>
    /// <param name="classes">Whether the answer is one of two classes rather than a number.</param>
    /// <param name="answers">Whether the rows carry the answer: they do to train from, and they do not to predict for.</param>
    /// <returns>The view.</returns>
    /// <exception cref="InvalidOperationException">The handover carries no answers, which a trainer cannot learn from.</exception>
    public static IDataView Of(MLContext context, Batch batch, bool classes, bool answers = true)
    {
        if (answers && batch.Labels is null)
        {
            throw new InvalidOperationException(
                "These rows are handed over without an answer, so nothing can be trained on them. A pipeline names what a model is "
                + "asked to predict with Target or Ahead, and a trainer takes exactly one answer.");
        }

        var names = batch.FeatureNames;
        var definition = SchemaDefinition.Create(typeof(HandedRow));

        definition[nameof(HandedRow.Features)].ColumnName = FeaturesColumn;
        definition[nameof(HandedRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, names.Count);
        definition[classes ? nameof(HandedRow.Class) : nameof(HandedRow.Label)].ColumnName = LabelColumn;

        var labels = batch.Labels ?? [.. Enumerable.Repeat(0.0, batch.Features.Count)];

        return context.Data.LoadFromEnumerable(
            batch.Features.Select((row, at) => new HandedRow
            {
                Features = [.. row.Select(value => (float)value)],
                Label = (float)labels[at],
                Class = labels[at] is not 0,
            }),
            definition);
    }
}
