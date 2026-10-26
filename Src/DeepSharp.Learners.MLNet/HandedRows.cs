// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
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
    /// <summary>The features of one row, as the rows name them.</summary>
    public float[] Features { get; init; } = [];

    /// <summary>The answer the row is labelled with, where the answer is a number.</summary>
    public float Label { get; init; }

    /// <summary>The answer the row is labelled with, where the answer is one of two classes.</summary>
    /// <remarks>ML.NET's binary trainers take a boolean and refuse a number, so a nought-or-one answer is handed over as one.</remarks>
    public bool Class { get; init; }
}

/// <summary>
/// The rows a <see cref="TrainingData"/> holds, as a view ML.NET can read.
/// </summary>
/// <remarks>
/// The one conversion both doors share: a pipeline's rows and a caller's own are both, by the time they reach here, a
/// <see cref="TrainingData"/> — a matrix of features and, for training, a matrix of answers one column wide. The order
/// is the handover's order and nothing may change it: a view that let a trainer shuffle would put the row order outside
/// the declaration, and two runs of one declaration could then differ.
/// </remarks>
internal static class HandedRows
{
    /// <summary>The name the features are handed over under.</summary>
    public const string FeaturesColumn = "Features";

    /// <summary>The name the answer is handed over under.</summary>
    public const string LabelColumn = "Label";

    /// <summary>A view of rows, as ML.NET reads them.</summary>
    /// <param name="context">The context the view belongs to.</param>
    /// <param name="features">The features, a row for each example.</param>
    /// <param name="classes">Whether the answer is one of two classes rather than a number.</param>
    /// <param name="answers">The answer for each row, one column wide; nothing to predict for, where every row is handed over as nought.</param>
    /// <returns>The view.</returns>
    public static IDataView Of(MLContext context, Tensor features, bool classes, Tensor? answers = null)
    {
        var width = features.Shape.Count / features.Shape[0];
        var rows = features.Shape[0];
        var values = features.Values.ToArray();
        var labels = answers?.Values.ToArray();

        var definition = SchemaDefinition.Create(typeof(HandedRow));

        definition[nameof(HandedRow.Features)].ColumnName = FeaturesColumn;
        definition[nameof(HandedRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, width);
        definition[classes ? nameof(HandedRow.Class) : nameof(HandedRow.Label)].ColumnName = LabelColumn;

        return context.Data.LoadFromEnumerable(
            Enumerable.Range(0, rows).Select(row => new HandedRow
            {
                Features = values[(row * width)..((row * width) + width)],
                Label = labels is null ? 0f : labels[row],
                Class = labels is not null && labels[row] is not 0f,
            }),
            definition);
    }
}
