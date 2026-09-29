// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>A network trained on the rows a pipeline prepared.</summary>
public static class CompiledNetworkExtensions
{
    /// <summary>
    /// Trains the network on the rows the pipeline hands over from its training part, judged by the rows of its validation
    /// part, then measures it as the pipeline's report declares.
    /// </summary>
    /// <param name="compiled">The network, compiled.</param>
    /// <param name="prepared">
    /// The pipeline, run: its training rows are what the network learns from and its validation rows what it is judged by,
    /// every feature on one scale. Its test rows are never read here, but by the report, once the network has learned.
    /// </param>
    /// <param name="options">The seed, the epochs, the batches, early stopping, checkpoints.</param>
    /// <returns>The network behind its pipeline: what it predicts, its file, what the run did and how the report measured it.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline names no answer; or a feature is not declared to land between minus one and one, as a network takes
    /// every feature.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// A training or validation row's answers are not ones the loss could have meant, the row named by where it was read; or
    /// anything else <see cref="CompiledNetwork.Fit"/> refuses.
    /// </exception>
    /// <remarks>
    /// The rows come over as doubles and are handed to the network as floats, once each part. A validation part of no rows
    /// leaves the run unjudged, which early stopping refuses.
    /// </remarks>
    public static TrainedNetwork Fit(this CompiledNetwork compiled, PreparedData prepared, FitOptions options)
    {
        ArgumentNullException.ThrowIfNull(compiled);
        ArgumentNullException.ThrowIfNull(prepared);
        ArgumentNullException.ThrowIfNull(options);

        if (prepared.Declaration.Output is null)
        {
            throw new InvalidOperationException("This pipeline names no answer, so there is nothing to train a network on: declare its output.");
        }

        var train = prepared.Batch(Part.Train, Needs.OneScale);
        var validation = prepared.Batch(Part.Validation, Needs.OneScale);
        var history = compiled.Fit(Handed(train, prepared), validation.RowCount > 0 ? Handed(validation, prepared) : null, options);
        var epoch = options.EarlyStopping is { RestoreBest: true } && history.BestEpoch is { } best ? best : history.Epochs[^1].Number;
        var measures = prepared.Declaration.Report is { } report
            ? prepared.Measure([.. report.Parts.Distinct().Select(part => Predicted(compiled, prepared.Batch(part, Needs.OneScale)))])
            : null;

        return new TrainedNetwork(compiled.Network, compiled.Loss, prepared, TrainedNetwork.TrainedOnOf(prepared, train, options.Seed, epoch))
        {
            History = history,
            Measures = measures,
        };
    }

    // What the network answers for a part's rows, as the report measures them.
    private static PartPredictions Predicted(CompiledNetwork compiled, Batch batch) =>
        new(batch, TrainedNetwork.Answered(compiled.Network, compiled.Loss, batch.Features));

    // The rows of a part as the loop takes them: floats, and each row named by where it was read.
    private static TrainingData Handed(Batch batch, PreparedData prepared) =>
        new(Floats(batch.Features, batch.Width), Floats(batch.Answers!, batch.AnswerNames!.Count))
        {
            RowNames =
            [
                .. Enumerable.Range(0, prepared.Table.RowCount)
                    .Where(row => prepared.Parts[row] == batch.Part)
                    .Select(row => string.Create(CultureInfo.InvariantCulture, $"Row {prepared.Table.Identities[row].ReadAt + 1}")),
            ],
        };

    private static Tensor Floats(IReadOnlyList<double[]> rows, int width) =>
        Tensor.From(new Shape(rows.Count, width), [.. rows.SelectMany(row => row.Select(value => (float)value))]);
}
