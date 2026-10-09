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
    extension(CompiledNetwork compiled)
    {
        /// <summary>
        /// Trains the network on the rows the pipeline hands over from its training part, judged by the rows of its validation
        /// part, then measures it as the pipeline's report declares.
        /// </summary>
        /// <param name="prepared">
        /// The pipeline, run: its training rows are what the network learns from and its validation rows what it is judged by,
        /// every feature on one scale. Its test rows are never read here, but by the report, once the network has learned.
        /// </param>
        /// <param name="options">
        /// The seed, the epochs, the batches, early stopping, checkpoints — and the engine, which trains the network, judges it
        /// and takes the report's measures.
        /// </param>
        /// <returns>The network behind its pipeline: what it predicts, its file, what the run did and how the report measured it.</returns>
        /// <exception cref="InvalidOperationException">
        /// The pipeline names no answer; it holds no rows, as one read from its file holds none, so it is run over its rows again
        /// to train a network behind it; the run was made for another learner and left out a step a network needs — a scale,
        /// or an encoder written one column per category — every such step named; or a feature is not declared to land between
        /// minus one and one, as a network takes every feature.
        /// </exception>
        /// <exception cref="ArgumentException">
        /// A training or validation row's answers are not ones the loss could have meant, the row named by where it was read; or
        /// anything else <see cref="CompiledNetwork.Fit"/> refuses.
        /// </exception>
        /// <remarks>
        /// The rows come over as doubles and are handed to the network as floats, once each part. A validation part of no rows
        /// leaves the run unjudged, which early stopping refuses. The report is part of the run, so it is measured on the run's
        /// engine — <see cref="FitOptions.Backend"/>, the light one unless the options name another — a chunk of a part's rows at
        /// a time, as many as <see cref="FitOptions.BatchSize"/> holds, by the evaluation pass
        /// <see cref="TrainedNetwork.Predict(IRowSource, ITensorBackend)"/> serves with; and each part it names counts its rows
        /// that move a feature away from the one value every training row held it at, which the network learned nothing about.
        /// </remarks>
        public TrainedNetwork Fit(PreparedData prepared, FitOptions options)
        {
            ArgumentNullException.ThrowIfNull(compiled);
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(options);

            if (prepared.Declaration.Output is null)
            {
                throw new InvalidOperationException("This pipeline names no answer, so there is nothing to train a network on: declare its output.");
            }

            if (compiled.Loss is EarthMoversDistance distance)
            {
                ThrowIfNotAlongItsOrder(distance, prepared.Declaration.Output);
            }

            var train = prepared.Batch(Part.Train, TrainedNetwork.FeatureNeeds);
            var validation = prepared.Batch(Part.Validation, TrainedNetwork.FeatureNeeds);
            var history = compiled.Fit(Handed(train, prepared), validation.RowCount > 0 ? Handed(validation, prepared) : null, options);
            var epoch = options.EarlyStopping is { RestoreBest: true } && history.BestEpoch is { } best ? best : history.Epochs[^1].Number;
            var trainedOn = TrainedNetwork.TrainedOnOf(prepared, train, options.Seed, epoch);
            var chunking = new Chunking(options.Backend, options.BatchSize);
            var measures = prepared.Declaration.Report is { } report
                ? prepared.Measure([.. report.Parts.Distinct().Select(part => Predicted(compiled, prepared.Batch(part, TrainedNetwork.FeatureNeeds), chunking, trainedOn))])
                : null;

            return new TrainedNetwork(compiled.Network, compiled.Loss, prepared, trainedOn)
            {
                History = history,
                Measures = measures,
            };
        }
    }

    // A distance along an order means something only of answers in one, and takes what is left of a whole for what it is only
    // when the output makes it: either is refused before anything is trained.
    private static void ThrowIfNotAlongItsOrder(EarthMoversDistance distance, INamesTheAnswer output)
    {
        if (!output.IsOrdered)
        {
            throw new InvalidOperationException(
                $"The loss '{EarthMoversDistance.Name}' measures how far the shares lie from the answer's along the order of the bands, and the answers '{output.Verb}' names "
                + "are no shares in an order: declare them with 'target.distribution' and 'ordered': true, or judge them by another loss.");
        }

        var left = output is DistributionStep { Remainder: { } remainder } ? remainder : null;

        if (left is not null && !distance.Remainder)
        {
            throw new InvalidOperationException(
                $"The output makes '{left}', what is left of the whole, which stands outside the order of the bands, and this loss takes it for one more band: "
                + "compile with new EarthMoversDistance(remainder: true).");
        }

        if (left is null && distance.Remainder)
        {
            throw new InvalidOperationException(
                $"This loss takes the last answer for what is left of the whole, and '{output.Verb}' makes no such answer: compile with new EarthMoversDistance().");
        }
    }

    // What the network answers for a part's rows, as the report measures them: on the run's engine, a chunk of rows at a
    // time, each row named with the features it moves away from the one value every training row held them at.
    private static PartPredictions Predicted(CompiledNetwork compiled, Batch batch, Chunking chunking, TrainedOn trainedOn) =>
        new(batch, TrainedNetwork.Answered(compiled.Network, compiled.Loss, batch.Features, chunking))
        {
            Unfamiliar = TrainedNetwork.Unfamiliar(trainedOn, batch.Features),
        };

    // The rows of a part as the loop takes them: floats, and each row named by where it was read.
    private static TrainingData Handed(Batch batch, PreparedData prepared) =>
        new(batch.Features.Floats(batch.Width), batch.Answers!.Floats(batch.AnswerNames!.Count))
        {
            RowNames =
            [
                .. Enumerable.Range(0, prepared.Table.RowCount)
                    .Where(row => prepared.Parts[row] == batch.Part)
                    .Select(row => string.Create(CultureInfo.InvariantCulture, $"Row {prepared.Table.Identities[row].ReadAt + 1}")),
            ],
        };
}
