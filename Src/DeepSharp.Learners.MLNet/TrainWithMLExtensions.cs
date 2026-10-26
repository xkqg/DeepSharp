// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// Training the trainer a pipeline declares, or a trainer fitted on rows a caller hands over directly.
/// </summary>
/// <remarks>
/// The door is not called <c>Train</c>: the networks package already offers that on a pipeline, and two packages
/// offering one name on one type either refuse to compile or quietly take each other's calls. <c>TrainWithML</c> says
/// which learner is meant at the place it is written.
/// </remarks>
public static class TrainWithMLExtensions
{
    extension(Pipeline pipeline)
    {
        /// <summary>Runs the pipeline for the trainer it declares, and trains that trainer on what it hands over.</summary>
        /// <returns>The trainer behind its pipeline: what it predicts, and the rows it learned from.</returns>
        /// <exception cref="ArgumentNullException">There is no pipeline.</exception>
        /// <exception cref="InvalidOperationException">The pipeline names no trainer from ML.NET, or anything the run itself refuses.</exception>
        public TrainedMLModel TrainWithML()
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            var step = Declared(pipeline.Declaration);

            return pipeline.RunFor(step.Needs).TrainWithML();
        }
    }

    extension(PreparedData prepared)
    {
        /// <summary>Trains the trainer a pipeline declares on the rows a run of it prepared.</summary>
        /// <returns>The trainer behind its pipeline.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared.</exception>
        /// <exception cref="InvalidOperationException">The run's declaration names no trainer from ML.NET.</exception>
        public TrainedMLModel TrainWithML()
        {
            ArgumentNullException.ThrowIfNull(prepared);

            var step = Declared(prepared.Declaration);
            var train = prepared.Batch(Part.Train, step.Needs);

            if (train.AnswerNames is not { Count: 1 })
            {
                throw new InvalidOperationException(
                    "A trainer from ML.NET learns one answer, and this pipeline names "
                    + $"{train.AnswerNames?.Count ?? 0}. Name one with Target or Ahead, or train what names several through a learner that takes several.");
            }

            var fitted = FittedMLModel.Fit(step.Trainer, step.Seed, Carried(train));

            return new TrainedMLModel(
                fitted,
                prepared,
                new TrainedOnRows(fitted.Rows, train.FeatureNames, train.AnswerNames[0], fitted.Classes));
        }
    }

    extension(TrainingData train)
    {
        /// <summary>Fits a trainer from ML.NET on rows handed over directly, with no pipeline behind them.</summary>
        /// <param name="trainer">The trainer, and the settings it takes.</param>
        /// <param name="seed">The number the trainer's own random draws are worked out from.</param>
        /// <returns>
        /// The fitted model: what it predicts for rows handed over the same way, and nothing of a pipeline it was never
        /// trained behind. It is not saved as a file — <see cref="CompiledNetwork"/> has none either — and the way to
        /// measure it, keep it behind a split, or serve it later is the pipeline-backed door, <c>Pipeline.TrainWithML()</c>.
        /// </returns>
        /// <exception cref="ArgumentNullException">There are no rows, or no sentence names the trainer.</exception>
        /// <exception cref="ArgumentException">
        /// There are no rows, the answer is not one column wide, or an answer is not a finite number.
        /// </exception>
        public FittedMLModel TrainWithML(Action<MLTrainerLine> trainer, int seed = 20260929)
        {
            ArgumentNullException.ThrowIfNull(train);

            return FittedMLModel.Fit(LearnMLStep.Declared(trainer, seed).Trainer, seed, train);
        }
    }

    // The trainer the declaration names, or the reason this door is the wrong one for it.
    private static LearnMLStep Declared(PipelineDeclaration declaration) =>
        declaration.Learner switch
        {
            LearnMLStep step => step,
            { } other => throw new InvalidOperationException(
                $"This pipeline is declared for '{other.Verb}', not for a trainer from ML.NET. Train what it names through that "
                + "learner's own door, or declare this one with .WithML(trainer => trainer.FastTree()), whose verb is 'learn.ml'."),
            _ => throw new InvalidOperationException(
                "This pipeline names no learner, so there is nothing here to train. Name one below the output — "
                + ".WithML(trainer => trainer.FastTree()) for a tree from ML.NET — and run it again."),
        };

    // The rows a pipeline's Batch hands over, as the TrainingData every fit goes through now: the one conversion both
    // doors share lives in FittedMLModel.Fit, and this is the pipeline side's own edge onto it.
    private static TrainingData Carried(Batch batch)
    {
        var width = batch.FeatureNames.Count;

        return new TrainingData(
            Tensor.From(new Shape([batch.Features.Count, width]), [.. batch.Features.SelectMany(row => row.Select(value => (float)value))]),
            Tensor.From(new Shape([batch.Features.Count, 1]), [.. batch.Labels!.Select(value => (float)value)]));
    }
}
