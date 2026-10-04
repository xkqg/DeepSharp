// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// Training the trainer a pipeline declares.
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
            var context = new MLContext(step.Seed);
            var train = prepared.Batch(Part.Train, step.Needs);

            if (train.AnswerNames is not { Count: 1 })
            {
                throw new InvalidOperationException(
                    "A trainer from ML.NET learns one answer, and this pipeline names "
                    + $"{train.AnswerNames?.Count ?? 0}. Name one with Target or Ahead, or train what names several through a learner that takes several.");
            }

            // Whether the answer is one of two classes or a number is decided from the training rows alone, and written
            // down with the model: a replay puts the same rows through the same trainer rather than deciding again.
            var classes = train.Labels!.All(label => label is 0 or 1);
            var model = Fitted(context, step, HandedRows.Of(context, train, classes), classes);

            return new TrainedMLModel(
                model,
                context,
                prepared,
                new TrainedOnRows(train.Features.Count, train.FeatureNames, train.AnswerNames[0], classes));
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

    // One trainer, pinned to the seed the declaration carries: every setting the word takes is written there, so nothing
    // about the model is decided by whatever the library defaults to today.
    private static ITransformer Fitted(MLContext context, LearnMLStep step, IDataView rows, bool classes)
    {
        var settings = step.Trainer.Settings.ToDictionary(setting => setting.Key, setting => setting.Value, StringComparer.Ordinal);

        int Whole(string key) => (int)settings[key].Number;

        return step.Trainer.Kind switch
        {
            "fastTree" when classes => context.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
            {
                FeatureColumnName = HandedRows.FeaturesColumn,
                LabelColumnName = HandedRows.LabelColumn,
                NumberOfLeaves = Whole("leaves"),
                NumberOfTrees = Whole("trees"),
                MinimumExampleCountPerLeaf = Whole("leastRows"),
                LearningRate = settings["rate"].Number,
                Seed = step.Seed,
            }).Fit(rows),

            "fastForest" when classes => context.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options
            {
                FeatureColumnName = HandedRows.FeaturesColumn,
                LabelColumnName = HandedRows.LabelColumn,
                NumberOfLeaves = Whole("leaves"),
                NumberOfTrees = Whole("trees"),
                MinimumExampleCountPerLeaf = Whole("leastRows"),
                Seed = step.Seed,
            }).Fit(rows),

            "fastTree" => context.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
            {
                FeatureColumnName = HandedRows.FeaturesColumn,
                LabelColumnName = HandedRows.LabelColumn,
                NumberOfLeaves = Whole("leaves"),
                NumberOfTrees = Whole("trees"),
                MinimumExampleCountPerLeaf = Whole("leastRows"),
                LearningRate = settings["rate"].Number,
                Seed = step.Seed,
            }).Fit(rows),

            // Every word the vocabulary offers is one of the four arms above, and a test holds the two lists to each
            // other, so there is no arm here for a word that cannot be written.
            _ => context.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options
            {
                FeatureColumnName = HandedRows.FeaturesColumn,
                LabelColumnName = HandedRows.LabelColumn,
                NumberOfLeaves = Whole("leaves"),
                NumberOfTrees = Whole("trees"),
                MinimumExampleCountPerLeaf = Whole("leastRows"),
                Seed = step.Seed,
            }).Fit(rows),
        };
    }
}
