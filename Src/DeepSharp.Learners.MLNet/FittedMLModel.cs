// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DeepSharp.Learners.MLNet;

/// <summary>What a <see cref="FittedMLModel"/> was fitted on: how many rows, how many features a row, and whether the answer is a class.</summary>
internal readonly record struct FittedOn(int Rows, int Width, bool Classes);

/// <summary>
/// A trainer from ML.NET, fitted on rows a caller handed over directly: no pipeline, and nothing of one kept with it.
/// </summary>
/// <remarks>
/// <para>
/// What <see cref="DeepSharp.Networks.CompiledNetwork"/> is to a network trained by hand, this is to a tree: a bare
/// primitive that fits and predicts, and promises nothing a bare primitive cannot keep. It is not saved as a file —
/// <see cref="CompiledNetwork"/> has none either — and it answers nothing about the rows that were never its own:
/// no measures, no interval, no replay of a pipeline's feature engineering on rows served later. A caller who wants
/// those keeps to the pipeline-backed door, <see cref="TrainWithMLExtensions"/>, which this type is built underneath.
/// </para>
/// <para>
/// Once fitted it answers only what it was fitted on: the same number of features a row, and the same choice between a
/// class and a number. Nothing about it changes after <see cref="Fit"/> returns.
/// </para>
/// </remarks>
public sealed class FittedMLModel
{
    private readonly ITransformer _model;
    private readonly MLContext _context;
    private readonly LearnMLStep _step;

    private FittedMLModel(ITransformer model, MLContext context, LearnMLStep step, FittedOn on)
    {
        _model = model;
        _context = context;
        _step = step;
        On = on;
    }

    private FittedOn On { get; }

    /// <summary>The trainer this was fitted with, and the settings it was given.</summary>
    public PartDeclaration Trainer => _step.Trainer;

    /// <summary>The number the trainer's random draws were worked out from.</summary>
    public int Seed => _step.Seed;

    /// <summary>How many rows this was fitted on.</summary>
    public int Rows => On.Rows;

    /// <summary>How many features a row holds.</summary>
    public int Width => On.Width;

    /// <summary>Whether the answer it learned is one of two classes rather than a number.</summary>
    public bool Classes => On.Classes;

    // ML.NET's own model and the context it was fitted under: not in this type's public surface, so a caller never
    // takes a dependency on ML.NET's own types through it. The pipeline-backed door keeps them to save its archive.
    internal ITransformer Model => _model;

    internal MLContext Context => _context;

    /// <summary>What this model answers for rows handed over the same way it was fitted on.</summary>
    /// <param name="features">The features, a row for each example, of the same width this was fitted on.</param>
    /// <returns>One answer a row, in the order the rows were handed in: a probability between nought and one for a class answer, the number itself otherwise.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="ArgumentException">A row holds another number of features than this was fitted on.</exception>
    public Tensor Predict(Tensor features)
    {
        ArgumentNullException.ThrowIfNull(features);

        var width = features.Shape.Count / features.Shape[0];

        if (width != Width)
        {
            throw new ArgumentException(
                $"This model was fitted on rows of {Width} features, and these rows hold {width}.", nameof(features));
        }

        var scored = _model.Transform(HandedRows.Of(_context, features, Classes));

        return Tensor.From(
            new Shape([features.Shape[0], 1]),
            [.. _context.Data.CreateEnumerable<ScoredRow>(scored, reuseRowObject: false, ignoreMissingColumns: true).Select(row => Classes ? row.Probability : row.Score)]);
    }

    /// <summary>Fits a trainer from ML.NET on rows handed over directly.</summary>
    /// <param name="trainer">The trainer, and the settings it takes.</param>
    /// <param name="seed">The number the trainer's own random draws are worked out from.</param>
    /// <param name="train">The rows: the features and the one answer every row names.</param>
    /// <returns>The fitted model.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="ArgumentException">
    /// There are no rows, the answer is not one column wide, or an answer is not a finite number.
    /// </exception>
    internal static FittedMLModel Fit(PartDeclaration trainer, int seed, TrainingData train)
    {
        ArgumentNullException.ThrowIfNull(train);

        if (train.Count == 0)
        {
            throw new ArgumentException("A trainer from ML.NET is fitted on at least one row.", nameof(train));
        }

        if (train.Answers.Shape[1] != 1)
        {
            throw new ArgumentException(
                $"A trainer from ML.NET learns one answer, and these rows name {train.Answers.Shape[1]}.", nameof(train));
        }

        if (train.Answers.Values.ToArray().Any(value => !float.IsFinite(value)))
        {
            throw new ArgumentException("A trainer from ML.NET learns from answers that are finite numbers.", nameof(train));
        }

        var step = new LearnMLStep(trainer) { Seed = seed };
        var classes = train.Answers.Values.ToArray().All(value => value is 0f or 1f);
        var width = train.Features.Shape.Count / train.Features.Shape[0];
        var context = new MLContext(seed);
        var model = Trained(context, step, HandedRows.Of(context, train.Features, classes, train.Answers), classes);

        return new FittedMLModel(model, context, step, new FittedOn(train.Count, width, classes));
    }

    // One trainer, pinned to the seed the declaration carries: every setting the word takes is written there, so nothing
    // about the model is decided by whatever the library defaults to today. A binary arm that ML.NET does not calibrate
    // on its own (FastForest) is given a Platt calibrator fitted on the same rows, so every class answer this hands back
    // is a probability between nought and one — the same promise the networks side's own activation keeps.
    private static ITransformer Trained(MLContext context, LearnMLStep step, IDataView rows, bool classes)
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
            }).Append(context.BinaryClassification.Calibrators.Platt()).Fit(rows),

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

/// <summary>One row as ML.NET hands a prediction back.</summary>
internal sealed class ScoredRow
{
    /// <summary>The number the model answers with, where the answer is a number.</summary>
    public float Score { get; init; }

    /// <summary>The probability the model answers with, where the answer is one of two classes.</summary>
    public float Probability { get; init; }
}
