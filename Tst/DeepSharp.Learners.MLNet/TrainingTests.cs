// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A trainer from ML.NET, learning from the rows a DeepSharp pipeline hands over.
/// </summary>
/// <remarks>
/// The declaration names the trainer and the seed it repeats from; the run is made for what that trainer needs, which
/// leaves the scalings out and writes down that it did; and what comes back is measured by the pipeline's own report, on
/// the same rows a network would have been measured on.
/// </remarks>
public class TrainingTests
{
    [Fact]
    public void ADeclarationNamingATree_TrainsIt_OnTheRowsThePipelineHandsOver()
    {
        var trained = Passengers().TrainWithML();

        Assert.Equal("fastTree", trained.Trainer.Kind);
        Assert.Equal(623, trained.TrainedOn.Rows);
        Assert.NotEmpty(trained.Predict(Part.Test));
    }

    [Fact]
    public void TheSameDeclarationTrainedTwice_GivesTheSameModel()
    {
        // What the seed in the declaration is for: a replay of the same text gives the same answers, which is the whole
        // promise a declared pipeline makes. The model's own bytes are not compared — ML.NET stamps its archive with the
        // clock it was saved at — so the answers are.
        var once = Passengers().TrainWithML();
        var again = Passengers().TrainWithML();

        Assert.Equal(once.Predict(Part.Test), again.Predict(Part.Test));
    }

    [Fact]
    public void WhereTheTrainerDrawsAtRandom_AnotherSeedGivesAnotherModel_AndWhereItDoesNotTheSeedChangesNothing()
    {
        // Measured, and worth saying plainly: boosted trees at these settings draw nothing at random, so their model is
        // the same whatever the seed says. A forest bags its rows, so its seed is the whole difference. Both are
        // replayable from the declaration, which is what the seed is there to promise.
        Assert.Equal(Passengers().TrainWithML().Predict(Part.Test), Passengers(seed: 7).TrainWithML().Predict(Part.Test));
        Assert.NotEqual(Forest().TrainWithML().Predict(Part.Test), Forest(seed: 7).TrainWithML().Predict(Part.Test));
    }

    [Fact]
    public void TheRunIsMadeForWhatATreeNeeds_SoTheScalingsAreLeftOutAndSaidSo()
    {
        var trained = Passengers().TrainWithML();

        Assert.NotEmpty(trained.Prepared.Skipped);
        Assert.All(
            trained.Prepared.Skipped,
            at => Assert.Equal("normalise", trained.Prepared.Declaration.Steps[at].Verb));
    }

    [Fact]
    public void APipelineThatNamesAnotherLearner_IsRefusedHereByName()
    {
        var network = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing(fill => fill.Median("age"))
            .Target("survived")
            .WithTensorflow(net => net.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy())
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(network.TrainWithML);

        Assert.Contains("learn.network", refused.Message, StringComparison.Ordinal);
        Assert.Contains("learn.ml", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineThatNamesNoLearnerAtAll_SaysSo()
    {
        var bare = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build();

        Assert.Throws<InvalidOperationException>(bare.TrainWithML);
    }

    [Fact]
    public void ATreeAndANetwork_AreMeasuredByOneReport_OnTheSameRows()
    {
        // The whole reason this package exists: two learners, one declaration shape, one set of measures over rows that
        // are the same rows — so what the numbers say is about the learners and not about two ways of measuring.
        var tree = Reported(forTree: true).TrainWithML();
        var measures = tree.Measures;

        Assert.NotNull(measures);
        Assert.Equal(
            Reported(forTree: false).Run().Batch(Part.Test).Keys,
            tree.Prepared.Batch(Part.Test, global::DeepSharp.Learners.ML.LearnMLStep.FeatureNeeds).Keys);
    }

    // The same passengers with a report declared, for a tree or for a network: the two differ in one line.
    private static Pipeline Reported(bool forTree)
    {
        var chain = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare")
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Precision).On(Part.Validation, Part.Test).As(Shown.Numbers));

        return (forTree
                ? chain.WithML(trainer => trainer.FastTree(trees: 30))
                : chain.WithTensorflow(net => net.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy()))
            .Build();
    }

    // The Titanic passengers, prepared for a tree: the scalings are declared and left out of the run this learner is
    // made for, which is what lets a tree and a network be measured on the same rows.
    private static Pipeline Passengers(int seed = 20260929) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema
                .Integer("survived", "sibsp", "parch")
                .Category("pclass", "sex")
                .Optional("age", ColumnKind.Number)
                .Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare", "sibsp", "parch")
            .Target("survived")
            .WithML(trainer => trainer.FastTree(trees: 50), seed)
            .Build();

    // The same passengers, learned by a forest, whose trees are grown from rows it bags at random.
    private static Pipeline Forest(int seed = 20260929) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare")
            .Target("survived")
            .WithML(trainer => trainer.FastForest(trees: 30), seed)
            .Build();
}
