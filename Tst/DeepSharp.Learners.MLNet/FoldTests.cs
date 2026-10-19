// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A tree is measured on folds as a network is: the folds, and what is laid beside one another, are the pipeline's, so the
/// second learner needs nothing of its own for them.
/// </summary>
public class FoldTests
{
    private const int Folds = 3;

    private static Pipeline Fold(int index) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", 0.80, new Fold(Folds, index), 7)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare")
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithML(trainer => trainer.FastTree(trees: 20))
            .Build();

    [Fact]
    public void ATreeTrainedOnEachFold_IsMeasuredOnTheSameTestRows_AndLaidBesideTheOthersByTheReport()
    {
        var across = MeasuresAcross.Of([.. Enumerable.Range(0, Folds).Select(index => Fold(index).TrainWithML().Measures!)]);
        var accuracy = across.Parts.Single(part => part.Part == Part.Test).Values.Single();

        Assert.Equal(Folds, across.Folds);
        Assert.Single(across.Parts.Single(part => part.Part == Part.Test).Rows.Distinct());
        Assert.InRange(accuracy.Mean, 0.6, 1.0);
        Assert.True(accuracy.Spread >= 0);
    }
}
