// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The deals of a study can be the folds of one pipeline: each deal is the pipeline with another slice of the rows as validation,
/// every candidate is judged by the mean of its validation loss over the folds, and the rows to be measured on are the same on every
/// one, which is what a study asks of its deals. What the folds measured is laid beside one another by the report, for any learner.
/// </summary>
public class StudyOverFoldsTests
{
    private const int Folds = 3;

    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Pipeline Fold(int deal) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.80, new Fold(Folds, deal), 100, 7)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build();

    private static StudyResult Run() =>
        new Study(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = 2,
            Deals = Folds,
        }.Run(Fold, TestContext.Current.CancellationToken);

    [Fact]
    public void TheDealsOfAStudy_AreTheFoldsOfOnePipeline_EachCandidateScoredOnEveryFoldsValidationRows()
    {
        var result = Run();

        Assert.All(result.Trials, trial => Assert.Equal(Folds, trial.Scores.Count));
        Assert.Equal(Folds, result.Winner.Count);
        Assert.Equal(Folds, result.Best.Scores.Distinct().Count());
    }

    [Fact]
    public void TheNetworksTheStudyKeeps_AreMeasuredOnTheSameTestRows_AndLaidBesideOneAnotherByTheReport()
    {
        var result = Run();
        var across = MeasuresAcross.Of([.. result.Winner.Select(network => network.Measures!)]);
        var test = across.Parts.Single(part => part.Part == Part.Test);

        Assert.Equal(Folds, across.Folds);
        Assert.Single(test.Rows.Distinct());
        Assert.Equal(Folds, test.Values.Single(value => value.Metric == Metric.Accuracy).Values.Count);
        Assert.InRange(test.Values.Single(value => value.Metric == Metric.Accuracy).Mean, 0.5, 1.0);
    }
}
