// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// What this package refuses, and the forms of trainer and answer it takes.
/// </summary>
public class RefusalTests
{
    [Fact]
    public void AForestLearnsToo_AndAnAnswerThatIsANumberIsLearnedAsOne()
    {
        // The other trainer word, and the other kind of answer: a number rather than one of two classes, which is a
        // different trainer inside ML.NET and the same declaration here.
        var forest = Prices("fastForest").TrainWithML();

        Assert.False(forest.TrainedOn.Classes);
        Assert.NotEmpty(forest.Predict(Part.Test));
        Assert.NotEmpty(Prices("fastTree").TrainWithML().Predict(Part.Test));
    }

    [Fact]
    public void APipelineWithNoReport_HasNoMeasures()
    {
        Assert.Null(Prices("fastTree").TrainWithML().Measures);
    }

    [Fact]
    public void NothingIsTrainedFromNothing()
    {
        Assert.Throws<ArgumentNullException>(() => ((Pipeline)null!).TrainWithML());
        Assert.Throws<ArgumentNullException>(() => ((PreparedData)null!).TrainWithML());
    }

    [Fact]
    public void RowsHandedOverWithoutAnAnswer_CannotBeTrainedFrom()
    {
        // A pipeline names what a model is asked to predict; rows handed over without one are the question, not the
        // lesson, and the refusal says so rather than training on a column of noughts.
        var refused = Assert.Throws<InvalidOperationException>(
            () => HandedRows.Of(new Microsoft.ML.MLContext(1), new Batch(["a"], [[1.0]], Labels: null), classes: false));

        Assert.Contains("Target", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryTrainerTheWordsOffer_IsOneThisPackageTrains()
    {
        // The two lists are the same list seen twice: the words a declaration may name, and the trainers behind them.
        // Holding them to each other here is what lets the fit switch have no arm for a word nobody can write.
        Assert.Equal(["fastForest", "fastTree"], MLWords.Trainers.Select(kind => kind.Name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ARowHandedOverToPredictFrom_CarriesNoAnswer_AndIsTakenAllTheSame()
    {
        var trained = Prices("fastTree").TrainWithML();

        Assert.Equal(trained.Predict(Part.Validation).Count, trained.Prepared.CountIn(Part.Validation));
        Assert.NotEqual(trained.Predict(Part.Validation), trained.Predict(Part.Test));
    }

    [Fact]
    public void AModelFileOfANumberAnswer_SaysItIsNotClasses_AndCarriesItsFeatures()
    {
        var read = MLModelFile.FromJson(Prices("fastForest").TrainWithML().ToJson());

        Assert.False(read.Classes);
        Assert.NotEmpty(read.Features);
        Assert.Equal("AAPL.Close.ahead5", read.Answer);
    }

    // A price five days on, which is a number: the other half of what a trainer here is asked for.
    private static Pipeline Prices(string trainer) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close", "AAPL.Volume"))
            .OrderBy("Date")
            .SplitByTime("Date", train: 0.70, validation: 0.15, gap: 5)
            .Ahead("AAPL.Close", 5, AheadAs.Return)
            .Normalise("AAPL.Close", "AAPL.Volume")
            .Drop("Date")
            .WithML(line => { if (trainer == "fastTree") { line.FastTree(trees: 10); } else { line.FastForest(trees: 10); } })
            .Build();
}
