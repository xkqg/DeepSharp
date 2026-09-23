// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// How the rows are divided. Two rules decide everything here. The share kept back to be measured on is
/// never written down — it is what is left, so nothing can add up to more than there is. And a slice can be
/// held back further still, to try a trained network on rows that took no part in anything at all.
/// </summary>
public class SplitSharesTests
{
    private static string Titanic => Path.Join(RepoRoot(), "Samples", "data", "titanic.csv");

    // ---- what the numbers mean -----------------------------------------------------------------------

    [Fact]
    public void PercentagesAndFractionsSayTheSameThing()
    {
        Assert.Equal(SplitShares.Of(0.80, 0.10), SplitShares.Of(80, 10));
    }

    [Fact]
    public void TheShareToBeMeasuredOnIsWhatIsLeft()
    {
        var shares = SplitShares.Of(70, 10);

        Assert.Equal(0.70, shares.Train, 9);
        Assert.Equal(0.10, shares.Validation, 9);
        Assert.Equal(0.20, shares.Test, 9);
        Assert.Equal(0, shares.Predict);
    }

    [Fact]
    public void OneNumberIsATwoWayDivision()
    {
        var shares = SplitShares.Of(80);

        Assert.Equal(0, shares.Validation);
        Assert.Equal(0.20, shares.Test, 9);
    }

    [Fact]
    public void AndAPredictSliceComesOffTheSameWhole()
    {
        // The owner's arithmetic: hold ten back to predict on, and training, validation and test are the
        // ninety that remain.
        var shares = SplitShares.Of(70, 10, predict: 10);

        Assert.Equal(0.10, shares.Test, 9);
        Assert.Equal(0.10, shares.Predict, 9);
        Assert.Equal(0.90, shares.Train + shares.Validation + shares.Test, 9);
    }

    [Fact]
    public void MoreThanThereIs_IsRefusedWithWhatWasAskedFor()
    {
        var refused = Assert.Throws<ArgumentException>(() => SplitShares.Of(80, 30));

        Assert.Contains("110", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EverythingAtOnce_LeavesNothingToBeMeasuredOn()
    {
        // A model measured on nothing scores perfectly on nothing, so the whole may not be spoken for.
        var refused = Assert.Throws<ArgumentException>(() => SplitShares.Of(80, 10, predict: 10));

        Assert.Contains("nothing", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NumbersInTwoDifferentUnits_AreRefusedRatherThanRead()
    {
        // 0.70 and 10 in one call is a slip, not a request for seven-tenths of one percent, and the run
        // that follows it would look entirely ordinary.
        var refused = Assert.Throws<ArgumentException>(() => SplitShares.Of(0.70, 0.10, predict: 10));

        Assert.Contains("same units", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(double.NaN, 0.1, 0)]
    [InlineData(0.7, double.NaN, 0)]
    [InlineData(0.7, 0.1, double.NaN)]
    [InlineData(double.PositiveInfinity, 0.1, 0)]
    public void AShareThatIsNotANumber_IsRefusedBeforeAnythingIsAddedUp(
        double train, double validation, double predict)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SplitShares.Of(train, validation, predict));
    }

    // ---- where the rows go ---------------------------------------------------------------------------

    [Fact]
    public void TheSlicesAreHandedOutInOrder_PredictLast()
    {
        var over = SplitShares.Of(70, 10, predict: 10).Over(10);

        Assert.Equal(7, over.Count(split => split == Part.Train));
        Assert.Equal(Part.Validation, over[7]);
        Assert.Equal(Part.Test, over[8]);
        Assert.Equal(Part.Predict, over[9]);
    }

    [Fact]
    public void SoASplitInTimeKeepsTheNewestRowsToPredictOn()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("t", ColumnKind.Integer, false)]),
            CsvRowSource.FromText("t\n" + string.Join("\n", Enumerable.Range(1, 10)) + "\n"));

        var parts = new SplitByTimeStep("t", SplitShares.Of(70, 10, predict: 10)).Assign(table);

        Assert.Equal(Part.Predict, parts[9]);
        Assert.Equal(Part.Train, parts[0]);
    }

    [Fact]
    public void AndNothingIsFittedOnThoseRowsEither()
    {
        // The same filter that keeps test out of a fit keeps predict out of it: a scale learned from the
        // rows you are about to predict on is the leak this library exists to prevent.
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n1\n2\n3\n4\n1000\n"));

        var learned = new NormaliseStep("a", Scale.MinMax).Fit(
            table, [Part.Train, Part.Train, Part.Train, Part.Train, Part.Predict]);

        // Four, not a thousand: the spike sits in the slice held back to predict on.
        Assert.Equal(1, learned.Number("centre"));
        Assert.Equal(3, learned.Number("spread"));
    }

    // ---- saying it in the chain ----------------------------------------------------------------------

    [Fact]
    public void ThePredictSliceIsWrittenOnceAndTheSplitCarriesIt()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Predict(10)
            .SplitByTime("t", 70, 10)
            .Declaration;

        var split = Assert.IsType<SplitByTimeStep>(declaration.Steps[1]);

        Assert.Equal(0.10, split.Shares.Predict, 9);
        Assert.Equal(0.10, split.Shares.Test, 9);
        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson()));
    }

    [Fact]
    public void EveryKindOfSplitTakesOne()
    {
        var atRandom = Pdd.Create().ReadCsv("x.csv").Predict(10).SplitAtRandom(70, 10).Declaration;
        var stratified = Pdd.Create().ReadCsv("x.csv").Predict(10).SplitStratified("g", 70, 10).Declaration;

        Assert.Equal(0.10, Assert.IsType<SplitAtRandomStep>(atRandom.Steps[1]).Shares.Predict, 9);
        Assert.Equal(0.10, Assert.IsType<SplitStratifiedStep>(stratified.Steps[1]).Shares.Predict, 9);
        Assert.Equal(atRandom, PipelineDeclaration.FromJson(atRandom.ToJson()));
        Assert.Equal(stratified, PipelineDeclaration.FromJson(stratified.ToJson()));
    }

    [Fact]
    public void AFileThatSaysNothingAboutPredicting_HoldsNothingBack()
    {
        // The share is the one thing in a split that may be absent from a file, because a pipeline without
        // a slice to predict on is the ordinary case.
        const string json = """
            {"declaration":[{"step":"split.byTime","column":"t","train":0.8,"validation":0.1,"test":0.1}]}
            """;

        var split = Assert.IsType<SplitByTimeStep>(PipelineDeclaration.FromJson(json).Steps[0]);

        Assert.Equal(0, split.Shares.Predict);
    }

    [Fact]
    public void SayingItTwice_IsRefusedWhereItIsWritten()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create().ReadCsv("x.csv").Predict(10).Predict(20));

        Assert.Contains("already", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SayingItAndThenNeverSplitting_IsRefusedToo()
    {
        // A share held back by a pipeline that never divides anything is a promise nothing keeps.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create().ReadCsv("x.csv").Predict(10).Build());

        Assert.Contains("split", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AShareThatIsNotAShare_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Pdd.Create().ReadCsv("x.csv").Predict(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pdd.Create().ReadCsv("x.csv").Predict(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pdd.Create().ReadCsv("x.csv").Predict(double.NaN));
    }

    // ---- and out the other end -----------------------------------------------------------------------

    [Fact]
    public void TheHeldBackRowsComeOutAsABatchOfTheirOwn()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Number("fare").Category("sex"))
            .Predict(10)
            .SplitAtRandom(70, 10, seed: 4)
            .Normalise("fare")
            .EncodeCategories()
            .Build()
            .Run();

        var predict = prepared.Batch(Part.Predict);

        Assert.Equal(89, predict.RowCount);
        Assert.Equal(891, prepared.CountIn(Part.Train)
                        + prepared.CountIn(Part.Validation)
                        + prepared.CountIn(Part.Test)
                        + prepared.CountIn(Part.Predict));
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Join(directory.FullName, "DeepSharp.slnx")))
        {
            directory = directory.Parent;
        }

        Assert.NotNull(directory);
        return directory!.FullName;
    }
}
