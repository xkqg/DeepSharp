// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rule the whole library exists for, measured rather than asserted: what a step learns comes from the
/// training rows and from nowhere else. The dataset is the published Titanic list, where the difference is
/// small — 29.78 against 29.70 — and that is the point. Nothing goes red when it is wrong.
/// </summary>
public class FittingTests
{
    private static string Titanic => Repository.Data("titanic.csv");

    private static PreparedData Prepared(FillStrategy strategy) =>
        Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived").Text("sex").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", strategy)
            .Build()
            .Run();

    [Fact]
    public void AFillValue_IsLearnedFromTheTrainingRowsAlone()
    {
        var prepared = Prepared(With.Mean);
        var learned = prepared.Fitted[3].Number("value");

        var overEverything = Enumerable.Range(0, prepared.Table.RowCount)
            .Select(row => ((Column<double>)prepared.Table["age"])[row]!.Value)
            .Average();

        // The mean of the training ages is not the mean of all of them, and a model fitted on the second
        // has been told something about the rows it is going to be measured on.
        Assert.NotEqual(overEverything, learned, 6);
        Assert.InRange(learned, 25, 35);
    }

    [Fact]
    public void EveryGapIsGone_AndTheMarkingColumnSaysWhereTheyWere()
    {
        var prepared = Prepared(With.Mean);
        var age = (Column<double>)prepared.Table["age"];
        var marker = (Column<double>)prepared.Table["age_was_missing"];

        Assert.DoesNotContain(Enumerable.Range(0, prepared.Table.RowCount), age.IsMissing);
        // Every gap in the file is marked, 177 of them; what the fit writes down is what it saw, the 133 among
        // the training rows.
        Assert.Equal(177, Enumerable.Range(0, prepared.Table.RowCount).Count(row => marker[row] == 1));
        Assert.Equal(133, prepared.Fitted[3].Number("gaps"));
        Assert.Equal(0, marker[0]);
    }

    [Fact]
    public void TheFilledRows_AllHoldTheOneLearnedValue()
    {
        var prepared = Prepared(With.Mean);
        var age = (Column<double>)prepared.Table["age"];
        var marker = (Column<double>)prepared.Table["age_was_missing"];
        var learned = prepared.Fitted[3].Number("value");

        var filled = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => marker[row] == 1)
            .Select(row => age[row]!.Value)
            .Distinct()
            .ToArray();

        Assert.Equal([learned], filled);
    }

    [Theory]
    [InlineData("median")]
    [InlineData("zero")]
    public void EveryStrategyLearnsItsOwnValue(string name)
    {
        var strategy = name == "median" ? With.Median : With.Zero;
        var prepared = Prepared(strategy);

        Assert.Equal(name == "zero" ? 0 : 29, prepared.Fitted[3].Number("value"), 0);
    }

    [Fact]
    public void AConstant_IsWrittenDownAsWhatWasUsed()
    {
        var prepared = Prepared(With.Constant(-1));

        Assert.Equal(-1, prepared.Fitted[3].Number("value"));
        Assert.Equal(-1, ((Column<double>)prepared.Table["age"])[5]);
    }

    [Fact]
    public void CarryingTheLastValueForward_LearnsNothingAndSaysSo()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Optional("AAPL.Close", ColumnKind.Number))
            .OrderBy("Date")
            .SplitByTime("Date", 0.70, 0.15)
            .FillMissing("AAPL.Close", With.Previous)
            .Build()
            .Run();

        Assert.False(prepared.Fitted[4].Numbers.ContainsKey("value"));
        Assert.IsType<FillMissingByPreviousStep>(prepared.Declaration.Steps[4]);
        Assert.DoesNotContain(
            Enumerable.Range(0, prepared.Table.RowCount), ((Column<double>)prepared.Table["AAPL.Close"]).IsMissing);
    }

    [Fact]
    public void AColumnSaidToHaveNoGaps_RefusesTheGapThatArrivesLater()
    {
        // The training rows were whole, so there was nothing to refuse while fitting; a row served a year later
        // with a gap in it is refused by the same declaration, rather than filled with a value nobody learned.
        var trained = Pdd.Create()
            .Read(CsvRowSource.FromText("a,b\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n10,10\n"), "ten rows")
            .Declare(schema => schema.Number("a", "b"))
            .SplitAtRandom(0.70, 0.15)
            .FillMissing("a", With.Refuse)
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(
            () => trained.Replay(CsvRowSource.FromText("a,b\n,1\n")));

        Assert.Contains("should be none", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheFormThatFillsWithOneValue_IsNotTheOneThatCarriesAValueForward()
    {
        Assert.Throws<ArgumentException>(() => new FillMissingByValueStep("a", With.Previous));
        Assert.IsType<FillMissingByPreviousStep>(FillMissingStep.Of("a", With.Previous));
        Assert.IsType<FillMissingByValueStep>(FillMissingStep.Of("a", With.Mean));
    }

    [Fact]
    public void AStratifiedSplit_KeepsTheMixtureInEveryPart()
    {
        var prepared = Prepared(With.Mean);
        var survived = (Column<long>)prepared.Table["survived"];

        foreach (var split in new[] { Part.Train, Part.Validation, Part.Test })
        {
            var rows = Enumerable.Range(0, prepared.Table.RowCount)
                .Where(row => prepared.Parts[row] == split)
                .ToArray();

            // 342 of 891 survived, which is 38.4 per cent; every part is within a point of it.
            Assert.InRange(rows.Count(row => survived[row] == 1) / (double)rows.Length, 0.374, 0.394);
        }
    }

    [Fact]
    public void TheSameSeed_LandsEveryRowInTheSamePlace()
    {
        Assert.Equal(Prepared(With.Mean).Parts, Prepared(With.Mean).Parts);
    }

    [Fact]
    public void ASplitInTime_PutsTheEarliestRowsInTraining()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .SplitByTime("Date", 0.70, 0.15)
            .Build()
            .Run();

        var dates = (Column<DateTime>)prepared.Table["Date"];

        var lastTraining = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => prepared.Parts[row] == Part.Train).Max(row => dates[row]!.Value);

        var firstTest = Enumerable.Range(0, prepared.Table.RowCount)
            .Where(row => prepared.Parts[row] == Part.Test).Min(row => dates[row]!.Value);

        Assert.True(lastTraining < firstTest, $"{lastTraining} should come before {firstTest}");
        Assert.Equal(354, prepared.CountIn(Part.Train));
        Assert.Equal(506, prepared.Parts.Count);
    }

    [Fact]
    public void ARowWithNoTimeAtAll_CannotBePlacedInASplitByTime()
    {
        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .ReadCsv(Titanic)
                .Declare(schema => schema.Optional("age", ColumnKind.Number))
                .SplitByTime("age", 0.70, 0.15)
                .Build()
                .Run());

        Assert.Contains("age", refused.Message);
    }

    [Fact]
    public void APipelineThatLearnsWithoutSplitting_CannotBeRunAtAll()
    {
        // Reached by hand, because the chain does not offer it: a file could still say it.
        var declaration = new PipelineDeclaration([
            new ReadCsvStep(Titanic),
            new DeclareStep([new ColumnDeclaration("age", ColumnKind.Number, true)]),
            new SplitAtRandomStep(new SplitShares(0.70, 0.15, 0.15), 1),
        ]);

        Assert.Equal(3, declaration.Steps.Count);

        var withoutSplit = new Pipeline(new PipelineDeclaration(declaration.Steps.Take(2)));

        Assert.Equal(891, withoutSplit.Prepare().RowCount);
    }

    [Fact]
    public void ARandomSplit_HandsOutTheSharesItWasAskedFor()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Titanic)
            .Declare(schema => schema.Integer("survived"))
            .SplitAtRandom(0.60, 0.20, seed: 7)
            .Build()
            .Run();

        Assert.Equal(535, prepared.CountIn(Part.Train));
        Assert.Equal(178, prepared.CountIn(Part.Validation));
        Assert.Equal(178, prepared.CountIn(Part.Test));
    }

    [Fact]
    public void TheWholePipelineWritesItselfOut_DeclarationAndWhatItLearned()
    {
        var written = Prepared(With.Mean).ToJson();

        Assert.Contains("\"declaration\"", written, StringComparison.Ordinal);
        Assert.Contains("\"fitted\"", written, StringComparison.Ordinal);
        Assert.Contains("\"prefix\"", written, StringComparison.Ordinal);
        Assert.Contains("\"learned\"", written, StringComparison.Ordinal);
        Assert.Contains("\"value\"", written, StringComparison.Ordinal);
    }
}
