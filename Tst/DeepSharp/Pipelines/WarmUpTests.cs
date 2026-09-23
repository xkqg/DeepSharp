// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// With indicators on the data, the honest row count is the rows minus the longest warm-up. An indicator of
/// period N does not say "nothing happened" about the first N rows, it says "there is not enough history
/// yet" — and filling that with a number learned from the training rows would invent a measurement nobody
/// took, which is the one thing this library refuses everywhere else.
/// </summary>
public class WarmUpTests
{
    private static string Apple => Path.Join(RepoRoot(), "Samples", "data", "apple.csv");

    private static PipelineBuilder Prices() =>
        Pdd.Create()
            .ReadCsv(Apple)
            .Declare(schema => schema
                .Timestamp("Date")
                .Number("AAPL.High", "AAPL.Low", "AAPL.Close"));

    [Fact]
    public void FiveHundredAndSixRowsWithATwentyPeriodAverage_AreFourHundredAndEightySeven()
    {
        var prepared = Prices()
            .AddIndicator("sma20", Indicator.Sma, ["AAPL.Close"], 20)
            .DropWarmUp()
            .SplitByTime("Date", 0.70, 0.15, 0.15)
            .Build()
            .Run();

        Assert.Equal(487, prepared.Table.RowCount);
        Assert.Equal(487, prepared.Splits.Count);
        Assert.DoesNotContain(
            Enumerable.Range(0, prepared.Table.RowCount), prepared.Table["sma20"].IsMissing);
    }

    [Fact]
    public void TheLongestWarmUpDecidesForEveryColumn()
    {
        // Three indicators with three warm-ups; the table has to stay rectangular, so the slowest wins.
        var prepared = Prices()
            .AddIndicator("sma5", Indicator.Sma, ["AAPL.Close"], 5)
            .AddIndicator("sma20", Indicator.Sma, ["AAPL.Close"], 20)
            .AddIndicator("rsi", Indicator.Rsi, ["AAPL.Close"], 14)
            .DropWarmUp()
            .SplitByTime("Date", 0.70, 0.15, 0.15)
            .Build()
            .Run();

        Assert.Equal(487, prepared.Table.RowCount);

        foreach (var column in prepared.Table.Columns)
        {
            Assert.Equal(0, column.LeadingGaps());
        }
    }

    [Fact]
    public void TheRowsThatSurvive_AreTheLastOnesAndStillInOrder()
    {
        var whole = Prices().Build().Prepare();
        var kept = Prices()
            .AddIndicator("sma20", Indicator.Sma, ["AAPL.Close"], 20)
            .DropWarmUp()
            .Build()
            .Run()
            .Table;

        var first = (Column<DateTime>)whole["Date"];
        var after = (Column<DateTime>)kept["Date"];

        // Nineteen rows gone from the front, and row 19 of the whole is row 0 of what is left.
        Assert.Equal(first[19], after[0]);
        Assert.Equal(first[505], after[486]);
    }

    [Fact]
    public void AColumnThatStartsLaterThanTheDataItself_StopsTheRun()
    {
        // The guard: a column empty from the top for some other reason would otherwise take the whole
        // dataset with it, and a run that shrank to nothing should stop rather than succeed.
        var refused = Assert.Throws<InvalidOperationException>(
            () => Prices()
                .AddIndicator("sma100", Indicator.Sma, ["AAPL.Close"], 100)
                .DropWarmUp(atMost: 50)
                .Build()
                .Run());

        Assert.Contains("sma100", refused.Message);
        Assert.Contains("99 rows", refused.Message);
    }

    [Fact]
    public void WithoutIndicatorsThereIsNothingToDrop()
    {
        var prepared = Prices().DropWarmUp().Build().Run();

        Assert.Equal(506, prepared.Table.RowCount);
    }

    [Fact]
    public void ATableOfNothingDropsNothing()
    {
        Assert.Equal(0, new DropWarmUpStep().FirstUsableRow(new Table([])));
        Assert.Throws<ArgumentNullException>(() => new DropWarmUpStep().FirstUsableRow(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DropWarmUpStep(-1));
    }

    [Fact]
    public void KeepingFromARowThatIsNotThere_IsRefused()
    {
        var table = new Table([new Column<double>("a", ColumnKind.Number, [1.0, 2.0])]);

        Assert.Throws<ArgumentOutOfRangeException>(() => table.From(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => table.From(3));
        Assert.Equal(0, table.From(2).RowCount);
    }

    [Fact]
    public void AColumnOfWordsIsCutTheSameWay()
    {
        var table = new Table([new TextColumn("a", ColumnKind.Category, [null, "x", "y"])]);

        Assert.Equal(1, table["a"].LeadingGaps());

        var kept = table.From(1);

        Assert.Equal(2, kept.RowCount);
        Assert.Equal(ColumnKind.Category, kept["a"].Kind);
        Assert.Equal("x", ((TextColumn)kept["a"])[0]);
    }

    [Fact]
    public void TheVerbSurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"))
            .DropWarmUp(atMost: 25)
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson());

        Assert.Equal(declaration, returned);
        Assert.Equal(25, ((DropWarmUpStep)returned.Steps[2]).AtMost);
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
