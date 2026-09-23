// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A run, a replay and the grid under one block of a notebook are the same walk over the same declaration:
/// the steps in the order they were written, fitting in the first and applying what was fitted in the other
/// two. There used to be two walks. The run lifted every feature above every dropped row and the replay
/// dropped nothing at all, so the same sixty rows came out as fifty-six from one and sixty from the other,
/// and a block's grid was not what the full run held at that block.
/// </summary>
public class OneWalkTests
{
    private static string Apple => Repository.Data("apple.csv");

    /// <summary>Sixty days of closing prices, one row a day, oldest first.</summary>
    private static InMemoryRowSource SixtyDays() => new(
        ["Date", "close"],
        [.. Enumerable.Range(0, 60).Select(day => (IReadOnlyList<string?>)
        [
            new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            (100 + day + (day % 7)).ToString(CultureInfo.InvariantCulture),
        ])]);

    private static PipelineBuilder Days() =>
        Pdd.Create()
            .Read(SixtyDays(), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("close"))
            .OrderBy("Date");

    [Fact]
    public void Replay_DropsTheWarmUpRowsRunDrops()
    {
        var trained = Days()
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Build()
            .Run();

        var served = trained.Replay(SixtyDays());

        Assert.Equal(56, trained.Table.RowCount);
        Assert.Equal(56, served.RowCount);
        Assert.Equal(((Column<double>)trained.Table["close"])[0], ((Column<double>)served["close"])[0]);
    }

    [Fact]
    public void AWarmUpDropTakesTheFeaturesWrittenAboveIt_AndNotTheOnesBelow()
    {
        // Written order is the order: the drop cuts the rows the five-day average has no value for, and the
        // ten-day average written after it is worked out over what is left, with its own warm-up still
        // showing. The run used to lift every feature above every drop and cut to the longest.
        var prepared = Days()
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .AddIndicator("sma10", Indicator.Sma, ["close"], 10)
            .Build()
            .Run();

        Assert.Equal(56, prepared.Table.RowCount);
        Assert.Equal(9, prepared.Table["sma10"].LeadingGaps());
        Assert.Equal(0, prepared.Table["sma5"].LeadingGaps());
    }

    [Fact]
    public void ThePrefixUpToABlock_HoldsWhatTheFullRunHoldsThere()
    {
        // The grid under a notebook block is the declaration run up to that block. A step that only looks is
        // put after each block of the full declaration in turn, and what it sees there has to be exactly
        // what the shorter declaration hands back.
        var steps = Pdd.Create()
            .ReadCsv(Apple)
            .Declare(schema => schema
                .Timestamp("Date")
                .Number("AAPL.High", "AAPL.Low", "AAPL.Close", "AAPL.Volume")
                .Category("direction"))
            .OrderBy("Date")
            .AddFeature("range", "AAPL.High", Arithmetic.Minus, "AAPL.Low")
            .AddIndicator("rsi", Indicator.Rsi, ["AAPL.Close"], 14)
            .DropWarmUp()
            .TimeParts("Date", TimePart.Quarter)
            .SplitByTime("Date", 70, 15)
            .Normalise("AAPL.Close", Scale.Robust)
            .Normalise("rsi", Scale.MinMax)
            .EncodeCategories()
            .Declaration
            .Steps;

        for (var n = 2; n <= steps.Count; n++)
        {
            var look = new Look();
            var whole = new PipelineDeclaration([.. steps.Take(n), look, .. steps.Skip(n)]);

            new Pipeline(whole).Run();

            var prefix = new Pipeline(new PipelineDeclaration(steps.Take(n))).Run().Table;

            Assert.Equal(Snapshot.Of(prefix), look.Seen);
        }
    }

    [Fact]
    public void AFitThatIsMissing_IsRefusedRatherThanSkipped()
    {
        // A declaration-only file used to serve age as it was read — 22 — where the model was trained on
        // it scaled: the replay skipped the step it had no fit for and said nothing.
        var declaration = Days().SplitByTime("Date", 0.70).Normalise("close").Declaration;
        var onlyTheSplit = new Dictionary<int, FittedStepValues> { [3] = new() };

        var refused = Assert.Throws<ArgumentException>(
            () => new PreparedData(declaration, new Table([]), [], onlyTheSplit));

        Assert.Contains("normalise", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFitForAStepThatLearnsNothing_IsRefusedToo()
    {
        var declaration = Days().SplitByTime("Date", 0.70).Normalise("close").Declaration;
        var fitted = new Dictionary<int, FittedStepValues> { [1] = new(), [3] = new() };

        var refused = Assert.Throws<ArgumentException>(
            () => new PreparedData(declaration, new Table([]), [], fitted));

        Assert.Contains("declare", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ServingARowWithoutTheAnswer_PreparesTheRestOfIt()
    {
        // The row a model is asked about has no answer yet: that is why it is asked. It used to be refused
        // at the door for lacking the column, so a host had to invent an answer to serve one.
        var trained = Pdd.Create()
            .Read(new InMemoryRowSource(["fare", "survived"], [["7.25", "0"], ["71.28", "1"], ["8.05", "0"], ["53.1", "1"]]), "four passengers")
            .Declare(schema => schema.Number("fare").Integer("survived"))
            .SplitAtRandom(0.50, seed: 1)
            .Normalise("fare")
            .Target("survived")
            .Build()
            .Run();

        var served = trained.Replay(new InMemoryRowSource(["fare"], [["7.25"]]));

        Assert.True(served["survived"].IsMissing(0));
        Assert.Equal(((Column<double>)trained.Table["fare"])[0], ((Column<double>)served["fare"])[0]);
    }

    [Fact]
    public void TheWayBackIsCheckedRowForRow_AfterAWarmUpDrop()
    {
        // With rows dropped at the start, the check compared the first row as read with a row four places
        // further on, and refused a pipeline whose way back was perfectly sound.
        var prepared = Days()
            .AddIndicator("sma5", Indicator.Sma, ["close"], 5)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("close")
            .Target("close")
            .Build()
            .Run();

        var close = (Column<double>)prepared.Table["close"];

        // The first row kept is the fifth day, whose close was 100 + 4 + 4.
        Assert.Equal(56, prepared.Table.RowCount);
        Assert.Equal(108, prepared.BackToOriginal(close[0]!.Value), 6);
    }

    /// <summary>What a table holds, cell by cell, so two tables can be compared as values.</summary>
    private sealed record Snapshot(string Written)
    {
        public static Snapshot Of(Table table)
        {
            var text = new StringBuilder();

            foreach (var column in table.Columns)
            {
                text.Append(column.Name).Append('|').Append(column.Kind).Append(':');

                for (var row = 0; row < table.RowCount; row++)
                {
                    text.Append(column.TextAt(row) ?? "\u0000").Append(';');
                }

                text.AppendLine();
            }

            return new Snapshot(text.ToString());
        }
    }

    /// <summary>A step that changes nothing and remembers what the table held when the walk reached it.</summary>
    private sealed class Look : IAddsColumns
    {
        public Snapshot? Seen { get; private set; }

        public string Verb => "test.look";

        public void AddTo(Table table) => Seen = Snapshot.Of(table);

        public void WriteTo(Utf8JsonWriter writer) => throw new NotSupportedException("A look is never written down.");
    }
}
