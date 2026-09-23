// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Some steps read the rows in their order — a moving average, a warm-up at the start, a gap filled with the
/// value before it — and the order a file happens to arrive in is not something anybody declared. The same
/// declaration over the same prices, reversed, gave a five-day average of 98.352 where the right one is 99.74.
/// So the order is a step of its own, and a step that reads it is refused unless the order stands above it.
/// </summary>
public class RowOrderTests
{
    private static InMemoryRowSource Apple(bool reversed)
    {
        var source = new CsvRowSource(Repository.Data("apple.csv"));

        return new InMemoryRowSource(source.ColumnNames, reversed ? source.Rows.Reverse() : source.Rows);
    }

    private static InMemoryRowSource Days(params string?[] values) => new(
        ["Date", "value"],
        [.. values.Select((value, day) => (IReadOnlyList<string?>)
            [new DateTime(2020, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), value])]);

    private static InMemoryRowSource Reversed(InMemoryRowSource rows) => new(rows.ColumnNames, rows.Rows.Reverse());

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheSameDeclarationOverAReversedFile_GivesTheSameMovingAverage(bool reversed)
    {
        var prepared = Pdd.Create()
            .Read(Apple(reversed), "apple prices")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .OrderBy("Date")
            .AddIndicator("sma5", Indicator.Sma, ["AAPL.Close"], 5)
            .Build()
            .Run();

        var dates = (Column<DateTime>)prepared.Table["Date"];
        var day = Enumerable.Range(0, prepared.Table.RowCount).Single(row => dates[row] == new DateTime(2016, 6, 1));

        Assert.Equal(99.74, ((Column<double>)prepared.Table["sma5"])[day]!.Value, 5);
        Assert.Equal(new DateTime(2015, 2, 17), dates[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ForwardFillOverAReversedFile_CarriesTheEarlierValue(bool reversed)
    {
        // The gap on the second day is filled with the first day's value — never with the third's, which is
        // what a file handed over newest first used to give it.
        var days = Days("1", null, "3", "4", "5", "6", "7", "8", "9", "10");

        var prepared = Pdd.Create()
            .Read(reversed ? Reversed(days) : days, "ten days")
            .Declare(schema => schema.Timestamp("Date").Optional("value", ColumnKind.Number))
            .OrderBy("Date")
            .SplitByTime("Date", 0.60, 0.20)
            .FillMissing("value", With.Previous)
            .Build()
            .Run();

        Assert.Equal(1, ((Column<double>)prepared.Table["value"])[1]);
    }

    [Theory]
    [InlineData("2020-01-01", "")]
    [InlineData("2020-01-01", "NaN")]
    [InlineData("2020-01-03", "2")]
    public void OrderBy_RefusesAGapANaNAndATie(string secondDate, string secondValue)
    {
        // A row with no key has no place in the order; two rows with the same key have no order between
        // them, and taking whichever came first in the file would be the accident this step exists to end.
        var rows = new InMemoryRowSource(
            ["Date", "value"],
            [["2020-01-01", "1"], [secondDate, secondValue], ["2020-01-03", "2"]]);

        var refused = Assert.Throws<InvalidOperationException>(
            () => Pdd.Create()
                .Read(rows, "three rows")
                .Declare(schema => schema.Timestamp("Date").Optional("value", ColumnKind.Number))
                .OrderBy("value", "Date")
                .Build()
                .Run());

        Assert.Contains("'value'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RowsWithTheSameFirstKey_AreOrderedByTheNext()
    {
        var rows = new InMemoryRowSource(
            ["group", "Date"],
            [["2", "2020-01-02"], ["1", "2020-01-03"], ["2", "2020-01-01"], ["1", "2020-01-01"]]);

        var ordered = Pdd.Create()
            .Read(rows, "four rows")
            .Declare(schema => schema.Integer("group").Timestamp("Date"))
            .OrderBy("group", "Date")
            .Build()
            .Run()
            .Table;

        Assert.Equal([3, 1, 2, 0], ordered.Identities.Select(identity => identity.ReadAt));
    }

    public static TheoryData<string> StepsThatReadRowOrder() => ["indicator", "warm-up", "forward fill"];

    [Theory]
    [MemberData(nameof(StepsThatReadRowOrder))]
    public void AStepThatReadsTheRowsInTheirOrder_WithoutAnOrderAboveIt_IsRefused(string which)
    {
        var start = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Optional("AAPL.Close", ColumnKind.Number));

        var refused = Assert.Throws<DeclarationException>(() =>
        {
            _ = which switch
            {
                "indicator" => start.AddIndicator("sma5", Indicator.Sma, ["AAPL.Close"], 5),
                "warm-up" => start.DropWarmUp(),
                _ => (object)start.SplitByTime("Date", 0.70).FillMissing("AAPL.Close", With.Previous),
            };
        });

        Assert.Contains("order.by", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclarationWithTwoOrders_IsRefused()
    {
        var refused = Assert.Throws<DeclarationException>(
            () => Pdd.Create().ReadCsv("apple.csv").Declare(schema => schema.Timestamp("Date")).OrderBy("Date").OrderBy("Date"));

        Assert.Equal("order.by", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void AnOrderBelowTheSplit_IsRefused()
    {
        // Rows are divided once. Putting them in another order afterwards would move rows under steps that
        // already learned from them in the first order.
        var refused = Assert.Throws<DeclarationException>(
            () => new PipelineDeclaration([
                new ReadCsvStep("apple.csv"),
                new DeclareStep([new ColumnDeclaration("Date", ColumnKind.Timestamp, false)]),
                new SplitByTimeStep("Date", new SplitShares(0.70, 0.15, 0.15)),
                new OrderByStep(["Date"]),
            ]));

        Assert.Equal(3, Assert.Single(refused.Faults).At);
    }

    [Fact]
    public void AnOrder_SurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("apple.csv")
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .OrderBy("Date", "AAPL.Close")
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(["Date", "AAPL.Close"], ((OrderByStep)returned.Steps[2]).Columns);
    }

    [Fact]
    public void TwoOrdersByTheSameColumns_AreTheSameStep()
    {
        var order = new OrderByStep(["Date", "value"]);

        Assert.Equal(order, new OrderByStep(["Date", "value"]));
        Assert.Equal(order.GetHashCode(), new OrderByStep(["Date", "value"]).GetHashCode());
        Assert.NotEqual(order, new OrderByStep(["value", "Date"]));
        Assert.False(order.Equals(null));
        Assert.Throws<ArgumentNullException>(() => new OrderByStep(null!));
        Assert.Throws<ArgumentException>(() => new OrderByStep([]));
    }

    [Fact]
    public void AServedRow_IsFoundAgainAfterTheOrderAndTheWarmUp()
    {
        // Served newest first, as a query without an ORDER BY might hand them over: the replay puts them in
        // the declared order, drops the warm-up, and each served row still says which handed-in row it is.
        var trained = Pdd.Create()
            .Read(Days([.. Enumerable.Range(1, 60).Select(day => (string?)day.ToString(CultureInfo.InvariantCulture))]), "sixty days")
            .Declare(schema => schema.Timestamp("Date").Number("value"))
            .OrderBy("Date")
            .AddIndicator("sma3", Indicator.Sma, ["value"], 3)
            .DropWarmUp()
            .SplitByTime("Date", 0.70)
            .Normalise("value")
            .Drop("Date")
            .Build()
            .Run();

        var served = trained.Served(Reversed(Days("1", "2", "3", "4", "5", "6")));

        Assert.Equal([3, 2, 1, 0], served.HandedInAt);
    }
}
