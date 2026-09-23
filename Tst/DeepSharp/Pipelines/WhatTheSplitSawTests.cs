// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A saved pipeline says what it learned, and now also what the split it learned behind saw: how many rows
/// went to each part, a digest of the rows it divided, and — for a split in time — where each part starts and
/// ends. Without that a model travels with numbers nobody can trace back to data. And a pipeline that never
/// splits calls its rows undivided rather than training rows, because nothing divided them.
/// </summary>
public class WhatTheSplitSawTests
{
    private static InMemoryRowSource Rows(string file)
    {
        var source = new CsvRowSource(Repository.Data(file));

        return new InMemoryRowSource(source.ColumnNames, source.Rows);
    }

    private static PreparedData Passengers(IRowSource rows) =>
        Pdd.Create()
            .Read(rows, "passengers")
            .Declare(schema => schema.Integer("survived", "pclass").Optional("age", ColumnKind.Number))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Build()
            .Run();

    private static FittedStepValues SplitEntry(PreparedData prepared) =>
        prepared.Fitted[Enumerable.Range(0, prepared.Declaration.Steps.Count).Single(at => prepared.Declaration.Steps[at] is ISplitStep)];

    [Fact]
    public void ARunWithoutASplit_LeavesEveryRowUndivided_NeverCallingThemTraining()
    {
        // A grid over rows nobody divided used to call every one of them a training row: 891 of 891.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("sibsp", "parch"))
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .Build()
            .Run();

        Assert.Equal(891, prepared.CountIn(Part.Undivided));
        Assert.Equal(0, prepared.CountIn(Part.Train));
        Assert.Equal(891, prepared.Batch(Part.Undivided).RowCount);
        Assert.Equal(0, prepared.Batch(Part.Train).RowCount);
    }

    [Fact]
    public void TheSplitWritesDownHowManyRowsEachPartHolds()
    {
        var prepared = Passengers(Rows("titanic.csv"));
        var seen = SplitEntry(prepared);

        Assert.Equal(prepared.CountIn(Part.Train), seen.Number("rows.train"));
        Assert.Equal(prepared.CountIn(Part.Validation), seen.Number("rows.validation"));
        Assert.Equal(prepared.CountIn(Part.Test), seen.Number("rows.test"));
        Assert.Equal(0, seen.Number("rows.predict"));
        Assert.Equal(891, seen.Numbers.Where(each => each.Key.StartsWith("rows.", StringComparison.Ordinal)).Sum(each => each.Value));
    }

    [Fact]
    public void TheDigestOfTheDividedRows_ChangesWhenARowChanges_AndNotWhenTheyArriveInAnotherOrder()
    {
        var rows = Rows("titanic.csv");
        var reversed = new InMemoryRowSource(rows.ColumnNames, rows.Rows.Reverse());
        var changed = new InMemoryRowSource(
            rows.ColumnNames,
            rows.Rows.Select((row, at) => at == 17 ? (IReadOnlyList<string?>)[.. row.Select((cell, column) => column == 6 ? "99.5" : cell)] : row));

        var digest = SplitEntry(Passengers(rows)).Text("digest");

        Assert.Equal(64, digest.Length);
        Assert.Equal(digest, SplitEntry(Passengers(reversed)).Text("digest"));
        Assert.NotEqual(digest, SplitEntry(Passengers(changed)).Text("digest"));
    }

    [Fact]
    public void ASplitInTime_WritesDownWhereEachPartStartsAndEnds()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .SplitByTime("Date", 0.70, 0.15)
            .Build()
            .Run();

        var dates = (Column<DateTime>)prepared.Table["Date"];

        foreach (var part in new[] { Part.Train, Part.Validation, Part.Test })
        {
            var inPart = Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => dates[row]!.Value).ToArray();

            Assert.Equal(
                [inPart.Min().ToString("o", CultureInfo.InvariantCulture), inPart.Max().ToString("o", CultureInfo.InvariantCulture)],
                SplitEntry(prepared).List($"moments.{part.ToString().ToLowerInvariant()}"));
        }

        Assert.False(SplitEntry(prepared).Lists.ContainsKey("moments.predict"));
    }

    [Fact]
    public void ASplitInTimeByAWholeNumber_WritesItsPartsAsNumbers()
    {
        var rows = new InMemoryRowSource(
            ["year", "value"],
            [.. Enumerable.Range(2000, 20).Select(year => (IReadOnlyList<string?>)[year.ToString(CultureInfo.InvariantCulture), "1"])]);

        var prepared = Pdd.Create()
            .Read(rows, "twenty years")
            .Declare(schema => schema.Integer("year").Number("value"))
            .SplitByTime("year", 0.70, 0.15)
            .Build()
            .Run();

        Assert.Equal(["2000", "2013"], SplitEntry(prepared).List("moments.train"));
    }

    [Fact]
    public void WhatTheSplitSaw_TravelsWithThePipeline()
    {
        var prepared = Passengers(Rows("titanic.csv"));

        var loaded = PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(SplitEntry(prepared).Text("digest"), SplitEntry(loaded).Text("digest"));
        Assert.Equal(SplitEntry(prepared).Number("rows.train"), SplitEntry(loaded).Number("rows.train"));
    }

    [Fact]
    public void APipelineWithoutWhatItsSplitSaw_IsRefused()
    {
        var prepared = Passengers(Rows("titanic.csv"));
        var withoutIt = prepared.Fitted.Where(each => prepared.Declaration.Steps[each.Key] is not ISplitStep).ToDictionary();

        var refused = Assert.Throws<ArgumentException>(
            () => new PreparedData(prepared.Declaration, prepared.Table, prepared.Parts, withoutIt));

        Assert.Contains("split.stratified", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWordAPipelineLearnedIsAskedForByName()
    {
        var seen = new FittedStepValues();
        seen.Learned("digest", "ab");

        Assert.Equal("ab", seen.Text("digest"));
        Assert.Equal("ab", Assert.Single(seen.Texts).Value);
        Assert.Throws<InvalidOperationException>(() => seen.Text("elsewhere"));
    }
}
