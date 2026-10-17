// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A window of rows a served model meets, measured against the training rows the pipeline kept: each watched column by the
/// distance its kind takes, the same two the likeness of the parts takes, flagged above the bound the watch was declared with,
/// beside how many rows the distance was worked out from and the first and last moment the window holds. The rows are replayed to
/// where the watch stands, so a category still is a word there and one the training rows never held is measured, not refused; and
/// nothing is learned from the window.
/// </summary>
public class WatchedWindowTests
{
    private const string Header = "t,day,round,farm,weight,y\n";

    // Four days of a farm's flocks: the first two train, the third validates, the fourth tests.
    private const string Rows =
        Header
        + "1,2025-01-01,1,A,10,0\n2,2025-01-01,2,B,20,1\n3,2025-01-02,1,A,20,0\n4,2025-01-02,2,A,,1\n"
        + "5,2025-01-03,1,B,30,0\n6,2025-01-03,2,B,40,1\n7,2025-01-04,1,A,50,0\n8,2025-01-04,2,C,60,1\n";

    private static PreparedData Prepared(params string[] columns) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Optional("weight", ColumnKind.Number).Number("y"))
            .OrderBy("day", "round")
            .SplitByTime("day", train: 0.5, validation: 0.25)
            .Target("y")
            .Likeness(0.2, columns)
            .Watch(0.2, columns)
            .EncodeCategories()
            .Build()
            .Run();

    private static IRowSource Window(params string[] rows) => CsvRowSource.FromText(Header + string.Join('\n', rows) + "\n");

    private static WindowColumn Column(WindowLikeness window, string name) => window.Columns.Single(column => column.Column == name);

    [Fact]
    public void AWindowLikeTheTrainingRows_IsNoDistanceAway_AndOneThatMovedOn_IsFlagged()
    {
        var prepared = Prepared("weight", "farm");
        var alike = prepared.Watched(Window("9,2025-02-01,1,A,10,", "10,2025-02-01,2,B,20,", "11,2025-02-02,1,A,20,"));
        var moved = prepared.Watched(Window("9,2025-02-01,1,C,90,", "10,2025-02-01,2,C,95,"));

        Assert.Equal(0, Column(alike, "weight").Apart, 12);
        Assert.Equal(1 / 12.0, Column(alike, "farm").Apart, 12);
        Assert.Equal(Distance.KolmogorovSmirnov, Column(alike, "weight").Distance);
        Assert.Equal(Distance.TotalVariation, Column(alike, "farm").Distance);
        Assert.False(Column(alike, "weight").Above);
        Assert.Equal(1, Column(moved, "weight").Apart, 12);
        Assert.Equal(1, Column(moved, "farm").Apart, 12);
        Assert.True(Column(moved, "weight").Above);
        Assert.Equal(0.2, moved.WarnAbove);
    }

    [Fact]
    public void AWindow_SaysWhenItWasAndOverHowManyRows()
    {
        var window = Prepared("weight").Watched(Window("9,2025-02-02,1,A,10,", "10,2025-02-01,1,B,,", "11,2025-02-03,1,A,20,"));
        var weight = Column(window, "weight");

        Assert.Equal(3, window.Rows);
        Assert.Equal(new DateTime(2025, 2, 1), window.From);
        Assert.Equal(new DateTime(2025, 2, 3), window.To);
        Assert.Equal((3, 1, 2, 1), (weight.TrainingRows, weight.TrainingGaps, weight.WindowRows, weight.WindowGaps));
    }

    [Fact]
    public void APipelineReadFromItsFile_MeasuresAWindowAsTheRunThatWroteIt()
    {
        var prepared = Prepared();
        var read = PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn());
        var window = Window("9,2025-02-01,1,A,15,", "10,2025-02-01,2,C,20,");
        var here = prepared.Watched(window);
        var there = read.Watched(window);

        Assert.Equal(here.Columns, there.Columns);
        Assert.Equal((here.Rows, here.From, here.To, here.WarnAbove), (there.Rows, there.From, there.To, there.WarnAbove));
        Assert.Equal(["t", "farm", "weight"], here.Columns.Select(column => column.Column));
    }

    [Fact]
    public void TheTestRowsWatched_AreAsFarFromTheTrainingRowsAsTheLikenessOfThePartsSaysTheyAre()
    {
        var prepared = Prepared("weight", "farm");
        var likeness = prepared.Evidence.Values.OfType<PartLikeness>().Single();
        var watched = prepared.Watched(Window("7,2025-01-04,1,A,50,", "8,2025-01-04,2,C,60,"));

        foreach (var column in likeness.Columns)
        {
            Assert.Equal(column.Pairs.Single(pair => pair is { First: Standing.Train, Second: Standing.Test }).Apart, Column(watched, column.Column).Apart, 15);
        }
    }

    [Fact]
    public void ARowAlone_StandsAsFarAsItsPlaceAmongTheTrainingRows_AndIsFlaggedByTheSameBound()
    {
        // A hundred training rows of the numbers nought to ninety-nine, and one served row at the middle: half the training rows lie
        // below it, so the distance is a half.
        var rows = string.Concat(Enumerable.Range(0, 120).Select(at => $"{at},2025-01-01,{at},A,{at % 100},1\n"));
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText(Header + rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Number("weight", "y"))
            .SplitAtRandom(0.95, 0.025, seed: 1)
            .Target("y")
            .Watch(0.2, "weight")
            .Build()
            .Run();
        var kept = prepared.Fitted[prepared.Declaration.Steps.ToList().FindIndex(step => step is WatchStep)];
        var all = kept.Curve("weight.counts").Sum();
        var below = kept.Curve("weight.values").Zip(kept.Curve("weight.counts")).Where(pair => pair.First < 50).Sum(pair => pair.Second) / all;
        var atOrBelow = kept.Curve("weight.values").Zip(kept.Curve("weight.counts")).Where(pair => pair.First <= 50).Sum(pair => pair.Second) / all;
        var alone = Column(prepared.Watched(Window("200,2025-03-01,1,A,50,")), "weight");

        Assert.Equal(Math.Max(below, 1 - atOrBelow), alone.Apart, 12);
        Assert.True(alone.Above);
        Assert.Equal(1, alone.WindowRows);
    }

    [Fact]
    public void AWindowOfGaps_HasNoDistance_IsNotFlagged_AndSaysHowManyGapsItHeld()
    {
        var window = Prepared("weight").Watched(Window("9,2025-02-01,1,A,,", "10,2025-02-01,2,A,,", "11,2025-02-01,3,A,5,"));
        var weight = Column(window, "weight");

        Assert.Equal((1, 2), (weight.WindowRows, weight.WindowGaps));

        var empty = Column(Prepared("weight").Watched(Window("9,2025-02-01,1,A,,")), "weight");

        Assert.True(double.IsNaN(empty.Apart));
        Assert.False(empty.Above);
    }

    [Fact]
    public void AnAnswerTheOutputMakes_ArrivesAsGaps_AndIsMeasuredAsNone()
    {
        // A row a day, so the rows ahead of a row are the days after it.
        var days = string.Concat(Enumerable.Range(0, 12).Select(at => $"{at},{new DateTime(2025, 1, 1).AddDays(at):yyyy-MM-dd},1,A,{10 + at},{at}\n"));
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText(Header + days), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Number("weight", "y"))
            .OrderBy("day")
            .SplitByTime("day", train: 0.5, validation: 0.25, gap: 1)
            .Ahead("y", 1, AheadAs.Value)
            .Watch(0.2, "y.ahead1", "weight")
            .Build()
            .Run();
        var answer = Column(prepared.Watched(Window("9,2025-02-01,1,A,10,", "10,2025-02-02,1,A,20,")), "y.ahead1");

        Assert.Equal(0, answer.WindowRows);
        Assert.True(double.IsNaN(answer.Apart));
    }

    [Fact]
    public void APipelineThatKeepsNoTrainingRows_RefusesToMeasureAWindow_SayingWhy()
    {
        var plain = Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Optional("weight", ColumnKind.Number).Number("y"))
            .SplitByTime("day", train: 0.5, validation: 0.25)
            .Target("y")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => plain.Watched(Window("9,2025-02-01,1,A,10,")));

        Assert.Contains("This pipeline keeps none of the rows it was fitted on, so the rows it is served have nothing to be measured against", refused.Message, StringComparison.Ordinal);
        Assert.Contains("watch.reference", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineThatKeepsTheTrainingRowsTwice_RefusesToMeasureAWindow_SayingAtHowManyPlaces()
    {
        var twice = Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Optional("weight", ColumnKind.Number).Number("y"))
            .SplitByTime("day", train: 0.5, validation: 0.25)
            .Target("y")
            .Watch(0.2, "weight")
            .Watch(0.3, "farm")
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => twice.Watched(Window("9,2025-02-01,1,A,10,")));

        Assert.Contains("at 2 places", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWindowTeachesTheWatchNothing()
    {
        var prepared = Prepared("weight");
        var before = prepared.ToJson();

        _ = prepared.Watched(Window("9,2025-02-01,1,C,900,"));

        Assert.Equal(before, prepared.ToJson());
        Assert.Equal(prepared.Watched(Window("9,2025-02-01,1,A,10,")), prepared.Watched(Window("9,2025-02-01,1,A,10,")), new SameWindow());
    }

    private sealed class SameWindow : IEqualityComparer<WindowLikeness>
    {
        public bool Equals(WindowLikeness? x, WindowLikeness? y) => x is not null && y is not null && x.Columns.SequenceEqual(y.Columns) && x.Rows == y.Rows;

        public int GetHashCode(WindowLikeness obj) => obj.Rows;
    }
}
