// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The rows a served model meets are measured against the rows it learned from, and a pipeline read from its file holds none of
/// those: so a step below the split keeps, as its fit sees them, the training rows of every column it watches — each value once,
/// with how many training rows hold it, and how many hold none. It learns nothing a replay reads and changes no column; it is
/// written into the eleventh version of the pipeline's file, which an earlier library refuses whole.
/// </summary>
public class WatchReferenceTests
{
    // Four days of a farm's flocks, ordered by day and by the round within it; the answer is y.
    private const string Rows =
        "t,day,round,farm,weight,y\n"
        + "1,2025-01-01,1,A,10,0\n2,2025-01-01,2,B,20,1\n3,2025-01-02,1,A,20,0\n4,2025-01-02,2,A,,1\n"
        + "5,2025-01-03,1,B,30,0\n6,2025-01-03,2,B,40,1\n7,2025-01-04,1,A,50,0\n8,2025-01-04,2,C,60,1\n";

    private static PreparedData Prepared(params string[] columns) =>
        Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Optional("weight", ColumnKind.Number).Number("y"))
            .OrderBy("day", "round")
            .SplitByTime("day", train: 0.5, validation: 0.25)
            .Target("y")
            .Watch(0.2, columns)
            .Build()
            .Run();

    private static FittedStepValues Kept(PreparedData prepared) =>
        prepared.Fitted[prepared.Declaration.Steps.ToList().FindIndex(step => step is WatchStep)];

    [Fact]
    public void TheFit_KeepsEachColumnOfTheTrainingRows_EveryValueOnceWithItsCount_AndTheGaps()
    {
        // The training rows are the first two days: weights 10, 20, 20 and a gap; farms A, B, A, A.
        var kept = Kept(Prepared("weight", "farm"));

        Assert.Equal(["weight", "farm"], kept.List("columns"));
        Assert.Equal([10, 20], kept.Curve("weight.values"));
        Assert.Equal([1, 2], kept.Curve("weight.counts"));
        Assert.Equal(1, kept.Number("weight.gaps"));
        Assert.Equal(["A", "B"], kept.List("farm.words"));
        Assert.Equal([3, 1], kept.Curve("farm.counts"));
        Assert.Equal(0, kept.Number("farm.gaps"));
    }

    [Fact]
    public void WhatItKeeps_IsAllSeen_SoNoReplayReadsIt_AndItChangesNoColumn()
    {
        var prepared = Prepared("weight", "farm");
        var kept = Kept(prepared);

        Assert.Equal(
            kept.Numbers.Keys.Concat(kept.Lists.Keys).Concat(kept.Curves.Keys).Order(StringComparer.Ordinal),
            kept.Counts.Order(StringComparer.Ordinal));
        Assert.Equal(["t", "round", "day", "farm", "weight", "y"], prepared.Table.Columns.Select(column => column.Name));
    }

    [Fact]
    public void LeftUnnamed_ItWatchesEveryColumnButTheAnswer_AndTheColumnsTheRowsAreOrderedAndDividedBy()
    {
        // The day and the round put the rows in order and the day divides them: every window would be unlike the training rows
        // there by construction, so they are no feature to watch.
        Assert.Equal(["t", "farm", "weight"], Kept(Prepared()).List("columns"));
    }

    [Fact]
    public void TheLikenessOfTheParts_StillMeasuresTheColumnsTheRowsAreDividedBy_AsItAlwaysHas()
    {
        // A verb never changes its meaning in place: the likeness left unnamed measured the split's column before the watch came,
        // so it still does, while the watch beside it leaves that column out.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t", "round").Timestamp("day").Category("farm").Optional("weight", ColumnKind.Number).Number("y"))
            .OrderBy("day", "round")
            .SplitByTime("day", train: 0.5, validation: 0.25)
            .Target("y")
            .Likeness(0.2)
            .Watch(0.2)
            .Build()
            .Run();

        Assert.Contains(prepared.Evidence.Values.OfType<PartLikeness>().Single().Columns, column => column.Column == "day");
        Assert.DoesNotContain("day", Kept(prepared).List("columns"));
    }

    [Fact]
    public void TheStep_HasItsBoundWithoutADefault_IsWrittenAndReadBack_AndIsNewInTheEleventhVersion()
    {
        var description = StepCatalog.BuiltIn().Describe(WatchStep.Name);
        var prepared = Prepared("weight");

        Assert.Equal("watch.reference", WatchStep.Name);
        Assert.Equal(11, WatchStep.Since);
        Assert.Equal(11, PipelineDeclaration.Version);
        Assert.Contains("warnAbove", description.Parameters.SelectMany(parameter => parameter.RequiredKeys));
        Assert.DoesNotContain("columns", description.Parameters.SelectMany(parameter => parameter.RequiredKeys));
        Assert.Equal(prepared.ToJson(), PreparedData.FromJson(prepared.ToJson(), StepCatalog.BuiltIn()).ToJson());
        Assert.Equal(new WatchStep(0.2, ["weight"]), PipelineDeclaration.FromJson(prepared.Declaration.ToJson(), StepCatalog.BuiltIn()).Steps.OfType<WatchStep>().Single());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1.5)]
    public void ABoundThatIsNoShareAboveNothing_IsRefusedWhereTheStepIsMade(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new WatchStep(wrong));

    [Fact]
    public void TwoStepsAreTheSame_WhenTheyWatchTheSameColumnsUnderTheSameBound()
    {
        var step = new WatchStep(0.2, ["weight"]);

        Assert.Equal(step, new WatchStep(0.2, ["weight"]));
        Assert.Equal(step.GetHashCode(), new WatchStep(0.2, ["weight"]).GetHashCode());
        Assert.NotEqual(step, new WatchStep(0.3, ["weight"]));
        Assert.NotEqual(step, new WatchStep(0.2));
        Assert.False(step.Equals(null));
    }

    [Fact]
    public void AStepThatKeepsTheTrainingRows_StandsBelowTheSplit()
    {
        // The declaration as a file writes it, with the watch moved above the split: a hand-edited file is the one door that can.
        var file = System.Text.Json.Nodes.JsonNode.Parse(Prepared("weight").Declaration.ToJson())!;
        var steps = file["declaration"]!.AsArray();
        var watch = steps.Single(step => (string)step!["step"]! == WatchStep.Name)!;
        var split = steps.Single(step => ((string)step!["step"]!).StartsWith("split.", StringComparison.Ordinal))!;

        steps.Remove(watch);
        steps.Insert(steps.IndexOf(split), watch);

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(file.ToJsonString(), StepCatalog.BuiltIn()));
        Assert.Contains("watch.reference", refused.Message, StringComparison.Ordinal);
    }
}
