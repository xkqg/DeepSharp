// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// How far a notebook has followed a course, read from its blocks alone: which steps of the course its blocks already say,
/// and which come next. A notebook may skip a step the course names and add one it does not; what it may not do is take
/// the steps in another order than the course teaches, because a course continued from there would say one thing and the
/// blocks another.
/// </summary>
public class CourseProgressTests
{
    private static readonly StepCatalog Catalog = NotebookVerbs.Catalog();

    private static CellModel Block(string verb) => new() { Type = StepCellType.StepType, Source = $"{{\"step\": \"{verb}\"}}" };

    private static CellModel Text(string source) => new() { Type = "markdown", Source = source };

    private static CourseProgress Of(PipelineCourse course, params CellModel[] cells) => CourseProgress.Of(course, cells, Catalog);

    [Fact]
    public void ANotebookWithNoBlock_FollowsTheCourse_AndWaitsForAllOfIt()
    {
        var progress = Of(PipelineCourse.Table);

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps, progress.Missing);
        Assert.Equal(0, progress.InsertAt);
    }

    [Fact]
    public void BlocksThatSayTheFirstSteps_LeaveTheRestMissing_AndTheNextBlockGoesAfterThem()
    {
        var progress = Of(PipelineCourse.Table, Text("# Passengers"), Block("read.csv"), Block("declare"), Text("Notes."));

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps.Skip(2), progress.Missing);
        Assert.Equal(3, progress.InsertAt);
    }

    [Fact]
    public void ANotebookWithNoBlockButText_PutsTheCourseAfterTheText()
    {
        var progress = Of(PipelineCourse.Table, Text("# Passengers"), Text("What follows reads the file."));

        Assert.Equal(2, progress.InsertAt);
    }

    [Fact]
    public void AStepTheNotebookLeftOut_IsNotMissing_BecauseItsPlaceIsBehindWhatStands()
    {
        // Settling, features and a scale are for the pipelines that need them; one that went straight to its split has
        // decided it does not, and the rest of the course goes on from there.
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("split.stratified"));

        Assert.True(progress.Follows);
        Assert.Equal(["target", "drop.columns", "fill.missing", "normalise", "evidence.report", "learn.network"], progress.Missing.Select(step => step.Verb));
    }

    [Fact]
    public void AStepTheCourseDoesNotName_IsLeftWhereItStands_AndTheCourseGoesOn()
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("evidence.profile"));

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps.Skip(2), progress.Missing);
        Assert.Equal(3, progress.InsertAt);
    }

    [Fact]
    public void ABlockOfAnotherKindOfSource_SaysTheCoursesSource()
    {
        var progress = Of(PipelineCourse.Table, Block("read.json"), Block("declare"));

        Assert.True(progress.Follows);
        Assert.DoesNotContain(progress.Missing, step => step.Verb == "read.csv");
    }

    [Theory]
    [InlineData("split.byTime", "split.stratified")]
    [InlineData("split.atRandom", "split.stratified")]
    [InlineData("target.labels", "target")]
    [InlineData("target.ahead", "target")]
    public void AStepOfAnotherKindOfTheSameThing_SaysTheCoursesStep(string standing, string row)
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block(standing));

        Assert.True(progress.Follows);
        Assert.DoesNotContain(progress.Missing, step => step.Verb == row);
    }

    [Fact]
    public void TwoStepsThatMayStandAnyNumberOfTimes_DoNotSayEachOther()
    {
        // The course holds one settling and one scale from bounds; a notebook that already has a feature and a settling
        // has not said the scale.
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("settle.gaps"), Block("feature.add"));

        Assert.Contains(progress.Missing, step => step.Verb == "scale.given");
    }

    [Fact]
    public void StepsInAnotherOrderThanTheCourseTeaches_DoNotFollowIt()
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("split.stratified"), Block("declare"));

        Assert.False(progress.Follows);
    }

    [Fact]
    public void ANotebookThatHoldsTheWholeCourse_HasNothingMissing()
    {
        var progress = Of(PipelineCourse.Table, [.. PipelineCourse.Table.Steps.Select(step => Block(step.Verb))]);

        Assert.True(progress.Follows);
        Assert.Empty(progress.Missing);
    }

    [Fact]
    public void ABlockWithNoVerbAtAll_IsNoStepOfAnyCourse_AndStopsNothing()
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), new CellModel { Type = StepCellType.StepType, Source = "not even json" });

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps.Skip(1), progress.Missing);
    }

    [Fact]
    public void ASeriesNotebook_SaysMoreOfTheSeriesCourse_ThanOfTheTable()
    {
        // The table's course names no order, so a block that orders the rows is nothing it says; the series' says it.
        var cells = new[] { Block("read.csv"), Block("declare"), Block("order.by") };

        Assert.Equal(3, Of(PipelineCourse.SeriesInTime, cells).Matched);
        Assert.Equal(2, Of(PipelineCourse.Table, cells).Matched);
    }

    [Fact]
    public void TheSameStepTwice_IsTheSameStepOfTheCourseSaidAgain_NotAStepOutOfOrder()
    {
        // A scale is written once a column, so a notebook holds as many as it has columns to scale.
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("normalise"), Block("normalise"));

        Assert.True(progress.Follows);
        Assert.Equal(["evidence.report", "learn.network"], progress.Missing.Select(step => step.Verb));
    }

    [Fact]
    public void AStepSaidAgainAfterTheStepsBehindIt_IsOutOfOrder()
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("fill.missing"), Block("normalise"), Block("fill.missing"));

        Assert.False(progress.Follows);
    }

    [Fact]
    public void ABlockOfAnyOtherKindOfCell_IsNotACourseStep()
    {
        var progress = Of(PipelineCourse.Table, new CellModel { Type = "code", Source = "{\"step\": \"split.stratified\"}" }, Block("read.csv"));

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps.Skip(1), progress.Missing);
    }

    [Fact]
    public void WhichCourseANotebookIsFollowing_IsTheTableUnlessItsBlocksOnlyFollowTheSeries()
    {
        Assert.Equal(PipelineCourse.Table, CourseProgress.FollowedBy(Array.Empty<CellModel>(), Catalog));
        Assert.Equal(PipelineCourse.Table, CourseProgress.FollowedBy([Block("read.csv"), Block("declare")], Catalog));
        Assert.Equal(PipelineCourse.SeriesInTime, CourseProgress.FollowedBy([Block("read.csv"), Block("declare"), Block("order.by")], Catalog));
        Assert.Null(CourseProgress.FollowedBy([Block("declare"), Block("read.csv")], Catalog));
    }

    [Fact]
    public void AStepAVerbOfThisCatalogDoesNotKnow_IsNoStepOfTheCourse()
    {
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("not.a.verb"));

        Assert.True(progress.Follows);
        Assert.Equal(PipelineCourse.Table.Steps.Skip(1), progress.Missing);
    }

    [Fact]
    public void ADropBeforeTheSplit_SaysTheCoursesDrop_AndMovesTheCourseOnNowhere()
    {
        // A drop may stand before what learns: one right after the schema is no reason to think the split and the answer,
        // which stand before the steps that learn, were said.
        var progress = Of(PipelineCourse.Table, Block("read.csv"), Block("declare"), Block("drop.columns"));

        Assert.True(progress.Follows);
        Assert.Equal(
            ["settle.gaps", "feature.add", "scale.given", "split.stratified", "target", "fill.missing", "normalise", "evidence.report", "learn.network"],
            progress.Missing.Select(step => step.Verb));
        Assert.Equal(2, progress.NextRow);
    }

    [Fact]
    public void ADropAfterWhatLearns_DoesNotTakeTheCourseAway()
    {
        var progress = Of(
            PipelineCourse.Table,
            [.. new[] { "read.csv", "declare", "split.stratified", "target", "fill.missing", "normalise", "drop.columns" }.Select(Block)]);

        Assert.True(progress.Follows);
        Assert.Equal(["evidence.report", "learn.network"], progress.Missing.Select(step => step.Verb));
    }

    [Fact]
    public void AShuffle_IsNotTheOrderOfASeries_SoItDoesNotMakeANotebookASeries()
    {
        var cells = new[] { Block("read.csv"), Block("declare"), Block("shuffle") };

        Assert.Equal(2, Of(PipelineCourse.SeriesInTime, cells).Matched);
        Assert.Equal(PipelineCourse.Table, CourseProgress.FollowedBy(cells, Catalog));
        Assert.True(CourseProgress.IsOfferedFor(PipelineCourse.Table, cells, Catalog));
    }

    [Fact]
    public void ABlockBetweenTwoOthers_IsTheStepThatBelongsThere()
    {
        var cells = new[] { Block("read.csv"), Block("declare"), Block("split.stratified") };

        Assert.Equal("settle.gaps", CourseProgress.NextAt(PipelineCourse.Table, cells, 2, Catalog)?.Verb);
        Assert.Equal("target", CourseProgress.NextAt(PipelineCourse.Table, cells, 3, Catalog)?.Verb);
    }

    [Fact]
    public void ABlockWhereNoStepOfTheCourseBelongs_StartsWithNone()
    {
        var cells = new[] { Block("read.csv"), Block("declare"), Block("split.stratified") };

        // Right after the source the schema comes, which the blocks already say below; before the source nothing stands.
        Assert.Null(CourseProgress.NextAt(PipelineCourse.Table, cells, 1, Catalog));
        Assert.Null(CourseProgress.NextAt(PipelineCourse.Table, cells, 0, Catalog));
    }

    [Fact]
    public void TextAboveTheBlocks_IsNoPlaceForAStep_AndTheEndOfThemIsStillTheNextStep()
    {
        var cells = new[] { Text("# Passengers"), Block("read.csv"), Block("declare") };

        Assert.Null(CourseProgress.NextAt(PipelineCourse.Table, cells, 1, Catalog));
        Assert.Equal("settle.gaps", CourseProgress.NextAt(PipelineCourse.Table, cells, 3, Catalog)?.Verb);
    }

    [Fact]
    public void ABlockAddedToANotebookThatWentItsOwnWay_StartsWithNoStep()
    {
        var cells = new[] { Block("split.stratified"), Block("read.csv") };

        Assert.Null(CourseProgress.NextAt(PipelineCourse.Table, cells, 2, Catalog));
    }

    [Fact]
    public void ADropAboveOrBelowThePlace_DoesNotLimitTheStepsThatBelongThere()
    {
        var cells = new[] { Block("read.csv"), Block("declare"), Block("drop.columns"), Block("split.stratified") };

        Assert.Equal("settle.gaps", CourseProgress.NextAt(PipelineCourse.Table, cells, 2, Catalog)?.Verb);
        Assert.Equal("settle.gaps", CourseProgress.NextAt(PipelineCourse.Table, cells, 3, Catalog)?.Verb);
    }

    [Fact]
    public void WhereverABlockIsAdded_TheStepItStartsAsKeepsTheNotebookOnTheCourse()
    {
        var cells = new[] { Block("read.csv"), Block("declare"), Block("settle.gaps"), Block("split.stratified"), Block("target") };

        for (var at = 0; at <= cells.Length; at++)
        {
            if (CourseProgress.NextAt(PipelineCourse.Table, cells, at, Catalog) is not { } next)
            {
                continue;
            }

            List<CellModel> added = [.. cells.Take(at), Block(next.Verb), .. cells.Skip(at)];

            Assert.True(CourseProgress.Of(PipelineCourse.Table, added, Catalog).Follows, $"{next.Verb} added at {at}");
        }
    }

    [Fact]
    public void EveryStepOfEveryNamedCourse_IsAVerbTheNotebookKnows()
    {
        // What is read of a course's verbs is asked of the notebook's catalog without asking first whether it knows them.
        Assert.All(PipelineCourse.Named.SelectMany(course => course.Steps), step => Assert.True(Catalog.Knows(step.Verb), step.Verb));
    }

    [Fact]
    public void TheStepsThatMayStandOnlyOnce_AreExactlyTheOnesTheRulesRefuseASecondOf()
    {
        // The notebook says which verbs fill the same place of a course by what a declaration keeps one of; the declaration
        // says it too, in its rules. Held to each other, so a step a rule starts to limit is limited here as well.
        foreach (var description in Catalog.Descriptions)
        {
            var step = Catalog.ReadStep(description.Template);
            var refusedTwice = PipelineDeclaration.FaultsIn([step, step]).Any(fault => fault.Message.StartsWith("a pipeline has one ", StringComparison.Ordinal));

            Assert.Equal(CourseProgress.OnceOnly.Any(kind => kind.IsInstanceOfType(step)), refusedTwice);
        }
    }
}
