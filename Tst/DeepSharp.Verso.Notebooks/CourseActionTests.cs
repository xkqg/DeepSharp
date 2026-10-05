// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The buttons that write a course into a notebook: every step the notebook does not hold yet, as its skeleton, in one
/// turn. A course is for starting a pipeline from nothing, so it goes on from where the blocks stand — never over a block,
/// never in another order than they stand in — and writes nothing a notebook that holds the whole course lacks.
/// </summary>
public sealed class CourseActionTests
{
    private static readonly StepCatalog Catalog = NotebookVerbs.Catalog();

    private static JsonObject Json(string text) => JsonNode.Parse(text)!.AsObject();

    // The table course filled in for the passenger list, as the core's tests fill it: a separate suite, so said again here.
    private static PipelineCourse Passengers() => PipelineCourse.Table
        .Say("read.csv", new JsonObject { ["path"] = Repository.Data("titanic.csv") })
        .Say("declare", Json("""
            {"columns":[{"name":"survived","kind":"integer","optional":false},{"name":"sibsp","kind":"integer","optional":false},
                        {"name":"parch","kind":"integer","optional":false},{"name":"fare","kind":"number","optional":false},
                        {"name":"age","kind":"number","optional":true}]}
            """))
        .Say("settle.gaps", Json("""{"column":"fare"}"""))
        .Say("feature.add", Json("""{"column":"family","left":"sibsp","arithmetic":"plus","right":"parch"}"""))
        .Say("scale.given", Json("""{"column":"fare","lowest":0,"highest":512}"""))
        .Say("split.stratified", Json("""{"column":"survived"}"""))
        .Say("target", Json("""{"column":"survived"}"""))
        .Say("drop.columns", Json("""{"columns":["sibsp","parch"]}"""))
        .Say("fill.missing", Json("""{"column":"age"}"""))
        .Say("normalise", Json("""{"column":"age"}"""));

    private static IEnumerable<string> Verbs(Notebook notebook) =>
        notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => StepText.Of(cell.Source).Verb!);

    private static IEnumerable<string> SourcesOfBlocks(Notebook notebook) =>
        notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => cell.Source);

    [Fact]
    public async Task AnEmptyNotebook_OffersBothCourses()
    {
        await using var notebook = await Notebook.OpenAsync();

        Assert.True(await notebook.EnabledAsync(TableCourseAction.Id));
        Assert.True(await notebook.EnabledAsync(SeriesCourseAction.Id));
    }

    [Fact]
    public async Task TheTwoCourseButtons_StandBesideTheOthers_AsPartsOfTheirOwn_WithTheirOwnWords()
    {
        await using var notebook = await Notebook.OpenAsync();
        var table = notebook.Host.GetToolbarActions().OfType<TableCourseAction>().Single();
        var series = notebook.Host.GetToolbarActions().OfType<SeriesCourseAction>().Single();

        Assert.Equal([ToolbarPlacement.MainToolbar, ToolbarPlacement.MainToolbar], [table.Placement, series.Placement]);
        Assert.Equal([TableCourseAction.Id, SeriesCourseAction.Id], [table.ActionId, series.ActionId]);
        Assert.Equal([table.ActionId, series.ActionId], [table.ExtensionId, series.ExtensionId]);
        Assert.Equal(["Course for a table", "Course for a series"], [table.DisplayName, series.DisplayName]);

        // After Run, Take over and Make blocks, in the order a person meets them.
        Assert.Equal([3, 4], [table.Order, series.Order]);

        foreach (var button in new CourseAction[] { table, series })
        {
            Assert.StartsWith("<svg", button.Icon, StringComparison.Ordinal);
            Assert.False(button.IconOnly);
            Assert.False(button.IsPrimary);
            Assert.Null(button.ConfirmationPrompt);
            Assert.False(string.IsNullOrWhiteSpace(button.Name));
            Assert.False(string.IsNullOrWhiteSpace(button.Description));
        }
    }

    [Fact]
    public async Task ADropBeforeTheSplit_DoesNotMakeTheCourseLeaveOutTheSplitAndTheAnswer()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock("""{"step": "read.csv", "path": "titanic.csv"}""");
        notebook.AddBlock("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");
        notebook.AddBlock("""{"step": "drop.columns", "columns": ["sibsp"]}""");

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.Equal(
            ["read.csv", "declare", "drop.columns", "settle.gaps", "feature.add", "scale.given", "split.stratified", "target", "fill.missing", "normalise", "evidence.report", "learn.network"],
            Verbs(notebook));
    }

    [Fact]
    public async Task PressingTheTableCourse_WritesEveryStepOfIt_AsItsSkeleton_InTheOrderTheyBelong()
    {
        await using var notebook = await Notebook.OpenAsync();

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.Equal(PipelineCourse.Table.Steps.Select(step => step.Verb), Verbs(notebook));
        Assert.Equal(PipelineCourse.Table.Steps.Select(step => step.AsBlockText(Catalog)), SourcesOfBlocks(notebook));
        Assert.All(notebook.Scaffold.Cells, cell => Assert.Equal(StepKernel.Language, cell.Language));
    }

    [Fact]
    public async Task PressingTheSeriesCourse_WritesTheSeriesSteps_WithTheGapTheCourseSays()
    {
        await using var notebook = await Notebook.OpenAsync();

        await notebook.PressAsync(SeriesCourseAction.Id);

        Assert.Equal(PipelineCourse.SeriesInTime.Steps.Select(step => step.Verb), Verbs(notebook));
        Assert.Contains("\"gap\": 1", notebook.Scaffold.Cells.Single(cell => StepText.Of(cell.Source).Verb == "split.byTime").Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheWholeCourse_IsWrittenInOneTurn_NotOneTurnForEachBlock()
    {
        // A turn publishes a few versions, however many blocks it writes; the course written a block at a time would publish
        // twice as many versions as it has blocks. How many of the engine's words one version gathers is a matter of timing,
        // so what is held is that the count does not grow with the blocks.
        await using var notebook = await Notebook.OpenAsync();
        var before = notebook.Opened.Current.Version;

        await notebook.PressAsync(TableCourseAction.Id);

        var published = notebook.Opened.Current.Version - before;

        Assert.Equal(PipelineCourse.Table.Steps.Count, notebook.Scaffold.Cells.Count);
        Assert.InRange(published, 1, PipelineCourse.Table.Steps.Count - 1);
    }

    [Fact]
    public async Task ABlockWaitingForSomething_ShowsItsOwnRefusal_AndNoBlockShowsARuleFault()
    {
        await using var notebook = await Notebook.OpenAsync();

        await notebook.PressAsync(TableCourseAction.Id);

        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        Assert.Equal(PipelineCourse.Table.Steps.Count, assembled.Blocks.Count);
        Assert.Empty(assembled.Readable.Steps);
        Assert.Contains("'path'", Assert.Single(assembled.Stopping), StringComparison.Ordinal);
        Assert.DoesNotContain(assembled.Blocks.SelectMany(block => block.Faults), fault => fault.Contains("no column of that name", StringComparison.Ordinal));
        Assert.All(assembled.Blocks.Take(10), block => Assert.NotEmpty(block.Faults));
    }

    [Fact]
    public async Task APressedCourse_IsNoLongerOffered_AndPressedAgainItWritesNothing()
    {
        await using var notebook = await Notebook.OpenAsync();

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.False(await notebook.EnabledAsync(TableCourseAction.Id));

        var before = notebook.Scaffold.Cells.Select(cell => (cell.Id, cell.Source)).ToArray();

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.Equal(before, notebook.Scaffold.Cells.Select(cell => (cell.Id, cell.Source)));
    }

    [Fact]
    public async Task ANotebookThatStartedTheCourse_GoesOnFromWhereItStands_AndNothingThatStoodChanges()
    {
        await using var notebook = await Notebook.OpenAsync();
        var read = notebook.AddBlock("""{"step": "read.csv", "path": "titanic.csv"}""");
        var declare = notebook.AddBlock("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");
        var before = new[] { (read.Id, read.Source), (declare.Id, declare.Source) };

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.Equal(PipelineCourse.Table.Steps.Select(step => step.Verb), Verbs(notebook));
        Assert.Equal(before, notebook.Scaffold.Cells.Take(2).Select(cell => (cell.Id, cell.Source)));
        Assert.Equal(PipelineCourse.Table.Steps.Skip(2).Select(step => step.AsBlockText(Catalog)), SourcesOfBlocks(notebook).Skip(2));
    }

    [Fact]
    public async Task TextAroundTheBlocks_StaysWhereItStands_AndTheCourseGoesRightAfterTheLastBlock()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.Scaffold.AddCell("markdown", source: "# The passenger list");
        notebook.AddBlock("""{"step": "read.csv", "path": "titanic.csv"}""");
        notebook.Scaffold.AddCell("markdown", source: "Notes at the end.");

        await notebook.PressAsync(TableCourseAction.Id);

        var kinds = notebook.Scaffold.Cells.Select(cell => cell.Type).ToArray();

        Assert.Equal("markdown", kinds[0]);
        Assert.Equal("markdown", kinds[^1]);
        Assert.Equal(PipelineCourse.Table.Steps.Count, kinds.Count(kind => kind == StepCellType.StepType));
        Assert.Equal("Notes at the end.", notebook.Scaffold.Cells[^1].Source);
    }

    [Fact]
    public async Task ASourceOfAnotherKind_StandsWhereTheCourseReadsACommaSeparatedFile()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock("""{"step": "read.json", "path": "titanic.json"}""");

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.DoesNotContain("read.csv", Verbs(notebook));
        Assert.Equal(PipelineCourse.Table.Steps.Count, Verbs(notebook).Count());
    }

    [Fact]
    public async Task ANotebookThatTookTheStepsInAnotherOrder_HasNoCourseOffered()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock("""{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""");
        notebook.AddBlock("""{"step": "read.csv", "path": "titanic.csv"}""");

        Assert.False(await notebook.EnabledAsync(TableCourseAction.Id));
        Assert.False(await notebook.EnabledAsync(SeriesCourseAction.Id));
    }

    [Fact]
    public async Task ASeriesNotebook_IsOfferedTheSeriesCourse_AndNotTheTable()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock("""{"step": "read.csv", "path": "apple.csv"}""");
        notebook.AddBlock("""{"step": "declare", "remainder": "drop", "columns": [{"name": "Date", "kind": "timestamp", "optional": false}]}""");
        notebook.AddBlock("""{"step": "order.by", "columns": ["Date"]}""");

        Assert.True(await notebook.EnabledAsync(SeriesCourseAction.Id));
        Assert.False(await notebook.EnabledAsync(TableCourseAction.Id));
    }

    [Fact]
    public async Task ANotebookHoldingTheWholeCourse_HasNothingToBeOffered()
    {
        await using var notebook = await Notebook.OpenAsync();

        await notebook.PressAsync(SeriesCourseAction.Id);

        Assert.False(await notebook.EnabledAsync(SeriesCourseAction.Id));
        Assert.False(await notebook.EnabledAsync(TableCourseAction.Id));
    }

    [Fact]
    public async Task ALayoutThatCannotAddABlock_HasNoCourseOffered_AndPressedAnywayItWritesNothing()
    {
        await using var notebook = await Notebook.OpenAsync();
        var layout = notebook.Host.GetLayouts().First(each => !each.Capabilities.HasFlag(LayoutCapabilities.CellInsert));
        notebook.Scaffold.NotebookOps.SetActiveLayout(layout.LayoutId);

        Assert.False(await notebook.EnabledAsync(TableCourseAction.Id));

        await notebook.PressAsync(TableCourseAction.Id);

        Assert.Empty(notebook.Scaffold.Cells);
    }

    [Fact]
    public async Task AFilledInCourseThatTheBlocksSay_RunsAsThePipelineItNames()
    {
        // A course written into a notebook is the pipeline once each block is filled in: here every block is written as the
        // step the filled course makes of it, and the toolbar's run of the whole pipeline reads the passenger list.
        await using var notebook = await Notebook.OpenAsync();
        await notebook.PressAsync(TableCourseAction.Id);

        var filled = Passengers();
        var cells = notebook.Scaffold.Cells.ToArray();

        for (var at = 0; at < cells.Length; at++)
        {
            cells[at].Source = filled.Steps[at].AsBlockText(Catalog);
        }

        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        Assert.True(assembled.Whole);
        Assert.Equal(PipelineCourse.Table.Steps.Count, assembled.Readable.Steps.Count);
    }
}
