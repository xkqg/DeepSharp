// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// New steps written into a notebook's blocks, whatever changed them: only the blocks whose steps changed are written,
/// each in its own place and keeping what the notebook holds about it; a step added goes after the block before it,
/// one taken away is removed, and every other block is left as it was — with its view. Steps the rules refuse are
/// not written at all, and the same steps again write nothing.
/// </summary>
public sealed class CommitTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-commit-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private const string PclassACategory =
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "category", "optional": false, "was": "integer"}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""";

    private const string FareByMinMax = """{"step": "normalise", "column": "fare", "scale": "minmax", "outOfRange": "pass"}""";

    private const string DropAge = """{"step": "drop.columns", "columns": ["age"]}""";

    public CommitTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static IReadOnlyList<IPipelineStep> Read(params string[] blocks) => [.. blocks.Select(block => StepCatalog.BuiltIn().ReadStep(block))];

    private static string[] Said(IReadOnlyList<IPipelineStep> before, IReadOnlyList<BlockChange> changes) =>
        [.. changes.Select(change => $"{change.Kind} {(change.To ?? before[change.From!.Value]).Verb}")];

    private static Gesture On(Notebook notebook, CellModel cell) => new(
        notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session,
        notebook.Scaffold.Notebook,
        notebook.Scaffold.NotebookOps,
        notebook.Scaffold.Variables,
        cell.Id,
        MayAddAndRemove: true);

    // ---- which blocks change

    [Fact]
    public void TheSameSteps_KeepEveryBlock()
    {
        var steps = Read(Titanic);

        Assert.All(StepCommit.Changes(steps, steps), change => Assert.Equal(BlockChangeKind.Kept, change.Kind));
    }

    [Fact]
    public void AStepChangedInItsPlace_IsRewrittenThere_AndNothingElseIs()
    {
        var before = Read(Titanic);

        Assert.Equal(
            ["Kept read.csv", "Kept declare", "Kept split.stratified", "Kept fill.missing", "Rewritten normalise"],
            Said(before, StepCommit.Changes(before, Read([.. Titanic[..4], FareByMinMax]))));
    }

    [Fact]
    public void AStepAdded_IsInsertedAfterTheStepBeforeIt()
    {
        var before = Read(Titanic);

        Assert.Equal(
            ["Kept read.csv", "Kept declare", "Kept split.stratified", "Kept fill.missing", "Inserted drop.columns", "Kept normalise"],
            Said(before, StepCommit.Changes(before, Read([.. Titanic[..4], DropAge, Titanic[4]]))));
    }

    [Fact]
    public void AStepTakenAway_IsRemoved()
    {
        var before = Read([.. Titanic[..4], DropAge, Titanic[4]]);

        Assert.Equal(
            ["Kept read.csv", "Kept declare", "Kept split.stratified", "Kept fill.missing", "Removed drop.columns", "Kept normalise"],
            Said(before, StepCommit.Changes(before, Read(Titanic))));
    }

    [Fact]
    public void TheSchemaAndTheOutputChanged_LeaveEveryStepBetweenThemAsItWas()
    {
        var before = Read([.. Titanic, """{"step": "target", "column": "survived"}"""]);
        var after = Read([Titanic[0], PclassACategory, .. Titanic[2..], """{"step": "target", "column": "pclass"}"""]);

        Assert.Equal(
            ["Kept read.csv", "Rewritten declare", "Kept split.stratified", "Kept fill.missing", "Kept normalise", "Rewritten target"],
            Said(before, StepCommit.Changes(before, after)));
    }

    [Fact]
    public void AStepAddedBeforeAStepRewritten_IsInserted_AndTheRewrittenStepStaysItsOwnBlock()
    {
        var before = Read(Titanic);

        Assert.Equal(
            ["Kept read.csv", "Kept declare", "Kept split.stratified", "Kept fill.missing", "Inserted drop.columns", "Rewritten normalise"],
            Said(before, StepCommit.Changes(before, Read([.. Titanic[..4], DropAge, FareByMinMax]))));
    }

    [Fact]
    public void EveryStepReplacedByOthers_IsRemoved_AndTheOthersInserted()
    {
        var before = Read(Titanic[0], Titanic[1], DropAge);
        var after = Read(Titanic[0], Titanic[1], Titanic[2]);

        Assert.Equal(["Kept read.csv", "Kept declare", "Removed drop.columns", "Inserted split.stratified"], Said(before, StepCommit.Changes(before, after)));
    }

    // ---- writing them into the notebook

    [Fact]
    public async Task CommittingTheStepsAsTheyAre_WritesNothing_AndRunsNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);
        Guid[] before = [.. notebook.Scaffold.Cells.Select(cell => cell.Id)];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        var changed = await StepCommit.CommitAsync(On(notebook, notebook.Scaffold.Cells[1]), assembled, assembled.Readable.Steps);

        Assert.False(changed);
        Assert.Equal(before, notebook.Scaffold.Cells.Select(cell => cell.Id));
        Assert.Empty(notebook.Pushed);
    }

    [Fact]
    public async Task ACommit_RewritesOnlyTheBlocksThatChanged_EachKeepingWhatTheNotebookHoldsAboutIt_AndRunsThem()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var cells = notebook.Scaffold.Cells;
        cells[1].Metadata["place"] = "left";
        cells[4].Metadata["place"] = "right";
        Guid[] before = [.. cells.Select(cell => cell.Id)];
        var assembled = NotebookPipeline.Of(cells);

        var changed = await StepCommit.CommitAsync(On(notebook, cells[1]), assembled, Read([Titanic[0], PclassACategory, .. Titanic[2..4], FareByMinMax]));

        // The scaffold hands out its cells as a copy, so the notebook is read again after the commit.
        cells = notebook.Scaffold.Cells;

        Assert.True(changed);
        Assert.Equal([before[0], before[2], before[3]], new[] { cells[0].Id, cells[2].Id, cells[3].Id });
        Assert.DoesNotContain(cells[1].Id, before);
        Assert.DoesNotContain(cells[4].Id, before);
        Assert.Equal("left", cells[1].Metadata["place"]);
        Assert.Equal("right", cells[4].Metadata["place"]);
        Assert.Contains(cells[1].Id, notebook.Pushed);
        Assert.Contains(cells[4].Id, notebook.Pushed);
        Assert.Equal(Read([Titanic[0], PclassACategory, .. Titanic[2..4], FareByMinMax]), NotebookPipeline.Of(cells).Readable.Steps);
    }

    [Fact]
    public async Task ACommitThatRemovesTheBlockItWasMadeOn_ShowsTheBlockBeforeIt()
    {
        await using var notebook = await NotebookAsync([.. Titanic[..4], DropAge, Titanic[4]]);
        var cells = notebook.Scaffold.Cells;
        var fill = cells[3];
        var assembled = NotebookPipeline.Of(cells);

        var changed = await StepCommit.CommitAsync(On(notebook, cells[4]), assembled, Read(Titanic));

        cells = notebook.Scaffold.Cells;

        Assert.True(changed);
        Assert.Equal(5, cells.Count);
        Assert.Equal(fill.Id, cells[3].Id);
        Assert.Contains(fill.Id, notebook.Pushed);
        Assert.Equal(Read(Titanic), NotebookPipeline.Of(cells).Readable.Steps);
    }

    [Fact]
    public async Task ACommitThatReplacesTheFirstBlock_PutsTheNewOneAtTheTop_AndShowsIt()
    {
        await using var notebook = await NotebookAsync(Titanic[0], Titanic[1]);
        var cells = notebook.Scaffold.Cells;
        var read = cells[0];
        var assembled = NotebookPipeline.Of(cells);

        var changed = await StepCommit.CommitAsync(On(notebook, read), assembled, Read("""{"step": "read.rows", "description": "rows"}""", Titanic[1]));

        cells = notebook.Scaffold.Cells;

        Assert.True(changed);
        Assert.Equal(["read.rows", "declare"], NotebookPipeline.Of(cells).Readable.Steps.Select(step => step.Verb));
        Assert.DoesNotContain(read.Id, cells.Select(cell => cell.Id));
        Assert.Contains(cells[0].Id, notebook.Pushed);
    }

    [Fact]
    public async Task ACommitTheRulesRefuse_WritesNothing_AndTheBlockSaysWhichRule()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var cells = notebook.Scaffold.Cells;
        string?[] before = [.. cells.Select(cell => cell.Source)];
        var assembled = NotebookPipeline.Of(cells);

        var changed = await StepCommit.CommitAsync(
            On(notebook, cells[1]), assembled, Read([.. Titanic[..4], """{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}"""]));

        Assert.False(changed);
        Assert.Equal(before, cells.Select(cell => cell.Source));
        Assert.Contains(cells[1].Outputs, output => output.IsError
            && output.Content.Contains("This change is not made", StringComparison.Ordinal)
            && output.Content.Contains("colour", StringComparison.Ordinal));
    }

    // ---- what a request says

    [Fact]
    public async Task ARequest_SaysWhatAskedForIt_AndWhetherTheBlocksMakeOnePipeline()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{ "step": "target", "column": """]);
        var cells = notebook.Scaffold.Cells;
        var halfTyped = NotebookPipeline.Of(cells);

        var run = halfTyped.RequestFor(cells[1].Id, ViewTrigger.Run, page: 0);
        var below = halfTyped.RequestFor(cells[5].Id, ViewTrigger.Show, page: 2);

        Assert.Equal(ViewTrigger.Run, run.Trigger);
        Assert.False(run.Whole);
        Assert.NotNull(run.Declaration);
        Assert.Null(below.Declaration);
        Assert.Equal(2, below.Page);
        Assert.NotEmpty(below.Faults);

        notebook.Scaffold.RemoveCell(cells[5].Id);

        Assert.True(NotebookPipeline.Of(notebook.Scaffold.Cells).RequestFor(cells[1].Id, ViewTrigger.Commit, page: 0).Whole);
        Assert.Empty(NotebookPipeline.Of(notebook.Scaffold.Cells).Stopping);
    }
}
