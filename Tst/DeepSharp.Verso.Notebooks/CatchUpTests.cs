// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A notebook's blocks can change where no gesture sees it: a host inserts, deletes or moves a cell, turns one into
/// another kind, or writes what a person typed. What was worked out from the blocks as they were — a grid, the pipeline
/// handed to C# cells with what a run learned — then no longer holds, and the notebook takes it back as soon as it is
/// told: at once, by a host that changes cells itself, and at the next gesture otherwise, which is when Verso's own
/// editors let it know. A stop is told too: what the stopped run would still write is never written, and the notebook
/// takes its next change at once.
/// </summary>
public sealed class CatchUpTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-catchup-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public CatchUpTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string NotebookPath => Path.Join(_folder, "titanic.verso");

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(NotebookPath);

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    // The block type Verso loaded, which keeps the notebook's session: found the way the notebook's parts find it.
    private static StepCellType Blocks(Notebook notebook) =>
        ((IExtensionHostContext)notebook.Host).GetLoadedExtensions().OfType<StepCellType>().Single();

    private static string? HandedOver(Notebook notebook) =>
        notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out var pipeline) ? pipeline : null;

    private static bool ShowsAGrid(CellModel cell) => cell.Outputs.Any(output => output.Content.Heads("fare"));

    // Tells the notebook its cells changed, as a host that changed them itself does.
    private static Task TellAsync(Notebook notebook) =>
        Blocks(notebook).BlocksChangedAsync(notebook.Scaffold.Notebook, notebook.Scaffold.Variables, notebook.Scaffold.NotebookOps);

    [Fact]
    public async Task ABlockTakenAwayAfterTheRun_TakesBackTheFitHandedToCSharpCells_OnceTheNotebookIsTold()
    {
        await using var notebook = await NotebookAsync(Titanic);
        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Contains("\"fitted\"", HandedOver(notebook), StringComparison.Ordinal);

        // Through the engine's own port, as a host takes a cell away.
        await notebook.Scaffold.NotebookOps.RemoveCellAsync(notebook.Scaffold.Cells[3].Id);
        await TellAsync(notebook);

        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ACellThatIsNoBlock_AddedAfterTheRun_LeavesWhatWasHandedOver_SinceTheBlocksStillMakeIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        await notebook.PressAsync(RunPipelineAction.Id);
        var handed = HandedOver(notebook);

        await notebook.Scaffold.NotebookOps.InsertCellAsync(0, "markdown", null!);
        await TellAsync(notebook);

        Assert.Equal(handed, HandedOver(notebook));
    }

    [Fact]
    public async Task TypingIntoABlockThroughTheHost_AfterTheRun_TakesBackTheFitOfTheStepItWas()
    {
        await using var notebook = await NotebookAsync(Titanic);
        await notebook.PressAsync(RunPipelineAction.Id);

        await notebook.Opened.EditAsync(notebook.Scaffold.Cells[3].Id, """{"step": "fill.missing", "column": "age", "with": "mean"}""");

        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ABlockTurnedIntoACSharpCell_KeepsWhatItPrinted_WhenAGridElsewhereChanges_AndSavesIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        Assert.True(ShowsAGrid(normalise));

        // As Verso's own editors turn a cell into another kind: in place, under the same id, its outputs cleared — and
        // telling no part of it.
        normalise.Type = "code";
        normalise.Language = "csharp";
        normalise.Outputs.Clear();
        normalise.Source = """Console.WriteLine("printed by the fifth cell");""";
        await notebook.RunAsync(normalise);

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);
        await notebook.Opened.SaveAsync();

        Assert.Contains(normalise.Outputs, output => output.Content.Contains("printed by the fifth cell", StringComparison.Ordinal));
        Assert.Contains("printed by the fifth cell", await File.ReadAllTextAsync(NotebookPath, TestContext.Current.CancellationToken), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABlockTurnedIntoACSharpCell_KeepsWhatItPrinted_WhenAFormChangesABlockAboveIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        normalise.Type = "code";
        normalise.Language = "csharp";
        normalise.Outputs.Clear();
        normalise.Source = """Console.WriteLine("printed by the fifth cell");""";
        await notebook.RunAsync(normalise);

        await notebook.Opened.SetPropertyAsync(notebook.Scaffold.Cells[3].Id, StepForm.Id, "with", "mean");

        Assert.Contains("\"mean\"", notebook.Scaffold.Cells[3].Source, StringComparison.Ordinal);
        Assert.Contains(normalise.Outputs, output => output.Content.Contains("printed by the fifth cell", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AShowAfterABlockNoGestureSawWasAdded_ClearsTheGridTheBlocksNoLongerMake()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var normalise = notebook.Scaffold.Cells[4];

        await notebook.GestureAsync(normalise, StepRenderer.Show);
        Assert.True(ShowsAGrid(normalise));

        // As Verso's own editors add a block: through the engine, telling no part of it.
        notebook.Scaffold.InsertCell(4, StepCellType.StepType, StepKernel.Language, """{"step": "drop.columns", "columns": ["pclass"]}""");
        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);

        Assert.False(ShowsAGrid(normalise));
        Assert.DoesNotContain(normalise.Id, Blocks(notebook).Session.Shown.Keys);
    }

    [Fact]
    public async Task TheToolbarsRunAfterABlockNoGestureSawWasAdded_ClearsTheGridTheBlocksNoLongerMake()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var fill = notebook.Scaffold.Cells[3];

        await notebook.GestureAsync(fill, StepRenderer.Show);
        Assert.True(ShowsAGrid(fill));

        // A block added above it in Verso's own editor, which tells no part of it; then the toolbar's run.
        notebook.Scaffold.InsertCell(2, StepCellType.StepType, StepKernel.Language, """{"step": "drop.columns", "columns": ["pclass"]}""");
        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.False(ShowsAGrid(fill));
        Assert.True(ShowsAGrid(notebook.Scaffold.Cells[^1]));
    }

    // Asks a block to show something in a turn of its own, as the toolbar's run or a gesture does, and stops the run the
    // moment the engine begins the block — as a person who pressed stop while it worked. The block works it all out, as
    // a run left behind does, and must write none of it.
    private static async Task AskAndStopAsync(Notebook notebook, CellModel block, ViewRequest request)
    {
        var blocks = Blocks(notebook);

        void StopAsItBegins(Guid cell)
        {
            if (cell == block.Id)
            {
                blocks.Stopped();
            }
        }

        notebook.Scaffold.OnCellExecuting += StopAsItBegins;

        try
        {
            await blocks.Session.OneAtATimeAsync(async turn =>
            {
                await blocks.Session.AskAsync(block.Id, request, turn, notebook.Scaffold.NotebookOps);

                return true;
            });
        }
        finally
        {
            notebook.Scaffold.OnCellExecuting -= StopAsItBegins;
        }
    }

    [Fact]
    public async Task ARunStoppedWhileItsBlockWorks_HandsNothingOver_AndShowsNothingMore()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var blocks = Blocks(notebook);
        var last = notebook.Scaffold.Cells[^1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        blocks.Session.Publish(assembled);
        await AskAndStopAsync(notebook, last, assembled.RequestFor(last.Id, ViewTrigger.Run, page: 0));

        Assert.Null(HandedOver(notebook));
        Assert.False(ShowsAGrid(last));
        Assert.Equal(0, blocks.Session.RunsFitted);
        Assert.DoesNotContain(last.Id, blocks.Session.Shown.Keys);
    }

    [Fact]
    public async Task AShowStoppedWhileItsBlockWorks_HandsNotEvenTheDeclarationOver()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var blocks = Blocks(notebook);
        var last = notebook.Scaffold.Cells[^1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        blocks.Session.Publish(assembled);
        await AskAndStopAsync(notebook, last, assembled.RequestFor(last.Id, ViewTrigger.Show, page: 0));

        Assert.Null(HandedOver(notebook));
        Assert.False(ShowsAGrid(last));
    }

    [Fact]
    public async Task AStoppedRunWhoseRowsAreGone_SaysNothingOfIt_AndHandsNothingOver()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var blocks = Blocks(notebook);
        var last = notebook.Scaffold.Cells[^1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        File.Delete(Path.Join(_folder, "titanic.csv"));
        blocks.Session.Publish(assembled);
        await AskAndStopAsync(notebook, last, assembled.RequestFor(last.Id, ViewTrigger.Show, page: 0));

        // The block's own card, written as it began, and nothing the stopped run met.
        Assert.Single(last.Outputs);
        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task AListStoppedWhileItsBlockWorks_IsNeitherDrawnNorSaved()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var blocks = Blocks(notebook);
        var declare = notebook.Scaffold.Cells[1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        blocks.Session.Publish(assembled);
        await AskAndStopAsync(notebook, declare, assembled.RequestFor(declare.Id, ViewTrigger.Show, page: 0) with { List = ListPicks.None });

        Assert.Single(declare.Outputs);
        Assert.False(File.Exists(Path.Join(_folder, "titanic.columns.json")));
    }

    [Fact]
    public async Task AStoppedListWhoseRowsAreGone_SaysNothingOfIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var blocks = Blocks(notebook);
        var declare = notebook.Scaffold.Cells[1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        File.Delete(Path.Join(_folder, "titanic.csv"));
        blocks.Session.Publish(assembled);
        await AskAndStopAsync(notebook, declare, assembled.RequestFor(declare.Id, ViewTrigger.Show, page: 0) with { List = ListPicks.None });

        Assert.Single(declare.Outputs);
    }

    [Fact]
    public async Task AChangeThatNeverEnds_HoldsTheNotebookUntilAStop_AndTheNextChangeRunsAtOnceAfterIt()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var session = Blocks(notebook).Session;
        var never = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        var held = session.OneAtATimeAsync(_ => never.Task);
        var next = session.OneAtATimeAsync(_ => Task.FromResult(true));

        Assert.False(next.IsCompleted);

        Blocks(notebook).Stopped();

        Assert.True(await next.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
        Assert.False(held.IsCompleted);

        // The change that was stopped still ends by itself, later, and holds nothing then.
        never.SetResult(false);
        Assert.False(await held);
    }

    [Fact]
    public async Task TheToolbarsRunStoppedThroughTheHost_WritesNothing_WhenItEndsByItselfLater()
    {
        // The passengers many times over: a run that works on long after it is stopped as its block begins.
        var lines = await File.ReadAllLinesAsync(Path.Join(_folder, "titanic.csv"), TestContext.Current.CancellationToken);
        await File.WriteAllLinesAsync(Path.Join(_folder, "titanic.csv"), [lines[0], .. Enumerable.Repeat(lines[1..], 400).SelectMany(rows => rows)], TestContext.Current.CancellationToken);
        await using var notebook = await NotebookAsync(Titanic);
        var last = notebook.Scaffold.Cells[^1];
        var ended = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        notebook.Scaffold.OnCellExecuting += cell =>
        {
            if (cell == last.Id)
            {
                notebook.Opened.Stop();
            }
        };
        notebook.Scaffold.OnCellExecuted += cell =>
        {
            if (cell == last.Id)
            {
                ended.TrySetResult();
            }
        };

        await notebook.PressAsync(RunPipelineAction.Id);
        await ended.Task.WaitAsync(TimeSpan.FromMinutes(2), TestContext.Current.CancellationToken);

        Assert.False(ShowsAGrid(last));
        Assert.Null(HandedOver(notebook));
        Assert.Equal(0, Blocks(notebook).Session.RunsFitted);
    }
}
