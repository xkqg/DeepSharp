// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook's session owns what only holds between calls: which block shows what, and whether a view there
/// still says what the blocks make; that one gesture on a notebook runs at a time, so a request is taken by the run
/// its own gesture started; and that a fit is handed on only for the steps the blocks declare.
/// </summary>
public sealed class SessionTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-session-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public SessionTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    [Fact]
    public void ForgettingStaleViews_ForgetsExactlyThoseWhoseStepsOrOffersChanged_AndSaysWhich()
    {
        CellModel[] cells = [.. Titanic.Select(Block)];
        var before = NotebookPipeline.Of(cells);
        var session = new NotebookSession();
        string[] sourceColumns = ["survived", "pclass", "sex", "age", "fare"];

        session.Showing(cells[0].Id, before.ViewKeyOf(cells[0].Id)!, DataGrid.HeaderOf(before.Readable, sourceColumns));
        session.Showing(cells[4].Id, before.ViewKeyOf(cells[4].Id)!, DataGrid.HeaderOf(before.Readable, ["survived", "pclass", "age", "fare", "age_was_missing"]));

        // The fill below the split changes: the source's rows are the same, and so are the offers their grid made.
        cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";

        Assert.Equal([cells[4].Id], session.ForgetStale(NotebookPipeline.Of(cells), except: null));
        Assert.Equal([cells[0].Id], session.Shown.Keys);

        // Age is dropped at the end: the source's rows are the same, but its grid offered to exclude age.
        cells[4].Source = """{"step": "drop.columns", "columns": ["age"]}""";

        Assert.Equal([cells[0].Id], session.ForgetStale(NotebookPipeline.Of(cells), except: null));
        Assert.Empty(session.Shown);

        // The block a gesture was made on is left to that gesture.
        session.Showing(cells[1].Id, "an old key", DataGrid.HeaderOf(before.Readable, []));

        Assert.Empty(session.ForgetStale(NotebookPipeline.Of(cells), except: cells[1].Id));
    }

    [Fact]
    public void ABlockAChangeTookAway_NoLongerStands_AndWhatTheSessionKnewOfItIsForgotten()
    {
        CellModel[] cells = [.. Titanic.Select(Block)];
        var assembled = NotebookPipeline.Of(cells);
        var session = new NotebookSession();
        var gone = cells[1];

        session.Showing(gone.Id, assembled.ViewKeyOf(gone.Id)!, DataGrid.HeaderOf(assembled.Readable, []));
        session.Refused(gone.Id, gone.Source, "the value is not one the step takes");

        Assert.True(session.Stands(gone.Id));

        session.Removed(gone.Id);

        Assert.False(session.Stands(gone.Id));
        Assert.True(session.Stands(cells[0].Id));
        Assert.DoesNotContain(gone.Id, session.Shown.Keys);
        Assert.Null(session.RefusalFor(gone.Id, gone.Source));
    }

    [Fact]
    public async Task GesturesOnOneNotebook_RunOneAtATime_InTheOrderTheyCame()
    {
        var session = new NotebookSession();
        var mayFinish = new TaskCompletionSource();
        var order = new List<string>();

        var first = session.OneAtATimeAsync(CancellationToken.None, async _ =>
        {
            order.Add("first begins");
            await mayFinish.Task;
            order.Add("first ends");

            return (string?)"first";
        });
        var second = session.OneAtATimeAsync(CancellationToken.None, _ =>
        {
            order.Add("second");

            return Task.FromResult<string?>("second");
        });

        Assert.Equal(["first begins"], order);
        Assert.False(second.IsCompleted);

        mayFinish.SetResult();

        Assert.Equal(["first", "second"], await Task.WhenAll(first, second));
        Assert.Equal(["first begins", "first ends", "second"], order);
    }

    [Fact]
    public async Task AGestureMadeWhileAnotherRuns_WaitsForIt_BeforeTouchingTheNotebook()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var declare = notebook.Scaffold.Cells[1];
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var mayFinish = new TaskCompletionSource();
        var running = session.OneAtATimeAsync(CancellationToken.None, async _ =>
        {
            await mayFinish.Task;

            return (string?)null;
        });

        var waiting = notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.False(waiting.IsCompleted);
        Assert.Empty(declare.Outputs);

        mayFinish.SetResult();
        await running;
        await waiting;

        Assert.True(declare.Outputs[^1].Content.Heads("fare"));
    }

    [Fact]
    public async Task AFitOfStepsTheBlocksNoLongerDeclare_IsNeverHandedOver()
    {
        // A run asked for one declaration, finished after the blocks became another: what it learned is not theirs.
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var last = notebook.Scaffold.Cells[^1];
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var asked = NotebookPipeline.Of(notebook.Scaffold.Cells);

        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        session.Publish(NotebookPipeline.Of(notebook.Scaffold.Cells));

        await session.OneAtATimeAsync(CancellationToken.None, async turn =>
        {
            await session.AskAsync(last.Id, asked.RequestFor(last.Id, ViewTrigger.Run, page: 0), turn, notebook.Scaffold.NotebookOps);

            return true;
        });

        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out var handed) && handed!.Contains("\"fitted\"", StringComparison.Ordinal));
        Assert.Equal(0, session.RunsFitted);
    }

    [Fact]
    public async Task AChangeAStopCameAfter_LeavesItsBlockNothing_AndNeverRunsIt()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var blocks = notebook.Host.GetCellTypes().OfType<StepCellType>().Single();
        var last = notebook.Scaffold.Cells[^1];
        var asked = NotebookPipeline.Of(notebook.Scaffold.Cells);

        await blocks.Session.OneAtATimeAsync(CancellationToken.None, async turn =>
        {
            // The stop comes after the change began, before it asks its block for anything.
            _ = blocks.StoppedAsync();
            await blocks.Session.AskAsync(last.Id, asked.RequestFor(last.Id, ViewTrigger.Show, page: 0), turn, notebook.Scaffold.NotebookOps);

            return true;
        });

        Assert.Empty(last.Outputs);
        Assert.Null(blocks.Session.Take(last.Id, blocks.Session.Enter(CancellationToken.None)));
    }

    [Fact]
    public async Task AFitThatEndsAfterAStop_IsNotHandedOver_ThoughTheBlocksStillDeclareIt()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var blocks = notebook.Host.GetCellTypes().OfType<StepCellType>().Single();
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);
        var fit = new Pipeline(assembled.Readable, rows: null, SourceFolder.Of(_folder)).Run();

        blocks.Session.Publish(assembled);

        // The turn of a change that began before the stop, and whose fit ends after it.
        var turn = await blocks.Session.OneAtATimeAsync(CancellationToken.None, turn => Task.FromResult(turn));
        await blocks.StoppedAsync();

        Assert.False(blocks.Session.HandOverFit(notebook.Scaffold.Variables, fit, "the bytes it read", turn, CancellationToken.None));
        Assert.Equal(0, blocks.Session.RunsFitted);
        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
    }

    [Fact]
    public async Task AnAskWhoseBlockIsRefused_LeavesNoRequestBehind()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var last = notebook.Scaffold.Cells[^1];
        var asked = NotebookPipeline.Of(notebook.Scaffold.Cells);

        // A look's operations refuse to run the block: the ask ends without it, and takes its request with it.
        await Assert.ThrowsAnyAsync<InvalidOperationException>(() => session.OneAtATimeAsync(CancellationToken.None, async turn =>
        {
            await session.AskAsync(last.Id, asked.RequestFor(last.Id, ViewTrigger.Show, page: 0), turn, notebook.ToolbarContext().Notebook);

            return true;
        }));

        Assert.Null(session.Take(last.Id, session.Enter(CancellationToken.None)));
    }

    [Fact]
    public async Task ABlocksRunLeftBehind_NeverTakesALaterChangesRequest()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var blocks = notebook.Host.GetCellTypes().OfType<StepCellType>().Single();
        var session = blocks.Session;
        var last = notebook.Scaffold.Cells[^1];
        var asked = NotebookPipeline.Of(notebook.Scaffold.Cells);
        var leftBehind = session.Enter(CancellationToken.None);
        ViewRequest? takenByTheRunLeftBehind = null;
        var looked = false;

        await blocks.StoppedAsync();

        // As the block begins for the change after the stop, the run a stop left behind looks for a request of its own.
        void LookAsItBegins(Guid cell)
        {
            if (cell == last.Id)
            {
                takenByTheRunLeftBehind = session.Take(last.Id, leftBehind);
                looked = true;
            }
        }

        notebook.Scaffold.OnCellExecuting += LookAsItBegins;

        await session.OneAtATimeAsync(CancellationToken.None, async turn =>
        {
            await session.AskAsync(last.Id, asked.RequestFor(last.Id, ViewTrigger.Show, page: 0), turn, notebook.Scaffold.NotebookOps);

            return true;
        });

        Assert.True(looked);
        Assert.Null(takenByTheRunLeftBehind);
        Assert.True(last.Outputs[^1].Content.Heads("fare"));
    }

    [Fact]
    public async Task AGestureAStopCameAfter_HandsNothingOver()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var blocks = notebook.Host.GetCellTypes().OfType<StepCellType>().Single();
        var session = blocks.Session;
        var last = notebook.Scaffold.Cells[^1];
        var assembled = NotebookPipeline.Of(notebook.Scaffold.Cells);

        await session.OneAtATimeAsync(CancellationToken.None, turn =>
        {
            // The stop comes after the gesture began, before it keeps the pipeline and hands it over.
            _ = blocks.StoppedAsync();

            var gesture = new Gesture(session, notebook.Scaffold.Notebook, notebook.Scaffold.NotebookOps, notebook.Scaffold.Variables, last.Id, MayAddAndRemove: true, turn);

            return StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Show, page: 0);
        });

        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
        Assert.Null(session.Assembled);
    }

    [Fact]
    public async Task AStopMadeWhileAWriteIsLetThrough_EndsOnlyOnceThatWriteLanded()
    {
        var session = new NotebookSession();
        Task? stopped = null;

        await session.OneAtATimeAsync(CancellationToken.None, async turn =>
        {
            // The stop comes while the write it was let through for is on its way: the stop waits for it.
            Assert.True(await session.LetThroughAsync(turn, () =>
            {
                stopped = session.Stopped();
                Assert.False(stopped.IsCompleted);

                return Task.CompletedTask;
            }));

            return true;
        });

        Assert.True(stopped!.IsCompletedSuccessfully);
    }

    [Fact]
    public async Task AWriteAfterAStop_IsNotLetThrough_AndTheStopWaitsForNothing()
    {
        var session = new NotebookSession();
        var written = false;
        var turn = await session.OneAtATimeAsync(CancellationToken.None, turn => Task.FromResult(turn));

        Assert.True(session.Stopped().IsCompletedSuccessfully);
        Assert.False(await session.LetThroughAsync(turn, () =>
        {
            written = true;

            return Task.CompletedTask;
        }));
        Assert.False(session.LetThrough(turn, () => written = true));
        Assert.False(written);
    }

    [Fact]
    public async Task AStopWaitsOnlyForWhatWasLetThroughBeforeIt()
    {
        var session = new NotebookSession();
        var mayLandBefore = new TaskCompletionSource();
        var mayLandAfter = new TaskCompletionSource();
        var letThroughAfter = new TaskCompletionSource();

        // A write of the first change is on its way when the stop comes: the change holds the notebook until it lands.
        var before = session.OneAtATimeAsync(CancellationToken.None, turn => session.LetThroughAsync(turn, () => mayLandBefore.Task));
        var stopped = session.Stopped();

        // The stop gave the notebook back: the next change begins, and a write of its own is let through and on its way too.
        var after = session.OneAtATimeAsync(CancellationToken.None, turn => session.LetThroughAsync(turn, () =>
        {
            letThroughAfter.SetResult();

            return mayLandAfter.Task;
        }));

        await letThroughAfter.Task;
        Assert.False(stopped.IsCompleted);

        mayLandBefore.SetResult();
        await before;

        // The stop ended with the write let through before it, and waited for none let through after it.
        Assert.True(stopped.IsCompletedSuccessfully);
        Assert.False(after.IsCompleted);

        mayLandAfter.SetResult();
        Assert.True(await after);
    }

    [Fact]
    public async Task AChangeWhoseRunWasStoppedBeforeItBegan_IsBornStopped_AndLetsTheNextChangeIn()
    {
        var session = new NotebookSession();
        var ran = false;
        using var stopped = new CancellationTokenSource();

        await stopped.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.OneAtATimeAsync(stopped.Token, _ =>
        {
            ran = true;

            return Task.FromResult(true);
        }));

        Assert.False(ran);
        Assert.True(await session.OneAtATimeAsync(CancellationToken.None, _ => Task.FromResult(true)));
    }
}
