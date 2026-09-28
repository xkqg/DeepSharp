// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What a stop leaves behind runs on with no run: its kernel was started afresh and the C# turn handed on, but the code
/// that did not stop goes on in the background until it ends by itself. Every version tells it, beside whatever the engine
/// runs that no run owns — a block a click runs — each as the engine says it began and ended, so neither overwrites the
/// other; and the grace, which closes a notebook nobody uses, never closes over it.
/// </summary>
[Collection(RunsLeftBehind.Name)]
public sealed class LeftBehindTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-left-behind-").FullName;

    public LeftBehindTests() => File.Copy(Repository.Data("titanic.csv"), At("titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    // A cell that says it began, goes on until it is let go, and says it ended — however it is stopped.
    private string Held(string name) =>
        $$"""System.IO.File.WriteAllText(@"{{At(name + "-began")}}", "on"); while (!System.IO.File.Exists(@"{{At(name + "-go")}}")) { await System.Threading.Tasks.Task.Delay(10); } System.IO.File.WriteAllText(@"{{At(name + "-ended")}}", "on");""";

    private Task LetGoAsync(string name) => File.WriteAllTextAsync(At(name + "-go"), "go", TestContext.Current.CancellationToken);

    private static async Task UntilAsync(Func<bool> holds, string what)
    {
        for (var waited = 0; !holds(); waited += 20)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    private async Task<string> WriteAsync(string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return At(name);
    }

    // Stops a run once its cell began, and waits for the run to end: what it ran is left behind.
    private async Task<HostedRun> StopOnceBegunAsync(NotebookHost host, Task running, string name)
    {
        await UntilAsync(() => File.Exists(At(name + "-began")), "the cell never began");

        var run = host.Running!.Value;

        Assert.True(host.Stop(run.Number));
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        return run;
    }

    [Fact]
    public async Task ACellAStopLeftBehind_IsToldRunningWithNoRun_UntilTheEngineSaysItEnded()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(await WriteAsync("held.verso", CSharp(Held("a"))), TestContext.Current.CancellationToken);
        var cell = host.Cells[0].Id;
        var whileItsKernelStartsAfresh = new ConcurrentQueue<Told>();

        host.Scaffold.OnKernelRestarting += _ => whileItsKernelStartsAfresh.Enqueue(new Told(host.Running, host.Executing));

        var stopped = await StopOnceBegunAsync(host, host.RunAsync(cell), "a");

        // Until its turn ends, the stopped run is the run under way, and nothing is told as left behind.
        var restarting = Assert.Single(whileItsKernelStartsAfresh);

        Assert.Equal(stopped.Number, restarting.Running?.Number);
        Assert.Empty(restarting.Executing);

        // Then no run is under way, and the cell runs on, as it began.
        Assert.Null(host.Current.Running);
        Assert.Equal([new HostedExecution(cell, stopped.Since, LeftBehind: true)], host.Current.Executing);

        await LetGoAsync("a");
        await UntilAsync(() => File.Exists(At("a-ended")), "the cell left behind never ended");
        await UntilAsync(() => host.Current.Executing.Count == 0, "the cell left behind was still told running after it ended");
    }

    [Fact]
    public async Task ACellLeftBehind_AndABlockAClickRuns_AreBothTold_AndTheBlocksEndLeavesTheOther()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(
            await WriteAsync(
                "both.verso",
                CSharp(Held("a")),
                Block("""{"step": "read.csv", "path": "titanic.csv"}"""),
                Block("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""")),
            TestContext.Current.CancellationToken);
        var held = host.Cells[0].Id;
        var block = host.Cells[2].Id;
        var atTheBlocksBegin = new ConcurrentQueue<IReadOnlyList<HostedExecution>>();
        var atTheBlocksEnd = new ConcurrentQueue<IReadOnlyList<HostedExecution>>();

        await StopOnceBegunAsync(host, host.RunAsync(held), "a");

        try
        {
            host.Scaffold.OnCellExecuting += cell =>
            {
                if (cell == block)
                {
                    atTheBlocksBegin.Enqueue(host.Executing);
                }
            };
            host.Scaffold.OnCellExecuted += cell =>
            {
                if (cell == block)
                {
                    atTheBlocksEnd.Enqueue(host.Executing);
                }
            };

            // A block's Show runs the block as the click's own, while the cell left behind still runs.
            await host.GestureAsync(new HostedGesture(block, StepRenderer.Id, "deepsharp.show", "")).WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.Equal([new Shown(held, LeftBehind: true), new Shown(block, LeftBehind: false)], Assert.Single(atTheBlocksBegin).Select(Shown.Of));
            Assert.Equal([new Shown(held, LeftBehind: true)], Assert.Single(atTheBlocksEnd).Select(Shown.Of));
            Assert.Equal([new Shown(held, LeftBehind: true)], host.Current.Executing.Select(Shown.Of));
        }
        finally
        {
            await LetGoAsync("a");
        }

        await UntilAsync(() => host.Current.Executing.Count == 0, "the cell left behind was still told running after it ended");
    }

    [Fact]
    public async Task CodeAButtonRanInNoCell_LeftBehindByAStop_IsToldUntilItEnds()
    {
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new CodeButton(Held("c")));

        var host = await NotebookHost.OpenAsync(await WriteAsync("code.verso", CSharp("1 + 1")), engine, TestContext.Current.CancellationToken);

        try
        {
            var stopped = await StopOnceBegunAsync(host, host.RunToolbarAsync(CodeButton.Id), "c");

            // Code in no cell, of which the engine says nothing: its run says what it runs, and the stop left it running.
            Assert.Null(host.Current.Running);
            Assert.Equal([new HostedExecution(null, stopped.Since, LeftBehind: true)], host.Current.Executing);

            await LetGoAsync("c");
            await UntilAsync(() => File.Exists(At("c-ended")), "the code left behind never ended");
            await UntilAsync(() => host.Current.Executing.Count == 0, "the code left behind was still told running after it ended");
        }
        finally
        {
            await LetGoAsync("c");
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ANotebookClosedWhileCodeAStopLeftBehindRuns_IsToldNothingMoreOnceThatCodeEnds()
    {
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new CodeButton(Held("d")));

        var host = await NotebookHost.OpenAsync(await WriteAsync("closed.verso", CSharp("1 + 1")), engine, TestContext.Current.CancellationToken);

        try
        {
            await StopOnceBegunAsync(host, host.RunToolbarAsync(CodeButton.Id), "d");
        }
        finally
        {
            await host.CloseAsync();
        }

        var atTheClose = host.Current;

        await LetGoAsync("d");
        await UntilAsync(() => File.Exists(At("d-ended")), "the code left behind never ended");

        // Nothing is published once the notebook is closed: the notebook stays as its close left it, however long one looks.
        for (var looked = 0; looked < 1000; looked += 20)
        {
            Assert.Equal(atTheClose, host.Current);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task TheGraceClose_DoesNotCloseOverACellAStopLeftBehind_AndClosesOnceItEnds()
    {
        var began = At("b-began");
        var go = At("b-go");
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new KernelledType(began, go));

        var host = await NotebookHost.OpenAsync(
            await WriteAsync("kernelled.verso", new CellModel { Type = "deepsharp.tests.kernelled", Source = "held" }),
            engine,
            TestContext.Current.CancellationToken);
        var closed = false;

        try
        {
            await StopOnceBegunAsync(host, host.RunAsync(host.Cells[0].Id), "b");

            // Saved, seen by nobody, nothing asked of it: only what the stop left behind keeps it open.
            Assert.False(host.Current.Unsaved);
            Assert.True(await host.StaysOpenAsync(TimeSpan.Zero, () => { }));

            await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
            await UntilAsync(() => host.Current.Executing.Count == 0, "the cell left behind was still told running after it ended");

            closed = !await host.StaysOpenAsync(TimeSpan.Zero, () => { });

            Assert.True(closed);
        }
        finally
        {
            if (!closed)
            {
                await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
                await host.CloseAsync();
            }
        }
    }

    // What the host told of what runs at one moment: the run under way, and what the engine runs that no run owns.
    private readonly record struct Told(HostedRun? Running, IReadOnlyList<HostedExecution> Executing);

    // What is told running with no run, but for when it began.
    private readonly record struct Shown(Guid? Cell, bool LeftBehind)
    {
        public static Shown Of(HostedExecution execution) => new(execution.Cell, execution.LeftBehind);
    }
}
