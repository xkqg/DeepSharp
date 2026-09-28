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
/// A change — a click on a control a cell drew, a field of a cell's properties changed — runs DeepSharp's blocks as its
/// own; anything else it asks to run is a run of its own, told to every view from its ask, in the C# turn when it runs
/// C#, and stopped as any run is; once that run is stopped, nothing else the change asks is done. Parts are real ones,
/// loaded into the engine as any extension is.
/// </summary>
[Collection(RunsLeftBehind.Name)]
public sealed class ChangeTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-changes-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private string At(string name) => Path.Join(_folder, name);

    // A cell that says it began, waits until it is let go, and says it ended.
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

    private async Task<string> SaveAsync(string name, CellModel[] cells)
    {
        var path = At(name);
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return path;
    }

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells) =>
        await notebooks.OpenAsync(await SaveAsync(name, cells), TestContext.Current.CancellationToken);

    // A notebook of the given cells, opened on an engine that also carries the given part.
    private async Task<NotebookHost> OpenWithAsync(IExtension part, params CellModel[] cells)
    {
        var path = await SaveAsync("b.verso", cells);
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(part);

        return await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);
    }

    private static HostedGesture Click(NotebookHost host) => new(host.Cells[0].Id, ClickingPart.Id, "act", "");

    // A click that runs the notebook's first cell.
    private static ClickingPart RunsTheFirstCell() => new(async context =>
    {
        await context.Notebook!.ExecuteCellAsync(context.NotebookModel!.Cells[0].Id);

        return "ran";
    });

    [Fact]
    public async Task AClicksCSharpCell_IsARunToldFromItsAsk_InTheCSharpTurn_AndStoppable()
    {
        // Notebook A's C# cell takes the process's C# turn until it is let go.
        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(Held("a")));
        var holding = a.RunAsync(a.Cells[0].Id);

        await UntilAsync(() => File.Exists(At("a-began")), "notebook A's run never began");

        var b = await OpenWithAsync(RunsTheFirstCell(), CSharp(Held("b")));

        try
        {
            var clicking = b.GestureAsync(Click(b));

            // The click's C# cell is a run from its ask: told to every view, waiting its turn among the process's C# runs.
            await UntilAsync(() => b.Current.Running is { Waits: true }, "the click's run was never told waiting for the C# turn");

            var run = b.Current.Running!.Value;

            Assert.Equal(b.Cells[0].Id, run.Cell);
            Assert.True(b.Stop(run.Number));

            // Stopped while it waited, the cell never ran, and the click ended as a stopped run ends.
            var clicked = await clicking.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.Null(clicked.Answer);
            Assert.Null(b.Current.Running);
            Assert.False(File.Exists(At("b-began")));
        }
        finally
        {
            await b.CloseAsync();
            await LetGoAsync("a");
            await holding;
        }
    }

    [Fact]
    public async Task APropertyChangeRunningCSharp_RunsItAsARun()
    {
        var b = await OpenWithAsync(new ChangingPanel((cell, context) => context.Notebook.ExecuteCellAsync(cell.Id)), CSharp(Held("b")));

        try
        {
            var changing = b.SetPropertyAsync(b.Cells[0].Id, ChangingPanel.Id, "field", "value");

            await UntilAsync(() => File.Exists(At("b-began")), "the change's cell never began");

            // Its C# cell runs as a run: told with the cell, and stopped as any run is.
            var run = b.Running!.Value;

            Assert.Equal(b.Cells[0].Id, run.Cell);
            Assert.False(run.Waits);
            Assert.True(b.Stop(run.Number));
            await changing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            Assert.Null(b.Running);
        }
        finally
        {
            await LetGoAsync("b");
            await b.CloseAsync();
        }
    }

    [Fact]
    public async Task AStoppedClick_IsRefusedWhatItAsksAfter()
    {
        var refused = new ConcurrentQueue<string>();
        var b = await OpenWithAsync(new ClickingPart(async context =>
        {
            try
            {
                await context.Notebook!.ExecuteCellAsync(context.NotebookModel!.Cells[0].Id);
            }
            catch (OperationCanceledException)
            {
                refused.Enqueue("run");
            }

            // What it asks once its run was stopped is refused, and the part goes on to its end.
            try
            {
                await context.Notebook!.InsertCellAsync(0, "markdown");
            }
            catch (OperationCanceledException)
            {
                refused.Enqueue("insert");
            }

            return "went on";
        }), CSharp(Held("b")));

        try
        {
            var clicking = b.GestureAsync(Click(b));

            await UntilAsync(() => File.Exists(At("b-began")), "the click's cell never began");
            Assert.True(b.Stop(b.Running!.Value.Number));

            var clicked = await clicking.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.Equal(["run", "insert"], refused);
            Assert.Equal("went on", clicked.Answer);
            Assert.Single(b.Cells);
        }
        finally
        {
            await LetGoAsync("b");
            await b.CloseAsync();
        }
    }

    [Fact]
    public async Task ClosingDuringAClicksRun_DoesNotWait()
    {
        var b = await OpenWithAsync(RunsTheFirstCell(), CSharp(Held("b")));
        var clicking = b.GestureAsync(Click(b));

        await UntilAsync(() => File.Exists(At("b-began")), "the click's cell never began");

        // A close stops the click's run as it stops any run, and waits only for the click to end.
        await b.CloseAsync().AsTask().WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await clicking.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await LetGoAsync("b");
    }

    [Fact]
    public async Task AClickAskingTwoRuns_RunsThemOneAtATime()
    {
        var order = new ConcurrentQueue<string>();
        var b = await OpenWithAsync(
            new ClickingPart(context =>
            {
                var cells = context.NotebookModel!.Cells;

                // Two runs asked at once, and neither awaited.
                _ = context.Notebook!.ExecuteCellAsync(cells[0].Id);
                _ = context.Notebook.ExecuteCellAsync(cells[1].Id);

                return Task.FromResult<string?>(null);
            }),
            CSharp("System.Threading.Thread.Sleep(300);"),
            CSharp("1 + 1"));

        int Index(Guid cell) => b.Scaffold.Cells.ToList().FindIndex(each => each.Id == cell);

        try
        {
            b.Scaffold.OnCellExecuting += cell => order.Enqueue($"{Index(cell)} begins as run {b.Running?.Number}");
            b.Scaffold.OnCellExecuted += cell => order.Enqueue($"{Index(cell)} ends");

            await b.GestureAsync(Click(b)).WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // The click's turn ended only once both had run, one after the other, each a run of its own.
            Assert.Equal(["0 begins as run 1", "0 ends", "1 begins as run 2", "1 ends"], order);
        }
        finally
        {
            await b.CloseAsync();
        }
    }

    [Fact]
    public async Task AClickRunsBlocksAsItsOwn_WithNoRun()
    {
        File.Copy(Repository.Data("titanic.csv"), At("titanic.csv"));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "titanic.verso",
            Block("""{"step": "read.csv", "path": "titanic.csv"}"""),
            Block("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}]}"""),
            Block("""{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}"""));
        var told = new ConcurrentQueue<HostedRun?>();

        host.Scaffold.OnCellExecuting += _ => told.Enqueue(host.Running);

        // A block's Show runs the block: the click's own, which no run owns and no stop reaches.
        await host.GestureAsync(new HostedGesture(host.Cells[2].Id, StepRenderer.Id, "deepsharp.show", ""));

        Assert.NotEmpty(told);
        Assert.All(told, run => Assert.Null(run));
    }
}
