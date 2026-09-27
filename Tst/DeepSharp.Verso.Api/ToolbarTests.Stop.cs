// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Contexts;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

// A button that runs cells is one run, stopped cell by cell: the cell under way is left behind, no cell the button would
// still run begins, and nothing else it asks of the notebook is done. Which cells begin is what the engine itself says as
// it begins one, before it looks at whether the run was stopped.
[Collection(RunsLeftBehind.Name)]
public sealed partial class ToolbarTests
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    // A cell that says it began, waits until it is let go, and says it ended.
    private string Held(string name) =>
        $$"""System.IO.File.WriteAllText(@"{{Path.Join(_folder, name + "-began")}}", "on"); while (!System.IO.File.Exists(@"{{Path.Join(_folder, name + "-go")}}")) { await System.Threading.Tasks.Task.Delay(10); } System.IO.File.WriteAllText(@"{{Path.Join(_folder, name + "-ended")}}", "on");""";

    private async Task UntilAsync(string name, string what)
    {
        for (var waited = 0; !File.Exists(Path.Join(_folder, name)); waited += 20)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    // The cells the engine begins, in the order it begins them.
    private static ConcurrentQueue<Guid> Begun(NotebookHost host)
    {
        var begun = new ConcurrentQueue<Guid>();

        host.Scaffold.OnCellExecuting += begun.Enqueue;

        return begun;
    }

    // Lets the cell held behind go, waits until it ended, and gives the press's loop the moment it would take to begin the
    // next cell.
    private async Task LetGoAsync(string name)
    {
        await File.WriteAllTextAsync(Path.Join(_folder, name + "-go"), "go", TestContext.Current.CancellationToken);
        await UntilAsync(name + "-ended", "the cell left behind never ended");
        await Task.Delay(TimeSpan.FromSeconds(1), TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task StoppingARunAll_BeginsNoFurtherCell()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "all.verso", CSharp(Held("first")), CSharp("var second = 2;"));
        var begun = Begun(host);
        var pressing = host.RunToolbarAsync("verso.action.run-all");

        await UntilAsync("first-began", "Run All's first cell never began");

        Assert.True(host.Stop(1));
        await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await LetGoAsync("first");

        Assert.Equal([host.Cells[0].Id], begun);
    }

    [Fact]
    public async Task StoppingRunCellForSeveralCells_BeginsNoFurtherCell()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "several.verso", CSharp(Held("first")), CSharp("var second = 2;"));
        var begun = Begun(host);
        var pressing = host.RunToolbarAsync("verso.action.run-cell", host.Cells[0].Id, host.Cells[1].Id);

        await UntilAsync("first-began", "the first of the cells never began");

        Assert.True(host.Stop(1));
        await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await LetGoAsync("first");

        Assert.Equal([host.Cells[0].Id], begun);
    }

    [Fact]
    public async Task AStopAsRunAllResets_BeginsNoCell()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "reset.verso", CSharp("var first = 1;"), CSharp("var second = 2;"));
        var begun = Begun(host);
        var stopped = false;

        // Run All clears the notebook's variables as it resets its kernels, before it begins a cell: the stop lands then.
        ((VariableStore)host.Scaffold.Variables).OnVariablesChanged += () =>
        {
            if (!stopped && begun.IsEmpty)
            {
                stopped = host.Stop(1);
            }
        };

        await host.RunToolbarAsync("verso.action.run-all").WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await Task.Delay(TimeSpan.FromSeconds(1), TestContext.Current.CancellationToken);

        Assert.True(stopped);
        Assert.Empty(begun);
    }

    [Fact]
    public async Task AStoppedButton_IsRefusedWhatItAsksAfter()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "refused.verso", CSharp("6 * 7"), CSharp("var other = 1;"));
        var cell = host.Cells[0].Id;

        await host.RunAsync(cell);

        var run = new Run(2, cell: null, takesTheCSharpTurn: true);

        run.Stop();

        var stopped = new RunOperations(host.Scaffold, run);
        var layout = stopped.ActiveLayoutId;
        var theme = stopped.ActiveThemeId;
        var begun = Begun(host);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.ExecuteCellAsync(cell));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(stopped.ExecuteAllAsync);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.ExecuteFromAsync(cell));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.ExecuteCodeAsync("var sent = 1;", "csharp", TestContext.Current.CancellationToken));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.ExecuteCodeCaptureOutputsAsync("1 + 1", "csharp", TestContext.Current.CancellationToken));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.ClearOutputAsync(cell));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(stopped.ClearAllOutputsAsync);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.RestartKernelAsync("csharp"));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.InsertCellAsync(0, "markdown"));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.RemoveCellAsync(cell));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stopped.MoveCellAsync(cell, 1));
        Assert.ThrowsAny<OperationCanceledException>(() => stopped.SetActiveLayout("dashboard"));
        Assert.ThrowsAny<OperationCanceledException>(() => stopped.SetActiveTheme("verso-dark"));

        // Nothing was done: no cell began, none came, went or moved, the cell still shows what it showed, and the layout
        // and the theme are as they were.
        Assert.Empty(begun);
        Assert.Equal([cell, host.Cells[1].Id], host.Scaffold.Cells.Select(each => each.Id));
        Assert.NotEmpty(host.Scaffold.Cells[0].Outputs);
        Assert.Equal(layout, host.Scaffold.NotebookOps.ActiveLayoutId);
        Assert.Equal(theme, host.Scaffold.NotebookOps.ActiveThemeId);
    }

    [Fact]
    public async Task ARunsOperations_DoWhatTheNotebooksOwnDo_WhileTheRunIsNotStopped()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "port.verso", CSharp("6 * 7"), CSharp("var second = 2;"));
        var first = host.Cells[0].Id;
        var second = host.Cells[1].Id;
        var port = new RunOperations(host.Scaffold, new Run(1, cell: null, takesTheCSharpTurn: true));

        await port.ExecuteCellAsync(first);
        Assert.NotEmpty(host.Scaffold.Cells[0].Outputs);

        await port.ClearOutputAsync(first);
        Assert.Empty(host.Scaffold.Cells[0].Outputs);

        await port.ExecuteFromAsync(first);
        Assert.NotEmpty(host.Scaffold.Cells[0].Outputs);
        await Assert.ThrowsAsync<InvalidOperationException>(() => port.ExecuteFromAsync(Guid.NewGuid()));

        await port.ClearAllOutputsAsync();
        Assert.Empty(host.Scaffold.Cells[0].Outputs);

        await port.ExecuteAllAsync();
        Assert.NotEmpty(host.Scaffold.Cells[0].Outputs);

        await port.ExecuteCodeAsync("var fromCode = 6 * 7;", "csharp", TestContext.Current.CancellationToken);
        Assert.True(host.Scaffold.Variables.TryGet<int>("fromCode", out var fromCode));
        Assert.Equal(42, fromCode);
        Assert.Contains("42", string.Concat((await port.ExecuteCodeCaptureOutputsAsync("6 * 7", "csharp", TestContext.Current.CancellationToken)).Select(output => output.Content)), StringComparison.Ordinal);

        await port.RestartKernelAsync("csharp");

        var added = Guid.Parse(await port.InsertCellAsync(2, "markdown"));

        await port.MoveCellAsync(added, 0);
        Assert.Equal([added, first, second], host.Scaffold.Cells.Select(cell => cell.Id));

        await port.RemoveCellAsync(added);
        Assert.Equal([first, second], host.Scaffold.Cells.Select(cell => cell.Id));

        port.SetActiveLayout("dashboard");
        port.SetActiveTheme("verso-dark");
        Assert.Equal("dashboard", port.ActiveLayoutId);
        Assert.Equal("verso-dark", port.ActiveThemeId);
    }

    [Fact]
    public async Task ARunsOperations_RunningFromACell_BeginNoFurtherCellOnceStopped()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "from.verso", CSharp(Held("first")), CSharp("var second = 2;"));
        var begun = Begun(host);
        var run = new Run(1, cell: null, takesTheCSharpTurn: true);
        var running = new RunOperations(host.Scaffold, run).ExecuteFromAsync(host.Cells[0].Id);

        await UntilAsync("first-began", "the first cell never began");
        run.Stop();
        await LetGoAsync("first");

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running.WaitAsync(AtOnce, TestContext.Current.CancellationToken));
        Assert.Equal([host.Cells[0].Id], begun);
    }

    private static string Printed(HostedCell cell) => string.Concat(cell.Outputs.Select(output => output.Content));

    [Fact]
    public async Task AStopBetweenCells_StartsNoKernelAfresh_AndKeepsTheVariables()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "between.verso", CSharp("var kept = 42;"), CSharp("var second = 2;"), CSharp("System.Console.Write(kept);"));
        var first = host.Cells[0].Id;
        var restarts = 0;
        var stopped = false;

        host.Scaffold.OnKernelRestarting += _ => Interlocked.Increment(ref restarts);

        // The stop lands as the first cell ends and before the second begins: nothing runs then.
        host.Scaffold.OnCellExecuted += cell =>
        {
            if (cell == first && !stopped)
            {
                stopped = host.Stop(1);
            }
        };

        await host.RunToolbarAsync("verso.action.run-all").WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.True(stopped);
        Assert.Equal(0, Volatile.Read(ref restarts));
        Assert.Equal("42", Printed(await host.RunAsync(host.Cells[2].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
    }

    [Fact]
    public async Task APress_IsToldByItsNumberFromItsStartAndBetweenCells_AndAsNoRunOnceStopped()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "told.verso", CSharp("var first = 1;"), CSharp("var second = 2;"));
        var first = host.Cells[0].Id;
        HostedRun? atTheReset = null;
        HostedRun? inTheFirstCell = null;
        HostedRun? betweenCells = null;
        var reset = false;

        // What every view is told as Run All resets its kernels, before any cell begins.
        ((VariableStore)host.Scaffold.Variables).OnVariablesChanged += () =>
        {
            if (!reset)
            {
                reset = true;
                atTheReset = host.Current.Running;
            }
        };

        // What the run is while the first cell runs.
        host.Scaffold.OnCellExecuting += cell =>
        {
            if (cell == first)
            {
                inTheFirstCell = host.Running;
            }
        };

        // What the run is as the first cell ends, before the second begins; the press is stopped there.
        host.Scaffold.OnCellExecuted += cell =>
        {
            if (cell == first && betweenCells is null)
            {
                betweenCells = host.Running;
                host.Stop(1);
            }
        };

        await host.RunToolbarAsync("verso.action.run-all").WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.Equal(1, atTheReset?.Number);
        Assert.Null(atTheReset?.Cell);
        Assert.False(atTheReset?.Waits);
        Assert.Equal(first, inTheFirstCell?.Cell);
        Assert.Equal(1, betweenCells?.Number);
        Assert.Null(betweenCells?.Cell);
        Assert.Null(host.Running);
        Assert.Null(host.Current.Running);
    }

    [Fact]
    public async Task AButtonsCodeWithNoCell_IsStoppedWithAFreshKernel()
    {
        var path = Path.Join(_folder, "code.verso");

        // The notebook's own kernel is the blocks', so only the code the button runs tells which kernel runs.
        var notebook = new NotebookModel { DefaultKernelId = "pdd" };

        notebook.Cells.Add(CSharp("6 * 7"));
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new CodeButton(
            $$"""System.IO.File.WriteAllText(@"{{Path.Join(_folder, "code-began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }"""));

        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var pressing = host.RunToolbarAsync(CodeButton.Id);

            await UntilAsync("code-began", "the button's code never began");

            Assert.True(host.Stop(1));
            await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // The C# kernel was started afresh, so a C# cell runs although the button's code never ends.
            Assert.Equal("42", Printed(await host.RunAsync(host.Cells[0].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AButtonStoppedAfterItsCodeEnded_StartsNoKernelAfresh()
    {
        var path = Path.Join(_folder, "after.verso");
        var notebook = new NotebookModel { DefaultKernelId = "pdd" };

        notebook.Cells.Add(CSharp("System.Console.Write(kept);"));
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();
        var waiting = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        // The button's code ends, and the button then waits for what never comes: the stop lands while nothing runs.
        await engine.LoadExtensionAsync(new CodeButton("var kept = 42;", () =>
        {
            waiting.TrySetResult();

            return Task.Delay(Timeout.Infinite, TestContext.Current.CancellationToken);
        }));

        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var pressing = host.RunToolbarAsync(CodeButton.Id);

            await waiting.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.True(host.Stop(1));
            await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // No kernel was started afresh, so what the button's code set is still there.
            Assert.Equal("42", Printed(await host.RunAsync(host.Cells[0].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ACellLeftBehindThatEndsLater_LeavesTheNextRunsStopWhole()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "late.verso", CSharp(Held("first")), CSharp(Held("second")));
        var first = host.Cells[0].Id;
        var ended = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var restarted = new ConcurrentQueue<string?>();

        host.Scaffold.OnCellExecuted += cell =>
        {
            if (cell == first)
            {
                ended.TrySetResult();
            }
        };

        // The first run is stopped while its cell runs, and the cell is left behind.
        var running = host.RunAsync(first);

        await UntilAsync("first-began", "the first cell never began");
        Assert.True(host.Stop(1));
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // The second run's cell runs when the cell left behind ends, and the engine says so.
        var second = host.RunAsync(host.Cells[1].Id);

        await UntilAsync("second-began", "the second cell never began");
        await File.WriteAllTextAsync(Path.Join(_folder, "first-go"), "go", TestContext.Current.CancellationToken);
        await ended.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // The second run's stop still starts afresh the kernel of the cell it runs.
        host.Scaffold.OnKernelRestarting += restarted.Enqueue;

        Assert.True(host.Stop(2));
        await second.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        Assert.Equal(["csharp"], restarted);
    }

    [Fact]
    public async Task AStoppedRunEndingDuringItsCellsNextRun_LeavesThatRunsStopWhole()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "same.verso", CSharp(Held("first")));
        var cell = host.Cells[0].Id;
        var ended = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var restarted = new ConcurrentQueue<string?>();

        // The first run is stopped while the cell runs, and the cell is left behind.
        var first = host.RunAsync(cell);

        await UntilAsync("first-began", "the cell never began");
        Assert.True(host.Stop(1));
        await first.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // The same cell runs again, as code that never ends, while the run left behind is still in it.
        await host.EditAsync(cell, $$"""System.IO.File.WriteAllText(@"{{Path.Join(_folder, "again-began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""");

        var again = host.RunAsync(cell);

        await UntilAsync("again-began", "the cell never ran again");

        // The run left behind ends, and the engine says the cell ended.
        host.Scaffold.OnCellExecuted += each =>
        {
            if (each == cell)
            {
                ended.TrySetResult();
            }
        };

        await File.WriteAllTextAsync(Path.Join(_folder, "first-go"), "go", TestContext.Current.CancellationToken);
        await ended.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // The second run's stop still starts afresh the kernel of the cell it runs, and the run ends.
        host.Scaffold.OnKernelRestarting += restarted.Enqueue;

        Assert.True(host.Stop(2));
        await again.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        Assert.Equal(["csharp"], restarted);
    }

    [Fact]
    public async Task ACellLeftBehindThatEndsWhileAButtonsCodeRuns_LeavesThatRunsStopWhole()
    {
        var path = Path.Join(_folder, "code-late.verso");

        // The notebook's own kernel is the blocks', so a stop that started it afresh would not end the button's C#.
        var notebook = new NotebookModel { DefaultKernelId = "pdd" };

        notebook.Cells.Add(CSharp(Held("first")));
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new CodeButton(
            $$"""System.IO.File.WriteAllText(@"{{Path.Join(_folder, "code-began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }"""));

        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var first = host.Cells[0].Id;
            var ended = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var restarted = new ConcurrentQueue<string?>();

            host.Scaffold.OnCellExecuted += cell =>
            {
                if (cell == first)
                {
                    ended.TrySetResult();
                }
            };

            // The first run is stopped while its cell runs, and the cell is left behind.
            var running = host.RunAsync(first);

            await UntilAsync("first-began", "the cell never began");
            Assert.True(host.Stop(1));
            await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // The button's code runs in no cell, and every view is told the run with no cell, when the cell left behind
            // ends, and the engine says so.
            var pressing = host.RunToolbarAsync(CodeButton.Id);

            await UntilAsync("code-began", "the button's code never began");
            Assert.Equal(2, host.Running?.Number);
            Assert.Null(host.Running?.Cell);
            await File.WriteAllTextAsync(Path.Join(_folder, "first-go"), "go", TestContext.Current.CancellationToken);
            await ended.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // The button's stop still starts afresh the kernel of the code it runs.
            host.Scaffold.OnKernelRestarting += restarted.Enqueue;

            Assert.True(host.Stop(2));
            await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            Assert.Equal(["csharp"], restarted);
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
