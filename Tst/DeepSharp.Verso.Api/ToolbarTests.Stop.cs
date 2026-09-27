// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using DeepSharp.Verso.Api;
using Verso.Contexts;

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

        using var stop = new CancellationTokenSource();

        await stop.CancelAsync();

        var stopped = new RunOperations(host.Scaffold, stop.Token);
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
        using var stop = new CancellationTokenSource();
        var port = new RunOperations(host.Scaffold, stop.Token);

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
        using var stop = new CancellationTokenSource();
        var running = new RunOperations(host.Scaffold, stop.Token).ExecuteFromAsync(host.Cells[0].Id);

        await UntilAsync("first-began", "the first cell never began");
        await stop.CancelAsync();
        await LetGoAsync("first");

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => running.WaitAsync(AtOnce, TestContext.Current.CancellationToken));
        Assert.Equal([host.Cells[0].Id], begun);
    }
}
