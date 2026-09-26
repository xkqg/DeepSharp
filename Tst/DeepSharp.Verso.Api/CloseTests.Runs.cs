// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

// No close waits for a run. A close stops the run under way as Stop stops it — one that waits for another notebook's C#
// run never runs, one that runs is left behind — while a change under way finishes and what was asked behind the close
// is refused. The holder stops the run of every notebook it holds before it closes any, so a run that waited never starts
// once another's stop hands the C# turn on. Only the grace, which closes a notebook nobody uses, never closes over a run.
[Collection(nameof(CloseTests))]
public sealed partial class CloseTests
{
    // Longer than any close takes that does not wait for a run; a close that waits for one never ends.
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private static string Endless(string started) =>
        $$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""";

    // A cell that leaves a mark once it runs.
    private static string Marking(string ran) => $$"""System.IO.File.WriteAllText(@"{{ran}}", "ran");""";

    private static string Printed(HostedCell cell) => string.Concat(cell.Outputs.Select(output => output.Content));

    private static async Task UntilAsync(Func<bool> holds, string what)
    {
        for (var waited = 0; !holds(); waited += 20)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task ClosingANotebookWhoseCSharpRunNeverEnds_ClosesAtOnce_AndLeavesTheOthers()
    {
        var started = At("started");

        await using var notebooks = new OpenNotebooks();
        var endless = await OpenAsync(notebooks, "endless.verso", CSharp(Endless(started)));
        var other = await OpenAsync(notebooks, "other.verso", CSharp("6 * 7"));
        var running = endless.RunAsync(endless.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "the endless cell never began");
        await notebooks.CloseAsync(endless).WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.DoesNotContain(endless.FilePath, notebooks.Paths);
        Assert.Throws<ObjectDisposedException>(endless.Subscribe);

        // The C# turn was handed on: another notebook's C# cell runs.
        Assert.Equal("42", Printed(await other.RunAsync(other.Cells[0].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
    }

    [Fact]
    public async Task ClosingANotebookWhoseRunWaits_ClosesAtOnce_AndItsCellNeverRuns()
    {
        var started = At("started");
        var ran = At("ran");

        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(Endless(started)));
        var b = await OpenAsync(notebooks, "b.verso", CSharp(Marking(ran)));
        var c = await OpenAsync(notebooks, "c.verso", CSharp("6 * 7"));
        var runningA = a.RunAsync(a.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "notebook A's run never began");

        var waiting = b.RunAsync(b.Cells[0].Id);

        await UntilAsync(() => b.Running is { Waits: true }, "notebook B's run never waited");

        try
        {
            await notebooks.CloseAsync(b).WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            await waiting.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        }
        finally
        {
            Assert.True(a.Stop(a.Running!.Value.Number));
        }

        await runningA.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // A C# run after it takes the C# turn after the one B's stopped run waited in, which came and started nothing.
        Assert.Equal("42", Printed(await c.RunAsync(c.Cells[0].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
        Assert.False(File.Exists(ran));
    }

    [Fact]
    public async Task WhatWasAskedBeforeACloseAndWaitedBehindIt_IsRefused_AndTheRunUnderWayIsStopped()
    {
        var started = At("started");
        var go = At("go");
        var ended = At("ended");
        var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "wait.verso",
            CSharp($$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); } System.IO.File.WriteAllText(@"{{ended}}", "on"); 1 + 1"""));

        var running = host.RunAsync(host.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "the cell never began");

        var asked = host.ToolbarAsync();

        await notebooks.DisposeAsync().AsTask().WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.DoesNotContain("2", Printed(await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken)), StringComparison.Ordinal);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => asked);

        // The run was left behind, not ended: let go, it ends by itself in the background.
        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
        await UntilAsync(() => File.Exists(ended), "the run left behind never ended");
    }

    [Fact]
    public async Task ClosingDuringARunAll_RunsNoFurtherCell()
    {
        var started = At("started");
        var go = At("go");
        var ended = At("ended");
        var ran = At("ran");
        var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "all.verso",
            CSharp($$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); } System.IO.File.WriteAllText(@"{{ended}}", "on");"""),
            CSharp(Marking(ran)));

        var pressing = host.RunToolbarAsync("verso.action.run-all");

        await UntilAsync(() => File.Exists(started), "Run All's first cell never began");
        await notebooks.DisposeAsync().AsTask().WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // The first cell was left behind; once it ends, Run All would go on to the next cell, of a notebook that is closed.
        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
        await UntilAsync(() => File.Exists(ended), "the cell left behind never ended");
        await Task.Delay(TimeSpan.FromSeconds(2), TestContext.Current.CancellationToken);

        Assert.False(File.Exists(ran));
    }

    [Fact]
    public async Task ClosingTheNotebooks_StartsNoRunThatWaited()
    {
        var started = At("started");
        var ran = At("ran");
        var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(Endless(started)), CSharp(Marking(ran)));
        var b = await OpenAsync(notebooks, "b.verso", CSharp(Endless(started)), CSharp(Marking(ran)));

        // The endless run goes in the notebook the holder reaches first as it closes them, so a close of one after another
        // would hand the C# turn on, as it stops that run, to the run waiting in the notebook it had not reached yet.
        var endless = string.Equals(notebooks.Paths.First(), a.FilePath, StringComparison.Ordinal) ? a : b;
        var waiting = ReferenceEquals(endless, a) ? b : a;
        var runningEndless = endless.RunAsync(endless.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "the endless cell never began");

        var runningWaiting = waiting.RunAsync(waiting.Cells[1].Id);

        await UntilAsync(() => waiting.Running is { Waits: true }, "the second run never waited");
        await notebooks.DisposeAsync().AsTask().WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await Task.WhenAll(runningEndless, runningWaiting).WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        // A C# run after the close takes the C# turn after the one the stopped run waited in.
        await using var after = new OpenNotebooks();
        var c = await OpenAsync(after, "c.verso", CSharp("6 * 7"));

        Assert.Equal("42", Printed(await c.RunAsync(c.Cells[0].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken)));
        Assert.False(File.Exists(ran));
    }

    [Fact]
    public async Task ARunAskedAsItsNotebookCloses_NeverHoldsTheCloseUp()
    {
        await using var notebooks = new OpenNotebooks();

        for (var round = 0; round < 12; round++)
        {
            var host = await OpenAsync(notebooks, $"race{round}.verso", CSharp(Endless(At($"started{round}"))));
            var go = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var running = Task.Run(async () =>
            {
                await go.Task;

                return await host.RunAsync(host.Cells[0].Id);
            }, TestContext.Current.CancellationToken);
            var closing = Task.Run(async () =>
            {
                await go.Task;
                await notebooks.CloseAsync(host);
            }, TestContext.Current.CancellationToken);

            go.SetResult();

            await closing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            // The run was refused, having come after the close, or stopped by it: either way it has ended with the close.
            Assert.Same(running, await Task.WhenAny(running, Task.Delay(AtOnce, TestContext.Current.CancellationToken)));
            Assert.True(running.IsCompletedSuccessfully || running.Exception!.GetBaseException() is ObjectDisposedException);
        }
    }

    [Fact]
    public async Task ARunThatWaits_KeepsTheNotebookOpen()
    {
        var started = At("started");

        await using var others = new OpenNotebooks();
        var a = await OpenAsync(others, "a.verso", CSharp(Endless(started)));
        var runningA = a.RunAsync(a.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "notebook A's run never began");

        // A cell that shows nothing, so its run leaves nothing unsaved.
        await using var notebooks = new OpenNotebooks(Grace);
        var b = await OpenAsync(notebooks, "b.verso", CSharp(Marking(At("ran"))));
        var waiting = b.RunAsync(b.Cells[0].Id);

        await UntilAsync(() => b.Running is { Waits: true }, "notebook B's run never waited");
        await Beyond();

        Assert.Contains(b.FilePath, notebooks.Paths);

        Assert.True(a.Stop(a.Running!.Value.Number));
        await runningA.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await waiting.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.True(File.Exists(At("ran")));

        await ForgottenAsync(notebooks, b);
    }
}

// A run left behind that later ends puts the process's console back as it found it when it began, under whatever C# run
// is under way then — Verso's C# kernel does so at the end of every run — so the closes, which let such runs end, run
// on their own, once every test run in parallel is done.
[CollectionDefinition(nameof(CloseTests), DisableParallelization = true)]
public sealed class CloseTestsRunAlone
{
}
