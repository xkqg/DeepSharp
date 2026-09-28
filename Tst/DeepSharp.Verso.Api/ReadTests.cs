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
/// What is only read of an open notebook — a cell's panel, what its kernel offers to write next, what a word in it means —
/// is answered beside whatever holds the notebook's turn, as Verso's editors read them while a cell runs: a run under way,
/// even one that never ends, holds none of them up, and a stop's fresh kernel faults none. A read answers for a cell the
/// last version holds, and while it reads the notebook stays open.
/// </summary>
[Collection(RunsLeftBehind.Name)]
public sealed class ReadTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-reads-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    // A C# cell that says it began, and never ends.
    private CellModel Endless() => CSharp($$"""System.IO.File.WriteAllText(@"{{At("began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""");

    private async Task<string> SaveAsync(CellModel[] cells)
    {
        var path = At("reads.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return path;
    }

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, params CellModel[] cells) =>
        await notebooks.OpenAsync(await SaveAsync(cells), TestContext.Current.CancellationToken);

    private async Task<NotebookHost> OpenWithAsync(IExtension part, params CellModel[] cells)
    {
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(part);

        return await NotebookHost.OpenAsync(await SaveAsync(cells), engine, TestContext.Current.CancellationToken);
    }

    private async Task BeganAsync()
    {
        for (var waited = 0; !File.Exists(At("began")); waited += 20)
        {
            Assert.True(waited < 30_000, "the endless cell never began");
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task APanelWhatIsOfferedAndWhatAWordMeans_AreAnsweredWhileARunHoldsTheTurn()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, Endless(), CSharp(""), Block("""{"step": "read.csv", "path": "titanic.csv"}"""));
        var code = host.Cells[1].Id;
        const string typed = "System.Console.";
        const string written = "System.Console.WriteLine(1);";
        var running = host.RunAsync(host.Cells[0].Id);

        await BeganAsync();

        // The run never ends, so a read that waited for the turn would never answer.
        var sections = await host.PropertiesAsync(host.Cells[2].Id).WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        var offered = await host.CompletionsAsync(code, typed, typed.Length).WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        var hover = await host.HoverAsync(code, written, written.IndexOf("WriteLine", StringComparison.Ordinal) + 2).WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.Contains(sections, section => section.Part == StepForm.Id);
        Assert.Contains(offered, completion => completion.Text == "WriteLine");
        Assert.Contains("WriteLine", hover?.Content, StringComparison.Ordinal);
        Assert.NotNull(host.Running);

        Assert.True(host.Stop(host.Running!.Value.Number));
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task WhatIsOffered_WhileAStopStartsTheKernelAfresh_IsAnsweredWithoutAFault()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, Endless(), CSharp(""));
        var code = host.Cells[1].Id;
        var running = host.RunAsync(host.Cells[0].Id);
        var faults = new ConcurrentQueue<Exception>();
        var answered = 0;

        await BeganAsync();

        using var asking = new CancellationTokenSource();
        var reads = Task.Run(
            async () =>
            {
                while (!asking.IsCancellationRequested)
                {
                    try
                    {
                        await host.CompletionsAsync(code, "System.Con", 10);
                        Interlocked.Increment(ref answered);
                    }
                    catch (Exception fault)
                    {
                        faults.Enqueue(fault);
                    }
                }
            },
            TestContext.Current.CancellationToken);

        // Asked before, through and after the fresh kernel the stop starts.
        await Task.Delay(100, TestContext.Current.CancellationToken);
        Assert.True(host.Stop(host.Running!.Value.Number));
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        await Task.Delay(300, TestContext.Current.CancellationToken);
        await asking.CancelAsync();
        await reads;

        Assert.Empty(faults);
        Assert.True(answered > 0);
    }

    [Fact]
    public async Task AReadOfACellNoVersionHolds_IsRefused()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, CSharp(""));
        var nowhere = Guid.NewGuid();

        Assert.Equal(nowhere, (await Assert.ThrowsAsync<CellGoneException>(() => host.PropertiesAsync(nowhere))).Cell);
        await Assert.ThrowsAsync<CellGoneException>(() => host.CompletionsAsync(nowhere, "1", 1));
        await Assert.ThrowsAsync<CellGoneException>(() => host.HoverAsync(nowhere, "1", 1));
    }

    [Fact]
    public async Task ACellAChangeAddedThatNoVersionHoldsYet_IsNotRead()
    {
        NotebookHost? host = null;
        CellGoneException? refused = null;

        host = await OpenWithAsync(
            new ClickingPart(async context =>
            {
                var added = Guid.Parse(await context.Notebook!.InsertCellAsync(0, "code", "csharp"));

                // Read while the change is under way: the engine has the cell, the notebook's last version does not.
                refused = await Assert.ThrowsAsync<CellGoneException>(() => host!.PropertiesAsync(added));

                return null;
            }),
            CSharp("1 + 1"));

        try
        {
            await host.GestureAsync(new HostedGesture(host.Cells[0].Id, ClickingPart.Id, "act", "")).WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.NotNull(refused);
            Assert.Equal(2, host.Cells.Count);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AReadOfAClosedNotebook_IsRefused()
    {
        var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, CSharp(""));
        var code = host.Cells[0].Id;

        await notebooks.DisposeAsync();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => host.PropertiesAsync(code));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => host.CompletionsAsync(code, "1", 1));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => host.HoverAsync(code, "1", 1));
    }

    [Fact]
    public async Task AReadUnderWay_KeepsTheNotebookOpen_UntilItHasAnswered()
    {
        var reading = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var host = await OpenWithAsync(new WaitingPanel(reading.Task), CSharp(""));
        var forgotten = 0;
        var closed = false;

        try
        {
            var read = host.PropertiesAsync(host.Cells[0].Id);

            // Nobody views it and nothing differs from its file, but a read is under way: it stays open.
            Assert.True(await host.StaysOpenAsync(TimeSpan.Zero, () => forgotten++));

            reading.SetResult();
            await read.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            closed = !await host.StaysOpenAsync(TimeSpan.Zero, () => forgotten++);

            Assert.True(closed);
            Assert.Equal(1, forgotten);
        }
        finally
        {
            if (!closed)
            {
                await host.CloseAsync();
            }
        }
    }

    [Fact]
    public async Task ACompletionItsKernelWasStartedAfreshDuring_OffersNothing()
    {
        var asked = new TaskCompletionSource();
        var go = new TaskCompletionSource();
        var host = await OpenWithAsync(new GatedKernel(asked, go.Task), new CellModel { Type = "code", Language = GatedKernel.Language, Source = "x" });

        try
        {
            var cell = host.Cells[0].Id;
            var offering = host.CompletionsAsync(cell, "x", 1);

            // The kernel is started afresh while it works out what to offer: what it offers belongs to a kernel now gone.
            await asked.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            await host.Scaffold.RestartKernelAsync(GatedKernel.Language);
            go.SetResult();

            Assert.Empty(await offering.WaitAsync(AtOnce, TestContext.Current.CancellationToken));

            // Asked again, with no start afresh under way, it offers what it offers.
            Assert.Equal(["offered"], (await host.CompletionsAsync(cell, "x", 1).WaitAsync(AtOnce, TestContext.Current.CancellationToken)).Select(each => each.Text));
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ACompletionItsKernelFailsAsItIsStartedAfresh_OffersNothing()
    {
        var asked = new TaskCompletionSource();
        var go = new TaskCompletionSource();
        var host = await OpenWithAsync(new GatedKernel(asked, go.Task), new CellModel { Type = "code", Language = GatedKernel.Language, Source = "x" });

        try
        {
            var offering = host.CompletionsAsync(host.Cells[0].Id, "x", 1);

            // The kernel is started afresh while it works out what to offer, and the read fails as a kernel put away fails
            // it: the failure belongs to a kernel now gone, so the asker is offered nothing rather than told of it.
            await asked.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            await host.Scaffold.RestartKernelAsync(GatedKernel.Language);
            go.SetException(new ObjectDisposedException(GatedKernel.Language));

            Assert.Empty(await offering.WaitAsync(AtOnce, TestContext.Current.CancellationToken));
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ACompletionItsKernelFailsByItself_HandsTheAskerTheFailure()
    {
        var asked = new TaskCompletionSource();
        var go = new TaskCompletionSource();
        var host = await OpenWithAsync(new GatedKernel(asked, go.Task), new CellModel { Type = "code", Language = GatedKernel.Language, Source = "x" });

        try
        {
            var offering = host.CompletionsAsync(host.Cells[0].Id, "x", 1);

            // No start afresh overlaps the asking, so the failure is the kernel's own and reaches whoever asked.
            await asked.Task.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            go.SetException(new InvalidOperationException("The kernel could not work out what to offer."));

            var failed = await Assert.ThrowsAsync<InvalidOperationException>(() => offering.WaitAsync(AtOnce, TestContext.Current.CancellationToken));

            Assert.Equal("The kernel could not work out what to offer.", failed.Message);
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
