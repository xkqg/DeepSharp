// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

// A run exists from the moment it is asked, whether it runs or waits for another notebook's C# run, and a Stop names the
// run it means. Which kernel runs a cell is the engine's rule, in the engine's order — the kernel of the cell's type, or
// none when the type only draws; else the kernel the cell's language names; else none when a renderer claims the type;
// else the notebook's default kernel — and it decides both whether the run takes the C# turn and which kernel a Stop
// starts afresh.
public sealed partial class VerbTests
{
    private static string Endless(string started) =>
        $$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""";

    private static string Printed(HostedCell cell) => string.Concat(cell.Outputs.Select(output => output.Content));

    private static async Task UntilAsync(Func<bool> holds, string what)
    {
        for (var waited = 0; !holds(); waited += 20)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    // Notebook A runs a C# cell that never ends, so the process's C# turn is taken until A is stopped.
    private async Task<Holding> HoldTheCSharpTurnAsync(OpenNotebooks notebooks)
    {
        var started = Path.Join(_folder, "holding");
        var a = await OpenAsync(notebooks, "a.verso", CSharp(Endless(started)));
        var running = a.RunAsync(a.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "notebook A's run never began");

        return new Holding(a, running);
    }

    [Fact]
    public async Task ACSharpRunWaitingForAnotherNotebooksCSharpRun_IsOnTheVersion_AndItsStopEndsTheWait_AndNothingRunsAfter()
    {
        var started = Path.Join(_folder, "started");

        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(Endless(started)));
        var b = await OpenAsync(notebooks, "b.verso", CSharp("var kept = 42;"), CSharp("""System.Console.Write("B ran");"""), CSharp("2 + 2"));

        await b.RunAsync(b.Cells[0].Id);

        var runningA = a.RunAsync(a.Cells[0].Id);

        await UntilAsync(() => File.Exists(started), "notebook A's run never began");

        var waiting = b.RunAsync(b.Cells[1].Id);

        await UntilAsync(() => b.Current.Running is { Waits: true }, "notebook B's version never showed its run waiting");

        var run = b.Current.Running!.Value;

        Assert.Equal(b.Cells[1].Id, run.Cell);
        Assert.True(b.Stop(run.Number));
        await waiting.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
        Assert.Null(b.Current.Running);

        // B's turn is free again, while A still runs.
        await b.EditAsync(b.Cells[2].Id, "System.Console.Write(kept);").WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.True(a.Stop(a.Running!.Value.Number));
        await runningA;

        // B's next C# run takes the C# turn after the one its stopped run waited in, which by then came and started
        // nothing: not the cell, and no fresh kernel either, so what B's kernel held is still there.
        Assert.Equal("42", Printed(await b.RunAsync(b.Cells[2].Id)));
        Assert.Empty(b.Cells[1].Outputs);
        Assert.Null(b.Cells[1].ExecutionCount);
    }

    [Fact]
    public async Task CodeCellsThatNameNoLanguage_RunInTheDefaultKernel_AndTakeTheCSharpTurnAsEveryCSharpRunDoes()
    {
        await using var notebooks = new OpenNotebooks();

        NotebookModel Printing(string letter)
        {
            var notebook = new NotebookModel { DefaultKernelId = "csharp" };

            notebook.Cells.Add(CSharp("var warm = 1;"));
            notebook.Cells.Add(new CellModel { Type = "code", Source = $$"""for (var i = 0; i < 20; i++) { System.Console.Write("{{letter}}"); await System.Threading.Tasks.Task.Delay(15); }""" });

            return notebook;
        }

        var a = await OpenAsync(notebooks, "a.verso", Printing("A"));
        var b = await OpenAsync(notebooks, "b.verso", Printing("B"));

        await a.RunAsync(a.Cells[0].Id);
        await b.RunAsync(b.Cells[0].Id);

        var ran = await Task.WhenAll(a.RunAsync(a.Cells[1].Id), b.RunAsync(b.Cells[1].Id));

        Assert.Equal(new string('A', 20), Printed(ran[0]));
        Assert.Equal(new string('B', 20), Printed(ran[1]));
    }

    [Fact]
    public async Task StoppingAnEndlessCSharpCell_StartsTheCSharpKernelAfresh_WhateverTheNotebooksDefaultKernel()
    {
        var started = Path.Join(_folder, "started");
        var notebook = new NotebookModel { DefaultKernelId = "pdd" };

        notebook.Cells.Add(CSharp(Endless(started)));
        notebook.Cells.Add(CSharp("System.Console.Write(6 * 7);"));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "stop.verso", notebook);
        var pressing = host.RunToolbarAsync("verso.action.run-all");

        await UntilAsync(() => File.Exists(started), "the endless cell never began");

        Assert.True(host.Stop(host.Running!.Value.Number));
        await pressing;

        var quick = await host.RunAsync(host.Cells[1].Id).WaitAsync(TimeSpan.FromSeconds(20), TestContext.Current.CancellationToken);

        Assert.Equal("42", Printed(quick));
    }

    [Fact]
    public async Task AStopNamingARunThatEnded_StopsNothing_AndTheRunUnderWayGoesOn()
    {
        var started = Path.Join(_folder, "started");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "runs.verso", CSharp("1 + 1"), CSharp(Endless(started)));

        await host.RunAsync(host.Cells[0].Id);

        var running = host.RunAsync(host.Cells[1].Id);

        await UntilAsync(() => File.Exists(started), "the endless cell never began");

        var now = host.Running!.Value;

        Assert.Equal(host.Cells[1].Id, now.Cell);
        Assert.False(now.Waits);
        Assert.False(host.Stop(now.Number - 1));
        await Task.Delay(300, TestContext.Current.CancellationToken);
        Assert.False(running.IsCompleted);
        Assert.True(host.Stop(now.Number));
        await running;
        Assert.False(host.Stop(now.Number));
    }

    [Fact]
    public async Task AMarkdownCell_TakesNoCSharpTurn()
    {
        await using var notebooks = new OpenNotebooks();
        var (a, holding) = await HoldTheCSharpTurnAsync(notebooks);
        var b = await OpenAsync(notebooks, "b.verso", new CellModel { Type = "markdown", Source = "# The passengers" });

        // The cell only draws: it runs no kernel, so it waits for no C# run.
        var drawn = await b.RunAsync(b.Cells[0].Id).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Contains("The passengers", Printed(drawn), StringComparison.Ordinal);
        Assert.True(a.Stop(a.Running!.Value.Number));
        await holding;
    }

    [Theory]
    [InlineData("python", "csharp")]
    [InlineData(null, "python")]
    public async Task ACellInALanguageNoKernelReads_TakesNoCSharpTurn(string? language, string defaultKernel)
    {
        await using var notebooks = new OpenNotebooks();
        var (a, holding) = await HoldTheCSharpTurnAsync(notebooks);
        var notebook = new NotebookModel { DefaultKernelId = defaultKernel };

        // The cell's own language, or else the notebook's default, is Python.
        notebook.Cells.Add(new CellModel { Type = "code", Language = language, Source = "print(1)" });

        var b = await OpenAsync(notebooks, "b.verso", notebook);

        // No kernel here reads Python, so the engine runs none and says so; nothing waits for a C# run.
        var ran = await b.RunAsync(b.Cells[0].Id).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Equal("Failed", ran.LastStatus);
        Assert.True(a.Stop(a.Running!.Value.Number));
        await holding;
    }

    [Fact]
    public async Task ACellOfAPartThatOnlyDraws_TakesNoCSharpTurn()
    {
        await using var notebooks = new OpenNotebooks();
        var (a, holding) = await HoldTheCSharpTurnAsync(notebooks);
        var path = Path.Join(_folder, "drawn.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = DrawingPart.Type, Source = "drawn" });
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new DrawingPart());

        var b = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            // A type a renderer claims, with no language, is drawn: no kernel runs it, whatever the default kernel.
            var drawn = await b.RunAsync(b.Cells[0].Id).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

            Assert.Equal("drawn", Printed(drawn));
        }
        finally
        {
            await b.CloseAsync();
        }

        Assert.True(a.Stop(a.Running!.Value.Number));
        await holding;
    }

    /// <summary>A notebook whose C# run never ends, and that run.</summary>
    /// <param name="A">The notebook.</param>
    /// <param name="Running">Its run, until it is stopped.</param>
    private readonly record struct Holding(NotebookHost A, Task<HostedCell> Running);
}
