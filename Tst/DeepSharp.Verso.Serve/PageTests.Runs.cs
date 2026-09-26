// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net.Http.Json;
using System.Text.RegularExpressions;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A run is offered as Verso's editor offers it: not while one is under way, since what a cell's ▶ or Shift+Enter asked
// then would only wait behind it and run after its Stop. And a run that waits for another notebook's C# run says so — on
// its cell, or on the page's own line for a button that runs no cell yet — and its Stop ends the wait.
public sealed partial class PageTests
{
    [Fact]
    public async Task WhileARunIsUnderWay_NoCellOffersARun_AndShiftEnterAsksNone()
    {
        var ran = At("ran");

        await SaveAsync(
            "runs.verso",
            new CellModel { Type = "code", Language = "csharp", Source = "while (true) { await System.Threading.Tasks.Task.Delay(10); }" },
            new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.AppendAllText(@"{{ran}}", "x");""" });
        await using var served = await StartAsync();

        try
        {
            var page = await OpenAsync(served, "runs.verso");
            var text = Cell(page, 1).Locator("textarea.source");

            await Cell(page, 0).Locator("button.run").ClickAsync();
            await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

            await Expect(Cell(page, 0).Locator("button.run")).ToBeDisabledAsync();
            await Expect(Cell(page, 1).Locator("button.run")).ToBeDisabledAsync();

            // Selecting a cell draws its bar again, and the run under way still holds its ▶.
            await Cell(page, 1).Locator(".cell-bar").ClickAsync();
            await Expect(Cell(page, 1)).ToHaveClassAsync(new Regex("selected"));
            await Expect(Cell(page, 1).Locator("button.run")).ToBeDisabledAsync();

            await text.PressAsync("Shift+Enter");
            await page.Locator("#stop").ClickAsync();
            await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
            await Expect(Cell(page, 1).Locator("button.run")).ToBeEnabledAsync();

            // The page sends one thing at a time, in the order asked: once the typing after the Stop reached the server, a run
            // the Shift+Enter had asked would have run before it.
            await text.FillAsync("var after = 1;");
            await text.BlurAsync();

            for (var waited = 0; (await CurrentAsync(served, "runs.verso")).Cells[1].Source != "var after = 1;"; waited += 50)
            {
                Assert.True(waited < 30_000, "the typing never reached the server");
                await Task.Delay(50, TestContext.Current.CancellationToken);
            }

            Assert.False(File.Exists(ran), "the Shift+Enter asked for a run while one was under way");
        }
        finally
        {
            await StopAnyRunAsync(served, "runs.verso");
        }
    }

    [Fact]
    public async Task ACellWhoseCSharpRunWaitsForAnotherNotebooks_SaysItWaits_AndItsStopEndsTheWait()
    {
        await SaveAAndBAsync();
        await using var served = await StartAsync();
        var runningA = await RunForeverInAAsync(served);

        try
        {
            await StopsTheWaitAsync(served, runningA);
        }
        finally
        {
            await StopAnyRunAsync(served, "b.verso");
            await StopAnyRunAsync(served, "a.verso");
        }
    }

    private async Task StopsTheWaitAsync(Served served, Task<HttpResponseMessage> runningA)
    {
        var page = await OpenAsync(served, "b.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex("waiting"), new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex("running"));
        await Expect(page.Locator("#status")).ToBeEmptyAsync();
        Assert.Contains(
            "waits for another notebook",
            await Cell(page, 0).Locator(".cell-bar .status").EvaluateAsync<string>("status => getComputedStyle(status, '::before').content"),
            StringComparison.Ordinal);

        await page.Locator("#stop").ClickAsync();

        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex("waiting"));
        await Expect(Cell(page, 0).Locator("button.run")).ToBeEnabledAsync();
        Assert.Empty((await CurrentAsync(served, "b.verso")).Cells[0].Outputs);

        var stopping = await served.Client.PostAsJsonAsync(
            "/api/notebooks/a.verso/stop",
            new { run = (await CurrentAsync(served, "a.verso")).Running!.Value.Number },
            TestContext.Current.CancellationToken);

        stopping.EnsureSuccessStatusCode();
        (await runningA).EnsureSuccessStatusCode();
    }

    [Fact]
    public async Task AButtonWhoseRunWaitsForAnotherNotebooksCSharpRun_SaysItWaits_AndItsStopEndsTheWait()
    {
        await SaveAAndBAsync();
        await using var served = await StartAsync();
        var runningA = await RunForeverInAAsync(served);

        try
        {
            var page = await OpenAsync(served, "b.verso");

            await page.Locator("#toolbar button[data-button='verso.action.run-all']").ClickAsync();

            // A button's run names no cell while it waits, so the page says it on its own line.
            await Expect(page.Locator("#status")).ToHaveTextAsync("Waits for another notebook's C# run to end…", new() { Timeout = 30_000 });

            await page.Locator("#stop").ClickAsync();

            await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
            await Expect(page.Locator("#status")).ToBeEmptyAsync();
            Assert.Empty((await CurrentAsync(served, "b.verso")).Cells[0].Outputs);
        }
        finally
        {
            await StopAnyRunAsync(served, "b.verso");
            await StopAnyRunAsync(served, "a.verso");
        }

        await runningA;
    }

    // Notebook A with a C# cell that never ends, and notebook B with a C# cell of its own.
    private async Task SaveAAndBAsync()
    {
        await SaveAsync(
            "a.verso",
            new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.WriteAllText(@"{{At("started")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""" });
        await SaveAsync("b.verso", new CellModel { Type = "code", Language = "csharp", Source = """System.Console.Write("B ran");""" });
    }

    // Runs notebook A's cell, which takes the C# turn of the whole process until it is stopped.
    private async Task<Task<HttpResponseMessage>> RunForeverInAAsync(Served served)
    {
        var endless = (await CurrentAsync(served, "a.verso")).Cells[0].Id;
        var running = served.Client.PostAsync($"/api/notebooks/a.verso/cells/{endless}/run", null, TestContext.Current.CancellationToken);

        for (var waited = 0; !File.Exists(At("started")); waited += 50)
        {
            Assert.True(waited < 30_000, "notebook A's run never began");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        return running;
    }

    // A run that never ends is stopped before the server is, whatever the test found: a close waits for the run under way.
    private static async Task StopAnyRunAsync(Served served, string notebook)
    {
        if ((await CurrentAsync(served, notebook)).Running is { } run)
        {
            await served.Client.PostAsJsonAsync($"/api/notebooks/{notebook}/stop", new { run = run.Number }, TestContext.Current.CancellationToken);
        }
    }
}
