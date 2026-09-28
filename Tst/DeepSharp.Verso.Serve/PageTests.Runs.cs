// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.RegularExpressions;
using DeepSharp.Tests.Serve.Parts;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A run is offered as Verso's editor offers it: not while one is under way, since what a cell's ▶ or Shift+Enter asked
// then would only wait behind it and run after its Stop. And a run that waits for another notebook's C# run says so — on
// its cell, or on the page's own line for a button that runs no cell yet — and its Stop ends the wait. A Stop of Run All
// leaves the cell under way behind — shown running, with nothing to stop, until it ends — and no cell after it begins,
// even once the one left behind ends.
[Collection(RunsLeftBehind.Name)]
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

        // Shift+Enter asks no run while one is under way, as Verso's editor starts none then; it selects the cell below all
        // the same — the last cell has none, so a code cell is added after the run, and selected.
        await text.PressAsync("Shift+Enter");
        await page.Locator("#stop").ClickAsync();
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(3, new() { Timeout = 30_000 });
        await Expect(Cell(page, 2)).ToHaveClassAsync(new Regex(@"\bselected\b"));
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

    [Fact]
    public async Task ACellWhoseCSharpRunWaitsForAnotherNotebooks_SaysItWaits_AndItsStopEndsTheWait()
    {
        await SaveAAndBAsync();
        await using var served = await StartAsync();
        var a = await RunForeverInAAsync(served);
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

        await StopAAsync(served, a);
    }

    [Fact]
    public async Task AButtonWhoseRunWaitsForAnotherNotebooksCSharpRun_SaysItWaits_AndItsStopEndsTheWait()
    {
        await SaveAAndBAsync();
        await using var served = await StartAsync();
        var a = await RunForeverInAAsync(served);

        var page = await OpenAsync(served, "b.verso");

        await page.Locator("#toolbar button[data-button='verso.action.run-all']").ClickAsync();

        // A button's run names no cell while it waits, so the page says it on its own line.
        await Expect(page.Locator("#status")).ToHaveTextAsync("Waits for another notebook's C# run to end…", new() { Timeout = 30_000 });

        await page.Locator("#stop").ClickAsync();

        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(page.Locator("#status")).ToBeEmptyAsync();
        Assert.Empty((await CurrentAsync(served, "b.verso")).Cells[0].Outputs);

        await StopAAsync(served, a);
    }

    [Fact]
    public async Task StoppingRunAllFromThePage_RunsNoFurtherCell_AndRunButtonsReturn()
    {
        var ran = At("ran");

        // The first cell says it began, waits until it is let go, and says it ended; the second leaves a mark if it runs.
        await SaveAsync(
            "all.verso",
            new CellModel
            {
                Type = "code",
                Language = "csharp",
                Source = $$"""System.IO.File.WriteAllText(@"{{At("began")}}", "on"); while (!System.IO.File.Exists(@"{{At("go")}}")) { await System.Threading.Tasks.Task.Delay(10); } System.IO.File.WriteAllText(@"{{At("ended")}}", "on");""",
            },
            new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.AppendAllText(@"{{ran}}", "x");""" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "all.verso");

        await page.Locator("#toolbar button[data-button='verso.action.run-all']").ClickAsync();
        await UntilAsync("began", "Run All's first cell never began");

        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex("running"));
        await Expect(Cell(page, 1).Locator("button.run")).ToBeDisabledAsync();

        await page.Locator("#stop").ClickAsync();

        // No run is under way: every run button returns, and the cell left behind is shown running until it ends.
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\brunning\b"));
        await Expect(Cell(page, 0).Locator("button.run")).ToBeEnabledAsync();
        await Expect(Cell(page, 1).Locator("button.run")).ToBeEnabledAsync();

        // The cell left behind is let go and ends, and the press's loop is given the moment it would take to begin the next.
        await File.WriteAllTextAsync(At("go"), "go", TestContext.Current.CancellationToken);
        await UntilAsync("ended", "the cell left behind never ended");
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\brunning\b"), new() { Timeout = 30_000 });
        await Task.Delay(TimeSpan.FromSeconds(1), TestContext.Current.CancellationToken);

        Assert.False(File.Exists(ran), "a cell after the stopped one ran");
        Assert.Empty((await CurrentAsync(served, "all.verso")).Cells[1].Outputs);
    }

    [Fact]
    public async Task ABlockAClickRuns_IsMarkedRunning_WithNoStop()
    {
        // A source large enough that its block runs long after a click.
        await File.WriteAllLinesAsync(
            At("large.csv"),
            ["survived,age", .. Enumerable.Range(0, 400_000).Select(row => $"{row % 2},{(row % 7 == 0 ? "" : (row % 80).ToString(CultureInfo.InvariantCulture))}")],
            TestContext.Current.CancellationToken);
        await SaveAsync("large.verso", Block("""{"step": "read.csv", "path": "large.csv"}"""));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "large.verso");
        var show = Cell(page, 0).Locator("[data-action='deepsharp.show']");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(show).ToHaveCountAsync(1, new() { Timeout = 30_000 });
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });

        // Every look the page gives the cell from here on: marked running or not, and whether a Stop is offered then.
        await page.EvaluateAsync("""
            () => {
              const cell = document.querySelectorAll('section.cell')[0];
              const stop = document.getElementById('stop');

              window.looks = [];
              new MutationObserver(() => window.looks.push((cell.classList.contains('running') ? 'running' : 'still') + (stop.hidden ? '' : ' with a stop')))
                .observe(cell, { attributes: true, attributeFilter: ['class'] });
            }
            """);

        // The Show runs the block as the click's own: marked running while it runs, and no run to stop.
        await show.ClickAsync();
        await Expect(Cell(page, 0).Locator("input[type=checkbox][data-action^='deepsharp.include']").First).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex("running"));

        var looks = await page.EvaluateAsync<string[]>("() => window.looks");

        Assert.Contains("running", looks);
        Assert.DoesNotContain(looks, look => look.EndsWith("with a stop", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AClicksCSharpRun_OffersAStop()
    {
        // The first cell draws a control naming a part the application carries, which runs the cell after it on a click.
        await SaveAsync(
            "click.verso",
            new CellModel
            {
                Type = "code",
                Language = "csharp",
                Source = $$"""new Verso.Abstractions.CellOutput("text/html", "<div data-extension-id='{{RunningPart.Id}}'><button data-action='run'>Run the next cell</button></div>")""",
            },
            new CellModel
            {
                Type = "code",
                Language = "csharp",
                Source = $$"""System.IO.File.WriteAllText(@"{{At("began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""",
            });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "click.verso");
        var control = Cell(page, 0).Locator($"[data-extension-id='{RunningPart.Id}'] button[data-action='run']");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(control).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });

        await control.ClickAsync();
        await UntilAsync("began", "the cell the click runs never began");

        // The C# cell a click runs is a run: marked running, with a Stop, which ends it while the click still waits.
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 1)).ToHaveClassAsync(new Regex("running"));

        await page.Locator("#stop").ClickAsync();

        // The run is over, so its Stop goes and the run buttons return; its cell, which never ends, runs on, shown running.
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 1)).ToHaveClassAsync(new Regex(@"\brunning\b"));
        await Expect(Cell(page, 1).Locator("button.run")).ToBeEnabledAsync();
    }

    [Fact]
    public async Task DuringARunThatNeverEnds_ThePanelDraws_WhatIsOfferedShows_AndANewTabDraws_WithNoToolbarAsked()
    {
        await SaveAsync(
            "reads.verso",
            new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.WriteAllText(@"{{At("began")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""" },
            new CellModel { Type = "code", Language = "csharp", Source = string.Empty },
            Block(Titanic[4]));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "reads.verso");
        var asked = new List<string>();

        page.Request += (_, request) => asked.Add(request.Url);

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await UntilAsync("began", "the endless cell never began");
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        // The run never ends, so nothing that waited for it would ever come.
        await page.Locator("#panels button[data-panel='properties']").ClickAsync();
        await Cell(page, 2).Locator(".cell-bar").ClickAsync();
        await Expect(page.Locator($"#panel select[data-part='{StepForm.Id}'][data-field='scale']")).ToBeVisibleAsync(new() { Timeout = 5_000 });

        var text = Cell(page, 1).Locator("textarea.source");

        await text.FocusAsync();
        await text.PressSequentiallyAsync("System.Console.");
        await Expect(Offered(Cell(page, 1)).Filter(new() { HasText = "WriteLine" }).First).ToBeVisibleAsync(new() { Timeout = 5_000 });

        var other = await OpenAsync(served, "reads.verso");

        await Expect(other.Locator("section.cell")).ToHaveCountAsync(3, new() { Timeout = 5_000 });
        await Expect(Cell(other, 0)).ToHaveClassAsync(new Regex("running"));

        // Its toolbar came with the notebook: the Stop in the place of Run All while the run goes on, and the rest as they are.
        await Expect(other.Locator("#toolbar #stop")).ToBeVisibleAsync(new() { Timeout = 5_000 });
        await Expect(other.Locator("#toolbar button[data-button='verso.action.restart-kernel']")).ToBeVisibleAsync();

        await page.Locator("#stop").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });

        // The buttons came with the versions: no page asked for them apart.
        Assert.DoesNotContain(asked, url => url.Contains("/toolbar", StringComparison.Ordinal) && !url.Contains("/toolbar/", StringComparison.Ordinal));
    }

    // Waits until a cell leaves the mark it is named by.
    private async Task UntilAsync(string mark, string what)
    {
        for (var waited = 0; !File.Exists(At(mark)); waited += 50)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }
    }

    // Notebook A with a C# cell that never ends, and notebook B with a C# cell of its own.
    private async Task SaveAAndBAsync()
    {
        await SaveAsync(
            "a.verso",
            new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.WriteAllText(@"{{At("started")}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""" });
        await SaveAsync("b.verso", new CellModel { Type = "code", Language = "csharp", Source = """System.Console.Write("B ran");""" });
    }

    // Runs notebook A's cell, which takes the C# turn of the whole process until it is stopped, from a page of its own.
    private async Task<Endless> RunForeverInAAsync(Served served)
    {
        var socket = await SocketAsync(served, "a.verso");
        var endless = (await socket.SnapshotAsync()).Version.Cells[0].Id;
        var running = await socket.SendAsync("run", new { cell = endless });

        await UntilAsync("started", "notebook A's run never began");

        return new Endless(socket, running);
    }

    // Stops notebook A's run from the page that ran it: the stop and the run are both answered.
    private static async Task StopAAsync(Served served, Endless a)
    {
        await using var socket = a.Socket;
        var stopped = await socket.AskAsync("stop", new { run = (await CurrentAsync(served, "a.verso")).Running!.Value.Number });

        Assert.False(stopped.Refused);
        Assert.False((await a.Running.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken)).Refused);
    }

    // Notebook A's run that never ends: the page that asked for it, and the answer the run comes to once stopped.
    private readonly record struct Endless(PageSocket Socket, Task<PageSocket.Answered> Running);
}
