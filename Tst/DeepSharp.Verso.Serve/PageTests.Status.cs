// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using DeepSharp.Tests.Serve.Parts;
using Verso.Abstractions;
using Verso.Serializers;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A page shows beside the notebook's name what its kernels do, as Verso's editor shows it — idle, running, being started
// afresh, or failed to start afresh and why — says when a kernel was started afresh, and marks its Save while anything
// differs from the notebook's file, in every tab. A cell a stop left behind shows running with nothing to stop, and the
// page says it goes on in the background until it ends; a close asks first only when something would be lost. The
// kernels that start afresh slowly, or not at all, are real parts, found beside the application.
public sealed partial class PageTests
{
    // A C# cell that says it began, goes on until it is let go, and ignores a stop.
    private CellModel HeldCell(string go) => new()
    {
        Type = "code",
        Language = "csharp",
        Source = $$"""System.IO.File.WriteAllText(@"{{At("began")}}", "on"); while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); }""",
    };

    // A notebook whose kernel is the one named, so Verso's Restart Kernel starts that one afresh.
    private async Task SaveOnAsync(string name, string kernel, params CellModel[] cells)
    {
        var notebook = new NotebookModel { DefaultKernelId = kernel };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TheSaveButton_CarriesADotWhileTheNotebookIsUnsaved_InEveryTab()
    {
        await SaveAsync("typed.verso", new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var first = await OpenAsync(served, "typed.verso");
        var second = await OpenAsync(served, "typed.verso");

        await Expect(second.Locator("section.cell")).ToHaveCountAsync(1);
        await Expect(first.Locator("#save .unsaved")).ToBeHiddenAsync();

        await Cell(first, 0).Locator("textarea.source").FillAsync("2 + 2");

        await Expect(first.Locator("#save .unsaved")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(second.Locator("#save .unsaved")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(second.Locator("#save")).ToHaveAttributeAsync("title", new Regex("unsaved changes"));

        await second.Locator("#save").ClickAsync();

        await Expect(first.Locator("#save .unsaved")).ToBeHiddenAsync(new() { Timeout = 10_000 });
        await Expect(second.Locator("#save .unsaved")).ToBeHiddenAsync(new() { Timeout = 10_000 });
    }

    [Fact]
    public async Task TheKernelStatus_SaysRunningWhileACellRuns_AndIdleOnceItEnds()
    {
        var go = At("go");

        await SaveAsync("held.verso", HeldCell(go));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "held.verso");
        var status = page.Locator("#kernel");

        await Expect(status).ToHaveTextAsync("Kernel idle");
        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(status).ToHaveTextAsync("Running…", new() { Timeout = 30_000 });
        await Expect(status).ToHaveClassAsync(new Regex("running"));

        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);

        await Expect(status).ToHaveTextAsync("Kernel idle", new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task TheKernelStatus_SaysRestartingWhileAKernelStartsAfresh_AndIdleOnceItHas()
    {
        await SaveOnAsync("slow.verso", SlowRestartKernel.Language, new CellModel { Type = "markdown", Source = "# A notebook" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "slow.verso");
        var status = page.Locator("#kernel");

        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();

        await Expect(status).ToHaveTextAsync("Kernel idle");
        await page.Locator("#toolbar button[data-button='verso.action.restart-kernel']").ClickAsync();

        await Expect(status).ToHaveTextAsync("Restarting…", new() { Timeout = 10_000 });
        await Expect(status).ToHaveClassAsync(new Regex("restarting"));
        await Expect(status).ToHaveTextAsync("Kernel idle", new() { Timeout = 30_000 });

        // As Verso's editor says it once a kernel has started afresh: a notice, gone again after a moment.
        await Expect(page.Locator("#notice")).ToHaveTextAsync("Kernel restarted");
    }

    [Fact]
    public async Task AFailedStartAfresh_TurnsTheKernelStatusToKernelError_SayingWhy_UntilACellRuns()
    {
        await SaveOnAsync("failing.verso", FailingRestartKernel.Language, new CellModel { Type = "markdown", Source = "# A notebook" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "failing.verso");
        var status = page.Locator("#kernel");

        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);
        await page.Locator("#toolbar button[data-button='verso.action.restart-kernel']").ClickAsync();

        await Expect(status).ToHaveTextAsync("Kernel error", new() { Timeout = 30_000 });
        await Expect(status).ToHaveAttributeAsync("title", new Regex(Regex.Escape(FailingRestartKernel.Why)));

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(status).ToHaveTextAsync("Kernel idle", new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task ACellAStopLeftBehind_IsMarkedRunning_WithNothingToStop_UntilItEnds()
    {
        var go = At("go");

        await SaveAsync("held.verso", HeldCell(go));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "held.verso");

        try
        {
            await Cell(page, 0).Locator("button.run").ClickAsync();

            // Stopped once its code runs: a stop before would end the run before the code began.
            for (var waited = 0; !File.Exists(At("began")); waited += 20)
            {
                Assert.True(waited < 30_000, "the cell's code never began");
                await Task.Delay(20, TestContext.Current.CancellationToken);
            }

            await page.Locator("#stop").ClickAsync();

            // No run is under way, so nothing is offered to stop and Run All stands in its place; the cell runs on, and is
            // shown running.
            await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
            await Expect(page.Locator("#toolbar button[data-button='verso.action.run-all']")).ToBeVisibleAsync();
            await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\brunning\b"));
            await Expect(page.Locator("#kernel")).ToHaveTextAsync("Running…");

            // Its kernel was started afresh, and the page says the stopped run goes on without it, for as long as it does.
            await Expect(page.Locator("#status")).ToHaveTextAsync("The stopped run goes on in the background until the tool stops.", new() { Timeout = 30_000 });
        }
        finally
        {
            await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
        }

        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\brunning\b"), new() { Timeout = 30_000 });
        await Expect(page.Locator("#kernel")).ToHaveTextAsync("Kernel idle");

        // Nothing a stop left behind runs any more, and the page no longer says that it goes on.
        await Expect(page.Locator("#status")).ToHaveTextAsync(string.Empty);
    }

    [Fact]
    public async Task ClosingANotebookWithNothingToLose_AsksNothing_AndOneWithUnsavedChanges_AsksFirst()
    {
        await SaveAsync("saved.verso", new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await SaveAsync("typed.verso", new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var asked = new ConcurrentQueue<string>();
        var saved = await OpenAsync(served, "saved.verso");

        saved.Dialog += async (_, dialog) =>
        {
            asked.Enqueue(dialog.Message);
            await dialog.AcceptAsync();
        };

        await Expect(saved.Locator("section.cell")).ToHaveCountAsync(1);
        await saved.Locator("#close").ClickAsync();

        await Expect(saved.Locator("#notebooks")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        Assert.Empty(asked);

        var typed = await OpenAsync(served, "typed.verso");

        typed.Dialog += async (_, dialog) =>
        {
            asked.Enqueue(dialog.Message);
            await dialog.DismissAsync();
        };

        // Closed straight after typing: what was typed is sent as the text leaves the cell, and asked about all the same.
        await Cell(typed, 0).Locator("textarea.source").FillAsync("2 + 2");
        await typed.Locator("#close").ClickAsync();

        for (var waited = 0; asked.IsEmpty; waited += 20)
        {
            Assert.True(waited < 10_000, "the close asked nothing");
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        // Asked, and told no: the notebook stays open, with what was typed.
        Assert.Contains("What is not saved is lost.", Assert.Single(asked), StringComparison.Ordinal);
        await Expect(Cell(typed, 0).Locator("textarea.source")).ToHaveValueAsync("2 + 2");
        await Expect(typed.Locator("#notebooks")).ToHaveCountAsync(0);
    }
}
