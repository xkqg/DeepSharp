// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Microsoft.Playwright;
using Verso.Abstractions;
using Verso.Serializers;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Writing a notebook in the page, as in Verso's own editor: a cell added after another or at the end, of any kind the
// notebook has, and selected with its text open; a cell taken away once the person says yes, moved past its neighbour,
// or turned into another kind — each only where the notebook's layout allows it. A cell shown rendered is rendered only
// while it has text and output and is not selected. A change asked for while a cell runs says it waits; a cell another
// page took away is said to be gone in the server's own words; and a folder is listed with a way to make a notebook.
public sealed partial class PageTests
{
    private async Task SaveAsync(string name, NotebookModel notebook) =>
        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

    // The line an event stream ends each of its lines with.
    private const char Line = '\n';

    private static ILocator Adding(IPage page, string type) => page.Locator($"#adding button[data-type='{type}']");

    [Fact]
    public async Task ACellAddedAtTheEnd_IsDrawnSelected_WithItsTextOpen()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        await Adding(page, "markdown").ClickAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(6);
        await Expect(Cell(page, 5)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(Cell(page, 5).Locator("textarea.source")).ToBeVisibleAsync();
        Assert.Equal("markdown", (await CurrentAsync(served, "titanic.verso")).Cells[5].Type);
    }

    [Fact]
    public async Task ACellAddedAfterTheSelectedOne_StandsRightAfterIt()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Cell(page, 1).Locator(".insert-row button[data-type='code']").ClickAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(6);
        await Expect(Cell(page, 2)).ToHaveAttributeAsync("data-type", "code");
        Assert.Equal("code", (await CurrentAsync(served, "titanic.verso")).Cells[2].Type);
    }

    [Fact]
    public async Task TakingACellAway_AsksFirst_AndOnlyAYesTakesIt()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var answer = false;

        page.Dialog += async (_, dialog) =>
        {
            if (answer)
            {
                await dialog.AcceptAsync();
            }
            else
            {
                await dialog.DismissAsync();
            }
        };

        await Cell(page, 4).Locator("button.delete").ClickAsync();
        await Task.Delay(TimeSpan.FromMilliseconds(500), TestContext.Current.CancellationToken);

        Assert.Equal(5, (await CurrentAsync(served, "titanic.verso")).Cells.Count);

        answer = true;
        await Cell(page, 4).Locator("button.delete").ClickAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(4);
        Assert.Equal(4, (await CurrentAsync(served, "titanic.verso")).Cells.Count);
    }

    [Fact]
    public async Task MovingACellDownAndUp_PassesTheNeighbourItWasShownBeside()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var ids = (await CurrentAsync(served, "titanic.verso")).Cells.Select(cell => cell.Id).ToArray();

        await Cell(page, 3).Locator("button.down").ClickAsync();

        await Expect(Cell(page, 4)).ToHaveAttributeAsync("data-cell-id", ids[3].ToString());

        await Cell(page, 4).Locator("button.up").ClickAsync();

        await Expect(Cell(page, 3)).ToHaveAttributeAsync("data-cell-id", ids[3].ToString());
        Assert.Equal(ids, (await CurrentAsync(served, "titanic.verso")).Cells.Select(cell => cell.Id));
    }

    [Fact]
    public async Task ACellsKind_IsChangedThroughItsSelect()
    {
        await SaveAsync("notes.verso", new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "notes.verso");

        await Cell(page, 0).Locator("select.kind").SelectOptionAsync("markdown|");

        await Expect(Cell(page, 0)).ToHaveAttributeAsync("data-type", "markdown");
        Assert.Equal("markdown", (await CurrentAsync(served, "notes.verso")).Cells[0].Type);
    }

    [Fact]
    public async Task ANotebookInTheDashboard_OffersNoWayToAddTakeAwayMoveOrWriteACell()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        foreach (var block in Titanic)
        {
            notebook.Cells.Add(Block(block));
        }

        await SaveAsync("dashboard.verso", notebook);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "dashboard.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        await Expect(page.Locator("#adding button")).ToHaveCountAsync(0);
        await Expect(page.Locator("section.cell button.delete, section.cell button.up, section.cell button.down, section.cell select.kind")).ToHaveCountAsync(0);
        await Expect(Cell(page, 3).Locator("textarea.source")).Not.ToBeEditableAsync();
        await Expect(Cell(page, 3).Locator("button.run")).ToBeVisibleAsync();
    }

    [Fact]
    public async Task AMarkdownCellWithTextAndOutput_IsShownRendered_UntilItIsSelected()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block), new CellModel { Type = "markdown", Source = "# The passengers" }]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Expect(Cell(page, 5).Locator(".outputs h1")).ToHaveTextAsync("The passengers");
        await Expect(Cell(page, 5).Locator("textarea.source")).ToBeHiddenAsync();

        await Cell(page, 5).Locator(".cell-bar").ClickAsync();

        await Expect(Cell(page, 5).Locator("textarea.source")).ToBeVisibleAsync();

        await Cell(page, 0).Locator(".cell-bar").ClickAsync();

        await Expect(Cell(page, 5).Locator("textarea.source")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task AChangeAskedForWhileACellRuns_SaysItWaitsForTheRun_AndIsMadeAfterIt()
    {
        await SaveAsync("slow.verso", new CellModel { Type = "code", Language = "csharp", Source = "await System.Threading.Tasks.Task.Delay(4000);" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "slow.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        await Adding(page, "markdown").ClickAsync();

        await Expect(page.Locator("#status")).ToHaveTextAsync(new Regex("Waits for the run"));
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(2, new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task ACellAnotherPageTookAway_IsSaidToBeGone_InTheServersOwnWords()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var snapshot = await served.Client.GetStringAsync("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken);
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var told = false;

        // The page hears the notebook as it stands and nothing after: its stream is the snapshot alone, and the stream it
        // opens again once that one ends is never answered.
        await page.RouteAsync("**/updates", async route =>
        {
            if (!told)
            {
                told = true;
                await route.FulfillAsync(new() { Status = 200, ContentType = "text/event-stream", Body = $"event: snapshot{Line}id: 0{Line}data: {snapshot}{Line}{Line}" });
            }
        });
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=titanic.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);

        var gone = (await CurrentAsync(served, "titanic.verso")).Cells[4].Id;

        await served.Client.PostAsync($"/api/notebooks/titanic.verso/cells/{gone}/remove", null, TestContext.Current.CancellationToken);

        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();
        await Cell(page, 4).Locator("button.delete").ClickAsync();

        await Expect(page.Locator("#status")).ToHaveTextAsync(new Regex($"The cell {gone} no longer stands"));
    }

    [Fact]
    public async Task AFolderOfOneNotebook_IsListed_WithANewNotebook_ThatOpensOnceMade()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served);

        await Expect(page.Locator("#notebooks a")).ToHaveCountAsync(1);

        await page.Locator("#new input[name='name']").FillAsync("made.verso");
        await page.Locator("#new button").ClickAsync();

        await Expect(page.Locator("#notebook-name")).ToHaveTextAsync("made.verso");
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);
        Assert.True(File.Exists(At("made.verso")));
    }

    [Fact]
    public async Task ANewNotebookWithANameThatIsNoBareVersoFileName_SaysWhy_AndMakesNothing()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served);

        await page.Locator("#new input[name='name']").FillAsync("made.ipynb");
        await page.Locator("#new button").ClickAsync();

        await Expect(page.Locator("#status")).ToHaveTextAsync(new Regex("not a name a new notebook can have"));
        Assert.False(File.Exists(At("made.ipynb")));
    }

    [Fact]
    public async Task AServerBesideOneNotebook_OpensIt_AndOffersNoNewOne()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync(At("titanic.verso"));
        var page = await OpenAsync(served);

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        await Expect(page.Locator("#new")).ToHaveCountAsync(0);
    }
}
