// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Which cell is chosen, as Verso's editor chooses it: a click or the focus in a cell's text, one click on what a cell
// shown rendered shows — or on anything a cell nobody writes shows — and, as the notebook opens, the first cell not shown
// rendered. A click beside the cells takes the choice away, and so does Run All pressed on the toolbar; a cell that goes
// hands the choice to the first cell, or to the cell a click put in its place. The properties panel follows the cell
// chosen. In a layout that lets nothing be edited — the dashboard, the presentation — no cell is drawn chosen, so a cell
// shown rendered stays shown; back in one that does, the choice is as it was.
public sealed partial class PageTests
{
    private static readonly Regex Chosen = new(@"\bselected\b");

    private static CellModel Rendered(string source, string html) => new() { Type = "markdown", Source = source, Outputs = { new CellOutput("text/html", html) } };

    private static ILocator OutputField(IPage page) => page.Locator("#panel select[data-part='verso.propertyprovider.display'][data-field='outputVisibility']");

    [Fact]
    public async Task InTheDashboard_ADoubleClickOnARenderedCell_LeavesItShown()
    {
        var notes = Rendered("# Notes", "<h1>Notes</h1>");

        await SaveInAsync("board.verso", "dashboard", notes);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "board.verso");
        var shown = Tile(page, notes.Id).Locator("section.cell .outputs");

        await Expect(shown).ToHaveTextAsync("Notes");
        await shown.DblClickAsync();

        await Expect(Tile(page, notes.Id).Locator("section.cell")).Not.ToHaveClassAsync(Chosen);
        await Expect(shown).ToBeVisibleAsync();
        await Expect(shown).ToHaveTextAsync("Notes");
    }

    [Fact]
    public async Task ACellChosenInTheNotebook_StaysShownInThePresentationAndTheDashboard_AndIsChosenAgainBack()
    {
        var notes = Rendered("# Notes", "<h1>Notes</h1>");

        await SaveAsync("carried.verso", notes, Showing("1 + 1", "2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "carried.verso");

        // One click on what it shows chooses it, and its text shows in place of its rendering.
        await Cell(page, 0).Locator(".outputs").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeVisibleAsync();

        await Toggle(page, "view").ClickAsync();
        await page.Locator("#panel .view-row[data-layout='presentation']").ClickAsync();

        var slide = page.Locator($"#arrangement [data-cell-slot='{notes.Id}'] section.cell .outputs");

        await Expect(slide).ToHaveTextAsync("Notes", new() { Timeout = 10_000 });
        await Expect(slide).ToBeVisibleAsync();
        await Expect(page.Locator($"#arrangement [data-cell-slot='{notes.Id}'] section.cell")).Not.ToHaveClassAsync(Chosen);

        await page.Locator("#panel .view-row[data-layout='dashboard']").ClickAsync();
        await Expect(Tile(page, notes.Id).Locator("section.cell .outputs")).ToHaveTextAsync("Notes", new() { Timeout = 10_000 });
        await Expect(Tile(page, notes.Id).Locator("section.cell .outputs")).ToBeVisibleAsync();

        await page.Locator("#panel .view-row[data-layout='notebook']").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen, new() { Timeout = 10_000 });
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeVisibleAsync();
    }

    [Fact]
    public async Task AClickOrTheFocusInACellsText_ChoosesIt_AndThePanelChangesThatCell()
    {
        await SaveAsync("two.verso", Code("var a = 1;"), Code("var b = 2;"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "two.verso");

        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Toggle(page, "properties").ClickAsync();
        await Expect(OutputField(page)).ToBeVisibleAsync(new() { Timeout = 10_000 });

        await Cell(page, 1).Locator("textarea.source").ClickAsync();
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(Chosen);

        // What the panel changes is the chosen cell's, and not the one chosen before.
        await OutputField(page).SelectOptionAsync("hidden");

        for (var waited = 0; !(await CurrentAsync(served, "two.verso")).Cells[1].Metadata.ContainsKey("verso:ui.outputVisibility"); waited += 50)
        {
            Assert.True(waited < 10_000, "the change never reached the cell being written in");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        Assert.False((await CurrentAsync(served, "two.verso")).Cells[0].Metadata.ContainsKey("verso:ui.outputVisibility"));

        // The focus alone chooses as well.
        await Cell(page, 0).Locator("textarea.source").FocusAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);

        // And a click in a text that kept its focus while another cell was chosen — Shift+Enter chooses the cell below and
        // leaves the focus where it was — chooses its cell again.
        await Cell(page, 0).Locator("textarea.source").PressAsync("Shift+Enter");
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeFocusedAsync();
        await Cell(page, 0).Locator("textarea.source").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);
    }

    [Fact]
    public async Task OneClickOnWhatACellNobodyWritesShows_ChoosesIt()
    {
        await SaveAsync("parameters.verso", Code("var a = 1;"), ParametersCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "parameters.verso");

        await Expect(Cell(page, 1).Locator("[data-action='parameter-add']")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Cell(page, 1).Locator(".outputs").ClickAsync(new() { Position = new() { X = 2, Y = 2 } });

        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);
    }

    [Fact]
    public async Task AClickBesideTheCells_TakesTheChoiceAway()
    {
        await SaveAsync("two.verso", Code("var a = 1;"), Code("var b = 2;"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "two.verso");

        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Toggle(page, "properties").ClickAsync();
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);

        await page.Locator("#notebook").ClickAsync(new() { Position = new() { X = 3, Y = 3 } });

        await Expect(page.Locator("section.cell.selected")).ToHaveCountAsync(0);
        await Expect(page.Locator("#panel")).ToContainTextAsync("Select a cell to view its properties.");
    }

    [Fact]
    public async Task AsTheNotebookOpens_TheFirstCellNotShownRenderedIsChosen_AndAReconnectKeepsTheChoice()
    {
        await SaveAsync("opened.verso", Rendered("# Title", "<h1>Title</h1>"), Code("var a = 1;"), Code("var b = 2;"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "opened.verso");

        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 0).Locator(".outputs")).ToHaveTextAsync("Title");

        await Cell(page, 2).Locator(".cell-bar").ClickAsync();
        await Expect(Cell(page, 2)).ToHaveClassAsync(Chosen);

        // Another page closes the notebook: this page opens it again and draws it whole, and the choice is as it was.
        await using (var other = await SocketAsync(served, "opened.verso"))
        {
            await other.AskAsync("close");
        }

        await Expect(page.Locator("#status")).ToBeEmptyAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 2)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 1)).Not.ToHaveClassAsync(Chosen);
    }

    [Fact]
    public async Task TheChosenCellTakenAway_HandsTheChoiceToTheFirst_WhoeverTookItAway()
    {
        var first = Code("var a = 1;");
        var second = Code("var b = 2;");
        var third = Code("var c = 3;");

        first.Metadata["verso:ui.outputVisibility"] = "preview";
        second.Metadata["verso:ui.outputVisibility"] = "hidden";
        third.Metadata["verso:ui.outputVisibility"] = "hidden";
        await SaveAsync("three.verso", first, second, third);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "three.verso");

        page.Dialog += async (_, dialog) => await dialog.AcceptAsync();

        await Toggle(page, "properties").ClickAsync();
        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Expect(OutputField(page)).ToHaveValueAsync("hidden", new() { Timeout = 10_000 });

        // Taken away on this page: the first cell is chosen, and the panel shows its fields.
        await Cell(page, 1).Locator(".cell-bar button.delete").ClickAsync();
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(2);
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);
        await Expect(OutputField(page)).ToHaveValueAsync("preview", new() { Timeout = 10_000 });

        // Taken away by another page: the same.
        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Expect(OutputField(page)).ToHaveValueAsync("hidden", new() { Timeout = 10_000 });

        await using (var other = await SocketAsync(served, "three.verso"))
        {
            await other.AskAsync("remove", new { cell = third.Id });
        }

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(1);
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);
        await Expect(OutputField(page)).ToHaveValueAsync("preview", new() { Timeout = 10_000 });
    }

    [Fact]
    public async Task ACardClickThatRewritesTheChosenBlock_KeepsTheChoiceOnTheBlockInItsPlace()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var before = await Cell(page, 1).GetAttributeAsync("data-cell-id");

        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Cell(page, 1).Locator("button.run").ClickAsync();
        await Cell(page, 1).Locator("[data-action='deepsharp.columns']").ClickAsync();
        await Cell(page, 1).Locator("tr[data-column='pclass'] select[data-action]").First.SelectOptionAsync("category");

        await Expect(Cell(page, 1).Locator("textarea.source")).ToHaveValueAsync(new Regex("\"pclass\"[^}]*\"category\""));

        // A new cell stands in the block's place, and it is the one chosen.
        Assert.NotEqual(before, await Cell(page, 1).GetAttributeAsync("data-cell-id"));
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);
    }

    [Fact]
    public async Task RunAllOnTheToolbar_TakesTheChoiceAway_WhileItsKeysKeepItButForACellShownRendered()
    {
        await SaveAsync("runs.verso", new CellModel { Type = "markdown", Source = "# Notes" }, Code("1 + 1"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "runs.verso");
        var runAll = page.Locator("#toolbar button[data-button='verso.action.run-all']");

        // Pressed on the toolbar: no cell is chosen after, so the Markdown shows rendered.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeVisibleAsync();
        await runAll.ClickAsync();

        await Expect(page.Locator("section.cell.selected")).ToHaveCountAsync(0);
        await Expect(Cell(page, 0).Locator(".outputs h1")).ToHaveTextAsync("Notes", new() { Timeout = 30_000 });
        await Expect(Cell(page, 1).Locator(".outputs")).ToHaveTextAsync("2", new() { Timeout = 30_000 });

        // By its keys: the code cell chosen stays chosen.
        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await page.Keyboard.PressAsync("Control+Alt+Enter");
        await Expect(Cell(page, 1).Locator(".cell-bar .status")).ToHaveTextAsync(new Regex(@"\[2\]"), new() { Timeout = 30_000 });
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);

        // And a Markdown cell chosen leaves the choice, as a cell shown rendered does when it runs.
        await Cell(page, 0).Locator(".outputs").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(Chosen);
        await page.Keyboard.PressAsync("Control+Alt+Enter");
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(Chosen, new() { Timeout = 30_000 });
        await Expect(Cell(page, 0).Locator(".outputs h1")).ToBeVisibleAsync();
    }
}
