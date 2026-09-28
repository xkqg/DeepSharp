// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tests.Serve.Parts;
using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A notebook shown in Verso's dashboard or presentation is drawn as the engine arranges it, as Verso's editors draw it:
// each cell's own element placed in the slot the arrangement names for it, showing only what the cell shows. A tile is
// moved and resized by its handles and run by its button, each an act on the arrangement the server hands to the layout's
// own part, and every tab is drawn the new arrangement; a file an act hands over is saved. Back in the notebook's own
// layout, the page's list of cells returns in the notebook's order; a layout that fails to draw shows the list as well, and
// the page says why.
public sealed partial class PageTests
{
    // Verso's dashboard lays its tiles on rows 50 pixels high, 8 apart.
    private const int TileRow = 58;

    private Task SaveInAsync(string name, string layout, params CellModel[] cells) => SaveInAsync(name, new LayoutReference($"verso.layout.{layout}", layout), cells);

    private Task SaveInAsync(string name, LayoutReference layout, params CellModel[] cells)
    {
        var notebook = new NotebookModel { ActiveLayout = layout };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        return SaveAsync(name, notebook);
    }

    // A C# cell that shows what it was saved showing.
    private static CellModel Showing(string source, string shown) =>
        new() { Type = "code", Language = "csharp", Source = source, Outputs = { new CellOutput("text/plain", shown) } };

    private static ILocator Tile(IPage page, Guid cell) => page.Locator($"#arrangement .verso-dashboard-cell[data-cell-slot='{cell}']");

    // Presses the mouse on a point of an element and lets it go further on, moving in steps as a hand does.
    private static async Task DragAsync(IPage page, ILocator from, double right, double down)
    {
        var box = await from.BoundingBoxAsync() ?? throw new InvalidOperationException("Nothing to drag is drawn.");
        var x = box.X + (box.Width / 2);
        var y = box.Y + (box.Height / 2);

        await page.Mouse.MoveAsync((float)x, (float)y);
        await page.Mouse.DownAsync();
        await page.Mouse.MoveAsync((float)(x + right), (float)(y + down), new() { Steps = 8 });
        await page.Mouse.UpAsync();
    }

    [Fact]
    public async Task ANotebookInTheDashboard_IsDrawnAsItsTiles_EachShowingOnlyWhatItsCellShows()
    {
        var first = Showing("1 + 1", "2");
        var second = Showing("2 + 2", "4");

        await SaveInAsync("tiles.verso", "dashboard", first, second);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "tiles.verso");

        await Expect(Tile(page, first.Id).Locator("section.cell .outputs")).ToHaveTextAsync("2");
        await Expect(Tile(page, second.Id).Locator("section.cell .outputs")).ToHaveTextAsync("4");
        await Expect(Tile(page, first.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 6;grid-row:1/span 4");
        await Expect(Tile(page, second.Id)).ToHaveAttributeAsync("style", "grid-column:7/span 6;grid-row:1/span 4");

        // Only what a cell shows: its text and its bar are not drawn in its tile, and the list is not drawn at all.
        await Expect(Tile(page, first.Id).Locator("textarea.source")).ToBeHiddenAsync();
        await Expect(Tile(page, first.Id).Locator(".cell-bar")).ToBeHiddenAsync();
        await Expect(page.Locator("#notebook")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task DraggingATile_MovesIt_InEveryTab()
    {
        var first = Showing("1 + 1", "2");
        var second = Showing("2 + 2", "4");

        await SaveInAsync("tiles.verso", "dashboard", first, second);
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var one = await context.NewPageAsync();
        var other = await context.NewPageAsync();

        foreach (var page in new[] { one, other })
        {
            await page.GotoAsync($"{served.Address}?token={Token}&notebook=tiles.verso");
            await Expect(Tile(page, first.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 6;grid-row:1/span 4");
        }

        // Dragged seven rows down by the empty middle of its bar, in the tab the person looks at.
        await one.BringToFrontAsync();
        await DragAsync(one, Tile(one, first.Id).Locator(".verso-dashboard-drag-handle"), right: 0, down: 7 * TileRow);

        foreach (var page in new[] { one, other })
        {
            await Expect(Tile(page, first.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 6;grid-row:8/span 4", new() { Timeout = 10_000 });
        }

        await Expect(Tile(one, first.Id).Locator("section.cell .outputs")).ToHaveTextAsync("2");

        // Dragged far to the right, it stops where it still fits the grid.
        await DragAsync(one, Tile(one, first.Id).Locator(".verso-dashboard-drag-handle"), right: 900, down: 0);
        await Expect(Tile(other, first.Id)).ToHaveAttributeAsync("style", "grid-column:7/span 6;grid-row:8/span 4", new() { Timeout = 10_000 });
    }

    [Fact]
    public async Task ADashboardDrawnAnew_KeepsWhereThePageWasScrolledTo()
    {
        // Enough tiles to scroll: two to a row of four rows each.
        CellModel[] cells = [.. Enumerable.Range(0, 16).Select(at => Showing($"{at} + 0", $"{at}"))];

        await SaveInAsync("tall.verso", "dashboard", cells);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "tall.verso");
        var last = Tile(page, cells[^1].Id);

        await Expect(last).ToHaveAttributeAsync("style", "grid-column:7/span 6;grid-row:29/span 4");
        await last.ScrollIntoViewIfNeededAsync();

        var scrolled = await page.EvaluateAsync<double>("() => document.scrollingElement.scrollTop");

        Assert.True(scrolled > 0, "the dashboard is not tall enough to scroll");

        // The last tile moved a column to the left: the dashboard is drawn anew, where the page was.
        await DragAsync(page, last.Locator(".verso-dashboard-drag-handle"), right: -300, down: 0);
        await Expect(last).Not.ToHaveAttributeAsync("style", "grid-column:7/span 6;grid-row:29/span 4", new() { Timeout = 10_000 });

        Assert.Equal(scrolled, await page.EvaluateAsync<double>("() => document.scrollingElement.scrollTop"));
    }

    [Fact]
    public async Task ResizingATile_ByItsCorner_ResizesIt()
    {
        var tile = Showing("1 + 1", "2");

        await SaveInAsync("tiles.verso", "dashboard", tile);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "tiles.verso");

        await Expect(Tile(page, tile.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 6;grid-row:1/span 4");

        // A column of the grid, as the dashboard measures it: its share of the width, the gap between two included.
        var column = await page.EvaluateAsync<double>(
            "() => { const grid = document.querySelector('.verso-dashboard-grid'); const style = getComputedStyle(grid); return (grid.clientWidth - parseFloat(style.paddingLeft) * 2 + parseFloat(style.columnGap)) / 12; }");

        await DragAsync(page, Tile(page, tile.Id).Locator(".verso-dashboard-resize-handle"), right: 3 * column, down: 2 * TileRow);

        await Expect(Tile(page, tile.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 9;grid-row:1/span 6", new() { Timeout = 10_000 });

        // Drawn back past its own corner, it keeps a column and a row.
        var corner = await Tile(page, tile.Id).Locator(".verso-dashboard-resize-handle").BoundingBoxAsync();

        await DragAsync(page, Tile(page, tile.Id).Locator(".verso-dashboard-resize-handle"), right: 5 - corner!.X, down: 5 - corner.Y);
        await Expect(Tile(page, tile.Id)).ToHaveAttributeAsync("style", "grid-column:1/span 1;grid-row:1/span 1", new() { Timeout = 10_000 });
    }

    [Fact]
    public async Task ATilesRunButton_RunsItsCell_AndItsCSharpRunCanBeStopped()
    {
        var endless = EndlessCell();

        await SaveInAsync("runaway.verso", "dashboard", endless);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "runaway.verso");

        await Tile(page, endless.Id).Locator("button[data-action='run']").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        await page.Locator("#stop").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task ThePresentation_ShowsWhatCellsShow_AndTheTextOfThoseShownWhole()
    {
        var shows = Showing("1 + 1", "2");
        var quiet = new CellModel { Type = "code", Language = "csharp", Source = "var nothing = 0;" };

        await SaveInAsync("slides.verso", "presentation", shows, quiet);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "slides.verso");

        await Expect(page.Locator("#arrangement .verso-presentation-input pre")).ToHaveTextAsync("1 + 1");
        await Expect(page.Locator($"#arrangement [data-cell-slot='{shows.Id}'] section.cell .outputs")).ToHaveTextAsync("2");
        await Expect(page.Locator($"#arrangement [data-cell-slot='{shows.Id}'] textarea.source")).ToBeHiddenAsync();
        await Expect(page.Locator($"section.cell[data-cell-id='{quiet.Id}']")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task ACellHiddenForTheDashboardInItsProperties_HasNoTileThere()
    {
        var hidden = Showing("1 + 1", "2");
        var shown = Showing("2 + 2", "4");
        var notebook = new NotebookModel();

        notebook.Cells.Add(hidden);
        notebook.Cells.Add(shown);
        await SaveAsync("tiles.verso", notebook);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "tiles.verso");

        // As Verso's properties panel sets it: the cell's visibility, in the dashboard.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Toggle(page, "properties").ClickAsync();
        await page.Locator("#panel select[data-part='verso.propertyprovider.visibility'][data-field='visibility:dashboard']").SelectOptionAsync("hidden");

        for (var waited = 0; !(await CurrentAsync(served, "tiles.verso")).Cells[0].Metadata.ContainsKey("verso:ui.layoutVisibility"); waited += 50)
        {
            Assert.True(waited < 10_000, "the visibility never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Toggle(page, "view").ClickAsync();
        await page.Locator("#panel .view-row[data-layout='dashboard']").ClickAsync();

        await Expect(Tile(page, shown.Id)).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(Tile(page, hidden.Id)).ToHaveCountAsync(0);
        await Expect(page.Locator($"section.cell[data-cell-id='{hidden.Id}']")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task AnActOnAnArrangement_HandsOverAFile_WhichThePageSaves()
    {
        await SaveInAsync("slots.verso", new LayoutReference(SlottingLayout.Part, SlottingLayout.Id), Showing("1 + 1", "2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "slots.verso");

        var download = await page.RunAndWaitForDownloadAsync(() => page.Locator("#arrangement button[data-action='hand']").ClickAsync());

        Assert.Equal(SlottingLayout.Handed, download.SuggestedFilename);
        Assert.Equal("the payload / the target", await File.ReadAllTextAsync(await download.PathAsync(), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ACellThatComesWhileAnArrangementShows_IsPlacedInItsSlot()
    {
        await SaveInAsync("slots.verso", new LayoutReference(SlottingLayout.Part, SlottingLayout.Id), Showing("1 + 1", "2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "slots.verso");

        await Expect(page.Locator("#arrangement .slot section.cell")).ToHaveCountAsync(1);

        // Another page adds a cell: it comes in its own slot.
        await using var other = await SocketAsync(served, "slots.verso");

        await other.SnapshotAsync();

        var added = (await other.AskAsync("add", new { type = "code", language = "csharp" })).Result<DeepSharp.Verso.Api.HostedCell>();

        await Expect(page.Locator($"#arrangement .slot[data-cell-slot='{added.Id}'] section.cell")).ToHaveCountAsync(1);
        await Expect(page.Locator("#notebook section.cell")).ToHaveCountAsync(0);
    }

    [Fact]
    public async Task ALayoutThatFailsToDraw_ShowsTheNotebooksList_AndThePageSaysWhy()
    {
        await SaveInAsync("failing.verso", new LayoutReference(FailingLayout.Part, FailingLayout.Id), Showing("1 + 1", "2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "failing.verso");

        await Expect(page.Locator("#error .text")).ToHaveTextAsync($"The layout could not be drawn: {FailingLayout.Why}");
        await Expect(page.Locator("#notebook")).ToBeVisibleAsync();
        await Expect(Cell(page, 0).Locator("textarea.source")).ToHaveValueAsync("1 + 1");
        await Expect(page.Locator("#arrangement")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task SwitchingBackToTheNotebook_DrawsItsListAgain_InOrder()
    {
        CellModel[] cells = [Showing("1 + 1", "2"), Showing("2 + 2", "4"), Showing("3 + 3", "6")];

        // The middle cell has no tile, so it waits in the list while the others are placed in theirs.
        cells[1].Metadata["verso:ui.layoutVisibility"] = new Dictionary<string, string> { ["dashboard"] = "Hidden" };

        await SaveInAsync("tiles.verso", "dashboard", cells);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "tiles.verso");

        await Expect(Tile(page, cells[2].Id)).ToBeVisibleAsync();

        await Toggle(page, "view").ClickAsync();
        await page.Locator("#panel .view-row[data-layout='notebook']").ClickAsync();

        await Expect(page.Locator("#notebook")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(page.Locator("#arrangement")).ToBeHiddenAsync();

        for (var at = 0; at < cells.Length; at++)
        {
            await Expect(page.Locator("#notebook > section.cell").Nth(at).Locator("textarea.source")).ToHaveValueAsync(cells[at].Source);
        }
    }
}
