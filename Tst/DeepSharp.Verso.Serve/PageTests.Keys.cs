// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A cell's text takes the keys Verso's editor takes: Shift+Enter runs the cell and selects the one below — a new code cell
// when it is the last — Ctrl+Enter runs it and stays on it, Alt+Enter runs it and inserts a code cell below, and
// Ctrl+Alt+Enter runs them all; none moves the text focus. Tab and Shift+Tab indent and outdent as the editor does, and
// Escape leaves the text, which sends what was typed and draws a rendered cell.
public sealed partial class PageTests
{
    private static CellModel Code(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static async Task UntilShownAsync(Microsoft.Playwright.ILocator cell, string text) =>
        await Expect(cell.Locator(".outputs")).ToContainTextAsync(text, new() { Timeout = 30_000 });

    [Fact]
    public async Task ShiftEnter_RunsTheCell_AndSelectsTheOneBelow_AndOnTheLastAddsACodeCell()
    {
        await SaveAsync("keys.verso", Code("1 + 1"), Code("2 + 2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");
        var first = Cell(page, 0).Locator("textarea.source");

        await first.PressAsync("Shift+Enter");

        await UntilShownAsync(Cell(page, 0), "2");
        await Expect(Cell(page, 1)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(first).ToBeFocusedAsync();

        await Cell(page, 1).Locator("textarea.source").PressAsync("Shift+Enter");

        await UntilShownAsync(Cell(page, 1), "4");
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(3);
        await Expect(Cell(page, 2)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(Cell(page, 2)).ToHaveAttributeAsync("data-type", "code");
        Assert.Equal("csharp", (await CurrentAsync(served, "keys.verso")).Cells[2].Language);
    }

    [Fact]
    public async Task CtrlEnter_RunsTheCell_AndStays()
    {
        await SaveAsync("keys.verso", Code("1 + 1"), Code("2 + 2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");

        await Cell(page, 0).Locator("textarea.source").PressAsync("Control+Enter");

        // The cell written in is the one chosen, and it stays chosen: no other is, and none is added.
        await UntilShownAsync(Cell(page, 0), "2");
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(page.Locator("section.cell.selected")).ToHaveCountAsync(1);
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(2);
    }

    [Fact]
    public async Task AltEnter_RunsTheCell_AndInsertsACodeCellBelow_Selected()
    {
        await SaveAsync("keys.verso", Code("1 + 1"), Code("2 + 2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");

        await Cell(page, 0).Locator("textarea.source").PressAsync("Alt+Enter");

        await UntilShownAsync(Cell(page, 0), "2");
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(3);
        await Expect(Cell(page, 1)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(Cell(page, 1).Locator("textarea.source")).ToHaveValueAsync(string.Empty);
        await Expect(Cell(page, 2).Locator("textarea.source")).ToHaveValueAsync("2 + 2");
        Assert.Equal("csharp", (await CurrentAsync(served, "keys.verso")).Cells[1].Language);
    }

    [Fact]
    public async Task CtrlAltEnter_RunsEveryCell()
    {
        await SaveAsync("keys.verso", Code("1 + 1"), Code("2 + 2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");

        await Cell(page, 1).Locator("textarea.source").PressAsync("Control+Alt+Enter");

        await UntilShownAsync(Cell(page, 0), "2");
        await UntilShownAsync(Cell(page, 1), "4");
    }

    [Fact]
    public async Task Tab_IndentsToTheNextStop_OverLinesShiftsEachLine_AndShiftTabOutdents()
    {
        await SaveAsync("keys.verso", Code("ab"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");
        var text = Cell(page, 0).Locator("textarea.source");

        await text.EvaluateAsync("text => { text.focus(); text.setSelectionRange(1, 1); }");
        await text.PressAsync("Tab");

        await Expect(text).ToHaveValueAsync("a   b");
        await Expect(text).ToBeFocusedAsync();

        // Over lines: each line's indentation goes to the next stop, an empty line stays empty, and back again.
        await text.FillAsync("x\n  y\n\nz");
        await text.EvaluateAsync("text => text.setSelectionRange(0, text.value.length)");
        await text.PressAsync("Tab");

        await Expect(text).ToHaveValueAsync("    x\n    y\n\n    z");

        await text.PressAsync("Shift+Tab");

        await Expect(text).ToHaveValueAsync("x\ny\n\nz");

        // What the keys wrote is typing like any other, sent as it is typed.
        for (var waited = 0; (await CurrentAsync(served, "keys.verso")).Cells[0].Source != "x\ny\n\nz"; waited += 50)
        {
            Assert.True(waited < 10_000, "the indented text never reached the server");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task ASelectedRenderedCell_ShowsItsText_NotItsRendering_AndRunningItShowsTheRendering()
    {
        await SaveAsync("keys.verso", new CellModel { Type = "markdown", Source = "# One" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");
        var text = Cell(page, 0).Locator("textarea.source");
        var rendering = Cell(page, 0).Locator(".outputs h1");

        await Expect(rendering).ToBeVisibleAsync();
        await Expect(text).ToBeHiddenAsync();

        // Selected, its text stands in for its rendering, as Verso's editor draws a cell being written.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();

        await Expect(text).ToBeVisibleAsync();
        await Expect(rendering).ToBeHiddenAsync();

        // Run, it is rendered: it leaves the selection, so the rendering shows.
        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\bselected\b"), new() { Timeout = 10_000 });
        await Expect(rendering).ToBeVisibleAsync();
        await Expect(text).ToBeHiddenAsync();
    }

    [Fact]
    public async Task Escape_LeavesTheText_SendsWhatWasTyped_AndARenderedCellDrawsItself()
    {
        await SaveAsync("keys.verso", new CellModel { Type = "markdown", Source = "# One" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "keys.verso");
        var text = Cell(page, 0).Locator("textarea.source");

        // One click on its rendering chooses it, as Verso's editor chooses it, and its text is written in.
        await Expect(Cell(page, 0).Locator(".outputs h1")).ToHaveTextAsync("One");
        await Cell(page, 0).Locator(".outputs").ClickAsync();
        await text.ClickAsync();
        await Expect(text).ToBeFocusedAsync();

        await text.FillAsync("# Two");
        await text.PressAsync("Escape");

        // Rendered, it leaves the selection and shows its rendering, as Verso's editor leaves a Markdown cell on Escape.
        await Expect(text).Not.ToBeFocusedAsync();
        await Expect(Cell(page, 0).Locator(".outputs h1")).ToHaveTextAsync("Two", new() { Timeout = 10_000 });
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new Regex(@"\bselected\b"));
        Assert.Equal("# Two", (await CurrentAsync(served, "keys.verso")).Cells[0].Source);
    }
}
