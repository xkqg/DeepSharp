// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A code cell's text folds and unfolds by the button on its bar, as Verso's editor folds it: folded, its first lines stand
// in its place — as many as the cell keeps, two when it keeps none that will do — and a click on them chooses the cell.
// The fold is the cell's own, saved with the notebook.
public sealed partial class PageTests
{
    [Fact]
    public async Task ACodeCellsText_FoldsToItsFirstLines_AndUnfolds()
    {
        await SaveAsync(
            "fold.verso",
            new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;\n\n  var b = 2;   \nvar c = 3;" },
            new CellModel { Type = "code", Language = "csharp", Source = "  \n" },
            new CellModel { Type = "markdown", Source = "# Notes" },
            new CellModel { Type = "code", Language = "csharp", Source = "l1\nl2\nl3\nl4", Metadata = { ["verso:ui.inputCollapsed"] = true, ["verso:ui.inputPreviewLineCount"] = 3 } },
            new CellModel { Type = "code", Language = "csharp", Source = "l1\nl2\nl3\nl4", Metadata = { ["verso:ui.inputCollapsed"] = true, ["verso:ui.inputPreviewLineCount"] = -1 } });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "fold.verso");
        var fold = Cell(page, 0).Locator(".cell-bar button.fold");
        var lines = Cell(page, 0).Locator(".source-preview pre");

        await Expect(fold).ToHaveAttributeAsync("aria-label", "Collapse Code");
        await fold.ClickAsync();

        // As Verso's editor draws a folded cell: its first two lines that say something, and a mark that there is more.
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeHiddenAsync(new() { Timeout = 10_000 });
        await Expect(lines).ToHaveTextAsync("var a = 1;\n  var b = 2;\n...");
        Assert.Equal("var a = 1;\n  var b = 2;\n...", await lines.TextContentAsync());
        await Expect(fold).ToHaveAttributeAsync("aria-label", "Expand Code");
        await Expect(fold).ToHaveClassAsync(new Regex(@"\bfolded\b"));
        Assert.Contains("true", (await CurrentAsync(served, "fold.verso")).Cells[0].Metadata["verso:ui.inputCollapsed"], StringComparison.Ordinal);

        // A click on the lines chooses the cell, which stays folded.
        await Cell(page, 0).Locator(".source-preview").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(new Regex(@"\bselected\b"));
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeHiddenAsync();

        await fold.ClickAsync();
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(Cell(page, 0).Locator(".source-preview")).ToBeHiddenAsync();
        await Expect(fold).Not.ToHaveClassAsync(new Regex(@"\bfolded\b"));

        // A cell with nothing written folds to saying so; Markdown has no fold of its own.
        await Cell(page, 1).Locator(".cell-bar button.fold").ClickAsync();
        await Expect(Cell(page, 1).Locator(".source-preview pre")).ToHaveTextAsync("(empty)", new() { Timeout = 10_000 });
        await Expect(Cell(page, 2).Locator(".cell-bar button.fold")).ToHaveCountAsync(0);

        // A cell folded in its file shows as many lines as it keeps, and two when what it keeps is no count above nought.
        Assert.Equal("l1\nl2\nl3\n...", await Cell(page, 3).Locator(".source-preview pre").TextContentAsync());
        Assert.Equal("l1\nl2\n...", await Cell(page, 4).Locator(".source-preview pre").TextContentAsync());
    }
}
