// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Verso's own buttons on a cell's bar can be pressed where Verso's editor lets them be, asked for that cell: running it
// wherever the layout lets cells run, clearing it once it shows something. And the line that says what a word means
// goes once the text is left.
public sealed partial class PageTests
{
    [Fact]
    public async Task VersosButtonsOnACellsBar_RunTheCell_AndClearWhatItShows_EachWhereItCanBePressed()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = """Console.WriteLine("printed");""" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "code.verso");
        var run = Cell(page, 0).Locator("button[data-button='verso.action.run-cell']");
        var clear = Cell(page, 0).Locator("button[data-button='verso.action.clear-cell-output']");

        await Expect(run).ToBeEnabledAsync();
        await Expect(clear).ToBeDisabledAsync();

        await run.ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs")).ToContainTextAsync("printed", new() { Timeout = 30_000 });
        await Expect(clear).ToBeEnabledAsync();

        await clear.ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs pre")).ToHaveCountAsync(0);
        await Expect(clear).ToBeDisabledAsync();
    }

    [Fact]
    public async Task TheLineThatSaysWhatAWordMeans_GoesOnceTheTextIsLeft()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var text = Cell(page, 3).Locator("textarea.source");

        await text.EvaluateAsync("text => { text.focus(); text.setSelectionRange(13, 13); }");
        await text.PressAsync("ArrowRight");
        await Expect(Cell(page, 3).Locator(".hover-line")).Not.ToBeEmptyAsync();

        await Cell(page, 0).Locator(".cell-bar").ClickAsync();

        await Expect(Cell(page, 3).Locator(".hover-line")).ToBeEmptyAsync();
    }
}
