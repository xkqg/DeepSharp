// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The engine's buttons are drawn as Verso's editor draws them: a cell's bar and the export menu hold only the buttons
// that can be pressed there now, and the engine's Run Cell is left off the bar, whose own button runs the cell. And the
// line that says what a word means goes once the text is left.
public sealed partial class PageTests
{
    [Fact]
    public async Task ACellsBar_HoldsVersosButtonsOnlyWhereTheyCanBePressedForIt_AndRunsTheCellFromItsOwnButton()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = """Console.WriteLine("printed");""" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "code.verso");
        var clear = Cell(page, 0).Locator("button[data-button='verso.action.clear-cell-output']");

        await Expect(page.Locator("#toolbar button[data-button='verso.action.run-all']")).ToBeVisibleAsync();
        await Expect(Cell(page, 0).Locator("button[data-button='verso.action.run-cell']")).ToHaveCountAsync(0);
        await Expect(clear).ToHaveCountAsync(0);

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs")).ToContainTextAsync("printed", new() { Timeout = 30_000 });
        await Expect(clear).ToBeEnabledAsync();

        await clear.ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs pre")).ToHaveCountAsync(0);
        await Expect(clear).ToHaveCountAsync(0);
    }

    [Fact]
    public async Task TheExportMenu_HoldsOnlyWhatCanBeExportedNow()
    {
        await SaveAsync("broken.verso", Block(Titanic[0]), Block("""{"step": "declare"}"""));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "broken.verso");

        await page.Locator("#export > summary").ClickAsync();

        await Expect(page.Locator("#export button[data-button='verso.action.export-html']")).ToBeVisibleAsync();
        await Expect(page.Locator($"#export button[data-button='{ExportPipelineAction.Id}']")).ToHaveCountAsync(0);
    }

    [Fact]
    public async Task ANotebookWithNothingToExport_HasNoExportMenu()
    {
        await SaveAsync("empty.verso");
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "empty.verso");

        await Expect(page.Locator("#toolbar button[data-button='verso.action.run-all']")).ToBeVisibleAsync();
        await Expect(page.Locator("#export")).ToBeHiddenAsync();
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
