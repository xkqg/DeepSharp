// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The run controls as Verso's editor draws them: while a run is under way the one Stop stands where the primary button —
// Run All — stood, so no second run is pressed there; a cell's run cannot be pressed then, and says why when the run is
// another cell's; leaving the text of a cell shown rendered renders it only when something is written there; and a cell
// nobody writes shows no run line.
public sealed partial class PageTests
{
    private const string RunSays = "Run the cell (Shift+Enter)";

    [Fact]
    public async Task WhileARunIsUnderWay_TheStopStandsWhereRunAllStood_AndACellsRunSaysWhyItCannotBePressed()
    {
        var go = At("go");

        await SaveAsync("held.verso", HeldCell(go), Code("2 + 2"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "held.verso");
        var runAll = page.Locator("#toolbar button[data-button='verso.action.run-all']");

        await Expect(runAll).ToBeVisibleAsync();
        await Expect(page.Locator("#stop")).ToBeHiddenAsync();

        try
        {
            await Cell(page, 0).Locator("button.run").ClickAsync();

            await Expect(page.Locator("#toolbar #stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });
            await Expect(runAll).ToHaveCountAsync(0);
            await Expect(Cell(page, 1).Locator("button.run")).ToBeDisabledAsync();
            await Expect(Cell(page, 1).Locator("button.run")).ToHaveAttributeAsync("title", $"{RunSays}. Another cell is running.");
            await Expect(Cell(page, 0).Locator("button.run")).ToHaveAttributeAsync("title", RunSays);
        }
        finally
        {
            await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
        }

        await Expect(runAll).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(page.Locator("#stop")).ToBeHiddenAsync();
        await Expect(Cell(page, 1).Locator("button.run")).ToBeEnabledAsync();
        await Expect(Cell(page, 1).Locator("button.run")).ToHaveAttributeAsync("title", RunSays);
    }

    [Fact]
    public async Task LeavingTheTextOfACellShownRendered_RendersItOnlyWhenSomethingIsWrittenThere()
    {
        await SaveAsync("empty.verso", new CellModel { Type = "markdown", Source = string.Empty }, Code("1 + 1"));
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "empty.verso");

        // Written in, then left for the next cell, whose typing reaches the notebook after anything leaving asked for.
        await Cell(page, 0).Locator("textarea.source").ClickAsync();
        await Cell(page, 1).Locator("textarea.source").ClickAsync();
        await Cell(page, 1).Locator("textarea.source").FillAsync("2 + 2");

        for (var waited = 0; (await CurrentAsync(served, "empty.verso")).Cells[1].Source != "2 + 2"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing never reached the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        Assert.Null((await CurrentAsync(served, "empty.verso")).Cells[0].ExecutionCount);
        await Expect(Cell(page, 0).Locator(".cell-bar .status")).ToHaveTextAsync(string.Empty);
    }

    [Fact]
    public async Task ACellNobodyWrites_ShowsNoRunLine()
    {
        await SaveAsync("parameters.verso", ParametersCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "parameters.verso");

        await Expect(Cell(page, 0).Locator("[data-action='parameter-add']")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0).Locator(".cell-bar .status")).ToHaveTextAsync(string.Empty);
    }
}
