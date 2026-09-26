// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Pipelines;
using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// As a cell's text is typed, the page asks the cell's kernel what may come next and what a word means, as Verso's
// editors ask it: Ctrl+Space, a dot or a quote lists what is offered, the arrows walk it, Enter or Tab takes one — in
// place of what was typed of it — and Escape closes it; a line under the text says what the word at the cursor means.
public sealed partial class PageTests
{
    private static ILocator Offered(ILocator cell) => cell.Locator(".completions li");

    [Fact]
    public async Task ABlocksVerbs_AreOfferedAsTheStepIsTyped_AndTheOneTakenIsWrittenInPlaceOfWhatWasTyped()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var text = Cell(page, 3).Locator("textarea.source");

        await text.FillAsync("""{"step": "fill""");
        await text.PressAsync("Control+Space");

        var missing = Offered(Cell(page, 3)).Filter(new() { HasText = "fill.missing" });

        await Expect(missing).ToBeVisibleAsync();

        await missing.ClickAsync();

        await Expect(text).ToHaveValueAsync(new Regex(Regex.Escape("""{"step": "fill.missing""")));
        await Expect(Cell(page, 3).Locator(".completions")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task TheArrowsWalkWhatIsOffered_EnterTakesIt_AndEscapeClosesIt()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var text = Cell(page, 3).Locator("textarea.source");

        await text.FillAsync("""{"step": "fill""");
        await text.PressAsync("Control+Space");
        await Expect(Offered(Cell(page, 3)).First).ToBeVisibleAsync();

        await text.PressAsync("Escape");

        await Expect(Cell(page, 3).Locator(".completions")).ToBeHiddenAsync();
        await Expect(text).ToHaveValueAsync("""{"step": "fill""");

        await text.PressAsync("Control+Space");
        await Expect(Offered(Cell(page, 3)).Nth(1)).ToBeVisibleAsync();

        var second = await Offered(Cell(page, 3)).Nth(1).TextContentAsync();

        await text.PressAsync("ArrowDown");
        await text.PressAsync("Enter");

        await Expect(text).ToHaveValueAsync(new Regex(Regex.Escape($"\"step\": \"{second}")));
    }

    [Fact]
    public async Task ACSharpCell_IsOfferedTheMembers_AfterADot()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = string.Empty });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "code.verso");
        var text = Cell(page, 0).Locator("textarea.source");

        await text.FocusAsync();
        await text.PressSequentiallyAsync("System.Console.");

        await Expect(Offered(Cell(page, 0)).Filter(new() { HasText = "WriteLine" }).First).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task TheLineUnderABlocksText_SaysWhatTheWordAtTheCursorMeans()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var text = Cell(page, 3).Locator("textarea.source");

        // The cursor inside the verb, where fill.missing's own words say what the step does.
        await text.EvaluateAsync("text => { text.focus(); text.setSelectionRange(13, 13); }");
        await text.PressAsync("ArrowRight");

        await Expect(Cell(page, 3).Locator(".hover-line")).ToHaveTextAsync(StepCatalog.BuiltIn().Describe("fill.missing").Purpose);
    }
}
