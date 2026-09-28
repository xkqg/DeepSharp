// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// How a cell shows what it shows, as the part that keeps it says and Verso's editor draws it. Hidden: a line in its place
// says how many outputs are hidden, and a click on that line chooses a cell shown rendered. Cut short: each text and each
// failure to as many lines as the cell keeps — five when it keeps no count above nought — and whatever is drawn as a page
// to 160 pixels; standard error, progress and widgets are never cut. The part's word is read without regard to case.
public sealed partial class PageTests
{
    private static readonly string TenLines = string.Join('\n', Enumerable.Range(1, 10).Select(line => $"line {line}"));

    private static CellModel ShownAs(string visibility, params CellOutput[] outputs)
    {
        var cell = Code("1 + 1");

        cell.Outputs.AddRange(outputs);
        cell.Metadata["verso:ui.outputVisibility"] = visibility;

        return cell;
    }

    // How many lines a text shows: its height over its line's.
    private static Task<int> LinesAsync(ILocator text) =>
        text.EvaluateAsync<int>("e => Math.round(e.getBoundingClientRect().height / parseFloat(getComputedStyle(e).lineHeight))");

    [Fact]
    public async Task HiddenOutputs_AreSaidToBeHiddenInTheirPlace_AndTheSayingChoosesACellShownRendered()
    {
        var notes = Rendered("# Notes", "<h1>Notes</h1>");

        notes.Metadata["verso:ui.outputVisibility"] = "hidden";
        await SaveAsync("hidden.verso", ShownAs("Hidden", CellOutput.Plain("one"), CellOutput.Plain("two")), ShownAs("hidden", CellOutput.Plain("one")), notes);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "hidden.verso");

        await Expect(Cell(page, 0).Locator(".output-summary")).ToHaveTextAsync("2 outputs hidden");
        await Expect(Cell(page, 0).Locator(".outputs")).ToBeHiddenAsync();
        await Expect(Cell(page, 1).Locator(".output-summary")).ToHaveTextAsync("1 output hidden");
        await Expect(Cell(page, 2).Locator(".output-summary")).ToHaveTextAsync("1 output hidden");
        await Expect(Cell(page, 2).Locator(".outputs")).ToBeHiddenAsync();

        // A code cell's saying chooses nothing; a cell shown rendered is chosen by it, and shows its text.
        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Cell(page, 0).Locator(".output-summary").ClickAsync();
        await Expect(Cell(page, 1)).ToHaveClassAsync(Chosen);

        await Cell(page, 2).Locator(".output-summary").ClickAsync();
        await Expect(Cell(page, 2)).ToHaveClassAsync(Chosen);
        await Expect(Cell(page, 2).Locator("textarea.source")).ToBeVisibleAsync();
        await Expect(Cell(page, 2).Locator(".output-summary")).ToBeHiddenAsync();

        // Shown in full again through the panel: the outputs are drawn, and the saying goes.
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Toggle(page, "properties").ClickAsync();
        await OutputField(page).SelectOptionAsync("expanded");
        await Expect(Cell(page, 0).Locator(".outputs")).ToHaveTextAsync("onetwo", new() { Timeout = 10_000 });
        await Expect(Cell(page, 0).Locator(".output-summary")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task OutputsCutShort_ShowEachTextAndFailureToTheLinesTheCellKeeps_AndWhatIsDrawnAsAPageTo160Pixels()
    {
        var kept = ShownAs(
            "preview",
            CellOutput.Plain(TenLines),
            new CellOutput("text/plain", TenLines, IsError: true, ErrorName: "Failure", ErrorStackTrace: TenLines),
            new CellOutput("text/plain", TenLines) { Channel = OutputChannel.Stderr },
            CellOutput.Html("<div style='height:400px'>tall</div>"));
        var none = ShownAs("Preview", CellOutput.Plain(TenLines));

        kept.Metadata["verso:ui.outputPreviewLineCount"] = 3;
        none.Metadata["verso:ui.outputPreviewLineCount"] = 0;
        await SaveAsync("preview.verso", kept, none);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "preview.verso");

        await Expect(Cell(page, 0).Locator(".outputs .error-name")).ToHaveTextAsync("Failure");
        Assert.Equal(3, await LinesAsync(Cell(page, 0).Locator(".outputs > pre.text")));
        Assert.Equal(3, await LinesAsync(Cell(page, 0).Locator(".outputs .error-content")));
        Assert.Equal(3, await LinesAsync(Cell(page, 0).Locator(".outputs .error-stack")));
        Assert.Equal(10, await LinesAsync(Cell(page, 0).Locator(".outputs .stderr pre")));
        Assert.Equal(160, await Cell(page, 0).Locator(".outputs > .html").EvaluateAsync<int>("e => Math.round(e.getBoundingClientRect().height)"));

        // A count that is not above nought is read as Verso reads it: five lines.
        Assert.Equal(5, await LinesAsync(Cell(page, 1).Locator(".outputs > pre.text")));
    }
}
