// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Microsoft.Playwright;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Diagrams and formulas are drawn as Verso's editors draw them — Mermaid's diagrams, KaTeX's formulas — from what the
// page carries, so they are drawn with every other address out of reach, and only once a diagram or a formula first
// shows: a page with neither never runs either library. A diagram out of sight waits until it shows, since Mermaid
// measures its text where it stands.
public sealed partial class PageTests
{
    private const string Diagram = "graph TD; A-->B";

    // A page on a notebook with every address but the server's out of reach, as on a computer with no network: what the
    // page draws, it draws from what it carries.
    private async Task<IPage> UnreachedAsync(Served served, string notebook)
    {
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var own = served.Address.GetLeftPart(UriPartial.Authority);

        await page.RouteAsync(address => !address.StartsWith(own, StringComparison.Ordinal), route => route.AbortAsync());
        await page.GotoAsync($"{served.Address}?token={Token}&notebook={notebook}");

        return page;
    }

    [Fact]
    public async Task AMermaidCell_IsDrawnAsItsDiagram_WithEveryOtherAddressOutOfReach()
    {
        await SaveAsync("diagram.verso", new CellModel { Type = "mermaid", Source = Diagram });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "diagram.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs .mermaid svg")).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task AMermaidFenceInMarkdown_IsDrawnAsItsDiagram()
    {
        await SaveAsync("fence.verso", new CellModel { Type = "markdown", Source = $"```mermaid\n{Diagram}\n```" });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "fence.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs .mermaid svg")).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task MathInMarkdown_IsTypesetAsKaTeXTypesetsIt_WithItsOwnFaces()
    {
        // A formula in the line, and one on lines of its own, as Markdown writes a displayed one.
        await SaveAsync("math.verso", new CellModel { Type = "markdown", Source = "The area is $\\pi r^2$.\n\n$$\ne^{i\\pi} + 1 = 0\n$$" });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "math.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs .math .katex")).ToHaveCountAsync(2, new() { Timeout = 30_000 });
        await Expect(Cell(page, 0).Locator(".outputs div.math .katex-display")).ToHaveCountAsync(1);

        // Drawn in KaTeX's own faces, which the page carries, not in a face the computer had.
        Assert.True(await page.EvaluateAsync<bool>("async () => { await document.fonts.ready; return document.fonts.check('16px KaTeX_Main') && [...document.fonts].some(face => face.family.includes('KaTeX_Main') && face.status === 'loaded'); }"));

        // Each typeset once, from what was written, however many versions come after: a save is one more.
        await page.Locator("#save").ClickAsync();
        await Expect(page.Locator("#notice")).ToHaveTextAsync(new Regex("^Saved to "));
        await Expect(Cell(page, 0).Locator(".outputs .math annotation")).ToHaveTextAsync(["\\pi r^2", "e^{i\\pi} + 1 = 0"]);
    }

    [Fact]
    public async Task ADiagramInAPaneThatUnfolds_IsDrawnWhenItUnfolds()
    {
        // What a cell shows keeps a diagram in a folded pane: nothing on the page is drawn again as the pane unfolds.
        await SaveAsync("pane.verso", new CellModel
        {
            Type = "code",
            Language = "csharp",
            Source = "1 + 1",
            Outputs = { new CellOutput("text/html", $"<details><summary>The diagram</summary><div class=\"mermaid\">{Diagram}</div></details>") },
        });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "pane.verso");

        await Expect(Cell(page, 0).Locator(".outputs summary")).ToBeVisibleAsync();
        await Expect(Cell(page, 0).Locator(".outputs .mermaid svg")).ToHaveCountAsync(0);

        await Cell(page, 0).Locator(".outputs summary").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs .mermaid svg")).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task ADiagramOutOfSight_IsDrawnOnceItShows()
    {
        // A diagram the dashboard has no tile for: its cell waits in the list, out of sight, until the notebook's own
        // layout shows it.
        var cell = new CellModel { Type = "mermaid", Source = Diagram, Outputs = { new CellOutput("text/x-verso-mermaid", Diagram) } };

        cell.Metadata["verso:ui.layoutVisibility"] = new Dictionary<string, string> { ["dashboard"] = "Hidden" };
        await SaveInAsync("hidden.verso", "dashboard", cell);
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "hidden.verso");

        await Expect(page.Locator("#arrangement .verso-dashboard-grid")).ToBeAttachedAsync();
        await Expect(page.Locator(".mermaid svg")).ToHaveCountAsync(0);

        await Toggle(page, "view").ClickAsync();
        await page.Locator("#panel .view-row[data-layout='notebook']").ClickAsync();

        await Expect(Cell(page, 0).Locator(".outputs .mermaid svg")).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task ADiagramInThePresentation_IsDrawnOnceItsCellIsPlaced()
    {
        await SaveInAsync("slides.verso", "presentation", new CellModel { Type = "mermaid", Source = Diagram, Outputs = { new CellOutput("text/x-verso-mermaid", Diagram) } });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "slides.verso");

        await Expect(page.Locator("#arrangement .mermaid svg")).ToBeVisibleAsync(new() { Timeout = 30_000 });
    }

    [Fact]
    public async Task APageWithNoDiagramOrFormula_RunsNeitherLibrary()
    {
        await SaveAsync("plain.verso", new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "plain.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(Cell(page, 0).Locator(".outputs")).ToHaveTextAsync("2", new() { Timeout = 30_000 });

        Assert.True(await page.EvaluateAsync<bool>("() => window.mermaid === undefined && window.katex === undefined"));
    }
}
