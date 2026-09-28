// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The Metadata panel shows what the notebook says of itself, as Verso's does — its title, which a person changes there and
// which names what the notebook is exported as, its default kernel, its file, when it was made and last saved, and the
// version of its format — in every tab. A title being written stays the person's own until they leave it.
public sealed partial class PageTests
{
    [Fact]
    public async Task TheMetadataPanel_ShowsWhatTheNotebookSaysOfItself_AndRetitlesItInEveryTab()
    {
        var notebook = new NotebookModel { Title = "Passengers", DefaultKernelId = "csharp", Created = new DateTimeOffset(2020, 1, 2, 3, 4, 0, TimeSpan.Zero) };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await SaveAsync("said.verso", notebook);
        await using var served = await StartAsync();
        var first = await OpenAsync(served, "said.verso");
        var second = await OpenAsync(served, "said.verso");

        // First among the panels, as Verso's editor has it.
        await Expect(first.Locator("#panels button").First).ToHaveAttributeAsync("data-panel", "metadata");
        await Toggle(first, "metadata").ClickAsync();

        var panel = first.Locator("#panel");
        var title = panel.Locator("input[data-meta='title']");

        await Expect(panel.Locator(".panel-title")).ToHaveTextAsync("Metadata");
        await Expect(title).ToHaveValueAsync("Passengers");
        await Expect(title).ToHaveAttributeAsync("placeholder", "Untitled");
        await Expect(panel.Locator("[data-meta='defaultKernel']")).ToHaveTextAsync("C# (Roslyn)");
        await Expect(panel.Locator("[data-meta='file']")).ToHaveTextAsync("said.verso");
        await Expect(panel.Locator("[data-meta='created']")).ToHaveTextAsync("2020-01-02 03:04");
        await Expect(panel.Locator("[data-meta='modified']")).ToHaveTextAsync("—");
        await Expect(panel.Locator("[data-meta='formatVersion']")).ToHaveTextAsync("1.1");

        // A title changed here is the notebook's, in every tab.
        await title.FillAsync("Survivors");
        await title.PressAsync("Enter");
        await TitledAsync(served, "said.verso", "Survivors");

        await Toggle(second, "metadata").ClickAsync();
        await Expect(second.Locator("#panel input[data-meta='title']")).ToHaveValueAsync("Survivors", new() { Timeout = 10_000 });
        await Expect(second.Locator("#save .unsaved")).ToBeVisibleAsync();

        // A title being written stays the person's own while the notebook tells something else of itself: saved, it tells
        // when, in every tab.
        await using var other = await SocketAsync(served, "said.verso");

        await title.FillAsync("Survivors of");
        await other.AskAsync("save");
        await Expect(panel.Locator("[data-meta='modified']")).Not.ToHaveTextAsync("—", new() { Timeout = 10_000 });
        await Expect(second.Locator("#panel [data-meta='modified']")).Not.ToHaveTextAsync("—", new() { Timeout = 10_000 });
        await Expect(title).ToHaveValueAsync("Survivors of");

        await title.PressAsync("Enter");
        await TitledAsync(served, "said.verso", "Survivors of");

        // Looked at and left as it was, the title shows what another tab made it meanwhile.
        await other.AskAsync("save");
        await Expect(first.Locator("#save .unsaved")).ToBeHiddenAsync(new() { Timeout = 10_000 });
        await title.BlurAsync();
        await title.FocusAsync();
        await other.AskAsync("title", new { title = "The survivors" });
        await Expect(first.Locator("#save .unsaved")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Expect(title).ToHaveValueAsync("Survivors of");

        await title.BlurAsync();
        await Expect(title).ToHaveValueAsync("The survivors");
    }

    [Fact]
    public async Task BesideItsName_TheNotebookSaysItsKernelAndHowManyCellsItHas_AsVersosToolbarSaysIt()
    {
        var titanic = new NotebookModel { DefaultKernelId = "csharp" };
        var two = new NotebookModel { DefaultKernelId = "csharp" };
        var one = new NotebookModel();

        titanic.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        titanic.Cells.Add(new CellModel { Type = "markdown", Source = "# Notes" });
        two.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        two.Cells.Add(new CellModel { Type = "code", Language = "python", Source = "1 + 1" });
        one.Cells.Add(new CellModel { Type = "markdown", Source = "# Notes" });
        await SaveAsync("passengers-of-the-titanic.verso", titanic);
        await SaveAsync("two.verso", two);
        await SaveAsync("one.verso", one);
        await using var served = await StartAsync();

        // A name past twenty characters is cut short, and whole in the tip.
        var page = await OpenAsync(served, "passengers-of-the-titanic.verso");

        await Expect(page.Locator("#doc-info .meta")).ToHaveTextAsync("C# (Roslyn) · 2 cells");
        await Expect(page.Locator("#notebook-name")).ToHaveTextAsync("passengers-of-the-ti…");
        await Expect(page.Locator("#doc-info")).ToHaveAttributeAsync("title", "passengers-of-the-titanic.verso");

        // A cell added is counted.
        await page.Locator("#adding button[data-type='markdown']").ClickAsync();
        await Expect(page.Locator("#doc-info .meta")).ToHaveTextAsync("C# (Roslyn) · 3 cells");

        // Code cells in another language are counted beside the kernel, and the tip names every kernel.
        var polyglot = await OpenAsync(served, "two.verso");

        await Expect(polyglot.Locator("#doc-info .meta")).ToHaveTextAsync("C# (Roslyn) +1 · 2 cells");
        await Expect(polyglot.Locator("#doc-info")).ToHaveAttributeAsync("title", "two.verso\nKernels: C# (Roslyn), python");

        // A notebook naming no kernel says so; one cell is a cell.
        var single = await OpenAsync(served, "one.verso");

        await Expect(single.Locator("#doc-info .meta")).ToHaveTextAsync("No kernel · 1 cell");
        await Expect(single.Locator("#doc-info")).Not.ToHaveAttributeAsync("title", new System.Text.RegularExpressions.Regex("."));
    }

    // Waits until the notebook the server holds has the title.
    private static async Task TitledAsync(Served served, string notebook, string title)
    {
        for (var waited = 0; (await CurrentAsync(served, notebook)).Metadata.Title != title; waited += 50)
        {
            Assert.True(waited < 10_000, $"the notebook was never titled '{title}'");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }
    }
}
