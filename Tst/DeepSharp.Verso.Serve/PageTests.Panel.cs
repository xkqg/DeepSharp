// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.RegularExpressions;
using DeepSharp.Tests.Serve.Parts;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The panels beside the notebook are Verso's editor's: none open at first, one at a time, each opened and closed from its
// toggle. A cell's properties panel is there only in a layout that has one — the notebook's own, not the dashboard or the
// presentation — and a switch to a layout that has one offers it again; it says so while it reads another cell's
// properties, and says a cell has none when they cannot be read. The View panel lists the layouts and the themes
// the engine has and switches them as chosen; the two buttons that only cycle through them are not on the toolbar; a
// theme chosen draws the page in it, in every tab, and is saved with the notebook — until one is chosen, the page draws
// in its own look.
public sealed partial class PageTests
{
    private static Microsoft.Playwright.ILocator Toggle(Microsoft.Playwright.IPage page, string panel) => page.Locator($"#panels button[data-panel='{panel}']");

    [Fact]
    public async Task ANotebookInTheDashboard_OffersNoPropertiesPanel_AndOnceSwitchedToItsOwnLayout_OffersItForTheCellChosen()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        foreach (var step in Titanic)
        {
            notebook.Cells.Add(Block(step));
        }

        await File.WriteAllTextAsync(At("dashboard.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "dashboard.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        await Expect(Toggle(page, "view")).ToBeVisibleAsync();
        await Expect(Toggle(page, "properties")).ToHaveCountAsync(0);

        // Chosen in the View panel: the notebook's own layout, which has a properties panel.
        await Toggle(page, "view").ClickAsync();
        await Expect(page.Locator("#panel .panel-title")).ToHaveTextAsync("View");
        await page.Locator("#panel .view-row[data-layout='notebook']").ClickAsync();

        await Expect(Toggle(page, "properties")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        await Toggle(page, "properties").ClickAsync();
        await Expect(page.Locator("#panel .panel-title")).ToHaveTextAsync("Cell Properties");

        // The cell chosen as the notebook opened — the first not shown rendered — is the one the panel shows.
        await Expect(page.Locator("#panel")).ToContainTextAsync("Reads the rows from a comma-separated file.", new() { Timeout = 10_000 });

        await Cell(page, 4).Locator(".cell-bar").ClickAsync();
        await Expect(page.Locator($"#panel select[data-part='{StepForm.Id}'][data-field='scale']")).ToBeVisibleAsync(new() { Timeout = 10_000 });
    }

    [Fact]
    public async Task ThePropertiesPanel_SaysItReadsAnotherCellsProperties_AndThatACellHasNone_WhenTheyCannotBeRead()
    {
        await SaveAsync("two.verso", Code("var a = 1;"), Code("var b = 2;"));
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var held = await HeldSocket.RouteAsync(page);

        await page.GotoAsync($"{served.Address}?token={Token}&notebook=two.verso");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "snapshot");

        // The panel opens on the cell chosen as the notebook opened, saying it reads its properties until they come.
        await Toggle(page, "properties").ClickAsync();
        await Expect(page.Locator("#panel .panel-body")).ToHaveTextAsync("Loading...");
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "answer");
        await Expect(OutputField(page)).ToBeVisibleAsync();

        // Another cell chosen: the first cell's fields go at once, so none of them changes a cell no longer chosen.
        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Expect(page.Locator("#panel .panel-body")).ToHaveTextAsync("Loading...");
        await Expect(OutputField(page)).ToHaveCountAsync(0);
        await held.LetThroughAsync(frame => HeldSocket.Type(frame) == "answer");
        await Expect(OutputField(page)).ToBeVisibleAsync();

        // The same cell chosen again: its fields stay, and they are not read again.
        var asked = held.Asked.Count(each => each == "properties");

        await Cell(page, 1).Locator(".cell-bar").ClickAsync();
        await Expect(OutputField(page)).ToBeVisibleAsync();
        Assert.Equal(asked, held.Asked.Count(each => each == "properties"));

        // With no connection a cell's properties cannot be read, and the panel says the cell has none.
        held.Refusing = true;
        await held.CloseAsync();
        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Expect(page.Locator("#panel .panel-body")).ToHaveTextAsync("No properties available for this cell.");
    }

    [Fact]
    public async Task APanelsFields_AreReadAndHandedOn_AsVersosBrowserEditorReadsAndHandsThem()
    {
        var cell = Code("1 + 1");

        cell.Metadata[ReadingPart.Marked] = true;
        cell.Metadata["verso:ui.layoutVisibility"] = new Dictionary<string, string> { ["dashboard"] = "Hidden" };
        await SaveAsync("fields.verso", cell);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "fields.verso");
        var reading = (string field) => page.Locator($"#panel [data-part='{ReadingPart.Part}'][data-field='{field}']");

        await Toggle(page, "properties").ClickAsync();

        // A select's value found among its choices without regard to case, a switch written as a word, and several
        // choices written as one line of words — trimmed, and without regard to case.
        await Expect(page.Locator("#panel select[data-part='verso.propertyprovider.visibility'][data-field='visibility:dashboard']")).ToHaveValueAsync("hidden", new() { Timeout = 10_000 });
        await Expect(reading("flag")).ToBeCheckedAsync();
        await Expect(reading("choices").Locator("input[value='a']")).ToBeCheckedAsync();
        await Expect(reading("choices").Locator("input[value='b']")).Not.ToBeCheckedAsync();

        // A field says what it is under its label.
        await Expect(page.Locator("#panel .field-description").First).ToHaveTextAsync(new Regex("^Default: "));

        // Handed on as Verso's browser editor hands them: a number as a double, and a blank one as its text; several
        // choices as the list they were, with the one ticked added and the word that is not among the choices kept; a
        // switch as a bool.
        await reading("count").FillAsync("3.5");
        await reading("count").BlurAsync();
        await GotAsync(served, "fields.verso", "count", "Double:3.5");

        await reading("count").FillAsync(string.Empty);
        await reading("count").BlurAsync();
        await GotAsync(served, "fields.verso", "count", "String:");

        await reading("choices").Locator("input[value='b']").CheckAsync();
        await GotAsync(served, "fields.verso", "choices", "List`1:A|zz|b");

        await reading("flag").UncheckAsync();
        await GotAsync(served, "fields.verso", "flag", "Boolean:False");
    }

    // Waits until the part writes down that it was handed the value; says what it last had when it never does.
    private static async Task GotAsync(Served served, string notebook, string field, string expected)
    {
        string? got = null;

        for (var waited = 0; waited < 10_000; waited += 50)
        {
            var cell = (await CurrentAsync(served, notebook)).Cells[0];

            got = cell.Metadata.TryGetValue(ReadingPart.Got + field, out var written) ? JsonSerializer.Deserialize<string>(written) : null;

            if (got == expected)
            {
                return;
            }

            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        Assert.Equal(expected, got);
    }

    [Fact]
    public async Task NoPanelIsOpenAtFirst_OneOpensAtATime_AndItsCollapseClosesIt()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        await Cell(page, 4).Locator(".cell-bar").ClickAsync();
        await Expect(page.Locator("#panel")).ToBeEmptyAsync();

        await Toggle(page, "properties").ClickAsync();
        await Expect(Toggle(page, "properties")).ToHaveAttributeAsync("aria-pressed", "true");
        await Expect(page.Locator($"#panel select[data-part='{StepForm.Id}'][data-field='scale']")).ToBeVisibleAsync(new() { Timeout = 10_000 });

        await Toggle(page, "view").ClickAsync();
        await Expect(page.Locator("#panel .panel-title")).ToHaveTextAsync("View");
        await Expect(Toggle(page, "properties")).ToHaveAttributeAsync("aria-pressed", "false");

        await page.Locator("#panel .panel-collapse").ClickAsync();
        await Expect(page.Locator("#panel")).ToBeEmptyAsync();
    }

    [Fact]
    public async Task TheViewPanel_ListsTheEnginesLayoutsAndThemes_AndSwitchesTheLayoutAsChosen()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Toggle(page, "view").ClickAsync();

        var layouts = page.Locator("#panel .view-row[data-layout]");

        // Verso's three, and the page suite's own two: one that fails to draw, one of slots.
        await Expect(layouts).ToHaveCountAsync(5);
        await Expect(page.Locator("#panel .view-row[data-layout='dashboard'] .chip")).ToHaveTextAsync("Read only");
        await Expect(page.Locator("#panel .view-row[data-layout='notebook'] .chip")).ToHaveCountAsync(0);
        await Expect(page.Locator("#panel .view-row[data-layout='notebook']")).ToHaveAttributeAsync("aria-pressed", "true");
        await Expect(page.Locator("#panel .view-row[data-theme]")).ToHaveCountAsync(3);

        await page.Locator("#panel .view-row[data-layout='dashboard']").ClickAsync();

        await Expect(page.Locator("#panel .view-row[data-layout='dashboard']")).ToHaveAttributeAsync("aria-pressed", "true", new() { Timeout = 10_000 });
        Assert.Equal("dashboard", (await CurrentAsync(served, "titanic.verso")).Layout.Id);
    }

    [Fact]
    public async Task ChoosingATheme_DrawsThePageInIt_InEveryTab_AndItIsSavedWithTheNotebook()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var first = await OpenAsync(served, "titanic.verso");
        var second = await OpenAsync(served, "titanic.verso");
        const string ground = "getComputedStyle(document.body).backgroundColor";

        await Expect(second.Locator("section.cell")).ToHaveCountAsync(5);

        // Until a theme is chosen the page draws in its own look, which a theme's background is not.
        Assert.NotEqual("rgb(30, 30, 30)", await second.EvaluateAsync<string>(ground));

        await Toggle(first, "view").ClickAsync();
        await first.Locator("#panel .view-row[data-theme='verso-dark']").ClickAsync();

        await Expect(first.Locator("#panel .view-row[data-theme='verso-dark']")).ToHaveAttributeAsync("aria-pressed", "true", new() { Timeout = 10_000 });

        foreach (var page in new[] { first, second })
        {
            for (var waited = 0; await page.EvaluateAsync<string>(ground) != "rgb(30, 30, 30)"; waited += 50)
            {
                Assert.True(waited < 10_000, "a tab was never drawn in the theme chosen");
                await Task.Delay(50, TestContext.Current.CancellationToken);
            }
        }

        await first.Locator("#save").ClickAsync();

        for (var waited = 0; (await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(At("titanic.verso"), TestContext.Current.CancellationToken))).PreferredThemeId != "verso-dark"; waited += 50)
        {
            Assert.True(waited < 10_000, "the theme chosen was never saved with the notebook");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task TheButtonsThatOnlyCycleLayoutsAndThemes_AreNotOnTheToolbar()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Expect(page.Locator("#toolbar button[data-button='verso.action.run-all']")).ToBeVisibleAsync();
        await Expect(page.Locator("#toolbar button[data-button='verso.switchLayout']")).ToHaveCountAsync(0);
        await Expect(page.Locator("#toolbar button[data-button='verso.switchTheme']")).ToHaveCountAsync(0);
        await Expect(Toggle(page, "view")).ToHaveAttributeAsync("aria-pressed", new Regex("false"));
    }
}
