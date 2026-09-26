// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The parameters form, as Verso's own editor hosts it: a cell nobody writes that shows nothing yet is run once when the
// notebook opens, so the form draws itself; and the form's controls, which name no part, reach Verso's parameters part
// the way Verso's own script sends them — a parameter added from its row, its value changed, and taken away.
public sealed partial class PageTests
{
    private static CellModel ParametersCell() => new() { Type = "parameters", Source = string.Empty };

    [Fact]
    public async Task AParametersCell_DrawsItsForm_OnceTheNotebookOpens()
    {
        await SaveAsync("parameters.verso", ParametersCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "parameters.verso");

        await Expect(Cell(page, 0).Locator("[data-action='parameter-add']")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0).Locator("textarea.source")).ToBeHiddenAsync();
    }

    [Fact]
    public async Task AParameterIsAdded_ItsValueChanged_AndTakenAway_ThroughTheForm()
    {
        await SaveAsync("parameters.verso", ParametersCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "parameters.verso");
        var form = Cell(page, 0);

        await form.Locator("[data-action='parameter-add']").ClickAsync(new() { Timeout = 30_000 });
        await form.Locator(".verso-parameter-add-row .verso-add-name").FillAsync("rate");
        await form.Locator(".verso-parameter-add-row .verso-add-default").FillAsync("0.5");
        await form.Locator(".verso-parameter-add-row [data-action='parameter-confirm-add']").ClickAsync();

        var rate = form.Locator("input[data-action='parameter-update'][data-param='rate']");

        await Expect(rate).ToHaveValueAsync("0.5");

        await rate.FillAsync("0.75");
        await rate.PressAsync("Enter");

        // The attribute, not the value typed into it: only the form the part drew again after the change carries it.
        await Expect(form.Locator("input[data-action='parameter-update'][data-param='rate']")).ToHaveAttributeAsync("value", "0.75");
        Assert.Contains("0.75", string.Concat((await CurrentAsync(served, "parameters.verso")).Cells[0].Outputs.Select(output => output.Content)), StringComparison.Ordinal);

        await form.Locator("[data-action='parameter-remove'][data-param='rate']").ClickAsync();

        await Expect(form.Locator("[data-param='rate']")).ToHaveCountAsync(0);
    }

    [Fact]
    public async Task TheFormsAddRow_OpensAndCloses_InThePageAlone()
    {
        await SaveAsync("parameters.verso", ParametersCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "parameters.verso");
        var form = Cell(page, 0);

        await form.Locator("[data-action='parameter-add']").ClickAsync(new() { Timeout = 30_000 });

        var before = (await CurrentAsync(served, "parameters.verso")).Version;

        await Expect(form.Locator(".verso-parameter-add-row")).ToBeVisibleAsync();

        await form.Locator(".verso-parameter-add-row [data-action='parameter-cancel-add']").ClickAsync();

        await Expect(form.Locator(".verso-parameter-add-row")).ToBeHiddenAsync();
        await Expect(form.Locator("[data-action='parameter-add']")).ToBeVisibleAsync();
        Assert.Equal(before, (await CurrentAsync(served, "parameters.verso")).Version);
    }
}
