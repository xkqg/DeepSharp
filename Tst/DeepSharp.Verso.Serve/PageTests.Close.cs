// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Microsoft.Extensions.DependencyInjection;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// A close goes past everything asked before it, as a stop does, since what waits may wait for the very run the close
// stops; it says the run under way is stopped before a person agrees; what was asked before it and waited behind the run
// is refused; and the page opens no socket after it, since that would open the closed notebook again.
public sealed partial class PageTests
{
    private static CellModel EndlessCell() =>
        new() { Type = "code", Language = "csharp", Source = "while (true) { await System.Threading.Tasks.Task.Delay(10); }" };

    [Fact]
    public async Task ClosingWhileARunNeverEnds_ClosesAtOnce_AndTheQuestionSaysTheRunIsStopped()
    {
        await SaveAsync("endless.verso", EndlessCell());
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "endless.verso");
        string? asked = null;

        page.Dialog += async (_, dialog) =>
        {
            asked = dialog.Message;
            await dialog.AcceptAsync();
        };

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        await page.Locator("#close").ClickAsync();

        await Expect(page.Locator("#notebooks")).ToBeVisibleAsync(new() { Timeout = 10_000 });
        Assert.Contains("The run under way is stopped", asked, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Closing_RefusesWhatWaitedBehindTheRun_AndOpensNoSocketAfter_SoNothingOpensTheNotebookAgain()
    {
        await SaveAsync("endless.verso", EndlessCell(), new CellModel { Type = "code", Language = "csharp", Source = "var kept = 1;" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "endless.verso");
        var opened = new List<string>();
        var closing = false;

        page.Dialog += async (_, dialog) =>
        {
            closing = true;
            await dialog.AcceptAsync();
        };
        page.WebSocket += (_, socket) =>
        {
            if (closing)
            {
                opened.Add(socket.Url);
            }
        };

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        // Typing while the cell runs is sent, and waits behind the run at the notebook, as every change does.
        var text = Cell(page, 1).Locator("textarea.source");

        await text.FillAsync("var typed = 2;");
        await text.BlurAsync();
        await Expect(page.Locator("#status")).ToHaveTextAsync("Waits for the run under way to end…");

        await page.Locator("#close").ClickAsync();
        await Expect(page.Locator("#notebooks")).ToBeVisibleAsync(new() { Timeout = 10_000 });

        // Longer than the page takes to open a socket again.
        await Task.Delay(TimeSpan.FromSeconds(2), TestContext.Current.CancellationToken);

        Assert.Empty(opened);

        // The notebook opens again as its file says: what waited behind the run was refused when the close came.
        var again = await served.App.Services.GetRequiredService<OpenNotebooks>().OpenAsync(At("endless.verso"), TestContext.Current.CancellationToken);

        Assert.Equal("var kept = 1;", again.Cells[1].Source);
    }
}
