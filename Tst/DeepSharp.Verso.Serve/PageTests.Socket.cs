// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Microsoft.Playwright;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Each tab holds one socket to its notebook, as Verso's editor holds one connection per tab, so nothing a page asks
// keeps anything else from the server: a Stop gets through however many asks wait behind the run it stops, and any
// number of tabs draw and are answered.
public sealed partial class PageTests
{
    [Fact]
    public async Task AStop_IsAnsweredInUnderASecond_WhileThirtyAsksWaitBehindTheRunItStops()
    {
        await SaveAsync("runaway.verso", EndlessCell());
        await using var served = await StartAsync();
        var context = await browsers.Browser.NewContextAsync();
        var page = await context.NewPageAsync();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        TimeSpan? stopped = null;

        // When the page is told the run it named is stopped: the Stop's own answer, on the page's socket.
        page.WebSocket += (_, socket) => socket.FrameReceived += (_, frame) =>
        {
            if (frame.Text?.Contains("\"stopped\":true", StringComparison.Ordinal) == true)
            {
                stopped ??= clock.Elapsed;
            }
        };
        await page.GotoAsync($"{served.Address}?token={Token}&notebook=runaway.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });

        // Thirty cells asked for while the cell runs, each waiting its turn behind the run.
        await page.EvaluateAsync("() => { const add = document.querySelector(\"#adding button[data-type='markdown']\"); for (let i = 0; i < 30; i++) add.click(); }");
        await Expect(page.Locator("#status")).ToHaveTextAsync("Waits for the run under way to end…");

        var pressed = clock.Elapsed;

        await page.Locator("#stop").ClickAsync();

        // The run ends once its kernel has started afresh, and every ask behind it is made after it.
        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(page.Locator("section.cell")).ToHaveCountAsync(31, new() { Timeout = 30_000 });

        Assert.NotNull(stopped);
        Assert.True(stopped - pressed < TimeSpan.FromSeconds(1), $"the stop was answered {(stopped - pressed)?.TotalMilliseconds:0} ms after it was pressed");
    }

    [Fact]
    public async Task FourteenTabsOnTwoNotebooks_AllDraw_AndASaveInTheFirstIsAnsweredAtOnce()
    {
        await SaveAsync("a.verso", [.. Titanic.Select(Block)]);
        await SaveAsync("b.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();

        // One browser, as a person has one: its tabs share its connections to the server.
        var context = await browsers.Browser.NewContextAsync();
        var tabs = new List<IPage>();

        for (var tab = 0; tab < 14; tab++)
        {
            var page = await context.NewPageAsync();

            await page.GotoAsync($"{served.Address}?token={Token}&notebook={(tab % 2 == 0 ? "a" : "b")}.verso");
            tabs.Add(page);
        }

        foreach (var page in tabs)
        {
            await Expect(page.Locator("section.cell")).ToHaveCountAsync(5, new() { Timeout = 30_000 });
        }

        await tabs[0].Locator("#save").ClickAsync();

        await Expect(tabs[0].Locator("#notice")).ToHaveTextAsync(new Regex("^Saved to "), new() { Timeout = 1_000 });
    }
}
