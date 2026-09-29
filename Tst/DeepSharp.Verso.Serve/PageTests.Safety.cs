// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// Opening a notebook is not running it. What a notebook file carries — an output a cell showed when it last ran — is drawn
// as the HTML it is, with what in it would run taken out first: a handler on an element, an address that is script. Such
// a handler once ran as the page itself, which holds the notebook's socket and can run its cells. A widget, which runs
// code on purpose, does so in a frame of its own, and takes its look only from the page that made it.
public sealed partial class PageTests
{
    [Fact]
    public async Task AnHtmlOutputANotebookFileCarries_IsDrawn_AndNothingInItRunsAsThePage()
    {
        await SaveAsync("carried-output.verso", new CellModel
        {
            Type = "code",
            Language = "csharp",
            Source = "1 + 1",
            Outputs =
            {
                new CellOutput(
                    "text/html",
                    "<p id='said'>Hello</p><img id='broken' src='missing.png' onerror=\"document.body.dataset.ran = 'yes'\"><a id='link' href=\"javascript:document.body.dataset.ran = 'yes'\">link</a>"),
            },
        });
        await using var served = await StartAsync();
        var page = await UnreachedAsync(served, "carried-output.verso");
        var outputs = Cell(page, 0).Locator(".outputs");

        await Expect(outputs.Locator("#said")).ToHaveTextAsync("Hello", new() { Timeout = 30_000 });

        // The image is drawn and its source asked for; its handler is not there to run once the asking fails.
        await Expect(outputs.Locator("#broken")).ToHaveCountAsync(1);
        await page.WaitForFunctionAsync("() => document.querySelector('.outputs #broken')?.complete === true");
        await Expect(outputs.Locator("#broken")).Not.ToHaveAttributeAsync("onerror", new Regex("."));
        await Expect(outputs.Locator("#link")).Not.ToHaveAttributeAsync("href", new Regex("^javascript:", RegexOptions.IgnoreCase));
        Assert.Null(await page.EvaluateAsync<string?>("() => document.body.dataset.ran ?? null"));
    }

    [Fact]
    public async Task AWidget_TakesItsLookOnlyFromThePageThatMadeIt()
    {
        await using var served = await StartAsync();
        var page = await RunOneAsync(
            served,
            "looks.verso",
            """Verso.Abstractions.CellOutput.Widget("<!doctype html><html><body><div id='drawn'>A widget</div></body></html>")""");

        await WidgetLoadedAsync(page);

        // Another frame of the page tells the widget a look, and then the page itself does: once the page's is taken, the
        // other frame's has been read before it.
        await page.EvaluateAsync("""
            () => new Promise(done => {
                const other = document.createElement('iframe');
                other.srcdoc = "<script>parent.document.querySelector('iframe.widget').contentWindow.postMessage({ type: 'verso/widget-theme', values: { '--told-by-another': 'another' } }, '*');<\/script>";
                other.onload = () => {
                    document.querySelector('iframe.widget').contentWindow.postMessage({ type: 'verso/widget-theme', values: { '--told-by-the-page': 'page' } }, '*');
                    done();
                };
                document.body.append(other);
            })
            """);

        var content = (await (await Cell(page, 0).Locator(".outputs iframe.widget").ElementHandleAsync()).ContentFrameAsync())!;

        await content.WaitForFunctionAsync("() => getComputedStyle(document.documentElement).getPropertyValue('--told-by-the-page').trim() === 'page'");
        Assert.Equal(string.Empty, await content.EvaluateAsync<string>("() => getComputedStyle(document.documentElement).getPropertyValue('--told-by-another').trim()"));
    }
}
