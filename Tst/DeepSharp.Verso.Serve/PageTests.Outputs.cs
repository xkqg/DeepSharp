// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// What a cell shows is drawn as Verso's editor draws it: a failure with its name and where it happened, standard error
// labelled and not a failure, JSON as a tree, CSV as a table, progress as a bar, and a widget — a whole document — in a
// frame of its own that runs its code and reaches nothing of the page, sized to what it draws, whose downloads the page
// hands over.
public sealed partial class PageTests
{
    // A notebook of one C# cell, run once the page draws it.
    private async Task<Microsoft.Playwright.IPage> RunOneAsync(Served served, string name, string source)
    {
        await SaveAsync(name, new CellModel { Type = "code", Language = "csharp", Source = source });

        var page = await OpenAsync(served, name);

        await Cell(page, 0).Locator("button.run").ClickAsync();

        return page;
    }

    // A widget's document has run its own script once it has said how tall it came to, which it says when it is loaded: the
    // page marks the frame sized then. Its height is no sign of it, since a frame whose document says nothing within six
    // seconds is given a height all the same.
    private static Task WidgetLoadedAsync(Microsoft.Playwright.IPage page) =>
        Expect(Cell(page, 0).Locator(".outputs iframe.widget")).ToHaveJSPropertyAsync("sized", true, new() { Timeout = 30_000 });

    [Fact]
    public async Task AFailure_IsDrawnWithItsName_AndWhereItHappened_WhenTheKernelKnowsIt()
    {
        await SaveAsync(
            "failure.verso",
            new CellModel { Type = "code", Language = "csharp", Source = """throw new System.InvalidOperationException("It went wrong.");""" },
            new CellModel { Type = "code", Language = "csharp", Source = "var missing = ;" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "failure.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Cell(page, 1).Locator("button.run").ClickAsync();

        var failure = Cell(page, 0).Locator(".outputs .error");

        await Expect(failure.Locator(".error-name")).ToHaveTextAsync("InvalidOperationException", new() { Timeout = 30_000 });
        await Expect(failure.Locator(".error-content")).ToHaveTextAsync("System.InvalidOperationException: It went wrong.");
        await Expect(failure.Locator(".error-stack")).Not.ToBeEmptyAsync();

        // A failure to compile has no stack to show.
        var compiling = Cell(page, 1).Locator(".outputs .error");

        await Expect(compiling.Locator(".error-name")).ToHaveTextAsync("CompilationError", new() { Timeout = 30_000 });
        await Expect(compiling.Locator(".error-stack")).ToHaveCountAsync(0);
    }

    [Fact]
    public async Task StandardError_IsLabelled_AndIsNoFailure()
    {
        await using var served = await StartAsync();
        var page = await RunOneAsync(served, "stderr.verso", """System.Console.Error.WriteLine("careful");""");
        var stderr = Cell(page, 0).Locator(".outputs .stderr");

        await Expect(stderr.Locator(".stderr-label")).ToHaveTextAsync("stderr", new() { Timeout = 30_000 });
        await Expect(stderr.Locator("pre")).ToContainTextAsync("careful");
        await Expect(stderr).ToHaveAttributeAsync("aria-label", "Standard error output");
        await Expect(Cell(page, 0).Locator(".outputs .error")).ToHaveCountAsync(0);
        await Expect(Cell(page, 0).Locator(".cell-bar .status")).ToContainTextAsync("✔");
    }

    [Fact]
    public async Task Json_IsDrawnAsATree_AndCsvAsATable()
    {
        await SaveAsync(
            "data.verso",
            new CellModel
            {
                Type = "code",
                Language = "csharp",
                Source = """Verso.Abstractions.CellOutput.Json("{\"name\": \"Titanic\", \"rows\": [1, 2, 3], \"fare\": 7.250, \"id\": 12345678901234567890, \"deep\": {\"deeper\": {\"deepest\": true}}}")""",
            },
            new CellModel { Type = "code", Language = "csharp", Source = """Verso.Abstractions.CellOutput.Csv("name,age\n\"Allen, Miss. Elisabeth\",29\n\"Brown, \"\"Molly\"\"\",38")""" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "data.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Cell(page, 1).Locator("button.run").ClickAsync();

        var tree = Cell(page, 0).Locator(".outputs .json");

        await Expect(tree.Locator(".jk").First).ToHaveTextAsync("\"name\"", new() { Timeout = 30_000 });
        await Expect(tree.Locator(".js").First).ToHaveTextAsync("\"Titanic\"");
        await Expect(tree.Locator("details summary .jcount").Nth(1)).ToHaveTextAsync("3 items");

        // A number as the JSON wrote it, however long; open two levels deep, closed below.
        await Expect(tree.Locator(".jn")).ToHaveTextAsync(["1", "2", "3", "7.250", "12345678901234567890"]);
        await Expect(tree.Locator("details")).ToHaveCountAsync(4);
        await Expect(tree.Locator("details[open]")).ToHaveCountAsync(3);

        var table = Cell(page, 1).Locator(".outputs table.csv");

        await Expect(table.Locator("th")).ToHaveTextAsync(["name", "age"], new() { Timeout = 30_000 });
        await Expect(table.Locator("td")).ToHaveTextAsync(["Allen, Miss. Elisabeth", "29", "Brown, \"Molly\"", "38"]);
    }

    [Fact]
    public async Task Progress_IsDrawnAsABar_OfUnknownLengthWhenNoShareIsGiven()
    {
        await SaveAsync(
            "progress.verso",
            new CellModel { Type = "code", Language = "csharp", Source = """Verso.Abstractions.CellOutput.Progress("Loading", "3 of 10", 30)""" },
            new CellModel { Type = "code", Language = "csharp", Source = """Verso.Abstractions.CellOutput.Progress("Waiting")""" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "progress.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Cell(page, 1).Locator("button.run").ClickAsync();

        var progress = Cell(page, 0).Locator(".outputs .progress");

        await Expect(progress.Locator(".progress-label")).ToContainTextAsync("Loading", new() { Timeout = 30_000 });
        await Expect(progress.Locator(".progress-status")).ToHaveTextAsync("3 of 10");
        await Expect(progress.Locator("[role=progressbar]")).ToHaveAttributeAsync("aria-valuenow", "30");

        var unknown = Cell(page, 1).Locator(".outputs .progress");

        await Expect(unknown.Locator(".progress-fill.unknown")).ToHaveCountAsync(1, new() { Timeout = 30_000 });
        await Expect(unknown.Locator("[role=progressbar]")).Not.ToHaveAttributeAsync("aria-valuenow", new System.Text.RegularExpressions.Regex("."));
    }

    [Fact]
    public async Task AWidget_IsDrawnInAFrameOfItsOwn_ThatRunsItsCode_ReachesNothingOfThePage_AndIsSizedToWhatItDraws()
    {
        await using var served = await StartAsync();
        var page = await RunOneAsync(
            served,
            "widget.verso",
            """Verso.Abstractions.CellOutput.Widget("<!doctype html><html><body><div id='drawn' style='height:300px'>A widget</div><script>document.getElementById('drawn').dataset.ran = 'yes';</script></body></html>")""");
        var frame = Cell(page, 0).Locator(".outputs iframe.widget");

        await Expect(frame).ToHaveAttributeAsync("sandbox", "allow-scripts", new() { Timeout = 30_000 });

        var inside = Cell(page, 0).FrameLocator(".outputs iframe.widget");

        await Expect(inside.Locator("#drawn")).ToHaveAttributeAsync("data-ran", "yes");

        // It reaches nothing of the page it is drawn in: not its document, not its cookie.
        var content = (await (await frame.ElementHandleAsync()).ContentFrameAsync())!;

        Assert.Equal("refused", await content.EvaluateAsync<string>("() => { try { return String(window.parent.document.title); } catch { return 'refused'; } }"));
        Assert.Equal("refused", await content.EvaluateAsync<string>("() => { try { return document.cookie; } catch { return 'refused'; } }"));

        // Sized to what it draws, and said to show only its saved state.
        for (var waited = 0; (await frame.BoundingBoxAsync())!.Height < 300; waited += 50)
        {
            Assert.True(waited < 10_000, "the frame never grew to what its document draws");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Expect(Cell(page, 0).Locator(".outputs .widget-inert")).ToBeVisibleAsync();
    }

    [Fact]
    public async Task AWidget_KeepsWhatItShows_WhileItsOutputStaysTheSame()
    {
        await using var served = await StartAsync();
        await SaveAsync(
            "counting.verso",
            new CellModel { Type = "code", Language = "csharp", Source = """Verso.Abstractions.CellOutput.Widget("<!doctype html><html><body><button id='more'>More</button><span id='count'>0</span><script>document.getElementById('more').onclick = function () { var count = document.getElementById('count'); count.textContent = String(Number(count.textContent) + 1); };</script></body></html>")""" });

        var page = await OpenAsync(served, "counting.verso");

        await page.EvaluateAsync("() => { window.__trace = []; const t0 = performance.now(); const stamp = (what) => window.__trace.push(Math.round(performance.now() - t0) + 'ms ' + what); new MutationObserver((records) => { for (const r of records) { for (const n of r.addedNodes) { if (n.nodeType === 1) { const frames = n.matches && n.matches('iframe') ? [n] : Array.from(n.querySelectorAll ? n.querySelectorAll('iframe') : []); for (const f of frames) { stamp('iframe added'); f.addEventListener('load', () => stamp('iframe load')); } } } for (const n of r.removedNodes) { if (n.nodeType === 1 && ((n.matches && n.matches('iframe')) || (n.querySelector && n.querySelector('iframe')))) stamp('iframe removed'); } } }).observe(document.body, { childList: true, subtree: true }); }");
        await Cell(page, 0).Locator("button.run").ClickAsync();

        var inside = Cell(page, 0).FrameLocator(".outputs iframe.widget");

        await WidgetLoadedAsync(page);
        await inside.Locator("#more").ClickAsync();
        await inside.Locator("#more").ClickAsync();

        try
        {
            await Expect(inside.Locator("#count")).ToHaveTextAsync("2");
        }
        catch (Microsoft.Playwright.PlaywrightException fault)
        {
            // Seen once in some forty full runs and never alone: whether the frame was drawn again between the two clicks is the thing to know.
            var trace = await page.EvaluateAsync<string[]>("() => window.__trace");

            throw new InvalidOperationException($"The widget did not count two clicks. Its frame, as the page drew it: {string.Join(" | ", trace)}.", fault);
        }

        await Cell(page, 0).Locator(".cell-bar").ClickAsync();
        await Expect(Cell(page, 0)).ToHaveClassAsync(new System.Text.RegularExpressions.Regex(@"\bselected\b"));

        await Expect(inside.Locator("#count")).ToHaveTextAsync("2");
    }

    [Fact]
    public async Task AWidgetsDownload_IsHandedToThePage_WhichSavesIt()
    {
        await using var served = await StartAsync();
        await SaveAsync(
            "saving.verso",
            new CellModel
            {
                Type = "code",
                Language = "csharp",
                Source = """Verso.Abstractions.CellOutput.Widget("<!doctype html><html><body><button id='save'>Save</button><script>document.getElementById('save').onclick = function () { var a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(['saved by a widget'], { type: 'text/plain' })); a.download = 'widget.txt'; a.click(); };</script></body></html>")""",
            });
        var page = await OpenAsync(served, "saving.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await WidgetLoadedAsync(page);

        var inside = Cell(page, 0).FrameLocator(".outputs iframe.widget");
        var download = await page.RunAndWaitForDownloadAsync(() => inside.Locator("#save").ClickAsync());

        Assert.Equal("widget.txt", download.SuggestedFilename);
        Assert.Equal("saved by a widget", await File.ReadAllTextAsync(await download.PathAsync(), TestContext.Current.CancellationToken));
    }
}
