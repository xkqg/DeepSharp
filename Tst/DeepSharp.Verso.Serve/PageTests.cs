// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Verso.Serve;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Playwright;
using Verso.Abstractions;
using Verso.Serializers;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// The page, in a real browser against the real server: it draws the notebook as Verso's editors draw one, sends what
/// a person does over its one socket the way Verso's own router means to — a button on its click, a box or a select on
/// its change, never on a key, in the order they came, the typing as it is typed and so before the click — draws each
/// change, and draws the notebook onto what it shows when its connection comes back. Its toolbar downloads what a button
/// hands over, its panel changes a field through its part, and it saves.
/// </summary>
public sealed partial class PageTests(Browsers browsers) : IClassFixture<Browsers>, IDisposable
{
    private const string Token = "00112233445566778899aabbccddeeff";

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-serve-page-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private async Task SaveAsync(string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
    }

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private async Task<Served> StartAsync(string? path = null)
    {
        if (!File.Exists(At("titanic.csv")))
        {
            File.Copy(Repository.Data("titanic.csv"), At("titanic.csv"));
        }

        var app = NotebookServer.Build(new ServeOptions(path ?? _folder, Port: 0, OpenBrowser: false, Help: false), Token, new StringWriter());

        await app.StartAsync(TestContext.Current.CancellationToken);

        var address = new Uri(app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single());
        var client = new HttpClient(new HttpClientHandler { UseCookies = false }) { BaseAddress = address };

        // As the server's own page asks: the cookie its first page set, and the page named, as a browser names it.
        client.DefaultRequestHeaders.Add("Cookie", $"deepsharp-serve-{address.Port}={Token}");
        client.DefaultRequestHeaders.Add("Origin", address.GetLeftPart(UriPartial.Authority));

        return new Served(app, client, address);
    }

    // A page opened on the address the server said, the notebook named when there is one.
    private async Task<IPage> OpenAsync(Served served, string? notebook = null)
    {
        var context = await browsers.Browser.NewContextAsync(new BrowserNewContextOptions { AcceptDownloads = true });
        var page = await context.NewPageAsync();
        var named = notebook is null ? string.Empty : $"&notebook={Uri.EscapeDataString(notebook)}";

        await page.GotoAsync($"{served.Address}?token={Token}{named}");

        return page;
    }

    // The notebook as a page opening it now is told it.
    private static async Task<NotebookVersion> CurrentAsync(Served served, string notebook)
    {
        await using var socket = await SocketAsync(served, notebook);

        return (await socket.SnapshotAsync()).Version;
    }

    // A notebook's socket, as another page of this server opens one.
    private static Task<PageSocket> SocketAsync(Served served, string notebook) => PageSocket.OpenAsync(served.Address, notebook, Token);

    private static ILocator Cell(IPage page, int index) => page.Locator("section.cell").Nth(index);

    [Fact]
    public async Task TheNotebook_IsDrawnACellASection_AndMarkdownIsShownRendered()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block), new CellModel { Type = "markdown", Source = "# The passengers" }]);
        await using var served = await StartAsync();

        var page = await OpenAsync(served, "titanic.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(6);
        await Expect(Cell(page, 5).Locator(".outputs h1")).ToHaveTextAsync("The passengers");
        await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(Titanic[3]);
    }

    [Fact]
    public async Task RunningABlock_ShowsItsCard_AndAButtonOnTheCardIsSentOnce_AndItsGridAppears()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        var show = Cell(page, 0).Locator("[data-action='deepsharp.show']");

        await Expect(show).ToHaveCountAsync(1);

        var before = (await CurrentAsync(served, "titanic.verso")).Cells[0].ExecutionCount;

        await show.ClickAsync();

        // Sent once: the Show ran its block once.
        await Expect(Cell(page, 0).Locator("input[type=checkbox][data-action^='deepsharp.include']").First).ToBeVisibleAsync();
        Assert.Equal(before + 1, (await CurrentAsync(served, "titanic.verso")).Cells[0].ExecutionCount);
    }

    [Fact]
    public async Task ABoxSendsItsStateOnItsChange_AndAKeyOnAFocusedButtonSendsNothing()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Cell(page, 0).Locator("[data-action='deepsharp.show']").ClickAsync();

        var deck = Cell(page, 0).Locator("input[type=checkbox][data-action^='deepsharp.include'][data-action*='\"deck\"']");

        await deck.CheckAsync();

        await Expect(page.Locator("section.cell").Nth(1).Locator("textarea.source")).ToHaveValueAsync(new System.Text.RegularExpressions.Regex("\"deck\""));

        var before = (await CurrentAsync(served, "titanic.verso")).Version;
        var button = Cell(page, 0).Locator("button[data-action]").First;

        await button.FocusAsync();
        await page.Keyboard.PressAsync("Tab");
        await page.Keyboard.PressAsync("Shift+Tab");
        // Longer than a change to the blocks waits before it reads them, so a send the keys made would have landed.
        await Task.Delay(TimeSpan.FromSeconds(1), TestContext.Current.CancellationToken);

        Assert.Equal(before, (await CurrentAsync(served, "titanic.verso")).Version);
    }

    [Fact]
    public async Task ASelectSendsTheValueItIsAt_OnItsChange()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Cell(page, 1).Locator("button.run").ClickAsync();
        await Cell(page, 1).Locator("[data-action='deepsharp.columns']").ClickAsync();

        var kind = Cell(page, 1).Locator("tr[data-column='pclass'] select[data-action]").First;

        await kind.SelectOptionAsync("category");

        await Expect(Cell(page, 1).Locator("textarea.source")).ToHaveValueAsync(new System.Text.RegularExpressions.Regex("\"pclass\"[^}]*\"category\""));
    }

    [Fact]
    public async Task TypingAndThenClicking_TheTypingReachesTheNotebookFirst()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await Cell(page, 0).Locator("button.run").ClickAsync();
        await Expect(Cell(page, 0).Locator("[data-action='deepsharp.show']")).ToHaveCountAsync(1);

        await Cell(page, 3).Locator("textarea.source").FillAsync(mean);
        await Cell(page, 0).Locator("[data-action='deepsharp.show']").ClickAsync();

        await Expect(Cell(page, 0).Locator("input[type=checkbox]").First).ToBeVisibleAsync();
        Assert.Equal(mean, (await CurrentAsync(served, "titanic.verso")).Cells[3].Source);
    }

    [Fact]
    public async Task ASecondPage_SeesWhatTheFirstChanged()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var first = await OpenAsync(served, "titanic.verso");
        var second = await OpenAsync(served, "titanic.verso");
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await Expect(Cell(second, 3).Locator("textarea.source")).ToHaveValueAsync(Titanic[3]);

        await Cell(first, 3).Locator("textarea.source").FillAsync(mean);
        await Cell(first, 3).Locator("textarea.source").BlurAsync();

        await Expect(Cell(second, 3).Locator("textarea.source")).ToHaveValueAsync(mean);
    }

    [Fact]
    public async Task APageWhoseSocketEnded_OpensItAgain_AndDrawsTheNotebookWhole()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(Titanic[3]);

        // Another page changes a cell, then closes the notebook without saving it: this page's socket ends with it.
        await using (var other = await SocketAsync(served, "titanic.verso"))
        {
            var fill = (await other.SnapshotAsync()).Version.Cells[3].Id;

            await other.AskAsync("edit", new { cell = fill, source = mean });
            await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(mean);
            await other.AskAsync("close");
        }

        // The page opens its socket again and draws the notebook whole, as its file holds it: the change it was told is gone.
        await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(Titanic[3], new() { Timeout = 30_000 });
        await Expect(page.Locator("#status")).ToBeEmptyAsync();
    }

    [Fact]
    public async Task ARunUnderWay_OffersAStop_AndStoppingEndsIt()
    {
        await SaveAsync(
            "runaway.verso",
            new CellModel { Type = "code", Language = "csharp", Source = "while (true) { await System.Threading.Tasks.Task.Delay(10); }" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "runaway.verso");

        await Cell(page, 0).Locator("button.run").ClickAsync();

        await Expect(page.Locator("#stop")).ToBeVisibleAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).ToHaveClassAsync(new System.Text.RegularExpressions.Regex("running"));

        await page.Locator("#stop").ClickAsync();

        await Expect(page.Locator("#stop")).ToBeHiddenAsync(new() { Timeout = 30_000 });
        await Expect(Cell(page, 0)).Not.ToHaveClassAsync(new System.Text.RegularExpressions.Regex("running"));
    }

    [Fact]
    public async Task ExportOnTheToolbar_DownloadsThePipelineFile_UnderItsName()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await page.Locator("#export > summary").ClickAsync();

        var download = await page.RunAndWaitForDownloadAsync(() => page.Locator($"#export button[data-button='{ExportPipelineAction.Id}']").ClickAsync());

        Assert.Equal("titanic.pipeline.json", download.SuggestedFilename);
    }

    [Fact]
    public async Task ThePanel_ChangesAFieldThroughItsPart()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");

        await Cell(page, 4).Locator(".cell-bar").ClickAsync();
        await page.Locator("#panels button[data-panel='properties']").ClickAsync();
        await page.Locator($"#panel select[data-part='{StepForm.Id}'][data-field='scale']").SelectOptionAsync("minmax");

        await Expect(Cell(page, 4).Locator("textarea.source")).ToHaveValueAsync(new System.Text.RegularExpressions.Regex("\"minmax\""));
    }

    [Fact]
    public async Task Saving_WritesTheNotebookToItsFile()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await Cell(page, 3).Locator("textarea.source").FillAsync(mean);
        await page.Locator("#save").ClickAsync();

        await Expect(page.Locator("#notice")).ToHaveTextAsync("Saved to titanic.verso");
        Assert.Contains("mean", await File.ReadAllTextAsync(At("titanic.verso"), TestContext.Current.CancellationToken), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AFoldersNotebooks_AreListed_AndOneOpens()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await SaveAsync("other.verso", Block(Titanic[0]));
        await using var served = await StartAsync();

        var page = await OpenAsync(served);

        await Expect(page.Locator("#notebooks a")).ToHaveCountAsync(2);

        await page.Locator("#notebooks a", new() { HasText = "titanic.verso" }).ClickAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
    }

    [Fact]
    public async Task TheToken_LeavesTheAddressBar_OnceTheCookieCarriesIt()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();

        var page = await OpenAsync(served, "titanic.verso");

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
        Assert.DoesNotContain(Token, page.Url, StringComparison.Ordinal);

        await page.ReloadAsync();

        await Expect(page.Locator("section.cell")).ToHaveCountAsync(5);
    }

    // A started server and a client that carries its token.
    private sealed class Served(WebApplication app, HttpClient client, Uri address) : IAsyncDisposable
    {
        public WebApplication App => app;

        public HttpClient Client => client;

        public Uri Address => address;

        public async ValueTask DisposeAsync()
        {
            client.Dispose();
            await app.DisposeAsync();
        }
    }
}

/// <summary>
/// The browser the page tests drive, started once for them: the installed Edge on Windows, as nothing needs
/// downloading there, and the Chromium the workflow installs everywhere else.
/// </summary>
public sealed class Browsers : IAsyncLifetime
{
    private IPlaywright? _playwright;

    /// <summary>The browser.</summary>
    public IBrowser Browser { get; private set; } = null!;

    public async ValueTask InitializeAsync()
    {
        _playwright = await Playwright.CreateAsync();
        Browser = await _playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions { Headless = true, Channel = OperatingSystem.IsWindows() ? "msedge" : null });
    }

    public async ValueTask DisposeAsync()
    {
        await Browser.DisposeAsync();
        _playwright?.Dispose();
    }
}
