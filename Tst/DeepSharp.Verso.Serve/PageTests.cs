// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net.Http.Json;
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
/// a person does the way Verso's own router means to — a button on its click, a box or a select on its change, never
/// on a key, one at a time in the order they came, the typing before the click — draws each change, and reads the whole
/// notebook again when its stream comes back. Its toolbar downloads what a button hands over, its panel changes a field
/// through its part, and it saves.
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

    private static Task<NotebookVersion> CurrentAsync(Served served, string notebook) =>
        served.Client.GetFromJsonAsync<NotebookVersion>($"/api/notebooks/{notebook}", TestContext.Current.CancellationToken);

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

        var before = (await CurrentAsync(served, "titanic.verso")).Version;

        await show.ClickAsync();

        await Expect(Cell(page, 0).Locator("input[type=checkbox][data-action^='deepsharp.include']").First).ToBeVisibleAsync();
        Assert.Equal(before + 1, (await CurrentAsync(served, "titanic.verso")).Version);
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
    public async Task APageWhoseStreamDropped_ReadsTheWholeNotebookAgain_WhenItIsBack()
    {
        await SaveAsync("titanic.verso", [.. Titanic.Select(Block)]);
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "titanic.verso");
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(Titanic[3]);

        await page.Context.SetOfflineAsync(true);

        var fill = (await CurrentAsync(served, "titanic.verso")).Cells[3].Id;

        await served.Client.PostAsJsonAsync($"/api/notebooks/titanic.verso/cells/{fill}/source", new { source = mean }, TestContext.Current.CancellationToken);
        await page.Context.SetOfflineAsync(false);

        await Expect(Cell(page, 3).Locator("textarea.source")).ToHaveValueAsync(mean, new() { Timeout = 30_000 });
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

        await Expect(page.Locator("#status")).ToHaveTextAsync(new System.Text.RegularExpressions.Regex("[Ss]aved"));
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
