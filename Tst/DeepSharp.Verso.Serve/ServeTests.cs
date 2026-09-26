// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.Sockets;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Serve;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// DeepSharp's server as a browser meets it: it listens on this computer alone, answers only a request that carries
/// the token it printed — in the address, or in the cookie the first page sets — and only under a name of this
/// computer, and it serves its own page and nothing from the folder it runs in. It says one line when it starts, on
/// the address it really bound, and nothing after; a port already taken stops it before it says anything.
/// </summary>
public sealed class ServeTests : IDisposable
{
    private const string Token = "0123456789abcdef0123456789abcdef";

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-serve-").FullName;

    public ServeTests()
    {
        File.WriteAllText(Path.Join(_folder, "private.txt"), "not for the browser");
        File.WriteAllText(Path.Join(_folder, "titanic.verso"), "{}");
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // The server, started for real on a port the system picks.
    private async Task<Started> StartAsync(StringWriter? said = null)
    {
        var app = NotebookServer.Build(new ServeOptions(_folder, Port: 0, OpenBrowser: false, Help: false), Token, said ?? new StringWriter());

        await app.StartAsync(TestContext.Current.CancellationToken);

        var address = app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single();

        return new Started(app, new Uri(address));
    }

    private static HttpClient Browser() => new(new HttpClientHandler { UseCookies = false });

    private static Task<HttpResponseMessage> GetAsync(Uri address, string path, Action<HttpRequestMessage>? dress = null)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, new Uri(address, path));

        dress?.Invoke(request);

        return Browser().SendAsync(request, TestContext.Current.CancellationToken);
    }

    private static string CookieFor(Uri address) => $"deepsharp-serve-{address.Port}";

    [Fact]
    public async Task ThePage_AnswersTheTokenInItsAddress_AndSetsACookieThatCarriesItAfter()
    {
        await using var started = await StartAsync();

        var first = await GetAsync(started.Address, $"/?token={Token}");
        var cookie = Assert.Single(first.Headers.GetValues("Set-Cookie"));

        Assert.Equal(HttpStatusCode.OK, first.StatusCode);
        Assert.Equal("text/html", first.Content.Headers.ContentType?.MediaType);
        Assert.StartsWith("<!doctype html>", await first.Content.ReadAsStringAsync(TestContext.Current.CancellationToken), StringComparison.OrdinalIgnoreCase);
        Assert.StartsWith($"{CookieFor(started.Address)}={Token}", cookie, StringComparison.Ordinal);
        Assert.Contains("httponly", cookie, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("samesite=strict", cookie, StringComparison.OrdinalIgnoreCase);

        var after = await GetAsync(started.Address, "/", request => request.Headers.Add("Cookie", $"{CookieFor(started.Address)}={Token}"));

        Assert.Equal(HttpStatusCode.OK, after.StatusCode);
    }

    [Fact]
    public async Task WithoutTheToken_NothingAnswers()
    {
        await using var started = await StartAsync();
        var otherPort = $"deepsharp-serve-{started.Address.Port + 1}={Token}";

        Assert.Equal(HttpStatusCode.Unauthorized, (await GetAsync(started.Address, "/")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await GetAsync(started.Address, "/?token=wrong")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await GetAsync(started.Address, "/anything")).StatusCode);
        Assert.Equal(HttpStatusCode.Unauthorized, (await GetAsync(started.Address, "/", request => request.Headers.Add("Cookie", otherPort))).StatusCode);
    }

    [Fact]
    public async Task AFileBesideTheNotebook_IsNeverServed()
    {
        await using var started = await StartAsync();

        Assert.Equal(HttpStatusCode.NotFound, (await GetAsync(started.Address, $"/private.txt?token={Token}")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await GetAsync(started.Address, $"/titanic.verso?token={Token}")).StatusCode);
    }

    [Fact]
    public async Task ARequestUnderANameThatIsNotThisComputers_IsRefused()
    {
        await using var started = await StartAsync();

        var rebound = await GetAsync(started.Address, $"/?token={Token}", request => request.Headers.Host = "evil.example");

        Assert.Equal(HttpStatusCode.BadRequest, rebound.StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await GetAsync(started.Address, $"/?token={Token}", request => request.Headers.Host = $"localhost:{started.Address.Port}")).StatusCode);
    }

    [Fact]
    public async Task TheServer_ListensOnThisComputerAlone_OnAPortTheSystemPicked()
    {
        await using var started = await StartAsync();

        Assert.Equal("127.0.0.1", started.Address.Host);
        Assert.True(started.Address.Port > 0);
    }

    [Fact]
    public async Task TheLineSaidAtTheStart_NamesWhatIsServed_AndTheAddressItBound_WithItsToken()
    {
        var said = new StringWriter();

        await using var started = await StartAsync(said);

        var line = Assert.Single(said.ToString().Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries));

        Assert.Contains(_folder, line, StringComparison.Ordinal);
        Assert.Contains($"{started.Address.GetLeftPart(UriPartial.Authority)}/?token={Token}", line, StringComparison.Ordinal);
    }

    [Fact]
    public async Task NothingIsLogged_SinceACSharpCellTakesOverTheConsoleWhileItRuns()
    {
        await using var started = await StartAsync();

        Assert.Empty(started.App.Services.GetServices<ILoggerProvider>());
    }

    [Fact]
    public async Task StoppingTheServer_ClosesTheNotebooksItOpened()
    {
        var started = await StartAsync();
        var notebook = await started.App.Services.GetRequiredService<OpenNotebooks>().OpenAsync(Path.Join(_folder, "titanic.verso"), TestContext.Current.CancellationToken);

        await started.DisposeAsync();

        Assert.Throws<ObjectDisposedException>(notebook.Subscribe);
    }

    [Fact]
    public async Task ThePortAsked_IsTheOneBound()
    {
        var port = FreePort();
        var app = NotebookServer.Build(new ServeOptions(_folder, port, OpenBrowser: false, Help: false), Token, new StringWriter());

        await using var started = new Started(app, new Uri($"http://127.0.0.1:{port}"));
        await app.StartAsync(TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.OK, (await GetAsync(started.Address, $"/?token={Token}")).StatusCode);
    }

    [Fact]
    public async Task APortAlreadyTaken_StopsTheToolBeforeItSaysAnything()
    {
        using var taken = new TcpListener(IPAddress.Loopback, 0);

        taken.Start();

        var port = ((IPEndPoint)taken.LocalEndpoint).Port;
        var said = new StringWriter();
        var errors = new StringWriter();

        var exit = await NotebookServer.RunAsync(["--port", $"{port}", "--no-browser", _folder], said, errors, TestContext.Current.CancellationToken);

        Assert.Equal(1, exit);
        Assert.Empty(said.ToString());
        Assert.Contains($"{port}", errors.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheTool_ServesUntilItIsStopped_AndSaysOneLineOnlyWhenItStarted()
    {
        var said = new StringWriter();
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        var running = NotebookServer.RunAsync(["--no-browser", _folder], said, new StringWriter(), stop.Token);

        for (var waited = 0; said.ToString().Length == 0; waited += 20)
        {
            Assert.True(waited < 30_000, "the tool said nothing");
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        var address = new Uri(said.ToString().Split(' ').Single(word => word.StartsWith("http://", StringComparison.Ordinal)).Trim());

        Assert.Equal(HttpStatusCode.OK, (await GetAsync(address, address.PathAndQuery)).StatusCode);

        await stop.CancelAsync();

        Assert.Equal(0, await running);
    }

    [Fact]
    public async Task ArgumentsItCannotUse_StopTheTool_WithWhatWasWrongAndTheUsage()
    {
        var errors = new StringWriter();
        var said = new StringWriter();

        Assert.Equal(2, await NotebookServer.RunAsync(["--nope"], said, errors, TestContext.Current.CancellationToken));
        Assert.Contains("--nope", errors.ToString(), StringComparison.Ordinal);
        Assert.Contains(ServeOptions.Usage, errors.ToString(), StringComparison.Ordinal);
        Assert.Empty(said.ToString());

        Assert.Equal(0, await NotebookServer.RunAsync(["--help"], said, errors, TestContext.Current.CancellationToken));
        Assert.Contains(ServeOptions.Usage, said.ToString(), StringComparison.Ordinal);
        Assert.Equal(0, await Program.Main(["--help"]));
    }

    [Fact]
    public async Task AChange_TheCookieCarriesFromAnotherPageOfThisComputer_IsRefused_AndMakesNothing()
    {
        await using var started = await StartAsync();
        var port = started.Address.Port + 1;

        Assert.Equal(HttpStatusCode.Forbidden, (await NewAsync(started, "a.verso", cookie: true, $"http://127.0.0.1:{port}")).StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, (await NewAsync(started, "b.verso", cookie: true, $"http://localhost:{port}")).StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, (await NewAsync(started, "c.verso", cookie: true, origin: null)).StatusCode);
        Assert.Empty(Directory.GetFiles(_folder, "?.verso"));
    }

    [Fact]
    public async Task AChange_FromTheServersOwnPage_IsAnswered_UnderEitherNameOfThisComputer()
    {
        await using var started = await StartAsync();

        Assert.Equal(HttpStatusCode.Created, (await NewAsync(started, "own.verso", cookie: true, $"http://127.0.0.1:{started.Address.Port}")).StatusCode);

        var underLocalhost = await PostAsync(started.Address, "/api/notebooks", """{"name": "local.verso"}""", request =>
        {
            request.Headers.Host = $"localhost:{started.Address.Port}";
            request.Headers.Add("Cookie", $"{CookieFor(started.Address)}={Token}");
            request.Headers.Add("Origin", $"http://localhost:{started.Address.Port}");
        });

        Assert.Equal(HttpStatusCode.Created, underLocalhost.StatusCode);
    }

    [Fact]
    public async Task AProgram_CarryingTheTokenInTheAddress_NamesNoPage_ButAPageItNamesMustBeTheServers()
    {
        await using var started = await StartAsync();

        Assert.Equal(HttpStatusCode.Created, (await NewAsync(started, "program.verso", cookie: false, origin: null)).StatusCode);
        Assert.Equal(HttpStatusCode.Forbidden, (await NewAsync(started, "other.verso", cookie: false, $"http://127.0.0.1:{started.Address.Port + 1}")).StatusCode);
        Assert.False(File.Exists(Path.Join(_folder, "other.verso")));
    }

    private static Task<HttpResponseMessage> PostAsync(Uri address, string path, string json, Action<HttpRequestMessage> dress)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, new Uri(address, path)) { Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json") };

        dress(request);

        return Browser().SendAsync(request, TestContext.Current.CancellationToken);
    }

    // A new notebook asked of the server as a script asks it: the token carried by the cookie or by the address, and the
    // page it comes from named, or none.
    private static Task<HttpResponseMessage> NewAsync(Started started, string name, bool cookie, string? origin) =>
        PostAsync(started.Address, cookie ? "/api/notebooks" : $"/api/notebooks?token={Token}", $$"""{"name": "{{name}}"}""", request =>
        {
            if (cookie)
            {
                request.Headers.Add("Cookie", $"{CookieFor(started.Address)}={Token}");
            }

            if (origin is not null)
            {
                request.Headers.Add("Origin", origin);
            }
        });

    private static int FreePort()
    {
        using var probe = new TcpListener(IPAddress.Loopback, 0);

        probe.Start();

        return ((IPEndPoint)probe.LocalEndpoint).Port;
    }

    // A started server, stopped and closed with the test.
    private sealed class Started(WebApplication app, Uri address) : IAsyncDisposable
    {
        public WebApplication App => app;

        public Uri Address => address;

        public async ValueTask DisposeAsync() => await app.DisposeAsync();
    }
}
