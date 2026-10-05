// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using DeepSharp.Tests.Serve.Parts;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Verso.Serve;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// What a page asks of a notebook the server serves: which notebooks there are, over HTTP; and over one socket per page,
/// the notebook as it stands and each change after it, and what a person does to it — typing, running, stopping, a click
/// on a block's control — each ask answered to that page alone, by its id, never before the change it made. A name the
/// server does not serve is not found, a cell a change replaced answers the version it is gone from, and a refusal says
/// why. A socket ends when its notebook closes, when the server stops, or when its page goes away.
/// </summary>
public sealed partial class NotebookEndpointTests : IDisposable
{
    private const string Token = "fedcba9876543210fedcba9876543210";

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-serve-notebooks-").FullName;

    public NotebookEndpointTests()
    {
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));
        File.WriteAllText(Path.Join(_folder, "private.txt"), "not a notebook");
        File.WriteAllText(Path.Join(_folder, "notes.md"), "# Notes");

        var notebook = new NotebookModel();

        foreach (var step in Titanic)
        {
            notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = step });
        }

        File.WriteAllText(Path.Join(_folder, "titanic.verso"), new VersoSerializer().SerializeAsync(notebook).GetAwaiter().GetResult());
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private async Task<Served> StartAsync(string? path = null, TimeSpan? grace = null)
    {
        var options = new ServeOptions(path ?? _folder, Port: 0, OpenBrowser: false, Help: false) { Grace = grace ?? TimeSpan.FromMinutes(1) };
        var app = NotebookServer.Build(options, Token, new StringWriter());

        await app.StartAsync(TestContext.Current.CancellationToken);

        var address = new Uri(app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single());
        var client = new HttpClient(new HttpClientHandler { UseCookies = false }) { BaseAddress = address };

        // As the server's own page asks: the cookie its first page set, and the page named, as a browser names it.
        client.DefaultRequestHeaders.Add("Cookie", $"deepsharp-serve-{address.Port}={Token}");
        client.DefaultRequestHeaders.Add("Origin", address.GetLeftPart(UriPartial.Authority));

        return new Served(app, client, address);
    }

    // A page's socket on a notebook the server serves.
    private static Task<PageSocket> SocketAsync(Served served, string name = "titanic.verso") => PageSocket.OpenAsync(served.Address, name, Token);

    // The notebook as a page opening it now sees it.
    private static async Task<NotebookVersion> NotebookAsync(Served served, string name = "titanic.verso")
    {
        await using var socket = await SocketAsync(served, name);

        return (await socket.SnapshotAsync()).Version;
    }

    // What a control a block drew carries, read the way a browser reads the attribute.
    private static string ActionOf(HostedCell cell, Func<string, bool> which) =>
        cell.Outputs.SelectMany(output => Action().Matches(output.Content)).Select(match => WebUtility.HtmlDecode(match.Groups[1].Value)).First(which);

    [GeneratedRegex("data-action=\"([^\"]*)\"")]
    private static partial Regex Action();

    [Fact]
    public async Task TheNotebooksOfTheFolder_AreTheFilesVersosEngineReads_AndAServedNotebookIsTheOnlyOne()
    {
        await using (var served = await StartAsync())
        {
            var names = await served.Client.GetFromJsonAsync<string[]>("/api/notebooks", TestContext.Current.CancellationToken);

            Assert.Equal(["notes.md", "titanic.verso"], names!);
        }

        await using var one = await StartAsync(At("titanic.verso"));

        Assert.Equal(["titanic.verso"], (await one.Client.GetFromJsonAsync<string[]>("/api/notebooks", TestContext.Current.CancellationToken))!);
    }

    [Theory]
    [InlineData("private.txt")]
    [InlineData("..%2Ftitanic.verso")]
    [InlineData("sub%2Ftitanic.verso")]
    [InlineData("missing.verso")]
    public async Task ANameTheServerDoesNotServe_IsNotFound_OnItsSocket(string name)
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, name);

        var refused = await socket.UnnamedAsync();

        Assert.Equal(404, refused.Status);
        await socket.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal((WebSocketCloseStatus)4404, socket.CloseStatus);
        Assert.NotEmpty((await NotebookAsync(served)).Cells);
    }

    [Fact]
    public async Task EveryPerNotebookRouteButTheSocket_IsGone()
    {
        await using var served = await StartAsync();
        var cell = (await NotebookAsync(served)).Cells[0].Id;

        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.GetAsync("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.GetAsync("/api/notebooks/titanic.verso/updates", TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.GetAsync("/api/notebooks/titanic.verso/toolbar", TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.PostAsync($"/api/notebooks/titanic.verso/cells/{cell}/run", null, TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.PostAsync("/api/notebooks/titanic.verso/save", null, TestContext.Current.CancellationToken)).StatusCode);

        // The socket's address answers a socket alone.
        Assert.Equal(HttpStatusCode.BadRequest, (await served.Client.GetAsync("/api/notebooks/titanic.verso/socket", TestContext.Current.CancellationToken)).StatusCode);
    }

    [Fact]
    public async Task ASocket_BeginsWithTheNotebookAsItStands_AndItsKinds_ThenEachChange_BeforeTheAnswerToTheAskThatMadeIt()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        var snapshot = await socket.SnapshotAsync();

        Assert.Equal(0, snapshot.Version.Version);
        Assert.Equal(Titanic, snapshot.Version.Cells.Select(cell => cell.Source));
        Assert.Contains(snapshot.Kinds, kind => kind.Type == StepCellType.StepType);

        var fill = snapshot.Version.Cells[3].Id;
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);
        var edited = await socket.AskAsync("edit", new { cell = fill, source = mean });

        // An edit is answered done: what it changed comes with the version, as every change's does.
        Assert.True(edited.IsNothing);

        var change = await socket.ChangeAsync();

        Assert.Equal(1, change.Version);
        Assert.Equal(mean, Assert.Single(change.Cells).Source);

        // The page is told the change before the answer to the ask that made it.
        var frames = socket.Frames.ToList();

        Assert.True(frames.IndexOf("change 1") < frames.IndexOf("answer 1"), string.Join(", ", frames));
    }

    [Fact]
    public async Task ARunAndAClickOnABlocksControl_AreMadeThroughTheSocket_AndThePageIsToldTheNewOrder()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cells = (await socket.SnapshotAsync()).Version.Cells;
        var read = cells[0].Id;

        var card = (await socket.AskAsync("run", new { cell = read })).Result<HostedCell>();

        Assert.NotEmpty(card.Outputs);

        await ClickAsync(socket, read, ActionOf(card, action => action == "deepsharp.show"), "");

        var grid = (await NotebookAsync(served)).Cells[0];
        var ticked = await ClickAsync(socket, read, ActionOf(grid, action => action.StartsWith("deepsharp.include ", StringComparison.Ordinal) && action.Contains("\"deck\"", StringComparison.Ordinal)), "true");

        Assert.True(ticked.StateChanged);

        NotebookChange change;

        do
        {
            change = await socket.ChangeAsync();
        }
        while (change.Order is null);

        Assert.DoesNotContain(cells[1].Id, change.Order);
    }

    [Fact]
    public async Task AClickNamingNoPayload_ReachesItsPart_WithAnEmptyOne_AsVersoHandsEveryPartOne()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cell = (await socket.SnapshotAsync()).Version.Cells[0].Id;

        var bare = (await socket.AskAsync("gesture", new { cell, extensionId = AnsweringPart.Id, action = "answer" })).Result<GestureResult>();
        var carrying = (await socket.AskAsync("gesture", new { cell, extensionId = AnsweringPart.Id, action = "answer", payload = "tick" })).Result<GestureResult>();

        Assert.Equal("[]", bare.Answer);
        Assert.Equal("[tick]", carrying.Answer);
    }

    private static async Task<GestureResult> ClickAsync(PageSocket socket, Guid cell, string action, string payload) =>
        (await socket.AskAsync("gesture", new HostedGesture(cell, StepRenderer.Id, action, payload))).Result<GestureResult>();

    [Fact]
    public async Task TypingIntoACellAChangeReplaced_IsRefused_NamingTheVersionItIsGoneFrom()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var gone = Guid.NewGuid();

        await socket.SnapshotAsync();

        var refused = await socket.AskAsync("edit", new { cell = gone, source = "x" });

        Assert.True(refused.Refused);
        Assert.Equal(409, refused.Status);

        // Said for the person who reads it; the cell and the version stay on the refusal for a program that reads those.
        Assert.Equal("The cell is no longer in the notebook: a change made before this one rewrote it or took it away.", refused.Frame.GetProperty("message").GetString());
        Assert.Equal(gone, refused.Frame.GetProperty("cell").GetGuid());
        Assert.Equal(0, refused.Frame.GetProperty("version").GetInt64());
        Assert.Equal(409, (await socket.AskAsync("run", new { cell = gone })).Status);
    }

    [Fact]
    public async Task AClickNamingNoPart_IsRefused_SayingWhy()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cell = (await socket.SnapshotAsync()).Version.Cells[0].Id;

        var refused = await socket.AskAsync("gesture", new HostedGesture(cell, "no.such.part", "deepsharp.show", ""));

        Assert.Equal(422, refused.Status);
        Assert.Contains("no.such.part", refused.Why, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnAskTheServerDoesNotKnow_OrCannotRead_IsRefusedAsABadRequest_AndTheSocketGoesOn()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        Assert.Equal(400, (await socket.AskAsync("fly")).Status);
        var unnamed = await socket.AskAsync(null);

        Assert.Equal(400, unnamed.Status);
        Assert.Equal("A page asks nothing called '' of a notebook.", unnamed.Why);

        await socket.SendAsync("this is not an ask");

        Assert.Equal(400, (await socket.UnnamedAsync()).Status);
        Assert.Equal(400, (await socket.AskAsync("move", new { cell = Guid.NewGuid() })).Status);
        Assert.False((await socket.AskAsync("stop", new { run = 1 })).Refused);
    }

    [Theory]
    [InlineData("run", "{}", "cell")]
    [InlineData("edit", """{"cell": "{cell}"}""", "text")]
    [InlineData("gesture", """{"cell": "{cell}", "action": "deepsharp.show"}""", "part")]
    [InlineData("gesture", """{"cell": "{cell}", "extensionId": "io.github.xkqg.deepsharp.notebooks.renderer"}""", "action")]
    [InlineData("add", "{}", "kind")]
    [InlineData("completions", """{"cell": "{cell}", "position": 0}""", "text")]
    [InlineData("hover", """{"cell": "{cell}", "code": ""}""", "position")]
    [InlineData("press", "{}", "button")]
    [InlineData("property", """{"cell": "{cell}", "field": "scale"}""", "part")]
    [InlineData("property", """{"cell": "{cell}", "part": "io.github.xkqg.deepsharp.notebooks.form"}""", "field")]
    [InlineData("interact", """{"action": "move"}""", "layout")]
    [InlineData("interact", """{"layout": "dashboard"}""", "action")]
    [InlineData("layout", "{}", "layout")]
    [InlineData("theme", "{}", "theme")]
    public async Task AnAskLackingWhatItNames_IsRefusedAsABadRequest_SayingWhatItLacks(string ask, string fields, string lacks)
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cell = (await socket.SnapshotAsync()).Version.Cells[0].Id;

        var refused = await socket.AskAsync(ask, JsonNode.Parse(fields.Replace("{cell}", cell.ToString(), StringComparison.Ordinal)));

        Assert.Equal(400, refused.Status);
        Assert.Equal($"'{ask}' names no {lacks}.", refused.Why);
    }

    [Fact]
    public async Task AFieldsValueNoPageSends_ReachesItsPart_AsNothing_AsWords_OrAsTheJsonItCameAs()
    {
        var cell = new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" };
        var notebook = new NotebookModel();

        cell.Metadata[ReadingPart.Marked] = true;
        notebook.Cells.Add(cell);
        File.WriteAllText(At("values.verso"), await new VersoSerializer().SerializeAsync(notebook));

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "values.verso");
        var id = (await socket.SnapshotAsync()).Version.Cells[0].Id;

        // What the part writes down it was handed: the value's type and the value.
        async Task<string?> HandedAsync(string field, object? value)
        {
            Assert.True((await socket.AskAsync("property", new { cell = id, part = ReadingPart.Part, field, value })).IsNothing);

            var metadata = (await NotebookAsync(served, "values.verso")).Cells[0].Metadata;

            return metadata.TryGetValue(ReadingPart.Got + field, out var got) ? JsonSerializer.Deserialize<string>(got) : null;
        }

        Assert.Equal(":", await HandedAsync("flag", null));
        Assert.Equal("List`1:1|b", await HandedAsync("choices", new object[] { 1, "b" }));
        Assert.Equal("JsonElement:{\"a\":1}", await HandedAsync("count", new { a = 1 }));
    }

    [Fact]
    public async Task TheLayoutAndTheTheme_AreSwitchedByTheIdAnAskNames_AndOneNamingNoneOrOneTheEngineLacksIsRefused()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var first = await socket.NextAsync();

        // The first frame lists what the notebook can be shown in — Verso's three layouts and the test suite's own two — each
        // theme with its block of custom properties.
        Assert.Equal(5, first.GetProperty("layouts").GetArrayLength());
        Assert.Contains("--verso-bg-default", first.GetProperty("themes")[0].GetProperty("css").GetString(), StringComparison.Ordinal);

        Assert.Equal(400, (await socket.AskAsync("layout")).Status);
        Assert.Equal(400, (await socket.AskAsync("theme")).Status);
        Assert.Equal(422, (await socket.AskAsync("layout", new { layout = "no-such-layout" })).Status);
        Assert.Equal(422, (await socket.AskAsync("theme", new { theme = "no-such-theme" })).Status);

        Assert.True((await socket.AskAsync("layout", new { layout = "presentation" })).IsNothing);
        Assert.True((await socket.AskAsync("theme", new { theme = "verso-dark" })).IsNothing);

        await using var other = await SocketAsync(served);
        var shown = (await other.SnapshotAsync()).Version;

        Assert.Equal("presentation", shown.Layout.Id);
        Assert.Equal("verso-dark", shown.ThemeId);
    }

    [Fact]
    public async Task ATitle_IsGivenThroughTheSocket_AndEveryPageIsToldIt()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        Assert.Null((await socket.SnapshotAsync()).Version.Metadata.Title);
        Assert.True((await socket.AskAsync("title", new { title = "Passengers" })).IsNothing);
        Assert.Equal("Passengers", (await socket.ChangeAsync()).Metadata?.Title);

        // One with no title takes it away.
        Assert.True((await socket.AskAsync("title")).IsNothing);
        Assert.Null((await socket.ChangeAsync()).Metadata?.Title);
    }

    [Fact]
    public async Task InTheDashboard_EveryPageIsToldItsArrangement_AndATileMovedThroughTheSocketIsArrangedAnew()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var tile = (await socket.SnapshotAsync()).Version.Cells[0].Id;

        Assert.True((await socket.AskAsync("layout", new { layout = "dashboard" })).IsNothing);
        Assert.Contains($"data-cell-slot=\"{tile}\"", (await socket.ChangeAsync()).Arrangement?.Html, StringComparison.Ordinal);

        var payload = $$"""{"cellId":"{{tile}}","row":3,"col":0,"width":6,"height":4}""";

        Assert.Equal(400, (await socket.AskAsync("interact", new { action = "updateCellPosition", payload, target = tile })).Status);
        Assert.Equal(400, (await socket.AskAsync("interact", new { layout = "dashboard", payload, target = tile })).Status);
        Assert.True((await socket.AskAsync("interact", new { layout = "dashboard", action = "updateCellPosition", payload, target = tile })).IsNothing);
        Assert.Contains("grid-column:1/span 6;grid-row:4/span 4", (await socket.ChangeAsync()).Arrangement?.Html, StringComparison.Ordinal);

        // An act on a layout the notebook is no longer shown in is refused, in words.
        var refused = await socket.AskAsync("interact", new { layout = "presentation", action = "run", target = tile });

        Assert.Equal(422, refused.Status);
        Assert.Contains("presentation", refused.Why, StringComparison.Ordinal);

        // A page that opens now is told the arrangement with the notebook.
        await using var later = await SocketAsync(served);

        Assert.Contains("grid-row:4/span 4", (await later.SnapshotAsync()).Version.Arrangement.Html, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ANotebookTheGuardRefusesToOpen_IsRefused_SayingWhy()
    {
        File.WriteAllText(
            At("old.ipynb"),
            $$"""{"cells": [{"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [{{JsonSerializer.Serialize(Titanic[0])}}]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}""");

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "old.ipynb");

        var refused = await socket.UnnamedAsync();

        Assert.Equal(422, refused.Status);
        Assert.Contains("read.csv", refused.Why, StringComparison.Ordinal);
        await socket.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal((WebSocketCloseStatus)4422, socket.CloseStatus);
    }

    [Fact]
    public async Task Stopping_IsAnswered_WhetherOrNotARunIsUnderWay()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        Assert.False((await socket.AskAsync("stop", new { run = 1 })).Result<JsonElement>().GetProperty("stopped").GetBoolean());
    }

    [Fact]
    public async Task ClosingANotebook_IsAnswered_ThenEndsEverySocket_AndWhatItHeldUnsavedIsGone()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        await using var other = await SocketAsync(served);
        var fill = (await socket.SnapshotAsync()).Version.Cells[3].Id;

        await other.SnapshotAsync();
        await socket.AskAsync("edit", new { cell = fill, source = "{}" });

        Assert.True((await socket.AskAsync("close")).IsNothing);

        foreach (var each in new[] { socket, other })
        {
            await each.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
            Assert.Equal(WebSocketCloseStatus.NormalClosure, each.CloseStatus);
        }

        // Opened again, it is the notebook the file holds.
        Assert.Equal(Titanic[3], (await NotebookAsync(served)).Cells[3].Source);
    }

    [Fact]
    public async Task APageThatNeverClosesItsSide_IsLetGo_OnceItsNotebookClosed()
    {
        await using var served = await StartAsync();
        using var silent = new ClientWebSocket();

        silent.Options.SetRequestHeader("Cookie", $"deepsharp-serve-{served.Address.Port}={Token}");
        silent.Options.SetRequestHeader("Origin", served.Address.GetLeftPart(UriPartial.Authority));
        await silent.ConnectAsync(new Uri($"ws://{served.Address.Authority}/api/notebooks/titanic.verso/socket"), TestContext.Current.CancellationToken);

        // A page that reads nothing more, and so never closes its side once the server closed its own.
        await using (var closing = await SocketAsync(served))
        {
            await closing.SnapshotAsync();
            await closing.AskAsync("close");
        }

        var stopping = System.Diagnostics.Stopwatch.StartNew();

        await served.App.StopAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

        // Let go after a moment, rather than kept until the server gives up on its last request.
        Assert.True(stopping.Elapsed < TimeSpan.FromSeconds(15), $"the server took {stopping.Elapsed.TotalSeconds:0.0} s to stop");
    }

    [Fact]
    public async Task StoppingTheServer_EndsAnOpenSocket_AtOnce()
    {
        var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        var stopping = served.App.StopAsync(TestContext.Current.CancellationToken);

        await socket.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        await stopping.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        await served.DisposeAsync();
    }

    // A page that leaves must leave nothing behind that fails later: a read of its socket that goes on past its leaving
    // fails on a socket that is gone, and since nothing waits for that failure the engine tells it to whichever run is
    // working when it is collected — in whichever test that is.
    [Fact]
    public async Task PagesThatComeAndGo_LeaveNoFailedReadBehind()
    {
        var left = new ConcurrentQueue<string>();

        void Saw(object? sender, UnobservedTaskExceptionEventArgs unobserved)
        {
            // Only what is the socket's own: other tests of this process may leave failures of theirs.
            if (unobserved.Exception.ToString().Contains(nameof(PageSocket), StringComparison.Ordinal))
            {
                left.Enqueue(unobserved.Exception.InnerException?.GetType().Name ?? unobserved.Exception.GetType().Name);
                unobserved.SetObserved();
            }
        }

        TaskScheduler.UnobservedTaskException += Saw;

        try
        {
            await using var served = await StartAsync();

            await Task.WhenAll(Enumerable.Range(0, 16).Select(async _ =>
            {
                for (var page = 0; page < 125; page++)
                {
                    await NotebookAsync(served);
                }
            }));

            // A task is told to have been waited for by nobody only when it is collected.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Assert.True(left.IsEmpty, $"{left.Count} of 2000 pages left a read behind that failed: {string.Join(", ", left.Distinct())}");
        }
        finally
        {
            TaskScheduler.UnobservedTaskException -= Saw;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task APageThatLeaves_EndsItsView_SoTheNotebookClosesOnceNoPageShowsIt(bool afterAChange)
    {
        await using var served = await StartAsync(grace: TimeSpan.FromMilliseconds(200));
        var host = await served.App.Services.GetRequiredService<OpenNotebooks>().OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken);
        var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        if (afterAChange)
        {
            // Left just after a change was told, saved first, so nothing unsaved keeps the notebook.
            await host.EditAsync(host.Cells[3].Id, Titanic[3].Replace("median", "mean", StringComparison.Ordinal));
            await host.SaveAsync();
        }

        // The page's tab closes: its socket goes with it.
        await socket.DisposeAsync();

        for (var waited = TimeSpan.Zero; ; waited += TimeSpan.FromMilliseconds(100))
        {
            Assert.True(waited < TimeSpan.FromSeconds(10), "the notebook stayed open");

            // Asked, not viewed: a view would be a page showing it.
            try
            {
                await host.PropertiesAsync(host.Cells[0].Id);
            }
            catch (ObjectDisposedException)
            {
                break;
            }

            await Task.Delay(100, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task TheToolbar_RidesTheVersion_EveryButtonTheEngineHas_EachSayingWhetherItCanBePressedThen()
    {
        await using var served = await StartAsync();

        var version = await NotebookAsync(served);

        Assert.True(version.Buttons.Single(button => button.Id == ExportPipelineAction.Id).IsEnabled);
        Assert.Contains(version.Buttons, button => button.Id == "verso.action.run-all");
    }

    [Fact]
    public async Task AFileAButtonHandsOver_GoesToThePageThatPressedIt_UnderItsOwnName_AndNothingIsWrittenBesideTheNotebook()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        var file = (await socket.AskAsync("press", new { button = ExportPipelineAction.Id, cells = Array.Empty<Guid>() })).Result<HostedFile>();

        Assert.Equal("titanic.pipeline.json", file.Name);
        Assert.Equal("application/json", file.ContentType);
        Assert.Contains("\"normalise\"", Encoding.UTF8.GetString(file.Bytes), StringComparison.Ordinal);
        Assert.False(File.Exists(At("titanic.pipeline.json")));
    }

    [Fact]
    public async Task AButtonThatHandsNothingOver_IsAnsweredWithNothing_AndOneTheEngineDoesNotHave_IsRefused()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        await socket.SnapshotAsync();

        Assert.True((await socket.AskAsync("press", new { button = RunPipelineAction.Id, cells = Array.Empty<Guid>() })).IsNothing);
        Assert.Equal(422, (await socket.AskAsync("press", new { button = "no.such.button", cells = Array.Empty<Guid>() })).Status);

        // A press that names no cells is pressed for none, as the notebook's own buttons are.
        Assert.True((await socket.AskAsync("press", new { button = RunPipelineAction.Id })).IsNothing);
    }

    [Fact]
    public async Task ACellsPanel_IsTheSectionsOfItsParts_AndAFieldIsChangedByItsPart_WithItsValueAsVersosBrowserEditorHandsItOn()
    {
        var notebook = new NotebookModel();

        foreach (var step in Titanic)
        {
            notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = step });
        }

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        File.WriteAllText(At("panel.verso"), await new VersoSerializer().SerializeAsync(notebook));

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "panel.verso");
        var cells = (await socket.SnapshotAsync()).Version.Cells;
        var normalise = cells[4].Id;
        var code = cells[5].Id;

        var sections = (await socket.AskAsync("properties", new { cell = normalise })).Result<JsonElement[]>();

        Assert.Contains(sections, section => section.GetProperty("part").GetString() == StepForm.Id);

        async Task ChangeAsync(Guid cell, string part, string field, object value) =>
            Assert.True((await socket.AskAsync("property", new { cell, part, field, value })).IsNothing);

        await ChangeAsync(normalise, StepForm.Id, "scale", "minmax");
        await ChangeAsync(code, "verso.propertyprovider.display", "inputCollapsed", true);
        await ChangeAsync(code, "verso.propertyprovider.display", "outputPreviewLineCount", 8);

        var after = (await NotebookAsync(served, "panel.verso")).Cells;

        Assert.Contains("\"minmax\"", after[4].Source, StringComparison.Ordinal);
        Assert.Equal("true", after[5].Metadata["verso:ui.inputCollapsed"]);
        Assert.Equal("8", after[5].Metadata["verso:ui.outputPreviewLineCount"]);

        // A number reaches the part as Verso's browser editor hands one on, as a double, which the part reads as it reads
        // one from that editor.
        await ChangeAsync(code, "verso.propertyprovider.display", "outputPreviewLineCount", 3.5);
        Assert.Equal("3", (await NotebookAsync(served, "panel.verso")).Cells[5].Metadata["verso:ui.outputPreviewLineCount"]);
    }

    [Fact]
    public async Task Saving_WritesTheNotebookToItsFile()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var fill = (await socket.SnapshotAsync()).Version.Cells[3].Id;
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);

        await socket.AskAsync("edit", new { cell = fill, source = mean });

        Assert.True((await socket.AskAsync("save")).IsNothing);
        Assert.Contains("mean", (await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(At("titanic.verso"), TestContext.Current.CancellationToken))).Cells[3].Source, StringComparison.Ordinal);
    }

    // A started server and a browser that carries its token, stopped and closed with the test.
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
