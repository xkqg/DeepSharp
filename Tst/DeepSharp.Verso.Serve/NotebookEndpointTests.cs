// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.Http.Json;
using System.Net.ServerSentEvents;
using System.Text.Json;
using System.Text.RegularExpressions;
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
/// What a page asks of a notebook the server serves: which notebooks there are, the notebook as it stands and each
/// change after it — a stream a page keeps open — and what a person does to it: typing, running, stopping, a click on a
/// block's control. A name the server does not serve is not found, a cell a change replaced answers the version it
/// is gone from, and a refusal says why. A stream ends when the server stops or the page goes away.
/// </summary>
public sealed partial class NotebookEndpointTests : IDisposable
{
    private const string Token = "fedcba9876543210fedcba9876543210";

    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

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

        client.DefaultRequestHeaders.Add("Cookie", $"deepsharp-serve-{address.Port}={Token}");

        return new Served(app, client);
    }

    // A view as a page opens one: the stream, read event by event.
    private static async Task<View> ViewAsync(Served served, string name)
    {
        var answer = await served.Client.SendAsync(
            new HttpRequestMessage(HttpMethod.Get, $"/api/notebooks/{name}/updates"), HttpCompletionOption.ResponseHeadersRead, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.OK, answer.StatusCode);
        Assert.Equal("text/event-stream", answer.Content.Headers.ContentType?.MediaType);

        var events = SseParser.Create(await answer.Content.ReadAsStreamAsync(TestContext.Current.CancellationToken))
            .EnumerateAsync(TestContext.Current.CancellationToken).GetAsyncEnumerator(TestContext.Current.CancellationToken);

        return new View(answer, events);
    }

    private static async Task<SseItem<string>> NextAsync(View view)
    {
        Assert.True(await view.Events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken));

        return view.Events.Current;
    }

    private static T Read<T>(SseItem<string> item) => JsonSerializer.Deserialize<T>(item.Data, Json)!;

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
    public async Task ANameTheServerDoesNotServe_IsNotFound(string name)
    {
        await using var served = await StartAsync();

        Assert.Equal(HttpStatusCode.NotFound, (await served.Client.GetAsync($"/api/notebooks/{name}", TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await served.Client.GetAsync("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).StatusCode);
    }

    [Fact]
    public async Task AView_BeginsWithTheNotebookAsItStands_ThenEachChange_NumberedByVersion()
    {
        await using var served = await StartAsync();
        await using var view = await ViewAsync(served, "titanic.verso");

        var first = await NextAsync(view);
        var snapshot = Read<NotebookVersion>(first);

        Assert.Equal("snapshot", first.EventType);
        Assert.Equal("0", first.EventId);
        Assert.Equal(Titanic, snapshot.Cells.Select(cell => cell.Source));

        var fill = snapshot.Cells[3].Id;
        var mean = Titanic[3].Replace("median", "mean", StringComparison.Ordinal);
        var edited = await served.Client.PostAsJsonAsync($"/api/notebooks/titanic.verso/cells/{fill}/source", new { source = mean }, TestContext.Current.CancellationToken);

        Assert.Equal(mean, (await edited.Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken)).Source);

        var next = await NextAsync(view);
        var change = Read<NotebookChange>(next);

        Assert.Equal("change", next.EventType);
        Assert.Equal("1", next.EventId);
        Assert.Equal(mean, Assert.Single(change.Cells).Source);
        Assert.Equal(1, (await served.Client.GetFromJsonAsync<NotebookVersion>("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).Version);
    }

    [Fact]
    public async Task ARunAndAClickOnABlocksControl_AreMadeThroughTheServer_AndTheViewIsToldTheNewOrder()
    {
        await using var served = await StartAsync();
        await using var view = await ViewAsync(served, "titanic.verso");
        var cells = Read<NotebookVersion>(await NextAsync(view)).Cells;
        var read = cells[0].Id;

        var card = await (await served.Client.PostAsync($"/api/notebooks/titanic.verso/cells/{read}/run", null, TestContext.Current.CancellationToken))
            .Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken);

        Assert.NotEmpty(card.Outputs);

        await Click(served, read, ActionOf(card, action => action == "deepsharp.show"), "");

        var grid = (await served.Client.GetFromJsonAsync<NotebookVersion>("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).Cells[0];
        var ticked = await Click(served, read, ActionOf(grid, action => action.StartsWith("deepsharp.include ", StringComparison.Ordinal) && action.Contains("\"deck\"", StringComparison.Ordinal)), "true");

        Assert.True(ticked.StateChanged);

        NotebookChange change;

        do
        {
            change = Read<NotebookChange>(await NextAsync(view));
        }
        while (change.Order is null);

        Assert.DoesNotContain(cells[1].Id, change.Order);
    }

    private static async Task<GestureResult> Click(Served served, Guid cell, string action, string payload)
    {
        var answer = await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/gestures", new HostedGesture(cell, StepRenderer.Id, action, payload), TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.OK, answer.StatusCode);

        return await answer.Content.ReadFromJsonAsync<GestureResult>(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TypingIntoACellAChangeReplaced_AnswersTheVersionItIsGoneFrom()
    {
        await using var served = await StartAsync();
        var gone = Guid.NewGuid();

        var answer = await served.Client.PostAsJsonAsync($"/api/notebooks/titanic.verso/cells/{gone}/source", new { source = "x" }, TestContext.Current.CancellationToken);
        var body = await answer.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.Conflict, answer.StatusCode);
        Assert.Equal(gone, body.GetProperty("cell").GetGuid());
        Assert.Equal(0, body.GetProperty("version").GetInt64());
        Assert.Equal(HttpStatusCode.Conflict, (await served.Client.PostAsync($"/api/notebooks/titanic.verso/cells/{gone}/run", null, TestContext.Current.CancellationToken)).StatusCode);
    }

    [Fact]
    public async Task AClickNamingNoPart_IsRefused_SayingWhy()
    {
        await using var served = await StartAsync();
        var cell = (await served.Client.GetFromJsonAsync<NotebookVersion>("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).Cells[0].Id;

        var answer = await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/gestures", new HostedGesture(cell, "no.such.part", "deepsharp.show", ""), TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.UnprocessableEntity, answer.StatusCode);
        Assert.Contains("no.such.part", (await answer.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken)).GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ANotebookTheGuardRefusesToOpen_IsRefused_SayingWhy()
    {
        File.WriteAllText(
            At("old.ipynb"),
            $$"""{"cells": [{"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [{{JsonSerializer.Serialize(Titanic[0])}}]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}""");

        await using var served = await StartAsync();

        var answer = await served.Client.GetAsync("/api/notebooks/old.ipynb", TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.UnprocessableEntity, answer.StatusCode);
        Assert.Contains("read.csv", (await answer.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken)).GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Stopping_IsAnswered_WhetherOrNotARunIsUnderWay()
    {
        await using var served = await StartAsync();

        Assert.Equal(HttpStatusCode.NoContent, (await served.Client.PostAsync("/api/notebooks/titanic.verso/stop", null, TestContext.Current.CancellationToken)).StatusCode);
    }

    [Fact]
    public async Task ClosingANotebook_EndsItsViews_AndWhatItHeldUnsavedIsGone()
    {
        await using var served = await StartAsync();
        await using var view = await ViewAsync(served, "titanic.verso");
        var fill = Read<NotebookVersion>(await NextAsync(view)).Cells[3].Id;

        await served.Client.PostAsJsonAsync($"/api/notebooks/titanic.verso/cells/{fill}/source", new { source = "{}" }, TestContext.Current.CancellationToken);
        await NextAsync(view);

        Assert.Equal(HttpStatusCode.NoContent, (await served.Client.PostAsync("/api/notebooks/titanic.verso/close", null, TestContext.Current.CancellationToken)).StatusCode);
        Assert.False(await view.Events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));

        // Opened again, it is the notebook the file holds.
        Assert.Equal(Titanic[3], (await served.Client.GetFromJsonAsync<NotebookVersion>("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken)).Cells[3].Source);
    }

    [Fact]
    public async Task APageThatLeavesAsSoonAsItHasTheNotebook_EndsItsStream_WithoutWaitingForAChange()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken);
        using var view = host.Subscribe();

        var events = NotebookEndpoints.Changes.Events(view, TestContext.Current.CancellationToken).GetAsyncEnumerator(TestContext.Current.CancellationToken);

        Assert.True(await events.MoveNextAsync());
        Assert.Equal("snapshot", events.Current.EventType);

        await events.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task APageThatLeavesAsSoonAsItHasAChange_EndsItsStream_WithoutWaitingForTheNext()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken);
        using var view = host.Subscribe();

        var events = NotebookEndpoints.Changes.Events(view, TestContext.Current.CancellationToken).GetAsyncEnumerator(TestContext.Current.CancellationToken);

        Assert.True(await events.MoveNextAsync());

        await host.EditAsync(host.Cells[3].Id, Titanic[3].Replace("median", "mean", StringComparison.Ordinal));

        Assert.True(await events.MoveNextAsync());
        Assert.Equal("change", events.Current.EventType);

        await events.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task StoppingTheServer_EndsAnOpenView_AtOnce()
    {
        var served = await StartAsync();
        await using var view = await ViewAsync(served, "titanic.verso");

        await NextAsync(view);

        var stopping = served.App.StopAsync(TestContext.Current.CancellationToken);

        Assert.False(await view.Events.MoveNextAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken));
        await stopping.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        await served.DisposeAsync();
    }

    [Fact]
    public async Task AViewWhosePageWentAway_Ends_SoTheNotebookClosesOnceNoPageShowsIt()
    {
        await using var served = await StartAsync(grace: TimeSpan.FromMilliseconds(200));
        var host = await served.App.Services.GetRequiredService<OpenNotebooks>().OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken);
        var view = await ViewAsync(served, "titanic.verso");

        await NextAsync(view);

        // The page's tab closes: its connection goes with it.
        await view.DisposeAsync();

        for (var waited = TimeSpan.Zero; ; waited += TimeSpan.FromMilliseconds(100))
        {
            Assert.True(waited < TimeSpan.FromSeconds(10), "the notebook stayed open");

            // Asked, not viewed: a view would be a page showing it.
            try
            {
                await host.ToolbarAsync();
            }
            catch (ObjectDisposedException)
            {
                break;
            }

            await Task.Delay(100, TestContext.Current.CancellationToken);
        }
    }

    // A page's stream: the answer it arrives in, closed with the view, as a tab closing closes its connection.
    private sealed class View(HttpResponseMessage answer, IAsyncEnumerator<SseItem<string>> events) : IAsyncDisposable
    {
        public IAsyncEnumerator<SseItem<string>> Events => events;

        public async ValueTask DisposeAsync()
        {
            await events.DisposeAsync();
            answer.Dispose();
        }
    }

    // A started server and a browser that carries its token, stopped and closed with the test.
    private sealed class Served(WebApplication app, HttpClient client) : IAsyncDisposable
    {
        public WebApplication App => app;

        public HttpClient Client => client;

        public async ValueTask DisposeAsync()
        {
            client.Dispose();
            await app.DisposeAsync();
        }
    }
}
