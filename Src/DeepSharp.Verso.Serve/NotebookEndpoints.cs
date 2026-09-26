// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net.ServerSentEvents;
using System.Text.Json;
using DeepSharp.Verso.Api;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// What a page asks of the notebooks a server serves: which there are, a notebook as it stands and each change after it
/// — a stream the page keeps open — and what a person does to one. A name the server does not serve is not found; a cell
/// a change replaced answers the version it is gone from, so the page knows which version to wait for; a refusal says why.
/// </summary>
internal static class NotebookEndpoints
{
    // As a browser's script reads JSON, and as the server answers it everywhere else.
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    /// <summary>Maps the notebooks' endpoints under <c>/api/notebooks</c>.</summary>
    /// <param name="app">The server.</param>
    public static void Map(WebApplication app)
    {
        var notebooks = app.MapGroup("/api/notebooks");

        notebooks.MapGet("/", async (ServedNotebooks served) => Results.Ok(await served.NamesAsync()));
        notebooks.MapGet("/{name}", (string name, HttpContext context) => WithAsync(name, context, host => Task.FromResult(Results.Ok(host.Current))));
        notebooks.MapGet("/{name}/updates", (string name, HttpContext context) => WithAsync(name, context, host => Task.FromResult<IResult>(new Changes(host))));
        notebooks.MapPost(
            "/{name}/cells/{cell:guid}/source",
            (string name, Guid cell, Typed typed, HttpContext context) => WithAsync(name, context, async host => Results.Ok(await host.EditAsync(cell, typed.Source))));
        notebooks.MapPost(
            "/{name}/cells/{cell:guid}/run",
            (string name, Guid cell, HttpContext context) => WithAsync(name, context, async host => Results.Ok(await host.RunAsync(cell))));
        notebooks.MapPost("/{name}/stop", (string name, HttpContext context) => WithAsync(name, context, host =>
        {
            host.Stop();

            return Task.FromResult(Results.NoContent());
        }));
        notebooks.MapPost(
            "/{name}/gestures",
            (string name, HostedGesture gesture, HttpContext context) => WithAsync(name, context, async host => Results.Ok(await host.GestureAsync(gesture))));

        // Closes it now, whatever it holds unsaved — what a person asks for who discards the changes; its views end.
        notebooks.MapPost("/{name}/close", (string name, HttpContext context) => WithAsync(name, context, async host =>
        {
            await context.RequestServices.GetRequiredService<OpenNotebooks>().CloseAsync(host);

            return Results.NoContent();
        }));
    }

    // Something asked of a notebook the server serves, and what came of it.
    private static async Task<IResult> WithAsync(string name, HttpContext context, Func<NotebookHost, Task<IResult>> asked)
    {
        if (await context.RequestServices.GetRequiredService<ServedNotebooks>().FileOfAsync(name) is not { } file)
        {
            return Results.NotFound();
        }

        try
        {
            return await asked(await context.RequestServices.GetRequiredService<OpenNotebooks>().OpenAsync(file, context.RequestAborted));
        }
        catch (CellGoneException gone)
        {
            return Results.Conflict(new { cell = gone.Cell, version = gone.Version, message = gone.Message });
        }
        catch (InvalidOperationException refused)
        {
            return Results.UnprocessableEntity(new { message = refused.Message });
        }
    }

    // A cell's new text, as a page sends it.
    internal readonly record struct Typed(string Source);

    // A notebook's stream: the notebook as it stands, then each change after it, until the page goes away, the server
    // stops, or the notebook closes. The page's view ends with it.
    internal sealed class Changes(NotebookHost host) : IResult
    {
        public async Task ExecuteAsync(HttpContext context)
        {
            using var view = host.Subscribe();
            using var ending = CancellationTokenSource.CreateLinkedTokenSource(
                context.RequestAborted, context.RequestServices.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping);

            context.Response.ContentType = "text/event-stream";
            context.Response.Headers.CacheControl = "no-cache";

            try
            {
                await SseFormatter.WriteAsync(Events(view, ending.Token), context.Response.Body, ending.Token);
            }
            catch (OperationCanceledException)
            {
                // The page went away or the server is stopping: the stream simply ends.
            }
        }

        /// <summary>The events of a view: the notebook as it stood when the view began, then each change after it.</summary>
        /// <param name="view">The view.</param>
        /// <param name="cancellationToken">Ends the events: the page went away or the server is stopping.</param>
        /// <returns>The events.</returns>
        internal static IAsyncEnumerable<SseItem<string>> Events(NotebookSubscription view, CancellationToken cancellationToken) =>
            view.ReadAllAsync(cancellationToken)
                .Select(change => Item(change, "change", change.Version))
                .Prepend(Item(view.Snapshot, "snapshot", view.Snapshot.Version));

        private static SseItem<string> Item<T>(T value, string type, long version) =>
            new(JsonSerializer.Serialize(value, Json), type) { EventId = version.ToString(CultureInfo.InvariantCulture) };
    }
}
