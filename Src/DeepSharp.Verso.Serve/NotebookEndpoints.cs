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
        notebooks.MapPost("/", NewAsync);

        // What the server serves: a folder, where a notebook can be made, or one notebook.
        app.MapGet("/api/served", (ServedNotebooks served) => Results.Ok(new { folder = served.IsFolder }));
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

        notebooks.MapGet("/{name}/kinds", (string name, HttpContext context) => WithAsync(name, context, host => Task.FromResult(Results.Ok(host.Kinds))));

        // A cell added after the one the page names, or at the end when it names none; it starts empty.
        notebooks.MapPost("/{name}/cells", (string name, Adding adding, HttpContext context) => WithAsync(name, context, async host =>
            Results.Ok(adding.After is { } after
                ? await host.InsertAsync(after, new HostedKind(adding.Type, adding.Language, adding.Type))
                : await host.AddAsync(new HostedKind(adding.Type, adding.Language, adding.Type)))));
        notebooks.MapPost("/{name}/cells/{cell:guid}/remove", (string name, Guid cell, HttpContext context) => WithAsync(name, context, async host =>
        {
            await host.RemoveAsync(cell);

            return Results.NoContent();
        }));

        // A move names the one neighbour it passes: the cell above for a move up, the cell below for a move down.
        notebooks.MapPost("/{name}/cells/{cell:guid}/move", (string name, Guid cell, Moving moving, HttpContext context) => WithAsync(name, context, async host =>
        {
            switch (moving)
            {
                case { Before: { } before, After: null }:
                    await host.MoveBeforeAsync(cell, before);

                    return Results.NoContent();

                case { Before: null, After: { } after }:
                    await host.MoveAfterAsync(cell, after);

                    return Results.NoContent();

                default:
                    return Results.BadRequest(new { message = "A move names the one neighbour it passes: the cell it goes before, or the cell it goes after." });
            }
        }));
        notebooks.MapPost(
            "/{name}/cells/{cell:guid}/kind",
            (string name, Guid cell, Kind kind, HttpContext context) => WithAsync(name, context, async host =>
                Results.Ok(await host.ChangeKindAsync(cell, new HostedKind(kind.Type, kind.Language, kind.Type)))));
        notebooks.MapPost(
            "/{name}/cells/{cell:guid}/completions",
            (string name, Guid cell, Asking asking, HttpContext context) => WithAsync(name, context, async host =>
                Results.Ok(await host.CompletionsAsync(cell, asking.Code, asking.Position))));
        notebooks.MapPost(
            "/{name}/cells/{cell:guid}/hover",
            (string name, Guid cell, Asking asking, HttpContext context) => WithAsync(name, context, async host =>
                await host.HoverAsync(cell, asking.Code, asking.Position) is { } hover ? Results.Ok(hover) : Results.NoContent()));

        notebooks.MapGet("/{name}/toolbar", (string name, HttpContext context) => WithAsync(name, context, async host => Results.Ok(await host.ToolbarAsync())));

        // A file a button hands over goes to the page that pressed it, under its own name, and never beside the notebook.
        notebooks.MapPost("/{name}/toolbar/{button}", (string name, string button, Pressed pressed, HttpContext context) => WithAsync(name, context, async host =>
            await host.RunToolbarAsync(button, [.. pressed.Cells]) is { } file ? Results.File(file.Bytes, file.ContentType, file.Name) : Results.NoContent()));
        notebooks.MapGet(
            "/{name}/cells/{cell:guid}/properties",
            (string name, Guid cell, HttpContext context) => WithAsync(name, context, async host => Results.Ok(await host.PropertiesAsync(cell))));

        // The value is handed to the part as it came, as JSON, the way Verso's own editors hand it on: each part reads it.
        notebooks.MapPost("/{name}/cells/{cell:guid}/properties", (string name, Guid cell, Changed changed, HttpContext context) => WithAsync(name, context, async host =>
        {
            await host.SetPropertyAsync(cell, changed.Part, changed.Field, changed.Value);

            return Results.NoContent();
        }));
        notebooks.MapPost("/{name}/save", (string name, HttpContext context) => WithAsync(name, context, async host =>
        {
            await host.SaveAsync();

            return Results.NoContent();
        }));

        // Closes it now, whatever it holds unsaved — what a person asks for who discards the changes; its views end.
        notebooks.MapPost("/{name}/close", (string name, HttpContext context) => WithAsync(name, context, async host =>
        {
            await context.RequestServices.GetRequiredService<OpenNotebooks>().CloseAsync(host);

            return Results.NoContent();
        }));
    }

    // A new notebook in the folder served, under a bare .verso file name; never over a file that is there.
    private static async Task<IResult> NewAsync(Naming naming, HttpContext context)
    {
        string? file;

        try
        {
            file = context.RequestServices.GetRequiredService<ServedNotebooks>().NewFile(naming.Name);
        }
        catch (ArgumentException refused)
        {
            return Results.BadRequest(new { message = refused.Message });
        }

        if (file is null)
        {
            return Results.NotFound();
        }

        try
        {
            await context.RequestServices.GetRequiredService<OpenNotebooks>().CreateAsync(file, context.RequestAborted);
        }
        catch (Exception there) when (there is IOException or InvalidOperationException)
        {
            return Results.Conflict(new { message = there.Message });
        }

        return Results.Created($"/api/notebooks/{Uri.EscapeDataString(naming.Name)}", new { name = naming.Name });
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

    // A cell to add: after which cell, or at the end with none, and of which kind.
    internal readonly record struct Adding(Guid? After, string Type, string? Language);

    // The neighbour a cell moves past: the one it goes before, or the one it goes after.
    internal readonly record struct Moving(Guid? Before, Guid? After);

    // The kind a cell is turned into.
    internal readonly record struct Kind(string Type, string? Language);

    // A cell's text as the person has it, and where the cursor stands in it.
    internal readonly record struct Asking(string Code, int Position);

    // The name a new notebook is to have.
    internal readonly record struct Naming(string Name);

    // The cells a toolbar button is pressed for; none for a button of the notebook as a whole.
    internal readonly record struct Pressed(IReadOnlyList<Guid> Cells);

    // A field of a cell's properties panel, changed: the part its section came from, the field, and what it now holds.
    internal readonly record struct Changed(string Part, string Field, object? Value);

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
