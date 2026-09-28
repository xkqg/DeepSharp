// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// What a page asks of the notebooks a server serves: which there are and whether a new one can be made, over HTTP; and
/// a notebook itself over one socket per page — the notebook as it stands and each change after it, and everything a
/// person does to it, each ask answered to that page alone. A name the server does not serve is not found; a notebook
/// that cannot be opened is refused, saying why.
/// </summary>
internal static class NotebookEndpoints
{
    /// <summary>Maps the notebooks' endpoints under <c>/api/notebooks</c>.</summary>
    /// <param name="app">The server.</param>
    public static void Map(WebApplication app)
    {
        var notebooks = app.MapGroup("/api/notebooks");

        notebooks.MapGet("/", async (ServedNotebooks served) => Results.Ok(await served.NamesAsync()));
        notebooks.MapPost("/", NewAsync);

        // What the server serves: a folder, where a notebook can be made, or one notebook.
        app.MapGet("/api/served", (ServedNotebooks served) => Results.Ok(new { folder = served.IsFolder }));
        notebooks.MapGet("/{name}/socket", SocketAsync);
    }

    // A page's socket on a notebook: opened first, so a refusal reaches the page in words it can show — a browser shows a
    // page nothing of an answer to a socket it refused.
    private static async Task SocketAsync(string name, HttpContext context)
    {
        if (!context.WebSockets.IsWebSocketRequest)
        {
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            await context.Response.WriteAsync("A notebook is served over a socket, which this is not a request for.");

            return;
        }

        using var socket = await context.WebSockets.AcceptWebSocketAsync();
        var notebooks = context.RequestServices.GetRequiredService<OpenNotebooks>();

        if (await context.RequestServices.GetRequiredService<ServedNotebooks>().FileOfAsync(name) is not { } file)
        {
            await NotebookSocket.RefuseAsync(socket, StatusCodes.Status404NotFound, $"No notebook named '{name}' is served here.");

            return;
        }

        NotebookHost host;

        try
        {
            host = await notebooks.OpenAsync(file, context.RequestAborted);
        }
        catch (InvalidOperationException refused)
        {
            await NotebookSocket.RefuseAsync(socket, StatusCodes.Status422UnprocessableEntity, refused.Message);

            return;
        }

        await new NotebookSocket(socket, host, notebooks).RunAsync();
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

        // Where a person opens it: the page, naming it.
        return Results.Created($"/?notebook={Uri.EscapeDataString(naming.Name)}", new { name = naming.Name });
    }

    // The name a new notebook is to have.
    internal readonly record struct Naming(string Name);
}
