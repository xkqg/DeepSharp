// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using DeepSharp.Verso.Api;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Primitives;

namespace DeepSharp.Verso.Serve;

/// <summary>DeepSharp's server: the notebook in a browser, on this computer alone.</summary>
/// <remarks>
/// A notebook runs code, so the server is shut to everyone but the person who started it: it listens on this computer
/// alone; it answers only a request that carries the token it said when it started — in the address, or in the cookie
/// its first page sets; it opens a notebook's socket and makes a change only for its own page, since a browser carries
/// this computer's cookie for a page from any of its ports, so no other page can drive it; and it answers only under a
/// name of this computer, so a site that makes its own name point here cannot reach it through the browser either. It
/// serves its own page, which it carries, and nothing from the folder it runs in. It writes nothing to the console once
/// it has said where it is, since a C# cell takes the console over while it runs.
/// </remarks>
public static class NotebookServer
{
    // A browser keeps a cookie for a computer whatever its port, so each server names its own after the port.
    private const string CookiePrefix = "deepsharp-serve-";

    /// <summary>Builds the server for what a command line said.</summary>
    /// <param name="options">What the command line said.</param>
    /// <param name="token">What every request must carry.</param>
    /// <param name="said">Where the one line said at the start goes.</param>
    /// <returns>The server, not yet started.</returns>
    public static WebApplication Build(ServeOptions options, string token, TextWriter said)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(token);
        ArgumentNullException.ThrowIfNull(said);

        // The tool's own folder, never the one it runs in: nothing there is served.
        var builder = WebApplication.CreateSlimBuilder(new WebApplicationOptions { ContentRootPath = AppContext.BaseDirectory });

        builder.Logging.ClearProviders();
        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Listen(IPAddress.Loopback, options.Port));
        builder.Services.AddHostFiltering(filtering => filtering.AllowedHosts = ["localhost", "127.0.0.1", "[::1]"]);
        // Made by the container, so the container closes it — and every notebook it holds — when the server stops; one
        // handed to it would be left open.
        builder.Services.AddSingleton(_ => new OpenNotebooks(options.Grace));
        builder.Services.AddSingleton(_ => new ServedNotebooks(options));

        var app = builder.Build();

        // Put together once: the page does not change while the server runs.
        var page = ServedPage.Carrying();

        app.UseHostFiltering();
        app.UseWebSockets();

        // A socket is asked for as a page is fetched, but it makes changes: it is held to the rule every change is.
        app.Use((context, next) => Carried(context, token) switch
        {
            TokenIn.Nothing => Refused(context),
            var carried when (!HttpMethods.IsGet(context.Request.Method) || context.WebSockets.IsWebSocketRequest) && !FromOwnPage(context, carried) => Forbidden(context),
            _ => next(context),
        });
        app.MapGet("/", page.Answer);
        NotebookEndpoints.Map(app);
        app.Lifetime.ApplicationStarted.Register(() => Started(app, options, token, said));

        // The notebooks close as soon as the server is told to stop, each run under way stopped as a close stops it, so a
        // request that waits for a run is answered, and the server does not wait out the time it gives a request to end.
        // Closing the notebooks is one close, which the container waits for as it closes them itself.
        app.Lifetime.ApplicationStopping.Register(() => _ = app.Services.GetRequiredService<OpenNotebooks>().DisposeAsync().AsTask());

        return app;
    }

    /// <summary>Runs <c>deepsharp-serve</c> with a command line, until it is stopped.</summary>
    /// <param name="args">The command line's words.</param>
    /// <param name="said">Where the one line said at the start goes, and the usage when it is asked for.</param>
    /// <param name="errors">Where what went wrong goes.</param>
    /// <param name="stop">Stops it; Ctrl+C does too.</param>
    /// <returns>0 once it stopped, or said its usage; 1 when it could not listen; 2 when its command line was refused.</returns>
    public static async Task<int> RunAsync(IReadOnlyList<string> args, TextWriter said, TextWriter errors, CancellationToken stop)
    {
        ArgumentNullException.ThrowIfNull(said);
        ArgumentNullException.ThrowIfNull(errors);

        ServeOptions options;

        try
        {
            options = ServeOptions.Parse(args, Environment.CurrentDirectory);
        }
        catch (ArgumentException refused)
        {
            await errors.WriteLineAsync(refused.Message);
            await errors.WriteLineAsync(ServeOptions.Usage);

            return 2;
        }

        if (options.Help)
        {
            await said.WriteLineAsync(ServeOptions.Usage);

            return 0;
        }

        await using var app = Build(options, RandomNumberGenerator.GetHexString(48, lowercase: true), said);

        try
        {
            await app.StartAsync(stop);
        }
        catch (IOException taken)
        {
            await errors.WriteLineAsync(taken.Message);

            return 1;
        }

        await app.WaitForShutdownAsync(stop);

        return 0;
    }

    // Where a request carried the token: nowhere, in its address as the tool said it, or in the cookie the first page set.
    private enum TokenIn
    {
        Nothing,
        Address,
        Cookie,
    }

    private static TokenIn Carried(HttpContext context, string token)
    {
        var cookie = $"{CookiePrefix}{context.Connection.LocalPort}";

        if (Same(context.Request.Query["token"], token))
        {
            context.Response.Cookies.Append(cookie, token, new CookieOptions { HttpOnly = true, SameSite = SameSiteMode.Strict, Path = "/" });

            return TokenIn.Address;
        }

        return Same(context.Request.Cookies[cookie], token) ? TokenIn.Cookie : TokenIn.Nothing;
    }

    // A change is made, and a notebook's socket opened, only for the server's own page. A browser names the page a
    // request comes from in its Origin, and carries this computer's cookie for a page from any of its ports — to a
    // browser they are one site — so a request the cookie carried must name the server's own page; a program that carries
    // the token in the address names no page, but one it names must be the server's.
    private static bool FromOwnPage(HttpContext context, TokenIn carried)
    {
        var origin = context.Request.Headers.Origin;

        return StringValues.IsNullOrEmpty(origin)
            ? carried == TokenIn.Address
            : string.Equals(origin, $"http://{context.Request.Host}", StringComparison.OrdinalIgnoreCase);
    }

    // Compared in the same time whatever the guess, so the time a refusal takes says nothing of the token.
    private static bool Same(string? given, string token) =>
        given is not null && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(given), Encoding.UTF8.GetBytes(token));

    private static Task Refused(HttpContext context)
    {
        context.Response.StatusCode = StatusCodes.Status401Unauthorized;

        return context.Response.WriteAsync("This server answers only the address it said when it started.");
    }

    private static Task Forbidden(HttpContext context)
    {
        context.Response.StatusCode = StatusCodes.Status403Forbidden;

        return context.Response.WriteAsync("This server makes a change only for its own page.");
    }

    // Once it listens: where, on the address it really bound — a port already taken stops it before it gets here — and a
    // browser on that address, unless it was told to open none.
    private static void Started(WebApplication app, ServeOptions options, string token, TextWriter said)
    {
        var bound = app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single();
        var address = $"{bound}/?token={token}";

        said.WriteLine($"DeepSharp serves {options.Path} at {address}");

        if (options.OpenBrowser)
        {
            Process.Start(new ProcessStartInfo(address) { UseShellExecute = true });
        }
    }
}
