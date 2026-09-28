// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Verso.Abstractions;
using static Microsoft.Playwright.Assertions;

namespace DeepSharp.Tests.Serve;

// The server's boundary in the browser that opened it: a browser keeps a cookie for this computer whatever the port, so a
// page from another port of it rides the same cookie — and whatever it asks to change is refused.
public sealed partial class PageTests
{
    [Fact]
    public async Task APageFromAnotherPortOfThisComputer_CannotOpenTheNotebooksSocket_InTheBrowserThatOpenedIt()
    {
        await SaveAsync("code.verso", new CellModel { Type = "code", Language = "csharp", Source = "var a = 1;" });
        await using var served = await StartAsync();
        var page = await OpenAsync(served, "code.verso");
        var text = Cell(page, 0).Locator("textarea.source");

        await text.FillAsync("var typed = 42;");
        await text.BlurAsync();

        for (var waited = 0; (await CurrentAsync(served, "code.verso")).Cells[0].Source != "var typed = 42;"; waited += 50)
        {
            Assert.True(waited < 10_000, "the typing never reached the server");
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        // Its socket, were it opened, would close the notebook and lose what was typed.
        await using var other = await OtherPageAsync($$"""
            <!doctype html><title>other</title>
            <script>
              const socket = new WebSocket('ws://{{served.Address.Authority}}/api/notebooks/code.verso/socket');

              socket.onopen = () => socket.send(JSON.stringify({ id: 1, ask: 'close' }));
              socket.onclose = () => document.title = 'sent';
            </script>
            """);
        var visitor = await page.Context.NewPageAsync();

        await visitor.GotoAsync(other.Address.ToString());
        await Expect(visitor).ToHaveTitleAsync("sent");

        Assert.Equal("var typed = 42;", (await CurrentAsync(served, "code.verso")).Cells[0].Source);
    }

    // Somebody else's page, served from another port of this computer.
    private static async Task<OtherPage> OtherPageAsync(string html)
    {
        var builder = WebApplication.CreateSlimBuilder();

        builder.Logging.ClearProviders();
        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Listen(IPAddress.Loopback, 0));

        var app = builder.Build();

        app.MapGet("/", () => Results.Content(html, "text/html"));
        await app.StartAsync(TestContext.Current.CancellationToken);

        return new OtherPage(app, new Uri(app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single()));
    }

    private sealed class OtherPage(WebApplication app, Uri address) : IAsyncDisposable
    {
        public Uri Address => address;

        public ValueTask DisposeAsync() => app.DisposeAsync();
    }
}
