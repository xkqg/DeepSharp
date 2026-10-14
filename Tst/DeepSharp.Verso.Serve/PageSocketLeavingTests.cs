// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.WebSockets;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// A page's socket in these tests leaves as a page does — a close frame, and the server's own in answer — and every frame it
/// was handed before then is whole. Closing a <c>ClientWebSocket</c> with <c>CloseAsync</c> receives, beside the read that is
/// already going, into a buffer no longer than a control frame, and throws away what it receives that is no close; a message
/// that was still arriving was split between the two, and the read was left holding a tail that is no JSON. The server here
/// tells a message in two fragments with a pause between, which is a message still arriving whenever the page leaves.
/// </summary>
public sealed class PageSocketLeavingTests
{
    private const string Token = "fedcba9876543210fedcba9876543210";

    [Fact]
    public async Task APageLeavingWhileAMessageIsStillArriving_IsHandedTheMessageWhole_AndLeavesWithoutAFault()
    {
        var builder = WebApplication.CreateBuilder();

        builder.WebHost.ConfigureKestrel(kestrel => kestrel.Listen(IPAddress.Loopback, 0));
        builder.Logging.ClearProviders();

        await using var app = builder.Build();

        app.UseWebSockets();
        app.Map("/api/notebooks/x/socket", Telling);

        await app.StartAsync(TestContext.Current.CancellationToken);

        try
        {
            var address = new Uri(app.Services.GetRequiredService<IServer>().Features.GetRequiredFeature<IServerAddressesFeature>().Addresses.Single());

            for (var leaving = 0; leaving < 6; leaving++)
            {
                var page = await PageSocket.OpenAsync(address, "x", Token);

                // The page leaves before the message begins to arrive, and the message is still arriving when its close is being waited for.
                await Task.Delay(60, TestContext.Current.CancellationToken);
                await page.DisposeAsync();

                // The message was handed whole, or not at all; had it been handed in pieces, leaving would have failed with it.
                Assert.Equal(["tick"], page.Frames);
            }
        }
        finally
        {
            await app.StopAsync(CancellationToken.None);
        }
    }

    // A server that tells one message in two fragments with a pause between — the second begins with the end of a string, so a
    // reader that is handed the second without its first bytes is handed no JSON — then waits for the page's close and answers it.
    private static async Task Telling(HttpContext context)
    {
        using var socket = await context.WebSockets.AcceptWebSocketAsync();

        await Task.Delay(150);
        await socket.SendAsync(Encoding.UTF8.GetBytes($"{{\"type\":\"tick\",\"padding\":\"{new string('x', 200)}"), WebSocketMessageType.Text, endOfMessage: false, CancellationToken.None);
        await Task.Delay(300);
        await socket.SendAsync(Encoding.UTF8.GetBytes($"{new string('x', 50)}\"{new string(' ', 200)}}}"), WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None);

        var buffer = new byte[256];

        try
        {
            while ((await socket.ReceiveAsync(buffer.AsMemory(), CancellationToken.None)).MessageType != WebSocketMessageType.Close)
            {
                // Whatever the page still sends before its close is read and let go.
            }

            await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "the page left", CancellationToken.None);
        }
        catch (WebSocketException)
        {
            // The page went without a word.
        }
    }
}
