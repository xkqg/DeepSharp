// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.Sockets;
using System.Text;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The stand-in venue the tests use, answering over a socket on this machine: what the package under check reaches as it
/// would reach the real one, without anything of the real one.
/// </summary>
/// <remarks>
/// A request is read to the end of its headers, handed to the stand-in as the message it is, and answered with the status,
/// the headers and the body the stand-in gave — the venue's own date, weight and identifier among them — and the connection
/// is closed once the client has read it. It listens on the loopback address alone.
/// </remarks>
internal sealed class StandIn : IDisposable
{
    private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
    private readonly HttpMessageInvoker _venue;
    private readonly CancellationTokenSource _stop = new();

    public StandIn(HttpMessageHandler venue)
    {
        _venue = new HttpMessageInvoker(venue);
        _listener.Start();
        _ = Task.Run(AcceptAsync);
    }

    /// <summary>The port it listens on.</summary>
    public int Port => ((IPEndPoint)_listener.LocalEndpoint).Port;

    public void Dispose()
    {
        _stop.Cancel();
        _listener.Stop();
        _venue.Dispose();
        _stop.Dispose();
    }

    private async Task AcceptAsync()
    {
        while (!_stop.IsCancellationRequested)
        {
            TcpClient client;

            try
            {
                client = await _listener.AcceptTcpClientAsync(_stop.Token);
            }
            catch (Exception ended) when (ended is OperationCanceledException or ObjectDisposedException or SocketException)
            {
                return;
            }

            _ = Task.Run(() => AnswerAsync(client));
        }
    }

    private async Task AnswerAsync(TcpClient client)
    {
        using (client)
        {
            var stream = client.GetStream();
            var head = new StringBuilder();
            var one = new byte[1];

            while (!head.ToString().EndsWith("\r\n\r\n", StringComparison.Ordinal) && await stream.ReadAsync(one, _stop.Token) == 1)
            {
                head.Append((char)one[0]);
            }

            var target = head.ToString().Split(' ')[1];

            using var request = new HttpRequestMessage(HttpMethod.Get, new Uri($"http://127.0.0.1:{Port}{target}"));
            using var response = await _venue.SendAsync(request, _stop.Token);

            var body = await response.Content.ReadAsByteArrayAsync(_stop.Token);
            var answer = new StringBuilder($"HTTP/1.1 {(int)response.StatusCode} {response.ReasonPhrase}\r\n");

            foreach (var header in response.Headers.Concat(response.Content.Headers).Where(header => header.Key != "Content-Length"))
            {
                answer.Append(header.Key).Append(": ").Append(string.Join(", ", header.Value)).Append("\r\n");
            }

            answer.Append("Content-Length: ").Append(body.Length).Append("\r\nConnection: close\r\n\r\n");

            await stream.WriteAsync(Encoding.ASCII.GetBytes(answer.ToString()), _stop.Token);
            await stream.WriteAsync(body, _stop.Token);
            await stream.FlushAsync(_stop.Token);

            // End as a server does: say nothing more, and let the socket go only once the client has read what it was given
            // and closed its side, so that an answer is never cut off by a reset.
            client.Client.Shutdown(SocketShutdown.Send);

            using var patience = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);

            patience.CancelAfter(TimeSpan.FromSeconds(2));

            try
            {
                while (await stream.ReadAsync(new byte[256], patience.Token) > 0)
                {
                }
            }
            catch (OperationCanceledException)
            {
                // The client keeps its side open: the answer was given whole, and the socket goes now.
            }
        }
    }
}
