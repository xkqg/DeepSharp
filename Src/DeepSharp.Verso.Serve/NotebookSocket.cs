// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net.WebSockets;
using System.Text.Json;
using System.Threading.Channels;
using DeepSharp.Verso.Api;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// One page's socket on one notebook — the page's view of it, and everything the page asks of it, as Verso's browser
/// editor has one connection per tab: the notebook as it stands and the kinds a cell can be first, then each change as
/// it comes, and each ask answered to this page alone, by the id the page gave it.
/// </summary>
/// <remarks>
/// <para>
/// What the page sends is read one ask at a time, and nothing is awaited between two but the next: a Stop is answered at
/// once, past everything asked before it, and every other ask is made of the notebook as it is read, so the notebook's
/// turn takes the asks in the order the page sent them. Each ask is owned until it is done — its answer, or why it was
/// refused — and a page that went away drops the answer, never the ask.
/// </para>
/// <para>
/// One writer tells the page everything: a change the notebook made goes before the answer to the ask that made it.
/// When the notebook closes, every ask made of it is answered first, and then the page is told the notebook was closed.
/// </para>
/// </remarks>
/// <param name="socket">The page's socket.</param>
/// <param name="host">The notebook.</param>
/// <param name="notebooks">The notebooks the server holds open.</param>
internal sealed class NotebookSocket(WebSocket socket, NotebookHost host, OpenNotebooks notebooks)
{
    // As a browser's script reads JSON.
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    // How long a page is given to close its side once the server closed its own.
    private static readonly TimeSpan Closing = TimeSpan.FromSeconds(5);

    // What a person is told of an ask on a cell a change before it rewrote or took away; the refusal still names the cell and
    // the version it is gone from, for a program that reads those.
    private const string GoneSays = "The cell is no longer in the notebook: a change made before this one rewrote it or took it away.";

    // The answers waiting for the writer, in the order the asks were done.
    private readonly Channel<object> _answers = Channel.CreateUnbounded<object>(new UnboundedChannelOptions { SingleReader = true });

    // Marked when the page has gone away: the writer stops waiting.
    private readonly CancellationTokenSource _gone = new();

    // How many asks are made and not answered yet.
    private int _outstanding;

    /// <summary>Refuses a page a notebook it asked for, saying why, and closes its socket.</summary>
    /// <param name="socket">The page's socket.</param>
    /// <param name="status">The refusal, as HTTP names it.</param>
    /// <param name="why">Why, in words a person reads.</param>
    /// <returns>When the socket is closed.</returns>
    /// <remarks>A browser shows a page nothing of an answer to a socket it refused, so the socket opens to say it.</remarks>
    public static async Task RefuseAsync(WebSocket socket, int status, string why)
    {
        await socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(new Refusal("refused", 0, status, why, null, null), Json), WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None);
        await socket.CloseOutputAsync((WebSocketCloseStatus)(4000 + status), "refused", CancellationToken.None);
        await ClosedAsync(socket);
    }

    /// <summary>Serves the page until it goes away, or the notebook closes.</summary>
    /// <returns>When the socket is closed.</returns>
    public async Task RunAsync()
    {
        using var view = host.Subscribe();
        var writing = WriteAsync(view);
        var reading = ReadAsync();

        if (await Task.WhenAny(reading, writing) == writing)
        {
            // The server closed its side, the notebook being closed, and the page is given a moment to close its own; one
            // that does not is let go.
            if (await Task.WhenAny(reading, Task.Delay(Closing)) != reading)
            {
                socket.Abort();
            }

            await reading;
            await writing;

            return;
        }

        // The page went away: the writer stops, and a page that closed its side is answered in kind.
        await _gone.CancelAsync();
        await writing;

        if (socket.State == WebSocketState.CloseReceived)
        {
            await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "the page left", CancellationToken.None);
        }
    }

    // The page's side of the socket, read ask by ask until it closes it or goes: nothing is awaited but the next ask.
    private async Task ReadAsync()
    {
        var buffer = new byte[16 * 1024];

        try
        {
            while (true)
            {
                using var frame = new MemoryStream();
                ValueWebSocketReceiveResult received;

                do
                {
                    received = await socket.ReceiveAsync(buffer.AsMemory(), CancellationToken.None);

                    if (received.MessageType == WebSocketMessageType.Close)
                    {
                        return;
                    }

                    frame.Write(buffer, 0, received.Count);
                }
                while (!received.EndOfMessage);

                Take(frame.GetBuffer().AsSpan(0, (int)frame.Length));
            }
        }
        catch (WebSocketException)
        {
            // The page went without closing its socket.
        }
    }

    // One ask, taken as it is read: a Stop answered at once; any other made of the notebook now, and owned until done.
    private void Take(ReadOnlySpan<byte> frame)
    {
        PageAsks.Asked asked;

        try
        {
            asked = JsonSerializer.Deserialize<PageAsks.Asked>(frame, Json);
        }
        catch (JsonException unread)
        {
            _answers.Writer.TryWrite(new Refusal("refused", 0, StatusCodes.Status400BadRequest, $"The page sent something that is not an ask: {unread.Message}", null, null));

            return;
        }

        if (asked.Ask == "stop")
        {
            _answers.Writer.TryWrite(new Answer("answer", asked.Id, new StopAnswer(host.Stop(asked.Run ?? 0))));

            return;
        }

        // Counted before it is made: an ask that closes the notebook ends this page's view as it is made, and the writer
        // answers every ask it counts before it tells the page the notebook was closed.
        Interlocked.Increment(ref _outstanding);

        Task<object?> doing;

        try
        {
            doing = PageAsks.Make(asked, host, notebooks);
        }
        catch (Exception refused)
        {
            doing = Task.FromException<object?>(refused);
        }

        _ = OwnAsync(asked.Id, doing);
    }

    // An ask, until it is done: its answer, or why it was refused, for the writer — even when the page is gone by then.
    private async Task OwnAsync(long id, Task<object?> doing)
    {
        object told;

        try
        {
            told = new Answer("answer", id, await doing);
        }
        catch (CellGoneException gone)
        {
            told = new Refusal("refused", id, StatusCodes.Status409Conflict, GoneSays, gone.Cell, gone.Version);
        }
        catch (ArgumentException wrong)
        {
            told = new Refusal("refused", id, StatusCodes.Status400BadRequest, wrong.Message, null, null);
        }
        catch (ObjectDisposedException)
        {
            told = new Refusal("refused", id, StatusCodes.Status422UnprocessableEntity, "The notebook was closed.", null, null);
        }
        catch (InvalidOperationException refused)
        {
            told = new Refusal("refused", id, StatusCodes.Status422UnprocessableEntity, refused.Message, null, null);
        }
        catch (Exception failed)
        {
            told = new Refusal("refused", id, StatusCodes.Status500InternalServerError, failed.Message, null, null);
        }

        _answers.Writer.TryWrite(told);
        Interlocked.Decrement(ref _outstanding);
    }

    // The one writer: the notebook as it stands, then every change and every answer, a change before the answer to the
    // ask that made it — the notebook offers the change before the ask is done, so it waits by the time the answer does.
    // Once the notebook is closed and every ask made of it is answered, the page is told so.
    private async Task WriteAsync(NotebookSubscription view)
    {
        try
        {
            await SendAsync(new Snapshot("snapshot", view.Snapshot, host.Kinds, host.Layouts, host.Themes));

            var viewEnded = false;
            Task<bool>? viewing = null;
            Task<bool>? answering = null;

            while (true)
            {
                if (view.TryRead(out var change))
                {
                    await SendAsync(new Changed("change", change));

                    continue;
                }

                if (_answers.Reader.TryRead(out var answer))
                {
                    if (view.TryRead(out var before))
                    {
                        await SendAsync(new Changed("change", before));
                    }

                    await SendAsync(answer);

                    continue;
                }

                if (viewEnded && Volatile.Read(ref _outstanding) == 0)
                {
                    break;
                }

                answering ??= _answers.Reader.WaitToReadAsync(_gone.Token).AsTask();

                if (viewEnded)
                {
                    await answering;
                    answering = null;

                    continue;
                }

                viewing ??= view.WaitToReadAsync(_gone.Token).AsTask();

                if (await Task.WhenAny(viewing, answering) == viewing)
                {
                    viewEnded = !await viewing;
                    viewing = null;
                }
                else
                {
                    await answering;
                    answering = null;
                }
            }

            await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "the notebook was closed", CancellationToken.None);
        }
        catch (OperationCanceledException)
        {
            // The page went away: nothing more is told.
        }
        catch (WebSocketException)
        {
            // The page went away mid-word.
        }
    }

    private Task SendAsync(object message) =>
        socket.SendAsync(JsonSerializer.SerializeToUtf8Bytes(message, Json), WebSocketMessageType.Text, endOfMessage: true, CancellationToken.None);

    // Waits a moment for the page's own close, so a socket ends as sockets do.
    private static async Task ClosedAsync(WebSocket socket)
    {
        var buffer = new byte[256];

        try
        {
            using var patience = new CancellationTokenSource(TimeSpan.FromSeconds(5));

            while ((await socket.ReceiveAsync(buffer.AsMemory(), patience.Token)).MessageType != WebSocketMessageType.Close)
            {
                // Whatever the page still sends before its close is read and let go.
            }
        }
        catch (Exception gone) when (gone is OperationCanceledException or WebSocketException)
        {
            // The page went, or did not answer in time.
        }
    }

    // The notebook as the socket's first frame tells it, with what does not change while it is open: the kinds a cell can
    // be, and the layouts and themes it can be shown in, each theme with its block of custom properties.
    private readonly record struct Snapshot(
        string Type, NotebookVersion Version, IReadOnlyList<HostedKind> Kinds, IReadOnlyList<HostedLayoutChoice> Layouts, IReadOnlyList<HostedTheme> Themes);

    // A change, as the page is told it.
    private readonly record struct Changed(string Type, NotebookChange Change);

    // An ask's answer: what its verb gave back, or nothing.
    private readonly record struct Answer(string Type, long Id, object? Result);

    // Why an ask was refused, and the status HTTP names the same refusal with; a cell that went names the version it went at.
    private readonly record struct Refusal(string Type, long Id, int Status, string Message, Guid? Cell, long? Version);

    // A stop's answer: whether it stopped the run it named.
    private readonly record struct StopAnswer(bool Stopped);
}
