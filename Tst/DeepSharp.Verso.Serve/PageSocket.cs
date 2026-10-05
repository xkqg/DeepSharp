// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;
using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// A notebook's socket, opened as the server's own page opens it — the cookie its first page set, the page named as a
/// browser names it: the notebook as it stands and the kinds a cell can be first, then each change as it comes, and an
/// answer to each ask by its id. Every frame is kept in the order it came, so a test can say which came first.
/// </summary>
internal sealed class PageSocket : IAsyncDisposable
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);

    private readonly ClientWebSocket _socket;
    private readonly Channel<JsonElement> _told = Channel.CreateUnbounded<JsonElement>();
    private readonly ConcurrentDictionary<long, TaskCompletionSource<Answered>> _asked = new();
    private readonly ConcurrentQueue<string> _frames = new();
    private readonly TaskCompletionSource _ended = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Task _reading;
    private long _next;

    private PageSocket(ClientWebSocket socket)
    {
        _socket = socket;
        _reading = ReadAsync();
    }

    /// <summary>Every frame the server sent, in the order it came: its type, and its id or version.</summary>
    public IReadOnlyCollection<string> Frames => _frames;

    /// <summary>Ends when the server has closed the socket.</summary>
    public Task Ended => _ended.Task;

    /// <summary>The status the server closed the socket with; nothing while it is open.</summary>
    public WebSocketCloseStatus? CloseStatus => _socket.CloseStatus;

    /// <summary>Why the server closed the socket, in its words.</summary>
    public string? CloseStatusDescription => _socket.CloseStatusDescription;

    /// <summary>Opens a notebook's socket.</summary>
    /// <param name="address">The server's address.</param>
    /// <param name="name">The notebook's name.</param>
    /// <param name="token">The server's token, as the cookie carries it.</param>
    /// <param name="origin">The page named; the server's own when none is given.</param>
    /// <returns>The socket.</returns>
    public static async Task<PageSocket> OpenAsync(Uri address, string name, string token, string? origin = null)
    {
        var socket = new ClientWebSocket();

        socket.Options.SetRequestHeader("Cookie", $"deepsharp-serve-{address.Port}={token}");
        socket.Options.SetRequestHeader("Origin", origin ?? address.GetLeftPart(UriPartial.Authority));
        socket.Options.CollectHttpResponseDetails = true;
        await socket.ConnectAsync(new Uri($"ws://{address.Authority}/api/notebooks/{name}/socket"), TestContext.Current.CancellationToken);

        return new PageSocket(socket);
    }

    /// <summary>
    /// Opens a notebook's socket as a program does: the token in the address the tool said, and no page named.
    /// </summary>
    /// <param name="said">The address the tool said, token and all.</param>
    /// <param name="name">The notebook's name.</param>
    /// <returns>The socket.</returns>
    public static async Task<PageSocket> ProgramAsync(Uri said, string name)
    {
        var socket = new ClientWebSocket();

        await socket.ConnectAsync(new Uri($"ws://{said.Authority}/api/notebooks/{name}/socket{said.Query}"), TestContext.Current.CancellationToken);

        return new PageSocket(socket);
    }

    /// <summary>What the server answers an upgrade it refuses.</summary>
    /// <param name="address">The server's address.</param>
    /// <param name="name">The notebook's name.</param>
    /// <param name="token">The server's token, as the cookie carries it.</param>
    /// <param name="origin">The page named.</param>
    /// <returns>The status the upgrade was answered with.</returns>
    public static async Task<HttpStatusCode> RefusedAsync(Uri address, string name, string token, string origin)
    {
        using var socket = new ClientWebSocket();

        socket.Options.SetRequestHeader("Cookie", $"deepsharp-serve-{address.Port}={token}");
        socket.Options.SetRequestHeader("Origin", origin);
        socket.Options.CollectHttpResponseDetails = true;

        await Assert.ThrowsAsync<WebSocketException>(() => socket.ConnectAsync(new Uri($"ws://{address.Authority}/api/notebooks/{name}/socket"), TestContext.Current.CancellationToken));

        return socket.HttpStatusCode;
    }

    /// <summary>The next thing the server told: the notebook as it stands first, then each change.</summary>
    /// <returns>The frame.</returns>
    public async Task<JsonElement> NextAsync()
    {
        using var patience = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        patience.CancelAfter(Patience);

        return await _told.Reader.ReadAsync(patience.Token);
    }

    /// <summary>The notebook as it stood when the socket opened: its first frame.</summary>
    /// <returns>The notebook's version, and the kinds a cell can be.</returns>
    public async Task<Snapshot> SnapshotAsync()
    {
        var first = await NextAsync();

        Assert.Equal("snapshot", first.GetProperty("type").GetString());

        return new Snapshot(Read<NotebookVersion>(first.GetProperty("version")), Read<HostedKind[]>(first.GetProperty("kinds")));
    }

    /// <summary>The next change the server told.</summary>
    /// <returns>The change.</returns>
    public async Task<NotebookChange> ChangeAsync()
    {
        var next = await NextAsync();

        Assert.Equal("change", next.GetProperty("type").GetString());

        return Read<NotebookChange>(next.GetProperty("change"));
    }

    /// <summary>Asks something of the notebook, as the page asks it.</summary>
    /// <param name="ask">What is asked; nothing for a frame that names no ask.</param>
    /// <param name="fields">What it names.</param>
    /// <returns>Its answer, or why it was refused.</returns>
    public async Task<Answered> AskAsync(string? ask, object? fields = null)
    {
        var id = Interlocked.Increment(ref _next);
        var answered = new TaskCompletionSource<Answered>(TaskCreationOptions.RunContinuationsAsynchronously);

        _asked[id] = answered;
        await SendAsync(Message(id, ask, fields));

        return await answered.Task.WaitAsync(Patience, TestContext.Current.CancellationToken);
    }

    /// <summary>Sends an ask without waiting for its answer.</summary>
    /// <param name="ask">What is asked.</param>
    /// <param name="fields">What it names.</param>
    /// <returns>Its answer, when it comes.</returns>
    public async Task<Task<Answered>> SendAsync(string ask, object? fields = null)
    {
        var id = Interlocked.Increment(ref _next);
        var answered = new TaskCompletionSource<Answered>(TaskCreationOptions.RunContinuationsAsynchronously);

        _asked[id] = answered;
        await SendAsync(Message(id, ask, fields));

        return answered.Task;
    }

    /// <summary>Sends a frame as it is, whatever it holds.</summary>
    /// <param name="text">The frame.</param>
    /// <returns>When it is sent.</returns>
    public Task SendAsync(string text) =>
        _socket.SendAsync(Encoding.UTF8.GetBytes(text), WebSocketMessageType.Text, endOfMessage: true, TestContext.Current.CancellationToken);

    /// <summary>The next answer to a frame that named no id the page can know: the server's word on a frame it could not read.</summary>
    /// <returns>The answer.</returns>
    public Task<Answered> UnnamedAsync()
    {
        var answered = _asked.GetOrAdd(0, _ => new TaskCompletionSource<Answered>(TaskCreationOptions.RunContinuationsAsynchronously));

        return answered.Task.WaitAsync(Patience, TestContext.Current.CancellationToken);
    }

    /// <summary>Leaves, as a page does: a close frame, and the server's own in answer.</summary>
    /// <returns>When it is closed, and the read of it has ended.</returns>
    /// <remarks>
    /// The socket is let go only once the read of it has ended. A read that went on past that would fail on a socket that is
    /// gone, and since nothing waits for it, the engine would tell that failure to whichever run is working when the task
    /// is collected — a run in another test.
    /// </remarks>
    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_socket.State == WebSocketState.Open)
            {
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "the page left", CancellationToken.None).WaitAsync(Patience, CancellationToken.None);
            }
        }
        catch (Exception done) when (done is WebSocketException or OperationCanceledException or TimeoutException)
        {
            // The server went first, or never answered: the socket is let go as it is, and the read with it.
            _socket.Abort();
        }

        try
        {
            await _reading.WaitAsync(Patience, CancellationToken.None);
        }
        finally
        {
            _socket.Dispose();
        }
    }

    private static T Read<T>(JsonElement element) => element.Deserialize<T>(Json)!;

    private static string Message(long id, string? ask, object? fields)
    {
        var message = fields is null ? new JsonObject() : JsonSerializer.SerializeToNode(fields, Json)!.AsObject();

        message["id"] = id;

        if (ask is not null)
        {
            message["ask"] = ask;
        }

        return message.ToJsonString();
    }

    private async Task ReadAsync()
    {
        var buffer = new byte[64 * 1024];

        try
        {
            while (true)
            {
                using var frame = new MemoryStream();
                WebSocketReceiveResult received;

                do
                {
                    received = await _socket.ReceiveAsync(buffer, CancellationToken.None);

                    // The server closed its side: the page closes its own, as a browser does.
                    if (received.MessageType == WebSocketMessageType.Close)
                    {
                        await _socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, null, CancellationToken.None);

                        return;
                    }

                    frame.Write(buffer, 0, received.Count);
                }
                while (!received.EndOfMessage);

                var told = JsonDocument.Parse(frame.ToArray()).RootElement.Clone();
                var type = told.GetProperty("type").GetString();

                switch (type)
                {
                    case "answer" or "refused":
                        var id = told.GetProperty("id").GetInt64();

                        _frames.Enqueue($"{type} {id}");
                        _asked.GetOrAdd(id, _ => new TaskCompletionSource<Answered>(TaskCreationOptions.RunContinuationsAsynchronously))
                            .TrySetResult(new Answered(type == "refused", told));
                        break;

                    default:
                        _frames.Enqueue(type == "change" ? $"change {told.GetProperty("change").GetProperty("version").GetInt64()}" : type!);
                        await _told.Writer.WriteAsync(told);
                        break;
                }
            }
        }
        catch (Exception done) when (done is WebSocketException or OperationCanceledException)
        {
            // The server went without a close frame, or the socket was let go as it was while the read waited.
        }
        finally
        {
            _told.Writer.TryComplete();
            _ended.TrySetResult();
        }
    }

    /// <summary>The notebook as a socket's first frame tells it.</summary>
    /// <param name="Version">The notebook as it stood.</param>
    /// <param name="Kinds">The kinds a cell can be.</param>
    public readonly record struct Snapshot(NotebookVersion Version, IReadOnlyList<HostedKind> Kinds);

    /// <summary>What the server answered an ask: its result, or why it refused it.</summary>
    /// <param name="Refused">Whether it was refused.</param>
    /// <param name="Frame">The frame it came in.</param>
    public readonly record struct Answered(bool Refused, JsonElement Frame)
    {
        /// <summary>The status a refusal names, as HTTP names the same refusal.</summary>
        public int Status => Frame.GetProperty("status").GetInt32();

        /// <summary>Why it was refused.</summary>
        public string? Why => Frame.GetProperty("message").GetString();

        /// <summary>The result, read as what it is.</summary>
        /// <typeparam name="T">What it is.</typeparam>
        /// <returns>It.</returns>
        public T Result<T>()
        {
            Assert.False(Refused, Refused ? Why : null);

            return Frame.GetProperty("result").Deserialize<T>(Json)!;
        }

        /// <summary>Whether the result is nothing: done, with nothing to hand back.</summary>
        public bool IsNothing => !Refused && Frame.GetProperty("result").ValueKind == JsonValueKind.Null;
    }
}
