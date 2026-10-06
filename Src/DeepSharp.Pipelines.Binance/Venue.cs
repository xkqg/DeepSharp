// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// The venue, for the length of one landing: a client, the pace it is held to and the courier that carries each unit of work.
/// </summary>
/// <remarks>
/// <para>
/// Made when a landing begins and disposed when it ends, nothing static and nothing shared with another landing: a landing
/// that ended, or was cancelled, or failed, leaves nothing of this running, since the pace has no thread and no timer and
/// the client is let go. A handler a caller handed in is theirs, and is kept; one this made itself is let go with the client.
/// </para>
/// <para>
/// The client waits for nothing on its own: its timeout is infinite, since the time one try may take is the courier's to
/// say, on the clock a landing is handed.
/// </para>
/// </remarks>
internal sealed class Venue : IDisposable
{
    private readonly HttpClient _client;
    private readonly Pacer _pace;
    private readonly Courier _courier;

    private Venue(HttpClient client, Pacer pace, Courier courier)
    {
        _client = client;
        _pace = pace;
        _courier = courier;
    }

    /// <summary>Opens the venue for a landing.</summary>
    /// <param name="handed">The handler requests go through, when a caller handed one in: kept, never disposed here.</param>
    /// <param name="host">The venue's address.</param>
    /// <param name="clock">The clock every wait and every timeout runs on.</param>
    /// <returns>The venue.</returns>
    public static Venue Open(HttpMessageHandler? handed, Uri host, TimeProvider clock)
    {
        var client = handed is null
            ? new HttpClient(new SocketsHttpHandler { PooledConnectionLifetime = TimeSpan.FromMinutes(2) }, disposeHandler: true)
            : new HttpClient(handed, disposeHandler: false);

        client.BaseAddress = host;
        client.Timeout = Timeout.InfiniteTimeSpan;

        var pace = new Pacer(clock);

        return new Venue(client, pace, new Courier(new VenueWire(client), pace, clock));
    }

    /// <summary>What the venue's own clock says.</summary>
    /// <param name="cancellation">Ends the request.</param>
    /// <returns>The moment, universal.</returns>
    public Task<DateTime> ServerTimeAsync(CancellationToken cancellation) =>
        _courier.AskAsync(VenueWire.Time, "the venue's clock", VenueWire.ServerTime, cancellation);

    /// <summary>Asks about a symbol: the venue refuses one it does not list, in its own words; and what an address may spend a minute is taken from its answer.</summary>
    /// <param name="symbol">The symbol.</param>
    /// <param name="cancellation">Ends the request.</param>
    /// <returns>When the venue has said what it lists.</returns>
    public async Task KnowAsync(string symbol, CancellationToken cancellation)
    {
        var allowed = await _courier.AskAsync(VenueWire.Information(symbol), "what the venue says of a symbol", VenueWire.WeightLimit, cancellation);

        if (allowed is { } limit)
        {
            _pace.Allowing(limit);
        }
    }

    /// <summary>One page of candles, judged against what was asked.</summary>
    /// <param name="ask">What is asked, and where the walk has come to.</param>
    /// <param name="cancellation">Ends the request, and any wait before it.</param>
    /// <returns>The page.</returns>
    public Task<Page> PageAsync(PageAsk ask, CancellationToken cancellation) =>
        _courier.AskAsync(VenueWire.Klines(ask.Symbol, ask.Interval.Venue, ask.Cursor, ask.End), "a page of candles", reply => Page.Of(reply, ask), cancellation);

    /// <inheritdoc />
    public void Dispose() => _client.Dispose();
}
