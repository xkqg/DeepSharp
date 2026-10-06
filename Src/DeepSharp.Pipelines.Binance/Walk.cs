// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>What a walk down a window brought back, before any of it is written.</summary>
/// <remarks>
/// A value of this type is a window walked to its end: a walk that stopped for any reason throws and returns nothing, so
/// there is no half of one to mistake for the whole, and what is written from it is never a marker beside work that was not
/// finished.
/// </remarks>
/// <param name="Candles">Every candle of the window the venue holds, in order, as it wrote them.</param>
/// <param name="Pages">Each page, as it came.</param>
/// <param name="AskedAt">When the walk began, by the clock it was handed.</param>
/// <param name="ServerTime">What the venue's own clock said, asked before anything else.</param>
/// <param name="PagesEstimated">How many pages the window was expected to take.</param>
internal sealed record Walked(IReadOnlyList<Candle> Candles, IReadOnlyList<PageReceipt> Pages, DateTime AskedAt, DateTime ServerTime, int PagesEstimated);

/// <summary>
/// The walk down a window: the venue's clock first, then what it says of the symbol, then the pages one at a time — each
/// from a millisecond after the last candle of the page before it — until the window is told.
/// </summary>
/// <remarks>
/// <para>
/// The window is half open and the venue's end is inclusive, so what is asked for ends a millisecond before the window does.
/// A window ends where the venue has closed its last candle at least a minute ago, by the venue's clock and not this one,
/// or it is refused before anything is spent: a candle that is still open changes with every request and would land as
/// whatever it was at the moment.
/// </para>
/// <para>
/// The cursor moves only past a page that was taken, which is a page the venue answered, in the shape it documents, inside
/// the window and in order; a page that is not one is asked for again, and after every try refused. Nothing the venue did not
/// hold is put in its place: a candle it lacks is a gap, counted in the manifest and not in the file.
/// </para>
/// </remarks>
/// <param name="ask">What is asked.</param>
/// <param name="venue">The venue, open for this landing.</param>
/// <param name="clock">The clock a landing is dated by.</param>
internal sealed class Walk(BinanceCandles ask, Venue venue, TimeProvider clock)
{
    /// <summary>Walks the window to its end.</summary>
    /// <param name="cancellation">Ends the walk, a wait or a request included.</param>
    /// <returns>What came back.</returns>
    /// <exception cref="InvalidOperationException">The window ends where the venue has not closed its last candle, or the venue holds no candle of it.</exception>
    /// <exception cref="BinanceException">The venue stopped the walk, or refused it in its own words.</exception>
    public async Task<Walked> RunAsync(CancellationToken cancellation)
    {
        var began = clock.GetUtcNow().UtcDateTime;
        var server = await venue.ServerTimeAsync(cancellation);

        RequireClosed(server);

        await venue.KnowAsync(ask.Symbol, cancellation);

        var candles = new List<Candle>();
        var receipts = new List<PageReceipt>();
        var cursor = new DateTimeOffset(ask.From).ToUnixTimeMilliseconds();
        long? previous = null;

        while (true)
        {
            var page = await venue.PageAsync(new PageAsk(ask.Symbol, ask.Length, ask.From, ask.To, cursor, previous), cancellation);

            if (page.Candles.Count == 0)
            {
                break;
            }

            receipts.Add(new PageReceipt(DateTimeOffset.FromUnixTimeMilliseconds(cursor).UtcDateTime, page.Candles.Count, page.Sha256, page.Date, page.Uuid));
            candles.AddRange(page.Candles);

            previous = page.Candles[^1].OpenTime;
            cursor = previous.Value + 1;

            // A short page is the end of what the venue holds; a full one ends the window when no candle can open before it does.
            if (page.Candles.Count < BinanceCandles.PageSize || ask.Length.After(DateTimeOffset.FromUnixTimeMilliseconds(previous.Value).UtcDateTime) >= ask.To)
            {
                break;
            }
        }

        if (candles.Count == 0)
        {
            throw new InvalidOperationException(
                $"Binance holds no candle of {ask.Symbol} at {ask.Interval} between {Spelled(ask.From)} and {Spelled(ask.To)}: the market may not have been listed yet, or the window may be one the venue has no history of.");
        }

        return new Walked(candles, receipts, began, server, ask.PagesEstimated);
    }

    // The last candle of the window closes where the window ends, and it is taken a minute later, by the venue's clock.
    private void RequireClosed(DateTime server)
    {
        var latest = server - LandingPace.SettleAfterClose;

        if (ask.To > latest)
        {
            throw new InvalidOperationException(
                $"The window ends at {Spelled(ask.To)}, and Binance's clock says it is {Spelled(server)}: a candle is taken {LandingPace.Said(LandingPace.SettleAfterClose)} after it closes, "
                + $"so the latest window of {ask.Interval} that can be landed now ends at {Spelled(ask.Length.Around(latest).Before)}.");
        }
    }

    private static string Spelled(DateTime moment) => moment.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);
}
