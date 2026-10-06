// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// One page of candles that is to be asked for: whose, at what interval, from where the window begins to where it ends, and
/// where the walk has come to.
/// </summary>
/// <remarks>
/// It is also what says whether a page that came is one to take, which is judged where the page is read, inside the unit
/// that asked, so a page that is not one is asked for again before the cursor has moved. Every candle must open inside the
/// window that the venue was asked about, on the interval's grid, and after the one before it — across pages too. The venue
/// is trusted to answer what it is asked and not to keep a unit or an order it has changed to itself: a candle in
/// microseconds, a window it has moved, a page out of order are all refused, and none is landed.
/// </remarks>
/// <param name="Symbol">The market.</param>
/// <param name="Interval">The candle length.</param>
/// <param name="From">The first moment of the window.</param>
/// <param name="To">The moment the window ends before.</param>
/// <param name="Cursor">Where this page starts, in milliseconds since the epoch.</param>
/// <param name="Previous">When the last candle taken opened, in milliseconds since the epoch; nothing before the first page.</param>
internal readonly record struct PageAsk(string Symbol, CandleInterval Interval, DateTime From, DateTime To, long Cursor, long? Previous)
{
    /// <summary>The last moment a candle may open at: a millisecond before the window ends, for the venue's end is inclusive.</summary>
    public long End => new DateTimeOffset(To).ToUnixTimeMilliseconds() - 1;

    /// <summary>Judges the candles of a page against what was asked.</summary>
    /// <param name="candles">The candles the page holds.</param>
    /// <exception cref="PageFormatException">One is outside the window, off the grid, or not after the one before it.</exception>
    public void Check(IReadOnlyList<Candle> candles)
    {
        var start = new DateTimeOffset(From).ToUnixTimeMilliseconds();
        var before = Previous;

        foreach (var candle in candles)
        {
            if (candle.OpenTime < start || candle.OpenTime > End)
            {
                throw new PageFormatException(
                    $"the candle that opens at {candle.OpenTime.ToString(CultureInfo.InvariantCulture)} milliseconds is outside the window {Spell(From)} to {Spell(To)} that Binance was asked about");
            }

            var opens = DateTimeOffset.FromUnixTimeMilliseconds(candle.OpenTime).UtcDateTime;

            if (before is { } last && candle.OpenTime <= last)
            {
                throw new PageFormatException($"candles are not in order: the one that opens at {Spell(opens)} follows the one that opens at {Spell(DateTimeOffset.FromUnixTimeMilliseconds(last).UtcDateTime)}");
            }

            if (!Interval.Opens(opens))
            {
                throw new PageFormatException($"a candle of {Interval.Venue} does not open at {Spell(opens)}");
            }

            before = candle.OpenTime;
        }
    }

    private static string Spell(DateTime moment) => moment.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);
}

/// <summary>A page that was taken: its candles, and what the venue said of the answer that held them.</summary>
/// <param name="Candles">The candles, in order.</param>
/// <param name="Sha256">The SHA-256 of the answer's body, as lower-case hexadecimal.</param>
/// <param name="Date">The venue's <c>Date</c> header as it was written, when it came with one.</param>
/// <param name="Uuid">The venue's <c>x-mbx-uuid</c>, when it came with one.</param>
internal sealed record Page(IReadOnlyList<Candle> Candles, string Sha256, string? Date, string? Uuid)
{
    /// <summary>Takes a page from an answer, judged against what was asked.</summary>
    /// <param name="reply">The answer.</param>
    /// <param name="ask">What was asked.</param>
    /// <returns>The page.</returns>
    /// <exception cref="PageFormatException">The answer is not a page that can be taken.</exception>
    public static Page Of(VenueReply reply, PageAsk ask)
    {
        var candles = VenueWire.Candles(reply);

        ask.Check(candles);

        return new Page(candles, reply.Body.Fingerprint(), reply.Date, reply.Uuid);
    }
}
