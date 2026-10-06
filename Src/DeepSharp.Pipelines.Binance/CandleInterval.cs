// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// The moments either side of one that stands on no interval's grid: where the candle it falls in opens, and where the next
/// one does.
/// </summary>
/// <param name="Before">The last moment a candle opens at or before it.</param>
/// <param name="After">The first moment a candle opens after it.</param>
internal readonly record struct Nearest(DateTime Before, DateTime After);

/// <summary>
/// A candle length the venue documents: how the venue spells it, how a file name spells it, and the moments at which a candle
/// of it opens.
/// </summary>
/// <remarks>
/// The venue opens a candle of a fixed length at a multiple of that length counted from the epoch — which is how a
/// three-day candle comes to open on every third day since 1970 — a week candle on a Monday and a month candle on the
/// first, always at midnight universal time. A window whose ends are not such moments is refused rather than rounded, since
/// the venue skips a start forward to the next candle and a rounded window would stop naming what it holds. The month is
/// spelled <c>1M</c> to the venue, which is a minute's <c>1m</c> in a file system that does not tell the cases apart, so
/// a file names it <c>1mo</c>, as the venue's own archive does.
/// </remarks>
internal sealed class CandleInterval
{
    private enum Opening { Multiple, Monday, FirstOfMonth }

    private readonly Opening _opening;
    private readonly TimeSpan _length;

    private CandleInterval(string venue, string file, Opening opening, TimeSpan length)
    {
        Venue = venue;
        File = file;
        _opening = opening;
        _length = length;
    }

    /// <summary>Every interval the venue documents, from the shortest to the longest.</summary>
    public static IReadOnlyList<CandleInterval> All { get; } =
    [
        Fixed("1s", TimeSpan.FromSeconds(1)),
        Fixed("1m", TimeSpan.FromMinutes(1)),
        Fixed("3m", TimeSpan.FromMinutes(3)),
        Fixed("5m", TimeSpan.FromMinutes(5)),
        Fixed("15m", TimeSpan.FromMinutes(15)),
        Fixed("30m", TimeSpan.FromMinutes(30)),
        Fixed("1h", TimeSpan.FromHours(1)),
        Fixed("2h", TimeSpan.FromHours(2)),
        Fixed("4h", TimeSpan.FromHours(4)),
        Fixed("6h", TimeSpan.FromHours(6)),
        Fixed("8h", TimeSpan.FromHours(8)),
        Fixed("12h", TimeSpan.FromHours(12)),
        Fixed("1d", TimeSpan.FromDays(1)),
        Fixed("3d", TimeSpan.FromDays(3)),
        new("1w", "1w", Opening.Monday, TimeSpan.FromDays(7)),
        new("1M", "1mo", Opening.FirstOfMonth, TimeSpan.FromDays(30)),
    ];

    /// <summary>The intervals, as the message of a refusal lists them.</summary>
    public static string Listed { get; } = string.Join(", ", All.Select(interval => interval.Venue));

    /// <summary>How the venue spells it, and what a request carries.</summary>
    public string Venue { get; }

    /// <summary>How a file name spells it: never two spellings that differ by case alone.</summary>
    public string File { get; }

    /// <summary>Where a candle of this interval opens, in words.</summary>
    public string Rule => _opening switch
    {
        Opening.Monday => "midnight on a Monday",
        Opening.FirstOfMonth => "midnight on the first of a month",
        _ when _length == TimeSpan.FromDays(3) => "midnight on every third day counted from the first of January 1970",
        _ when _length == TimeSpan.FromDays(1) => "midnight",
        _ => "a multiple of its length counted from midnight",
    };

    /// <summary>The interval the venue spells so.</summary>
    /// <param name="venue">The spelling, with its case.</param>
    /// <returns>The interval, or nothing when the venue documents none by that spelling.</returns>
    public static CandleInterval? Named(string venue) => All.FirstOrDefault(interval => string.Equals(interval.Venue, venue, StringComparison.Ordinal));

    private static CandleInterval Fixed(string venue, TimeSpan length) => new(venue, venue, Opening.Multiple, length);

    /// <summary>Whether a candle of this interval opens at the moment.</summary>
    /// <param name="moment">A universal moment.</param>
    /// <returns><see langword="true"/> when it stands on this interval's grid.</returns>
    public bool Opens(DateTime moment) => _opening switch
    {
        Opening.Multiple => (moment - DateTime.UnixEpoch).Ticks % _length.Ticks == 0,
        Opening.Monday => moment.DayOfWeek == DayOfWeek.Monday && moment.TimeOfDay == TimeSpan.Zero,
        _ => moment.Day == 1 && moment.TimeOfDay == TimeSpan.Zero,
    };

    /// <summary>Where the candle a moment falls in opens, and where the next one does.</summary>
    /// <param name="moment">A universal moment.</param>
    /// <returns>The two moments either side of it, or the moment itself and the one after it when it stands on the grid.</returns>
    public Nearest Around(DateTime moment)
    {
        switch (_opening)
        {
            case Opening.Multiple:
                var ticks = (moment - DateTime.UnixEpoch).Ticks;
                var before = DateTime.UnixEpoch.AddTicks(ticks - (((ticks % _length.Ticks) + _length.Ticks) % _length.Ticks));

                return new Nearest(before, before + _length);
            case Opening.Monday:
                var monday = moment.Date.AddDays(-(((int)moment.DayOfWeek + 6) % 7));

                return new Nearest(monday, monday.AddDays(7));
            default:
                var first = new DateTime(moment.Year, moment.Month, 1, 0, 0, 0, DateTimeKind.Utc);

                return new Nearest(first, first.AddMonths(1));
        }
    }

    /// <summary>Where the candle after the one that opens at a moment opens.</summary>
    /// <param name="open">A moment a candle of this interval opens at.</param>
    /// <returns>The next such moment.</returns>
    public DateTime After(DateTime open) => _opening switch
    {
        Opening.Multiple => open + _length,
        Opening.Monday => open.AddDays(7),
        _ => open.AddMonths(1),
    };

    /// <summary>How many candles of this interval a window holds.</summary>
    /// <param name="from">The first moment, on the grid.</param>
    /// <param name="to">The moment the window ends before, on the grid.</param>
    /// <returns>The candles that open at or after <paramref name="from"/> and before <paramref name="to"/>.</returns>
    public long Candles(DateTime from, DateTime to) => _opening switch
    {
        Opening.Multiple => (to - from).Ticks / _length.Ticks,
        Opening.Monday => (to - from).Days / 7,
        _ => ((to.Year - from.Year) * 12L) + to.Month - from.Month,
    };
}
