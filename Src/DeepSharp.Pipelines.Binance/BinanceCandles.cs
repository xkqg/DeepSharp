// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// What is asked of Binance: the candles of one symbol at one interval over a closed window.
/// </summary>
/// <remarks>
/// <para>
/// The window is closed, with a <c>from</c> and a <c>to</c> and never "the most recent thousand", which is a moving target
/// dressed as a source: it is half open, so the candle that opens at <c>from</c> is in it and the one that opens at
/// <c>to</c> is not, and two windows side by side tile without a candle in both. Both ends stand on the interval's own
/// grid, since the venue opens a candle only there, and a window that does not is refused rather than rounded: a start the
/// venue skips forward would stop naming what the landing holds.
/// </para>
/// <para>
/// Nothing is sent until the window is landed. A moment that says nothing of its zone is universal, as a pipeline's own
/// are, never the machine's; one that says it is local is the instant it stands for.
/// </para>
/// <para>
/// How it is sent is chosen beside the window and never part of what it names: <see cref="Through"/> the handler its
/// requests go through, <see cref="On"/> the clock every wait and every stamp runs on, <see cref="At"/> the address of the
/// venue. Each gives another window of the same candles, and none is needed by a caller who is content with the venue's own
/// public address and this machine's clock.
/// </para>
/// </remarks>
public sealed class BinanceCandles
{
    /// <summary>The most pages of a thousand candles that one landing asks for.</summary>
    /// <remarks>
    /// The venue's budget is counted for the address, and the owner's other programs may be spending it: a window of one
    /// second candles for a year is more than thirty thousand pages and hours of it, so a window that would take more than
    /// this is refused where it is written, naming how many pages it would take.
    /// </remarks>
    public const int MostPages = 2_000;

    /// <summary>How many candles one page of the venue's answer holds, at most.</summary>
    internal const int PageSize = 1_000;

    private static readonly Uri PublicMarketData = new("https://data-api.binance.vision");

    private readonly HttpMessageHandler? _handler;
    private readonly TimeProvider _clock;
    private readonly Uri _host;

    /// <summary>The candles of one symbol at one interval, from the moment a candle opens to the moment one opens that is not in it.</summary>
    /// <param name="symbol">The symbol Binance names the market by, such as <c>BTCEUR</c>; kept in capitals.</param>
    /// <param name="interval">How long a candle is, spelled as Binance spells it: <c>1s</c>, <c>1m</c>, <c>3m</c>, <c>5m</c>, <c>15m</c>, <c>30m</c>, <c>1h</c>, <c>2h</c>, <c>4h</c>, <c>6h</c>, <c>8h</c>, <c>12h</c>, <c>1d</c>, <c>3d</c>, <c>1w</c> or <c>1M</c>, a month — with its case, which tells it from a minute.</param>
    /// <param name="from">The first moment: a moment a candle of this interval opens at.</param>
    /// <param name="to">The moment the window ends before: another such moment, after <paramref name="from"/>.</param>
    /// <exception cref="ArgumentNullException">A symbol or an interval is missing.</exception>
    /// <exception cref="ArgumentException">
    /// The symbol is none Binance could name, the interval is none it documents, an end of the window is not a moment a
    /// candle opens at (the message names the two nearest), the window does not end after it begins, or it would take more
    /// than <see cref="MostPages"/> pages.
    /// </exception>
    public BinanceCandles(string symbol, string interval, DateTime from, DateTime to)
    {
        ArgumentNullException.ThrowIfNull(symbol);
        ArgumentNullException.ThrowIfNull(interval);

        Length = CandleInterval.Named(interval)
                 ?? throw new ArgumentException($"Binance documents no candle of '{interval}': the intervals are {CandleInterval.Listed}.", nameof(interval));

        if (!IsSymbol(symbol))
        {
            throw new ArgumentException($"'{symbol}' is not a symbol Binance names a market by: up to twenty letters, digits, dots, hyphens and underscores.", nameof(symbol));
        }

        Symbol = symbol.ToUpperInvariant();
        From = OnTheGrid(Universal(from), nameof(from));
        To = OnTheGrid(Universal(to), nameof(to));

        if (To <= From)
        {
            throw new ArgumentException($"The window must end after it begins: {Spelled(To)} is not after {Spelled(From)}.", nameof(to));
        }

        var candles = Length.Candles(From, To);

        PagesEstimated = (int)Math.Min(int.MaxValue, (candles + PageSize - 1) / PageSize);

        if (PagesEstimated > MostPages)
        {
            throw new ArgumentException(
                $"The window is {candles.ToString("N0", CultureInfo.InvariantCulture)} candles of {interval}, {PagesEstimated.ToString("N0", CultureInfo.InvariantCulture)} pages "
                + $"of {PageSize.ToString("N0", CultureInfo.InvariantCulture)}, and one landing asks for at most {MostPages.ToString("N0", CultureInfo.InvariantCulture)}: "
                + "land it in pieces, each ending where the next begins.",
                nameof(to));
        }

        _clock = TimeProvider.System;
        _host = PublicMarketData;
    }

    private BinanceCandles(BinanceCandles window, HttpMessageHandler? handler, TimeProvider clock, Uri host)
    {
        Symbol = window.Symbol;
        Length = window.Length;
        From = window.From;
        To = window.To;
        PagesEstimated = window.PagesEstimated;
        _handler = handler;
        _clock = clock;
        _host = host;
    }

    /// <summary>The symbol Binance names the market by, in capitals.</summary>
    public string Symbol { get; }

    /// <summary>How long a candle is, spelled as Binance spells it: a month is <c>1M</c>, a minute <c>1m</c>.</summary>
    public string Interval => Length.Venue;

    /// <summary>The first moment of the window: where a candle opens that is in it. Universal.</summary>
    public DateTime From { get; }

    /// <summary>The moment the window ends before: where a candle opens that is not in it. Universal.</summary>
    public DateTime To { get; }

    /// <summary>The interval, as the code that walks it knows it.</summary>
    internal CandleInterval Length { get; }

    /// <summary>The address the requests go to, as the record of a landing says it: the scheme and the host.</summary>
    internal string Host => _host.GetLeftPart(UriPartial.Authority);

    /// <summary>How many pages the window is expected to take.</summary>
    internal int PagesEstimated { get; }

    /// <summary>
    /// The name of the file this window lands in, which is what was asked: the symbol, the interval as a file spells it, and
    /// the two moments without a colon in them.
    /// </summary>
    internal string LandingName => $"{Stem}.csv";

    /// <summary>The name of the record beside it of what was asked and what came back.</summary>
    internal string ManifestName => $"{Stem}.manifest.json";

    private string Stem => $"{Symbol}-{Length.File}-{Stamp(From)}-{Stamp(To)}";

    /// <summary>The same window, its requests going through a handler of the caller's.</summary>
    /// <param name="handler">The handler: a proxy's, a recording one a test hands in. It is the caller's, and a landing never disposes it.</param>
    /// <returns>The window, asked that way.</returns>
    /// <exception cref="ArgumentNullException">No handler is handed in.</exception>
    public BinanceCandles Through(HttpMessageHandler handler)
    {
        ArgumentNullException.ThrowIfNull(handler);

        return new BinanceCandles(this, handler, _clock, _host);
    }

    /// <summary>The same window, every wait and every stamp of the landing running on a clock of the caller's.</summary>
    /// <param name="clock">The clock, such as the one a test moves on by hand.</param>
    /// <returns>The window, kept to that clock.</returns>
    /// <exception cref="ArgumentNullException">No clock is handed in.</exception>
    public BinanceCandles On(TimeProvider clock)
    {
        ArgumentNullException.ThrowIfNull(clock);

        return new BinanceCandles(this, _handler, clock, _host);
    }

    /// <summary>The same window, asked of another address than Binance's public market data.</summary>
    /// <param name="host">The address, with a scheme of <c>http</c> or <c>https</c> and nothing after the host: a stand-in on this machine, say.</param>
    /// <returns>The window, asked there.</returns>
    /// <exception cref="ArgumentNullException">No address is handed in.</exception>
    /// <exception cref="ArgumentException">The address is not an <c>http</c> or <c>https</c> address of a host alone.</exception>
    public BinanceCandles At(Uri host)
    {
        ArgumentNullException.ThrowIfNull(host);

        if (!host.IsAbsoluteUri || host.Scheme is not ("http" or "https") || host.AbsolutePath != "/" || host.Query.Length > 0 || host.Fragment.Length > 0 || host.UserInfo.Length > 0)
        {
            // Spelled without the credentials it may carry: a message is logged, and a secret does not belong in one.
            var spelled = host.IsAbsoluteUri ? host.GetComponents(UriComponents.SchemeAndServer | UriComponents.Path, UriFormat.Unescaped) : host.ToString();

            throw new ArgumentException($"'{spelled}' is not an address of a host alone: an http or https address with no path, no query and no credentials.", nameof(host));
        }

        return new BinanceCandles(this, _handler, _clock, host);
    }

    /// <summary>Lands the window in a folder as a file of its candles and a record of what was asked and what came back, once.</summary>
    /// <param name="folder">The folder the landing goes in: the one the pipeline that reads it reads relative paths from.</param>
    /// <param name="cancellation">Ends the landing — a request or a wait — before anything is written.</param>
    /// <returns>The name of the landing's file, relative to the folder: what a pipeline reads.</returns>
    /// <exception cref="ArgumentNullException">No folder is handed in.</exception>
    /// <exception cref="DirectoryNotFoundException">The folder is not there; nothing is asked of Binance until it is.</exception>
    /// <exception cref="InvalidOperationException">The window ends where Binance has not yet closed its last candle, or Binance holds no candle of it.</exception>
    /// <exception cref="BinanceException">Binance stopped the landing, asked it to wait longer than a landing waits, refused it in its own words, or could not be made to answer.</exception>
    /// <exception cref="InvalidDataException">What stands in the folder is not a landing of this window that can be reused — edited, torn, of a newer version or another window — or the venue now answers otherwise than a file that stands there says it did.</exception>
    /// <remarks>
    /// <para>
    /// The window is asked of the venue once. A landing that is there is reused, never replaced: its bytes are held to the
    /// fingerprint its record names, and the venue is not asked again. What is asked is paced and obeys the venue's own count
    /// of what this address has spent, and nothing is written until the whole window has been walked: a landing that was
    /// cancelled, or stopped, or refused, leaves nothing in the folder.
    /// </para>
    /// <para>
    /// A pipeline then reads the file as it reads any: <c>.ReadCsv(name)</c>. The file holds the venue's own numbers as it
    /// wrote them — the data is Binance's, under Binance's terms, which the record beside it points to.
    /// </para>
    /// </remarks>
    public Task<string> LandAsync(SourceFolder folder, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return new Landing(this, folder).RunAsync(cancellation);
    }

    /// <summary>Walks the window to its end and brings back the candles, without writing any of them.</summary>
    /// <param name="cancellation">Ends the walk, a wait or a request included.</param>
    /// <returns>What came back.</returns>
    internal async Task<Walked> WalkAsync(CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();

        using var venue = Venue.Open(_handler, _host, _clock);

        return await new Walk(this, venue, _clock).RunAsync(cancellation);
    }

    private static bool IsSymbol(string symbol) =>
        symbol.Length is > 0 and <= 20 && symbol.All(letter => letter is (>= 'A' and <= 'Z') or (>= 'a' and <= 'z') or (>= '0' and <= '9') or '.' or '-' or '_');

    private static DateTime Universal(DateTime moment) =>
        moment.Kind == DateTimeKind.Local ? moment.ToUniversalTime() : DateTime.SpecifyKind(moment, DateTimeKind.Utc);

    private DateTime OnTheGrid(DateTime moment, string end)
    {
        if (Length.Opens(moment))
        {
            return moment;
        }

        var near = Length.Around(moment);

        throw new ArgumentException(
            $"A candle of {Length.Venue} opens at {Length.Rule}, and {Spelled(moment)} is not such a moment; the nearest are {Spelled(near.Before)} and {Spelled(near.After)}.",
            end);
    }

    private static string Spelled(DateTime moment) => moment.ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture);

    private static string Stamp(DateTime moment) => moment.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture);
}
