// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;

namespace DeepSharp.Tests.Pipelines;

/// <summary>An answer the stand-in venue was told to give, and the path it is for: any, when none is named.</summary>
/// <param name="Path">The path the answer is for.</param>
/// <param name="Answer">The answer.</param>
internal readonly record struct Scripted(string? Path, Func<CancellationToken, Task<HttpResponseMessage>> Answer);

/// <summary>One request the stand-in venue was sent.</summary>
/// <param name="At">When it came, by the venue's clock before any skew.</param>
/// <param name="Path">The path asked for.</param>
/// <param name="Query">What the query said, each key once.</param>
/// <param name="UserAgent">What the request called itself, when it said.</param>
internal readonly record struct VenueRequest(DateTime At, string Path, IReadOnlyDictionary<string, string> Query, string? UserAgent);

/// <summary>
/// A stand-in for Binance's public market-data host, written from what the venue's documentation says and what was measured
/// of it, so that what a landing does can be tested without a connection and without ever being a guest of the real one.
/// </summary>
/// <remarks>
/// <para>
/// What it answers is generated from a seed, never captured: the same ask gives the same bytes whichever way the window is
/// paged, another seed gives other numbers for a candle that opens at the same moment, and none of it was ever Binance's. What
/// it does as the venue does, each held by a test of its own: the end of a window is inclusive of a candle that opens at that
/// moment, a start between two candles skips forward to the next, at most a thousand candles come in an answer, a candle still
/// open is in the answer, a request costs weight that is counted by the venue's own minute and told in <c>x-mbx-used-weight-1m</c>,
/// the venue's clock is its own, and a refusal is written in the venue's words.
/// </para>
/// <para>
/// It is a message handler and nothing more, written against the framework alone, so a test hands it to a client and a
/// program that stands in front of it with a listener answers over a socket in the same words.
/// </para>
/// </remarks>
/// <param name="clock">The venue's clock, which a test advances.</param>
internal sealed class FakeBinanceVenue(TimeProvider clock) : HttpMessageHandler
{
    private const string UsedWeight = "x-mbx-used-weight-1m";

    private static readonly string[] Symbols = ["BTCEUR", "ETHEUR", "BTCUSDT", "ETHUSDT"];

    private static readonly Dictionary<string, long> FixedSpans = new(StringComparer.Ordinal)
    {
        ["1s"] = 1_000, ["1m"] = 60_000, ["3m"] = 180_000, ["5m"] = 300_000, ["15m"] = 900_000, ["30m"] = 1_800_000,
        ["1h"] = 3_600_000, ["2h"] = 7_200_000, ["4h"] = 14_400_000, ["6h"] = 21_600_000, ["8h"] = 28_800_000, ["12h"] = 43_200_000,
        ["1d"] = 86_400_000, ["3d"] = 259_200_000,
    };

    private readonly object _gate = new();
    private readonly List<VenueRequest> _seen = [];
    private readonly List<Scripted> _scripted = [];
    private readonly HashSet<long> _holes = [];
    private long _minute = long.MinValue;
    private int _weight;
    private int _requests;

    /// <summary>What the venue's numbers are generated from: another seed is the venue answering other numbers for the same candle.</summary>
    public int Seed { get; set; } = 7;

    /// <summary>When the symbols' history begins: nothing is answered from before it.</summary>
    public DateTime ListedAt { get; set; } = new(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    /// <summary>How far the venue's own clock stands from the clock handed in.</summary>
    public TimeSpan ClockSkew { get; set; }

    /// <summary>The weight the venue lets an address spend in a minute, as its exchange information says.</summary>
    public int WeightLimit { get; set; } = 6000;

    /// <summary>Every request the venue was sent, in the order they came.</summary>
    public IReadOnlyList<VenueRequest> Seen
    {
        get
        {
            lock (_gate)
            {
                return [.. _seen];
            }
        }
    }

    /// <summary>Candles the venue does not hold, each by the moment it would have opened.</summary>
    /// <param name="opens">The moments.</param>
    public void HoleAt(params DateTime[] opens)
    {
        lock (_gate)
        {
            foreach (var open in opens)
            {
                _holes.Add(new DateTimeOffset(DateTime.SpecifyKind(open, DateTimeKind.Utc)).ToUnixTimeMilliseconds());
            }
        }
    }

    /// <summary>The next requests are answered with a refusal, whatever they ask for.</summary>
    /// <param name="status">The status.</param>
    /// <param name="times">How many requests.</param>
    /// <param name="retryAfter">What the answer says to wait, when it says.</param>
    /// <param name="body">What the body says; the venue's own words for a refused request when none is given.</param>
    /// <param name="path">The path the refusal is for; any when none is given.</param>
    public void Fail(HttpStatusCode status, int times = 1, TimeSpan? retryAfter = null, string? body = null, string? path = null)
    {
        for (var each = 0; each < times; each++)
        {
            Script(path, _ =>
            {
                var response = new HttpResponseMessage(status)
                {
                    Content = new StringContent(body ?? "{\"code\":-1003,\"msg\":\"Too many requests; current limit of IP(s) is 6000 requests per minute.\"}", Encoding.UTF8, "application/json"),
                };

                if (retryAfter is { } wait)
                {
                    response.Headers.RetryAfter = new RetryConditionHeaderValue(wait);
                }

                return Task.FromResult(response);
            });
        }
    }

    /// <summary>Weight somebody else spent from the same address, in the venue's minute it is now.</summary>
    /// <param name="weight">How much.</param>
    public void Spend(int weight)
    {
        lock (_gate)
        {
            Tick(clock.GetUtcNow().UtcDateTime + ClockSkew);
            _weight += weight;
        }
    }

    /// <summary>The next requests are answered with this body and nothing wrong with the answer but what it holds.</summary>
    /// <param name="body">What the body says.</param>
    /// <param name="times">How many requests.</param>
    /// <param name="path">The path the body is for; any when none is given.</param>
    public void Respond(string body, int times = 1, string? path = null)
    {
        for (var each = 0; each < times; each++)
        {
            Script(path, _ => Task.FromResult(Json(HttpStatusCode.OK, body)));
        }
    }

    /// <summary>The next requests are never answered, until the caller gives up.</summary>
    /// <param name="times">How many requests.</param>
    /// <param name="path">The path the stall is for; any when none is given.</param>
    public void Stall(int times = 1, string? path = null)
    {
        for (var each = 0; each < times; each++)
        {
            Script(path, async cancellation =>
            {
                await Task.Delay(Timeout.Infinite, cancellation);

                return new HttpResponseMessage(HttpStatusCode.OK);
            });
        }
    }

    /// <summary>The next requests are answered with a page that is not JSON.</summary>
    /// <param name="times">How many requests.</param>
    /// <param name="path">The path the page is for; any when none is given.</param>
    public void Garble(int times = 1, string? path = null)
    {
        for (var each = 0; each < times; each++)
        {
            Script(path, _ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent("<html>not what the venue answers with</html>", Encoding.UTF8, "text/html") }));
        }
    }

    private void Script(string? path, Func<CancellationToken, Task<HttpResponseMessage>> answer)
    {
        lock (_gate)
        {
            _scripted.Add(new Scripted(path, answer));
        }
    }

    /// <inheritdoc />
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();

        var uri = request.RequestUri ?? throw new InvalidOperationException("The venue was sent a request with no address.");
        var query = QueryOf(uri);
        var now = clock.GetUtcNow().UtcDateTime;
        var venueNow = now + ClockSkew;
        Func<CancellationToken, Task<HttpResponseMessage>>? scripted = null;
        int weight;
        Guid identifier;

        lock (_gate)
        {
            _seen.Add(new VenueRequest(now, uri.AbsolutePath, query, request.Headers.UserAgent.ToString() is { Length: > 0 } agent ? agent : null));

            Tick(venueNow);

            _weight += WeightOf(uri.AbsolutePath, query);
            weight = _weight;
            identifier = new Guid(Seed, 0, 0, [0, 0, 0, 0, 0, 0, 0, (byte)++_requests]);

            var at = _scripted.FindIndex(each => each.Path is null || each.Path == uri.AbsolutePath);

            if (at >= 0)
            {
                scripted = _scripted[at].Answer;
                _scripted.RemoveAt(at);
            }
        }

        var response = scripted is not null ? await scripted(cancellation) : Answer(uri.AbsolutePath, query, venueNow);

        response.Headers.Date = new DateTimeOffset(venueNow, TimeSpan.Zero);
        response.Headers.TryAddWithoutValidation("x-mbx-uuid", identifier.ToString());
        response.Headers.TryAddWithoutValidation(UsedWeight, weight.ToString(CultureInfo.InvariantCulture));

        return response;
    }

    // The venue counts weight by the wall-clock minute: it starts again at nought as a new minute begins.
    private void Tick(DateTime venueNow)
    {
        var minute = new DateTimeOffset(DateTime.SpecifyKind(venueNow, DateTimeKind.Utc)).ToUnixTimeMilliseconds() / 60_000;

        if (minute != _minute)
        {
            _minute = minute;
            _weight = 0;
        }
    }

    private static int WeightOf(string path, Dictionary<string, string> query) => path switch
    {
        "/api/v3/klines" => 2,
        "/api/v3/exchangeInfo" => query.ContainsKey("symbol") ? 20 : 40,
        _ => 1,
    };

    private HttpResponseMessage Answer(string path, Dictionary<string, string> query, DateTime venueNow) => path switch
    {
        "/api/v3/time" => Json(HttpStatusCode.OK, $"{{\"serverTime\":{new DateTimeOffset(venueNow, TimeSpan.Zero).ToUnixTimeMilliseconds()}}}"),
        "/api/v3/exchangeInfo" => ExchangeInfo(query),
        "/api/v3/klines" => Klines(query, new DateTimeOffset(venueNow, TimeSpan.Zero).ToUnixTimeMilliseconds()),
        _ => Json(HttpStatusCode.NotFound, string.Empty),
    };

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    private static HttpResponseMessage Refused(int code, string message) =>
        Json(HttpStatusCode.BadRequest, $"{{\"code\":{code.ToString(CultureInfo.InvariantCulture)},\"msg\":\"{message}\"}}");

    private HttpResponseMessage ExchangeInfo(Dictionary<string, string> query)
    {
        if (!query.TryGetValue("symbol", out var symbol) || !Symbols.Contains(symbol, StringComparer.Ordinal))
        {
            return Refused(-1121, "Invalid symbol.");
        }

        return Json(
            HttpStatusCode.OK,
            "{\"timezone\":\"UTC\",\"serverTime\":0,\"rateLimits\":["
            + $"{{\"rateLimitType\":\"REQUEST_WEIGHT\",\"interval\":\"MINUTE\",\"intervalNum\":1,\"limit\":{WeightLimit.ToString(CultureInfo.InvariantCulture)}}},"
            + "{\"rateLimitType\":\"ORDERS\",\"interval\":\"SECOND\",\"intervalNum\":10,\"limit\":100},"
            + "{\"rateLimitType\":\"RAW_REQUESTS\",\"interval\":\"MINUTE\",\"intervalNum\":5,\"limit\":300000}],"
            + $"\"exchangeFilters\":[],\"symbols\":[{{\"symbol\":\"{symbol}\",\"status\":\"TRADING\"}}]}}");
    }

    private HttpResponseMessage Klines(Dictionary<string, string> query, long nowMs)
    {
        if (!query.TryGetValue("symbol", out var symbol) || symbol.Length == 0)
        {
            return Refused(-1102, "Mandatory parameter 'symbol' was not sent, was empty/null, or malformed.");
        }

        if (!Symbols.Contains(symbol, StringComparer.Ordinal))
        {
            return Refused(-1121, "Invalid symbol.");
        }

        if (!query.TryGetValue("interval", out var interval) || interval.Length == 0)
        {
            return Refused(-1102, "Mandatory parameter 'interval' was not sent, was empty/null, or malformed.");
        }

        if (!FixedSpans.ContainsKey(interval) && interval is not ("1w" or "1M"))
        {
            return Refused(-1120, "Invalid interval.");
        }

        var start = query.TryGetValue("startTime", out var from) && long.TryParse(from, NumberStyles.None, CultureInfo.InvariantCulture, out var s) ? s : 0;
        var end = query.TryGetValue("endTime", out var to) && long.TryParse(to, NumberStyles.None, CultureInfo.InvariantCulture, out var e) ? e : long.MaxValue;
        var limit = query.TryGetValue("limit", out var count) && int.TryParse(count, NumberStyles.None, CultureInfo.InvariantCulture, out var l) ? Math.Clamp(l, 1, 1000) : 500;

        if (start > end)
        {
            return Refused(-1128, "Combination of optional parameters invalid.");
        }

        var listed = new DateTimeOffset(ListedAt, TimeSpan.Zero).ToUnixTimeMilliseconds();
        var body = new StringBuilder("[");
        var written = 0;

        foreach (var open in Opens(interval, Math.Max(start, listed), Math.Min(end, nowMs)))
        {
            bool held;

            lock (_gate)
            {
                held = !_holes.Contains(open);
            }

            if (!held)
            {
                continue;
            }

            if (written == limit)
            {
                break;
            }

            body.Append(written++ == 0 ? string.Empty : ",").Append(Row(symbol, interval, open));
        }

        return Json(HttpStatusCode.OK, body.Append(']').ToString());
    }

    // The moments candles of an interval open at, from the first at or after a start to a last at or before an end, written
    // out here from the venue's own rules and not taken from the code under test.
    private static IEnumerable<long> Opens(string interval, long startMs, long endMs)
    {
        if (FixedSpans.TryGetValue(interval, out var span))
        {
            for (var open = (startMs + span - 1) / span * span; open <= endMs; open += span)
            {
                yield return open;
            }

            yield break;
        }

        var at = DateTime.UnixEpoch.AddMilliseconds(startMs);

        if (interval == "1w")
        {
            var monday = at.Date == at ? at : at.Date.AddDays(1);

            while (monday.DayOfWeek != DayOfWeek.Monday)
            {
                monday = monday.AddDays(1);
            }

            for (; Ms(monday) <= endMs; monday = monday.AddDays(7))
            {
                yield return Ms(monday);
            }

            yield break;
        }

        var first = new DateTime(at.Year, at.Month, 1, 0, 0, 0, DateTimeKind.Utc);

        for (first = first < at ? first.AddMonths(1) : first; Ms(first) <= endMs; first = first.AddMonths(1))
        {
            yield return Ms(first);
        }
    }

    private static long Ms(DateTime moment) => new DateTimeOffset(DateTime.SpecifyKind(moment, DateTimeKind.Utc)).ToUnixTimeMilliseconds();

    private static long Closes(string interval, long open) => interval switch
    {
        "1w" => Ms(DateTime.UnixEpoch.AddMilliseconds(open).AddDays(7)) - 1,
        "1M" => Ms(DateTime.UnixEpoch.AddMilliseconds(open).AddMonths(1)) - 1,
        _ => open + FixedSpans[interval] - 1,
    };

    // One candle in the twelve fields the venue writes: numbers for the moments and the count, strings of eight digits
    // for everything else, and a nought in the last for what the venue's documentation says it ignores.
    private string Row(string symbol, string interval, long open)
    {
        var state = Mix(((ulong)(uint)Seed * 0x100000001B3UL) ^ Hash(symbol) ^ (Hash(interval) << 13) ^ (ulong)open);
        var next = () => state = Mix(state);
        var opens = 10_000m + (next() % 5_000_000_000_000UL) / 100_000_000m;
        var closes = opens + (((long)(next() % 400_000_000_000UL) - 200_000_000_000L) / 100_000_000m);
        var high = Math.Max(opens, closes) + (next() % 100_000_000_000UL) / 100_000_000m;
        var low = Math.Min(opens, closes) - (next() % 100_000_000_000UL) / 100_000_000m;
        var volume = (next() % 1_000_000_000_000UL) / 100_000_000m;
        var quote = Math.Round(volume * (opens + closes) / 2m, 8);
        var trades = 1 + (long)(next() % 100_000UL);
        var takerShare = (next() % 1000UL) / 1000m;

        string D(decimal value) => value.ToString("F8", CultureInfo.InvariantCulture);

        return string.Create(
            CultureInfo.InvariantCulture,
            $"[{open},\"{D(opens)}\",\"{D(high)}\",\"{D(low)}\",\"{D(closes)}\",\"{D(volume)}\",{Closes(interval, open)},\"{D(quote)}\",{trades},\"{D(Math.Round(volume * takerShare, 8))}\",\"{D(Math.Round(quote * takerShare, 8))}\",\"0\"]");
    }

    private static ulong Mix(ulong value)
    {
        value += 0x9E3779B97F4A7C15UL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;

        return value ^ (value >> 31);
    }

    private static ulong Hash(string text) => text.Aggregate(14695981039346656037UL, (hash, letter) => (hash ^ letter) * 1099511628211UL);

    private static Dictionary<string, string> QueryOf(Uri uri) =>
        uri.Query.TrimStart('?').Split('&', StringSplitOptions.RemoveEmptyEntries)
            .Select(pair => pair.Split('=', 2))
            .GroupBy(pair => Uri.UnescapeDataString(pair[0]), StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => Uri.UnescapeDataString(group.Last().ElementAtOrDefault(1) ?? string.Empty), StringComparer.Ordinal);
}
