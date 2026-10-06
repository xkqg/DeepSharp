// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The venue's dialect, in one place: the addresses and parameters it is asked with, the twelve fields it answers a candle
/// with, the headers that tell what has been spent, and its own words for a request it refuses.
/// </summary>
/// <remarks>
/// A page is read exactly as the venue wrote it: the open time and the count of trades as numbers, every price and volume as
/// the text it came as, so a candle goes into the landing in the venue's own spelling. Nothing here knows what a window is or
/// whether a request is worth asking again; it asks, and it reads what came.
/// </remarks>
/// <param name="client">The client the requests go through, whose address is the venue's.</param>
internal sealed class VenueWire(HttpClient client)
{
    private static readonly string Agent =
        $"DeepSharp.Pipelines.Binance/{typeof(VenueWire).Assembly.GetName().Version!.ToString(3)} (+https://github.com/xkqg/DeepSharp)";

    /// <summary>The path and query that ask for the venue's own clock.</summary>
    public const string Time = "/api/v3/time";

    /// <summary>The path and query that ask what a symbol is and what an address may spend.</summary>
    /// <param name="symbol">The symbol.</param>
    /// <returns>The path and query.</returns>
    public static string Information(string symbol) => $"/api/v3/exchangeInfo?symbol={Uri.EscapeDataString(symbol)}";

    /// <summary>The path and query that ask for a page of candles.</summary>
    /// <param name="symbol">The symbol.</param>
    /// <param name="interval">The interval, as the venue spells it.</param>
    /// <param name="start">The first moment the page may open at, in milliseconds since the epoch.</param>
    /// <param name="end">The last moment the page may open at, inclusive, in milliseconds since the epoch.</param>
    /// <returns>The path and query.</returns>
    public static string Klines(string symbol, string interval, long start, long end) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"/api/v3/klines?symbol={Uri.EscapeDataString(symbol)}&interval={Uri.EscapeDataString(interval)}&startTime={start}&endTime={end}&limit={BinanceCandles.PageSize}");

    /// <summary>Asks the venue for something and keeps what came back, whatever it was.</summary>
    /// <param name="pathAndQuery">What is asked.</param>
    /// <param name="cancellation">Ends the request.</param>
    /// <returns>The answer, undecided.</returns>
    public async Task<VenueReply> GetAsync(string pathAndQuery, CancellationToken cancellation)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, pathAndQuery);

        request.Headers.UserAgent.ParseAdd(Agent);
        request.Headers.Accept.ParseAdd("application/json");

        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellation);
        var body = await response.Content.ReadAsByteArrayAsync(cancellation);

        return new VenueReply(response.StatusCode, body, UsedBy(response), WaitOf(response), Written(response, "Date"), Written(response, "x-mbx-uuid"));
    }

    /// <summary>The venue's clock, from its answer to the question what time it is.</summary>
    /// <param name="reply">The answer.</param>
    /// <returns>The moment, universal.</returns>
    /// <exception cref="PageFormatException">The answer is not that.</exception>
    public static DateTime ServerTime(VenueReply reply)
    {
        try
        {
            using var document = JsonDocument.Parse(reply.Body);

            return document.RootElement.ValueKind == JsonValueKind.Object
                   && document.RootElement.TryGetProperty("serverTime", out var time)
                   && time.ValueKind == JsonValueKind.Number
                   && time.TryGetInt64(out var milliseconds)
                ? DateTimeOffset.FromUnixTimeMilliseconds(milliseconds).UtcDateTime
                : throw new PageFormatException("it is not the venue's clock, which is an object with a serverTime");
        }
        catch (JsonException)
        {
            throw new PageFormatException("it is not JSON");
        }
        catch (ArgumentOutOfRangeException)
        {
            throw new PageFormatException("it names a moment that is not one");
        }
    }

    /// <summary>What the venue lets an address spend a minute, from its answer about a symbol.</summary>
    /// <param name="reply">The answer.</param>
    /// <returns>The weight; nothing when the answer says none.</returns>
    public static int? WeightLimit(VenueReply reply)
    {
        try
        {
            using var document = JsonDocument.Parse(reply.Body);

            if (document.RootElement.ValueKind != JsonValueKind.Object || !document.RootElement.TryGetProperty("rateLimits", out var limits) || limits.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            foreach (var limit in limits.EnumerateArray())
            {
                if (limit.TextOf("rateLimitType") == "REQUEST_WEIGHT"
                    && limit.TextOf("interval") == "MINUTE"
                    && limit.WholeOf("intervalNum") == 1
                    && limit.WholeOf("limit") is > 0 and var allowed)
                {
                    return allowed;
                }
            }

            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>The candles of a page, each as the venue wrote it.</summary>
    /// <param name="reply">The answer.</param>
    /// <returns>The candles, in the order the page holds them.</returns>
    /// <exception cref="PageFormatException">The answer is not a list of candles of twelve fields in the types the venue documents.</exception>
    public static IReadOnlyList<Candle> Candles(VenueReply reply)
    {
        try
        {
            using var document = JsonDocument.Parse(reply.Body);

            if (document.RootElement.ValueKind != JsonValueKind.Array)
            {
                throw new PageFormatException("it is not a list of candles");
            }

            return [.. document.RootElement.EnumerateArray().Select(CandleOf)];
        }
        catch (JsonException)
        {
            throw new PageFormatException("it is not JSON");
        }
    }

    // One candle: the open time and the count of trades are numbers, and every price and volume the text the venue wrote.
    private static Candle CandleOf(JsonElement row)
    {
        if (row.ValueKind != JsonValueKind.Array || row.GetArrayLength() < 12)
        {
            throw new PageFormatException($"a candle is not a list of the twelve fields the venue documents ({(row.ValueKind == JsonValueKind.Array ? row.GetArrayLength().ToString(CultureInfo.InvariantCulture) + " fields" : "it is not a list")})");
        }

        var open = Whole(row, 0, "its open time");

        Whole(row, 6, "its close time");

        return new Candle(
            open,
            Decimal(row, 1, "its open"),
            Decimal(row, 2, "its high"),
            Decimal(row, 3, "its low"),
            Decimal(row, 4, "its close"),
            Decimal(row, 5, "its volume"),
            Decimal(row, 7, "its quote volume"),
            Whole(row, 8, "its count of trades").ToString(CultureInfo.InvariantCulture),
            Decimal(row, 9, "its taker buy volume"),
            Decimal(row, 10, "its taker buy quote volume"));
    }

    private static long Whole(JsonElement row, int at, string what) =>
        row[at].ValueKind == JsonValueKind.Number && row[at].TryGetInt64(out var number)
            ? number
            : throw new PageFormatException($"{what} is not a whole number");

    private static string Decimal(JsonElement row, int at, string what)
    {
        var text = row[at].ValueKind == JsonValueKind.String ? row[at].GetString() : null;

        return text is not null && decimal.TryParse(text, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out _)
            ? text
            : throw new PageFormatException($"{what} is not a decimal number written as text ('{(row[at].ValueKind == JsonValueKind.String ? text : row[at].GetRawText())}')");
    }

    private static int? UsedBy(HttpResponseMessage response) =>
        response.Headers.TryGetValues("x-mbx-used-weight-1m", out var used) && int.TryParse(used.FirstOrDefault(), NumberStyles.None, CultureInfo.InvariantCulture, out var weight) ? weight : null;

    private static TimeSpan? WaitOf(HttpResponseMessage response) => response.Headers.RetryAfter switch
    {
        { Delta: { } delta } => delta,
        { Date: { } date } => date - (response.Headers.Date ?? DateTimeOffset.UtcNow),
        _ => null,
    };

    private static string? Written(HttpResponseMessage response, string header) =>
        response.Headers.NonValidated.TryGetValues(header, out var values) ? string.Join(", ", values) : null;
}
