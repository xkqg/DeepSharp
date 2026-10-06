// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.Json;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The venue the landing's tests talk to, held to what the venue's documentation says and what was measured of it: a test
/// that fetched from the real one would be a test of the weather, and one against a stand-in that answers as the code
/// expects proves nothing. So the stand-in is written from the venue's own rules, and these tests are what hold it to them.
/// What it holds is generated from a seed, never captured: no number in it was ever Binance's.
/// </summary>
public sealed class FakeBinanceVenueTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 0, TimeSpan.Zero);

    private readonly FakeTimeProvider _clock = new(Now);

    private (FakeBinanceVenue Venue, HttpClient Client) Venue(int seed = 7)
    {
        var venue = new FakeBinanceVenue(_clock) { Seed = seed };

        return (venue, new HttpClient(venue) { BaseAddress = new Uri("https://venue.test") });
    }

    private static async Task<JsonElement> RowsAsync(HttpClient client, string query)
    {
        using var response = await client.GetAsync($"/api/v3/klines?{query}", TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        return JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement.Clone();
    }

    private static long[] Opens(JsonElement rows) => [.. rows.EnumerateArray().Select(row => row[0].GetInt64())];

    [Fact]
    public async Task TheFakeVenueAnswersTheMeasuredBoundaryLikeTheRealOne()
    {
        // Measured on the real venue: a start at one day's open and an end at the next day's open answered with two candles,
        // so the end is inclusive of a candle that opens at that very moment.
        var (_, client) = Venue();
        var rows = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704153600000&limit=10");

        Assert.Equal([1704067200000, 1704153600000], Opens(rows));
        Assert.Equal([1704153599999, 1704239999999], rows.EnumerateArray().Select(row => row[6].GetInt64()));
    }

    [Fact]
    public async Task AStartBetweenTwoCandles_SkipsForwardToTheNextOne()
    {
        var (_, client) = Venue();
        var rows = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200001&endTime=1704326400000&limit=10");

        Assert.Equal([1704153600000, 1704240000000, 1704326400000], Opens(rows));
    }

    [Fact]
    public async Task AnAnswerHoldsAtMostTheLimit_AndNeverMoreThanAThousand()
    {
        var (_, client) = Venue();

        Assert.Equal(3, Opens(await RowsAsync(client, "symbol=BTCEUR&interval=1m&startTime=1704067200000&endTime=1704153600000&limit=3")).Length);
        Assert.Equal(1000, Opens(await RowsAsync(client, "symbol=BTCEUR&interval=1m&startTime=1704067200000&endTime=1704153600000&limit=1000")).Length);
        Assert.Equal(1000, Opens(await RowsAsync(client, "symbol=BTCEUR&interval=1m&startTime=1704067200000&endTime=1704153600000&limit=1001")).Length);
        Assert.Equal(500, Opens(await RowsAsync(client, "symbol=BTCEUR&interval=1m&startTime=1704067200000&endTime=1704153600000")).Length);
    }

    [Fact]
    public async Task AWindowBeforeTheListing_StartsWhereTheListingDoes_AndOneAfterTheEnd_AnswersNothing()
    {
        var (venue, client) = Venue();

        venue.ListedAt = new DateTime(2020, 1, 3, 0, 0, 0, DateTimeKind.Utc);

        var first = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1483228800000&limit=2");
        var future = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1893456000000&endTime=1893542400000&limit=10");

        Assert.Equal([1578009600000, 1578096000000], Opens(first));
        Assert.Empty(future.EnumerateArray());
    }

    [Fact]
    public async Task TheCandleThatIsStillOpen_IsInTheAnswer_WithAClosingMomentStillToCome()
    {
        var (_, client) = Venue();
        var today = new DateTimeOffset(2026, 10, 6, 0, 0, 0, TimeSpan.Zero).ToUnixTimeMilliseconds();
        var rows = await RowsAsync(client, $"symbol=BTCEUR&interval=1d&startTime={today - 86_400_000}&limit=10");

        Assert.Equal([today - 86_400_000, today], Opens(rows));
        Assert.True(rows[1][6].GetInt64() > Now.ToUnixTimeMilliseconds());
    }

    [Fact]
    public async Task EveryCandleHasTheTwelveFieldsInTheTypesTheVenueWritesThem()
    {
        var (_, client) = Venue();
        var row = (await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&limit=1"))[0];

        Assert.Equal(12, row.GetArrayLength());
        Assert.Equal(
            [JsonValueKind.Number, JsonValueKind.String, JsonValueKind.String, JsonValueKind.String, JsonValueKind.String, JsonValueKind.String, JsonValueKind.Number, JsonValueKind.String, JsonValueKind.Number, JsonValueKind.String, JsonValueKind.String, JsonValueKind.String],
            row.EnumerateArray().Select(field => field.ValueKind));
        Assert.Equal("0", row[11].GetString());
    }

    [Fact]
    public async Task ADecimalIsSpelledWithEightDigits_AsTheVenueSpellsOne()
    {
        var (_, client) = Venue();
        var row = (await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&limit=1"))[0];

        foreach (var at in new[] { 1, 2, 3, 4, 5, 7, 9, 10 })
        {
            var text = row[at].GetString()!;

            Assert.Matches(@"^\d+\.\d{8}$", text);
        }
    }

    [Fact]
    public async Task ACandleIsInternallyConsistent_TheHighIsTheHighestAndTheLowTheLowest()
    {
        var (_, client) = Venue();

        foreach (var row in (await RowsAsync(client, "symbol=BTCEUR&interval=1h&startTime=1704067200000&limit=200")).EnumerateArray())
        {
            var (open, high, low, close) = (decimal.Parse(row[1].GetString()!), decimal.Parse(row[2].GetString()!), decimal.Parse(row[3].GetString()!), decimal.Parse(row[4].GetString()!));

            Assert.True(high >= Math.Max(open, close) && low <= Math.Min(open, close), $"{open} {high} {low} {close}");
        }
    }

    [Fact]
    public async Task TheSameAsk_GivesTheSameBytes_AndTwoWindowsSideBySideMakeTheirUnion()
    {
        var (_, client) = Venue();
        var whole = Opens(await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704672000000&limit=1000"));
        var first = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704326399999&limit=1000");
        var second = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704326400000&endTime=1704672000000&limit=1000");

        long[] union = [.. Opens(first), .. Opens(second)];

        Assert.Equal(whole, union);
        Assert.Equal(first.GetRawText(), (await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704326399999&limit=1000")).GetRawText());
    }

    [Fact]
    public async Task AnotherSeed_AnswersOtherNumbers_ForTheCandleThatOpensAtTheSameMoment()
    {
        var one = await RowsAsync(Venue(seed: 1).Client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&limit=1");
        var other = await RowsAsync(Venue(seed: 2).Client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&limit=1");

        Assert.Equal(Opens(one), Opens(other));
        Assert.NotEqual(one.GetRawText(), other.GetRawText());
    }

    [Fact]
    public async Task ACandleTheVenueDoesNotHold_IsNotInTheAnswer_AndNothingIsPutInItsPlace()
    {
        var (venue, client) = Venue();

        venue.HoleAt(new DateTime(2024, 1, 2, 0, 0, 0, DateTimeKind.Utc));

        var rows = await RowsAsync(client, "symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704326400000&limit=10");

        Assert.Equal([1704067200000, 1704240000000, 1704326400000], Opens(rows));
    }

    [Fact]
    public async Task AnUnknownSymbol_IsRefusedInTheVenuesWords()
    {
        var (_, client) = Venue();
        using var response = await client.GetAsync("/api/v3/klines?symbol=NOPE&interval=1d", TestContext.Current.CancellationToken);
        var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(-1121, body.GetProperty("code").GetInt32());
        Assert.Equal("Invalid symbol.", body.GetProperty("msg").GetString());
    }

    [Fact]
    public async Task AnUnknownInterval_IsRefusedInTheVenuesWords()
    {
        var (_, client) = Venue();
        using var response = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=2d", TestContext.Current.CancellationToken);
        var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal(-1120, body.GetProperty("code").GetInt32());
        Assert.Equal("Invalid interval.", body.GetProperty("msg").GetString());
    }

    [Fact]
    public async Task TheTimeEndpoint_AnswersTheVenuesOwnClock()
    {
        var (venue, client) = Venue();

        venue.ClockSkew = TimeSpan.FromMilliseconds(-770);

        using var response = await client.GetAsync("/api/v3/time", TestContext.Current.CancellationToken);
        var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;

        Assert.Equal(Now.ToUnixTimeMilliseconds() - 770, body.GetProperty("serverTime").GetInt64());
    }

    [Fact]
    public async Task TheWeightSpentIsCountedByTheVenuesMinute_ATimeCallOneAndAKlinesCallTwo()
    {
        var (_, client) = Venue();

        using var time = await client.GetAsync("/api/v3/time", TestContext.Current.CancellationToken);
        using var klines = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);
        using var again = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);

        Assert.Equal(["1"], time.Headers.GetValues("x-mbx-used-weight-1m"));
        Assert.Equal(["3"], klines.Headers.GetValues("x-mbx-used-weight-1m"));
        Assert.Equal(["5"], again.Headers.GetValues("x-mbx-used-weight-1m"));

        _clock.Advance(TimeSpan.FromSeconds(61));

        using var later = await client.GetAsync("/api/v3/time", TestContext.Current.CancellationToken);

        Assert.Equal(["1"], later.Headers.GetValues("x-mbx-used-weight-1m"));
    }

    [Fact]
    public async Task WeightSomebodyElseSpends_IsInTheSameMinuteAndTheSameCount()
    {
        var (venue, client) = Venue();

        venue.Spend(500);

        using var time = await client.GetAsync("/api/v3/time", TestContext.Current.CancellationToken);

        Assert.Equal(["501"], time.Headers.GetValues("x-mbx-used-weight-1m"));
    }

    [Fact]
    public async Task TheExchangeInformationSaysWhatAnAddressMaySpendAMinute()
    {
        var (venue, client) = Venue();

        venue.WeightLimit = 1200;

        using var response = await client.GetAsync("/api/v3/exchangeInfo?symbol=BTCEUR", TestContext.Current.CancellationToken);
        var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken)).RootElement;
        var limit = body.GetProperty("rateLimits").EnumerateArray().Single(each => each.GetProperty("rateLimitType").GetString() == "REQUEST_WEIGHT");

        Assert.Equal(1200, limit.GetProperty("limit").GetInt32());
        Assert.Equal("MINUTE", limit.GetProperty("interval").GetString());
        Assert.Equal(1, limit.GetProperty("intervalNum").GetInt32());
        Assert.Equal(["20"], response.Headers.GetValues("x-mbx-used-weight-1m"));
    }

    [Fact]
    public async Task AnAnswerCanBeScriptedToHoldWhatTheTestNeeds()
    {
        var (venue, client) = Venue();

        venue.Respond("[[1,2]]");

        using var response = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);

        Assert.Equal("[[1,2]]", await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task AnswersCarryTheVenuesDate_AndAnIdentifierOfTheRequest()
    {
        var (_, client) = Venue();
        using var response = await client.GetAsync("/api/v3/time", TestContext.Current.CancellationToken);

        Assert.Equal(Now, response.Headers.Date);
        Assert.True(Guid.TryParse(response.Headers.GetValues("x-mbx-uuid").Single(), out _));
    }

    [Fact]
    public async Task AScriptedFailure_IsAnsweredAsScripted_ThenTheVenueAnswersAgain()
    {
        var (venue, client) = Venue();

        venue.Fail(HttpStatusCode.TooManyRequests, times: 2, retryAfter: TimeSpan.FromSeconds(3));

        using var first = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);
        using var second = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);
        using var third = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.TooManyRequests, first.StatusCode);
        Assert.Equal(TimeSpan.FromSeconds(3), first.Headers.RetryAfter!.Delta);
        Assert.Equal(HttpStatusCode.TooManyRequests, second.StatusCode);
        Assert.Equal(HttpStatusCode.OK, third.StatusCode);
    }

    [Fact]
    public async Task AStalledAnswer_NeverComes_UntilTheCallerGivesUp()
    {
        var (venue, client) = Venue();

        venue.Stall(times: 1);

        using var gave = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.GetAsync("/api/v3/time", gave.Token));
    }

    [Fact]
    public async Task AnAnswerThatIsNotJson_IsScriptable_ToSeeWhatARefusalSaysOfIt()
    {
        var (venue, client) = Venue();

        venue.Garble(times: 1);

        using var response = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&limit=1", TestContext.Current.CancellationToken);

        var body = await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.ThrowsAny<JsonException>(() => JsonDocument.Parse(body));
    }

    [Fact]
    public async Task EveryRequestIsKept_WithWhenItCameAndWhatItAskedFor()
    {
        var (venue, client) = Venue();

        using var _ = await client.GetAsync("/api/v3/klines?symbol=BTCEUR&interval=1d&startTime=1704067200000&limit=2", TestContext.Current.CancellationToken);

        var seen = Assert.Single(venue.Seen);

        Assert.Equal("/api/v3/klines", seen.Path);
        Assert.Equal("BTCEUR", seen.Query["symbol"]);
        Assert.Equal("1704067200000", seen.Query["startTime"]);
        Assert.Equal(Now.UtcDateTime, seen.At);
    }

    [Fact]
    public void NoTestReachesTheNetwork_ForTheDeadProxyAnswersEveryHostButTheMachineItself()
    {
        // Set once for the whole assembly before any test runs: a program a suite runs that fetched from the venue would
        // find nothing listening, not the venue, and a loopback stand-in would still answer.
        var proxy = HttpClient.DefaultProxy;

        Assert.Equal(new Uri("http://127.0.0.1:9"), proxy.GetProxy(new Uri("https://data-api.binance.vision/api/v3/time")));
        Assert.Equal(new Uri("http://127.0.0.1:9"), proxy.GetProxy(new Uri("https://api.binance.com/api/v3/time")));
        Assert.True(proxy.IsBypassed(new Uri("http://127.0.0.1:5000/api/v3/time")));
        Assert.True(proxy.IsBypassed(new Uri("http://localhost:5000/api/v3/time")));
    }
}
