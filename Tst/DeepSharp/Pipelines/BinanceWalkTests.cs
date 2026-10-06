// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Security.Cryptography;
using DeepSharp.Pipelines;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The walk down a window: the venue's clock is asked before anything is spent, the venue's limits are read, and pages are
/// asked for one at a time, each starting a millisecond after the last candle of the page before it, until the window is
/// told. A page is taken only if it is a page of candles in the window the venue was asked about, in order; the cursor
/// moves only past a page that was taken; and nothing that the venue did not hold is put in its place.
/// </summary>
public sealed class BinanceWalkTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 0, TimeSpan.Zero);

    private static readonly DateTime Jan1 = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private const string Klines = "/api/v3/klines";

    private readonly FakeTimeProvider _clock = new(Now);

    private FakeBinanceVenue _venue = null!;

    private BinanceCandles Asked(string interval, DateTime from, DateTime to, int seed = 7)
    {
        _venue = new FakeBinanceVenue(_clock) { Seed = seed };

        return new BinanceCandles("BTCEUR", interval, from, to).Through(_venue).On(_clock).At(new Uri("https://venue.test"));
    }

    private BinanceCandles Again(string interval, DateTime from, DateTime to) =>
        new BinanceCandles("BTCEUR", interval, from, to).Through(_venue).On(_clock).At(new Uri("https://venue.test"));

    private static long Ms(DateTime moment) => new DateTimeOffset(moment).ToUnixTimeMilliseconds();

    private string[] Paths() => [.. _venue.Seen.Select(request => request.Path)];

    private IEnumerable<VenueRequest> KlineRequests() => _venue.Seen.Where(request => request.Path == "/api/v3/klines");

    [Fact]
    public async Task TheVenuesClockIsAskedFirst_ThenItsLimits_ThenThePages()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));

        await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(["/api/v3/time", "/api/v3/exchangeInfo", "/api/v3/klines"], Paths());
        Assert.Equal("BTCEUR", _venue.Seen[1].Query["symbol"]);
    }

    [Fact]
    public async Task TheEndOfTheRequestIsAMillisecondBeforeTheEndOfTheWindow_ForTheVenuesEndIsInclusive()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));
        var asked = Assert.Single(KlineRequests());

        Assert.Equal((Ms(Jan1.AddDays(3)) - 1).ToString(System.Globalization.CultureInfo.InvariantCulture), asked.Query["endTime"]);
        Assert.Equal(Ms(Jan1).ToString(System.Globalization.CultureInfo.InvariantCulture), asked.Query["startTime"]);
        Assert.Equal("1000", asked.Query["limit"]);
        Assert.Equal([Ms(Jan1), Ms(Jan1.AddDays(1)), Ms(Jan1.AddDays(2))], walked.Candles.Select(candle => candle.OpenTime));
    }

    [Fact]
    public async Task TwoWindowsSideBySide_LandRowsThatShareNoCandle_AndMakeTheLandingOfTheirUnion()
    {
        var first = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(10)).WalkAsync(TestContext.Current.CancellationToken));
        var second = await _clock.RunAsync(Again("1d", Jan1.AddDays(10), Jan1.AddDays(25)).WalkAsync(TestContext.Current.CancellationToken));
        var whole = await _clock.RunAsync(Again("1d", Jan1, Jan1.AddDays(25)).WalkAsync(TestContext.Current.CancellationToken));

        Assert.Empty(first.Candles.Select(candle => candle.OpenTime).Intersect(second.Candles.Select(candle => candle.OpenTime)));
        Candle[] union = [.. first.Candles, .. second.Candles];

        Assert.Equal(whole.Candles, union);
    }

    [Fact]
    public async Task AWindowOfMoreThanOnePage_IsWalkedPageByPage_EachStartingAMillisecondAfterTheLastCandleBefore()
    {
        // 2,500 hours: two full pages and a short one.
        var candles = Asked("1h", Jan1, Jan1.AddHours(2500));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));
        var asked = KlineRequests().ToArray();

        Assert.Equal(3, asked.Length);
        Assert.Equal(2500, walked.Candles.Count);
        Assert.Equal(Ms(Jan1).ToString(System.Globalization.CultureInfo.InvariantCulture), asked[0].Query["startTime"]);
        Assert.Equal((Ms(Jan1.AddHours(999)) + 1).ToString(System.Globalization.CultureInfo.InvariantCulture), asked[1].Query["startTime"]);
        Assert.Equal((Ms(Jan1.AddHours(1999)) + 1).ToString(System.Globalization.CultureInfo.InvariantCulture), asked[2].Query["startTime"]);
        Assert.Equal(3, walked.PagesEstimated);
        Assert.Equal([1000, 1000, 500], walked.Pages.Select(page => page.Rows));
    }

    [Fact]
    public async Task AWindowThatIsExactlyOnePage_IsAskedOnce_NotAgainForAnEmptyOne()
    {
        var candles = Asked("1h", Jan1, Jan1.AddHours(1000));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(1000, walked.Candles.Count);
        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task AWindowThatBeginsBeforeTheListing_LandsWhatTheVenueHolds_AndCountsNothingInPlaceOfTheRest()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(10));

        _venue.ListedAt = Jan1.AddDays(4);

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(6, walked.Candles.Count);
        Assert.Equal(Ms(Jan1.AddDays(4)), walked.Candles[0].OpenTime);
    }

    [Fact]
    public async Task ACandleTheVenueDoesNotHold_IsNotThere_AndIsNotFilled()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(5));

        _venue.HoleAt(Jan1.AddDays(2));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal([Ms(Jan1), Ms(Jan1.AddDays(1)), Ms(Jan1.AddDays(3)), Ms(Jan1.AddDays(4))], walked.Candles.Select(candle => candle.OpenTime));
    }

    [Fact]
    public async Task EveryNumberIsAsTheVenueWroteIt_ATradeCountIncluded()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(1));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));
        var candle = walked.Candles[0];

        Assert.Matches(@"^\d+\.\d{8}$", candle.Open);
        Assert.Matches(@"^\d+\.\d{8}$", candle.TakerBuyQuoteVolume);
        Assert.Matches(@"^\d+$", candle.Trades);
    }

    [Fact]
    public async Task EachPageIsRecordedAsItCame_ItsBodyHashedAndTheVenuesOwnHeadersKept()
    {
        var candles = Asked("1h", Jan1, Jan1.AddHours(1500));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(2, walked.Pages.Count);
        Assert.Equal(Jan1, walked.Pages[0].From);
        Assert.Equal(Jan1.AddHours(999).AddMilliseconds(1), walked.Pages[1].From);
        Assert.All(walked.Pages, page =>
        {
            Assert.Matches("^[0-9a-f]{64}$", page.Sha256);
            Assert.False(string.IsNullOrEmpty(page.Date));
            Assert.True(Guid.TryParse(page.Uuid, out _));
        });
        Assert.NotEqual(walked.Pages[0].Uuid, walked.Pages[1].Uuid);
    }

    [Fact]
    public async Task APagesHash_IsTheSha256OfTheBodyTheVenueSent()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(2));

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));
        using var client = new HttpClient(new FakeBinanceVenue(_clock), disposeHandler: true);
        var body = await client.GetByteArrayAsync(
            new Uri($"https://venue.test/api/v3/klines?symbol=BTCEUR&interval=1d&startTime={Ms(Jan1)}&endTime={Ms(Jan1.AddDays(2)) - 1}&limit=1000"),
            TestContext.Current.CancellationToken);

        Assert.Equal(Convert.ToHexString(SHA256.HashData(body)).ToLowerInvariant(), walked.Pages[0].Sha256);
    }

    [Fact]
    public async Task TheClockTheLandingBeganByAndTheVenuesOwn_AreBothKept_AndAreNotTheSame()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(1));

        _venue.ClockSkew = TimeSpan.FromMilliseconds(-770);

        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(Now.UtcDateTime, walked.AskedAt);
        Assert.Equal(_venue.Seen[0].At.AddMilliseconds(-770), walked.ServerTime);
    }

    [Fact]
    public async Task AWindowThatEndsAfterTheVenueHasClosedItsLastCandle_IsRefused_NamingTheLatestThatCanLand()
    {
        // The last candle of the window closes at its end; a candle is taken only a minute after it closes.
        var to = new DateTime(2026, 10, 6, 0, 0, 0, DateTimeKind.Utc).AddDays(1);
        var candles = Asked("1d", to.AddDays(-3), to);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("2026-10-07T00:00:00.000Z", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2026-10-06T00:00:00.000Z", refused.Message, StringComparison.Ordinal);
        Assert.Equal(["/api/v3/time"], Paths());
    }

    [Fact]
    public async Task AWindowThatEndsExactlyAMinuteAfterItsLastCandleClosed_Lands()
    {
        // The venue's clock says eight o'clock; the last candle of the window closes at a minute before.
        var candles = Asked("1m", new DateTime(2026, 10, 6, 7, 56, 0, DateTimeKind.Utc), new DateTime(2026, 10, 6, 7, 59, 0, DateTimeKind.Utc));
        var walked = await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(3, walked.Candles.Count);
    }

    [Fact]
    public async Task AWindowThatEndsASecondTooSoonAfterItsLastCandleClosed_IsRefused()
    {
        var candles = Asked("1m", new DateTime(2026, 10, 6, 7, 56, 0, DateTimeKind.Utc), new DateTime(2026, 10, 6, 8, 0, 0, DateTimeKind.Utc));

        await Assert.ThrowsAsync<InvalidOperationException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));
    }

    [Fact]
    public async Task AWindowTheVenueHoldsNothingOf_IsRefused_NotLandedAsAHeaderAlone()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));

        _venue.ListedAt = Jan1.AddYears(5);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("holds no candle", refused.Message, StringComparison.Ordinal);
        Assert.Contains("BTCEUR", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ASymbolTheVenueDoesNotKnow_IsRefusedInTheVenuesOwnWords_BeforeAPageIsAsked()
    {
        _venue = new FakeBinanceVenue(_clock);

        var candles = new BinanceCandles("NOPE", "1d", Jan1, Jan1.AddDays(3)).Through(_venue).On(_clock).At(new Uri("https://venue.test"));
        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal(HttpStatusCode.BadRequest, refused.Status);
        Assert.Equal(-1121, refused.Code);
        Assert.Contains("Invalid symbol.", refused.Message, StringComparison.Ordinal);
        Assert.Empty(KlineRequests());
    }

    [Theory]
    [InlineData("a string, not a list")]
    [InlineData("{\"code\":0}")]
    [InlineData("[1]")]
    [InlineData("[[1704067200000,\"1\",\"2\"]]")]
    [InlineData("[[\"1704067200000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",1704153599999,\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]")]
    [InlineData("[[1704067200000,1.0,\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",1704153599999,\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]")]
    [InlineData("[[1704067200000,\"abc\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",1704153599999,\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]")]
    public async Task APageThatIsNotAPageOfCandles_IsNotTaken_AndIsRetriedThenRefused(string body)
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));

        _venue.Respond(body, times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("not a page of candles", refused.Message, StringComparison.Ordinal);
        Assert.Equal(LandingPace.MostAttempts, KlineRequests().Count());
    }

    [Fact]
    public async Task ACandleOutsideTheWindowThatWasAskedAbout_IsRefused_ForTheVenueHasChangedItsMind()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));
        var outside = $"[[{Ms(Jan1.AddDays(3))},\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",{Ms(Jan1.AddDays(4)) - 1},\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]";

        _venue.Respond(outside, times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("outside the window", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACandleInAnotherUnit_Microseconds_IsRefused_NotLandedAsTheYearTenThousand()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));
        var micro = $"[[{Ms(Jan1) * 1000},\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",{Ms(Jan1) * 1000 + 86_399_999_999},\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]";

        _venue.Respond(micro, times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("outside the window", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CandlesThatDoNotFollowEachOther_AreRefused()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));
        string Row(int day) => $"[{Ms(Jan1.AddDays(day))},\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",{Ms(Jan1.AddDays(day + 1)) - 1},\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]";

        _venue.Respond($"[{Row(1)},{Row(0)}]", times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("not in order", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACandleOffTheGridOfTheInterval_IsRefused()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(3));
        var off = $"[[{Ms(Jan1) + 3_600_000},\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",\"1.00000000\",{Ms(Jan1) + 90_000_000},\"1.00000000\",1,\"1.00000000\",\"1.00000000\",\"0\"]]";

        _venue.Respond(off, times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("does not open", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACancelledWalk_AsksNothingFurther()
    {
        var candles = Asked("1h", Jan1, Jan1.AddHours(2500));
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        await stop.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _clock.RunAsync(candles.WalkAsync(stop.Token)));

        Assert.Empty(_venue.Seen);
    }

    [Fact]
    public async Task ACancellationBetweenPages_EndsTheWalk_WithoutAPageMore()
    {
        var candles = Asked("1h", Jan1, Jan1.AddHours(2500));
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var walking = candles.WalkAsync(stop.Token);

        // Let the first page be asked, then give up while the pace holds the second.
        await AdvanceUntil(() => KlineRequests().Count() == 1);
        await stop.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _clock.RunAsync(walking));

        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task AHandlerHandedIn_IsKept_NotDisposedByTheLanding()
    {
        var venue = new FakeBinanceVenue(_clock);
        using var handler = new TrackedHandler(venue);

        _venue = venue;

        var candles = new BinanceCandles("BTCEUR", "1d", Jan1, Jan1.AddDays(2)).Through(handler).On(_clock).At(new Uri("https://venue.test"));

        await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.False(handler.Disposed);
    }

    [Fact]
    public async Task EveryRequestSaysWhoAsks_ByAnHonestProductToken()
    {
        var candles = Asked("1d", Jan1, Jan1.AddDays(2));

        await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.All(_venue.Seen, request =>
        {
            Assert.StartsWith("DeepSharp.Pipelines.Binance/", request.UserAgent, StringComparison.Ordinal);
            Assert.Contains("github.com/xkqg/DeepSharp", request.UserAgent, StringComparison.Ordinal);
        });
    }

    [Fact]
    public async Task TheSymbolIsEscapedInTheRequest_NeverTrustedAsItStands()
    {
        // A symbol is checked when the window is made; the request still carries it escaped.
        var candles = Asked("1d", Jan1, Jan1.AddDays(2));

        await _clock.RunAsync(candles.WalkAsync(TestContext.Current.CancellationToken));

        Assert.All(KlineRequests(), request => Assert.Equal("BTCEUR", request.Query["symbol"]));
    }

    // Moves the clock on in small steps until the walk has come to what the test waits for, and stops moving it there.
    private async Task AdvanceUntil(Func<bool> condition)
    {
        for (var waited = 0; !condition() && waited < 2000; waited++)
        {
            await Task.Delay(1, TestContext.Current.CancellationToken);

            if (!condition())
            {
                _clock.Advance(TimeSpan.FromMilliseconds(100));
            }
        }

        Assert.True(condition(), "The walk did not come to what the test waited for.");
    }

    private sealed class TrackedHandler(HttpMessageHandler inner) : DelegatingHandler(inner)
    {
        public bool Disposed { get; private set; }

        protected override void Dispose(bool disposing)
        {
            Disposed = true;
            base.Dispose(disposing);
        }
    }
}
