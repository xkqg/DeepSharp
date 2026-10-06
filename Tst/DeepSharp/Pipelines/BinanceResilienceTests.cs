// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// How a landing treats a budget it shares with programs it knows nothing of: every request is paced and the venue's own
/// count of what this address has spent is obeyed; a refusal is met by waiting longer each time, by as long as the venue
/// says, and only up to a ceiling; one unit of work fails once however many tries it took; and what says stop — a ban, a
/// wall, a region — stops at once, since a walk that marched on after a refusal is how one rejected call becomes a storm.
/// All of it runs on a clock the test holds, so none of it is waited for.
/// </summary>
public sealed class BinanceResilienceTests
{
    private const string Klines = "/api/v3/klines";

    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 0, TimeSpan.Zero);

    private static readonly DateTime Jan1 = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private readonly FakeTimeProvider _clock = new(Now);

    private readonly FakeBinanceVenue _venue;

    public BinanceResilienceTests() => _venue = new FakeBinanceVenue(_clock);

    private BinanceCandles Asked(string interval, DateTime from, DateTime to, HttpMessageHandler? through = null) =>
        new BinanceCandles("BTCEUR", interval, from, to).Through(through ?? _venue).On(_clock).At(new Uri("https://venue.test"));

    private IEnumerable<VenueRequest> KlineRequests() => _venue.Seen.Where(request => request.Path == Klines);

    private async Task Until(Func<bool> condition)
    {
        for (var waited = 0; !condition() && waited < 2000; waited++)
        {
            await Task.Delay(1, TestContext.Current.CancellationToken);
        }

        Assert.True(condition(), "The walk did not come to what the test waited for.");
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

    [Fact]
    public async Task EveryRequestIsPaced_NoTwoCloserThanTheFixedPace()
    {
        var walked = await _clock.RunAsync(Asked("1h", Jan1, Jan1.AddHours(2500)).WalkAsync(TestContext.Current.CancellationToken));
        var seen = _venue.Seen;

        Assert.Equal(2500, walked.Candles.Count);
        Assert.Equal(5, seen.Count);

        for (var at = 1; at < seen.Count; at++)
        {
            Assert.True(seen[at].At - seen[at - 1].At >= LandingPace.PerRequest, $"Request {at} came {seen[at].At - seen[at - 1].At} after the one before.");
        }
    }

    [Fact]
    public async Task NothingIsAskedUntilThePaceHasPassed()
    {
        var walking = Asked("1h", Jan1, Jan1.AddHours(2500)).WalkAsync(TestContext.Current.CancellationToken);

        await Until(() => _venue.Seen.Count == 1);
        await Task.Delay(30, TestContext.Current.CancellationToken);

        Assert.Single(_venue.Seen);

        _clock.Advance(LandingPace.PerRequest - TimeSpan.FromMilliseconds(1));
        await Task.Delay(30, TestContext.Current.CancellationToken);

        Assert.Single(_venue.Seen);

        _clock.Advance(TimeSpan.FromMilliseconds(1));
        await Until(() => _venue.Seen.Count == 2);

        await _clock.RunAsync(walking);
    }

    [Fact]
    public async Task AnAddressThatHasSpentMoreThanHalfOfItsMinute_IsGivenTheRestOfItByTheWalk()
    {
        // Somebody else spent 60 of the 100 an address may spend a minute. This address waits for the venue's next minute
        // before it asks for anything of its own, whatever the venue's clock says the time is.
        _venue.WeightLimit = 100;
        _venue.Spend(60);

        var walked = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));
        var seen = _venue.Seen;

        Assert.Equal(3, walked.Candles.Count);
        Assert.Equal(["/api/v3/time", "/api/v3/exchangeInfo", Klines], seen.Select(request => request.Path));
        Assert.True(seen[1].At < new DateTime(2026, 10, 6, 8, 1, 0, DateTimeKind.Utc), $"The request that learns what an address may spend came at {seen[1].At:O}.");
        Assert.True(seen[2].At >= new DateTime(2026, 10, 6, 8, 1, 0, DateTimeKind.Utc), $"The first page came at {seen[2].At:O}, inside the minute that was spent.");
        Assert.True(seen[2].At < new DateTime(2026, 10, 6, 8, 1, 10, DateTimeKind.Utc), $"The first page came at {seen[2].At:O}, long after the minute was over.");
    }

    [Fact]
    public async Task TheWaitOfTheBrake_IsNotATry_SoARefusalThatFollowsIt_IsStillAskedAgain()
    {
        // The brake holds a request for most of a minute. That wait is not a try, and no try's timeout runs while it lasts: were
        // it timed, each ten seconds of it would use up one of the five tries, and the one refusal that follows would end the
        // landing with Binance having been asked once.
        _venue.WeightLimit = 100;
        _venue.Spend(60);
        _venue.Fail(HttpStatusCode.ServiceUnavailable, path: Klines);

        var walked = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(3, walked.Candles.Count);
        Assert.Equal(2, KlineRequests().Count());
    }

    [Fact]
    public async Task AnAddressThatHasSpentLessThanHalf_IsNotHeldBack()
    {
        _venue.WeightLimit = 100;
        _venue.Spend(10);

        await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken), TimeSpan.FromMilliseconds(100));

        Assert.True(_venue.Seen[^1].At - _venue.Seen[0].At < TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task ARefusalThatSaysHowLongToWait_IsWaitedOutByThatLong()
    {
        _venue.Fail(HttpStatusCode.TooManyRequests, retryAfter: TimeSpan.FromSeconds(7), path: Klines);

        await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));

        var asked = KlineRequests().ToArray();

        Assert.Equal(2, asked.Length);
        Assert.True(asked[1].At - asked[0].At >= TimeSpan.FromSeconds(7), $"The retry came {asked[1].At - asked[0].At} after the refusal.");
    }

    [Fact]
    public async Task WithoutSayingHowLong_EachRetryWaitsLongerThanTheLast()
    {
        _venue.Fail(HttpStatusCode.ServiceUnavailable, times: 3, path: Klines);

        await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken), TimeSpan.FromMilliseconds(100));

        var asked = KlineRequests().ToArray();
        var gaps = asked.Zip(asked.Skip(1), (before, after) => after.At - before.At).ToArray();

        Assert.Equal(4, asked.Length);
        Assert.True(gaps[0] >= LandingPace.FirstBackoff, $"{gaps[0]}");
        Assert.True(gaps[1] >= gaps[0] + (LandingPace.FirstBackoff / 2), $"{gaps[1]} after {gaps[0]}");
        Assert.True(gaps[2] >= gaps[1] + LandingPace.FirstBackoff, $"{gaps[2]} after {gaps[1]}");
    }

    [Fact]
    public async Task ARefusalThatAsksForMoreThanALandingWaits_StopsTheWalk_InsteadOfWaiting()
    {
        _venue.Fail(HttpStatusCode.TooManyRequests, retryAfter: TimeSpan.FromDays(3), path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal(HttpStatusCode.TooManyRequests, refused.Status);
        Assert.Contains("wait", refused.Message, StringComparison.Ordinal);
        Assert.Contains("3 days", refused.Message, StringComparison.Ordinal);
        Assert.Single(KlineRequests());
    }

    [Theory]
    [InlineData(HttpStatusCode.Forbidden)]
    [InlineData((HttpStatusCode)418)]
    [InlineData((HttpStatusCode)451)]
    public async Task WhatSaysStop_StopsAtOnce_WithoutARetry(HttpStatusCode status)
    {
        _venue.Fail(status, retryAfter: TimeSpan.FromSeconds(120), path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal(status, refused.Status);
        Assert.Contains("stopped", refused.Message, StringComparison.Ordinal);
        Assert.Single(KlineRequests());
        Assert.Equal(1, _venue.Seen.Count(request => request.Path == "/api/v3/exchangeInfo"));
    }

    [Fact]
    public async Task ABan_SaysHowLongItLasts_WhenTheVenueSaysSo()
    {
        _venue.Fail((HttpStatusCode)418, retryAfter: TimeSpan.FromMinutes(5), path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Contains("banned", refused.Message, StringComparison.Ordinal);
        Assert.Contains("5 minutes", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AStalledPage_IsGivenUpOn_AfterTheTimeAnAttemptMayTake_AndAskedAgain()
    {
        _venue.Stall(path: Klines);

        var walked = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));
        var asked = KlineRequests().ToArray();

        Assert.Equal(3, walked.Candles.Count);
        Assert.Equal(2, asked.Length);
        Assert.True(asked[1].At - asked[0].At >= LandingPace.AttemptTimeout, $"{asked[1].At - asked[0].At}");
    }

    [Fact]
    public async Task APageThatNeverComes_IsOneFailureOfTheUnit_NotOneForEachTry()
    {
        _venue.Fail(HttpStatusCode.ServiceUnavailable, times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal(HttpStatusCode.ServiceUnavailable, refused.Status);
        Assert.Contains($"{LandingPace.MostAttempts} tries", refused.Message, StringComparison.Ordinal);
        Assert.Contains("startTime=1704067200000", refused.Message, StringComparison.Ordinal);
        Assert.Equal(LandingPace.MostAttempts, KlineRequests().Count());
    }

    [Fact]
    public async Task TheClockCall_IsRetriedToo()
    {
        _venue.Fail(HttpStatusCode.BadGateway, path: "/api/v3/time");

        var walked = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(3, walked.Candles.Count);
        Assert.Equal(2, _venue.Seen.Count(request => request.Path == "/api/v3/time"));
    }

    [Fact]
    public async Task TheVenuesOwnWordsForAWrongRequest_AreCarried_AndTheRequestIsNotAskedAgain()
    {
        _venue.Fail(HttpStatusCode.BadRequest, body: "{\"code\":-1100,\"msg\":\"Illegal characters found in parameter 'symbol'.\"}", path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal(-1100, refused.Code);
        Assert.Contains("Illegal characters found in parameter 'symbol'.", refused.Message, StringComparison.Ordinal);
        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task ACallThatNeverReachedTheVenue_IsRetried_ThenRefusedWithWhatWentWrong()
    {
        using var flaky = new FlakyHandler(_venue, failures: LandingPace.MostAttempts);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3), flaky).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Null(refused.Status);
        Assert.IsType<HttpRequestException>(refused.InnerException);
        Assert.Equal(LandingPace.MostAttempts, flaky.Calls);
    }

    [Fact]
    public async Task ACallThatNeverReachedTheVenueOnce_IsRetriedAndTheWalkGoesOn()
    {
        using var flaky = new FlakyHandler(_venue, failures: 1);

        var walked = await _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3), flaky).WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(3, walked.Candles.Count);
    }

    [Fact]
    public async Task ACancellationWhileARetryWaits_EndsTheWalk_AndAsksNothingMore()
    {
        _venue.Fail(HttpStatusCode.ServiceUnavailable, times: 3, path: Klines);

        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var walking = Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(stop.Token);

        await AdvanceUntil(() => KlineRequests().Count() == 1);
        await stop.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _clock.RunAsync(walking));

        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task APageThatIsNeverAnswered_IsOneFailureOfTheUnit_SayingHowLongATryMayTake_AndCarryingNothingOfPolly()
    {
        _venue.Stall(times: LandingPace.MostAttempts, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Null(refused.Status);
        Assert.Null(refused.Code);
        Assert.Contains($"within {LandingPace.Said(LandingPace.AttemptTimeout)}", refused.Message, StringComparison.Ordinal);
        Assert.Contains($"any of {LandingPace.MostAttempts} tries", refused.Message, StringComparison.Ordinal);
        Assert.Equal(LandingPace.MostAttempts, KlineRequests().Count());
        Assert.Null(refused.InnerException);
    }

    [Fact]
    public async Task ABanThatSaysNoLength_IsStillABan_AndNamesNoLength()
    {
        _venue.Fail((HttpStatusCode)418, path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Equal((HttpStatusCode)418, refused.Status);
        Assert.Contains("banned this address (418): the landing stopped", refused.Message, StringComparison.Ordinal);
        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task ARefusalThatGivesWordsButNoCode_IsToldInItsWords_WithNoCodeMadeUp()
    {
        _venue.Fail(HttpStatusCode.BadRequest, body: "{\"msg\":\"Something is wrong with the request.\"}", path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Null(refused.Code);
        Assert.Contains("Its words: Something is wrong with the request.", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("(code", refused.Message, StringComparison.Ordinal);
        Assert.Single(KlineRequests());
    }

    [Fact]
    public async Task ARefusalThatGivesNothingToRead_IsToldByItsStatusAlone()
    {
        _venue.Fail(HttpStatusCode.BadRequest, body: "<html>Bad Request</html>", path: Klines);

        var refused = await Assert.ThrowsAsync<BinanceException>(() => _clock.RunAsync(Asked("1d", Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken)));

        Assert.Null(refused.Code);
        Assert.Equal(HttpStatusCode.BadRequest, refused.Status);
        Assert.EndsWith("(400).", refused.Message, StringComparison.Ordinal);
    }

    private sealed class FlakyHandler(HttpMessageHandler inner, int failures) : DelegatingHandler(inner)
    {
        public int Calls { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellation)
        {
            if (request.RequestUri!.AbsolutePath == "/api/v3/time" || request.RequestUri.AbsolutePath == "/api/v3/exchangeInfo" || Calls++ >= failures)
            {
                return base.SendAsync(request, cancellation);
            }

            throw new HttpRequestException("No connection could be made because the target machine actively refused it.");
        }
    }
}
