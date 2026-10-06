// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.Http.Headers;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The venue's dialect, held apart from any walk: what a request calls itself and asks for, how the headers that say what
/// has been spent and how long to wait are read, and how each kind of answer is read — each as far as it goes, and an answer
/// that is not what it should be refused by name rather than thrown out of the package as something nobody named.
/// </summary>
public sealed class VenueWireTests
{
    private static readonly DateTimeOffset VenueSaid = new(2026, 10, 6, 8, 0, 1, TimeSpan.Zero);

    private static VenueReply Reply(string body) => new(HttpStatusCode.OK, Encoding.UTF8.GetBytes(body), Used: null, RetryAfter: null, Date: null, Uuid: null);

    private static string Limits(params string[] entries) => $$"""{"rateLimits":[{{string.Join(',', entries)}}]}""";

    private static async Task<VenueReply> AskedAsync(Answering venue)
    {
        using var client = new HttpClient(venue) { BaseAddress = new Uri("https://venue.test") };

        return await new VenueWire(client).GetAsync("/api/v3/time", TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TheRequest_IsAGet_NamingItsPackageAndWhereItComesFrom_AndAsksForJson()
    {
        using var venue = new Answering();

        await AskedAsync(venue);

        Assert.Equal("GET", venue.Method);
        Assert.Equal("/api/v3/time", venue.Target);
        Assert.Matches("""^DeepSharp\.Pipelines\.Binance/\d+\.\d+\.\d+ \(\+https://github\.com/xkqg/DeepSharp\)$""", venue.UserAgent);
        Assert.Equal("application/json", venue.Accept);
    }

    [Fact]
    public async Task WhatTheVenueSaysOfItsBudgetAndItsClock_IsKeptWithTheAnswer_AsItCame()
    {
        using var venue = new Answering(HttpStatusCode.OK, """{"serverTime":1}""", response =>
        {
            response.Headers.TryAddWithoutValidation("x-mbx-used-weight-1m", "42");
            response.Headers.TryAddWithoutValidation("x-mbx-uuid", "5e7d0c6a-3b2f-4a8e-9d11-0a1b2c3d4e5f");
            response.Headers.Date = VenueSaid;
        });

        var reply = await AskedAsync(venue);

        Assert.Equal(HttpStatusCode.OK, reply.Status);
        Assert.Equal("""{"serverTime":1}""", Encoding.UTF8.GetString(reply.Body));
        Assert.Equal(42, reply.Used);
        Assert.Equal("5e7d0c6a-3b2f-4a8e-9d11-0a1b2c3d4e5f", reply.Uuid);
        Assert.Equal("Tue, 06 Oct 2026 08:00:01 GMT", reply.Date);
        Assert.Null(reply.RetryAfter);
    }

    [Fact]
    public async Task WhatTheVenueDidNotSay_IsNothing_NotAGuess()
    {
        using var venue = new Answering();

        var reply = await AskedAsync(venue);

        Assert.Null(reply.Used);
        Assert.Null(reply.Uuid);
        Assert.Null(reply.Date);
        Assert.Null(reply.RetryAfter);
    }

    [Theory]
    [InlineData("many")]
    [InlineData("-5")]
    [InlineData("4.5")]
    [InlineData("")]
    public async Task AWeightThatIsNoWholeNumberOfNoughtOrMore_IsNothing(string said)
    {
        using var venue = new Answering(HttpStatusCode.OK, "{}", response => response.Headers.TryAddWithoutValidation("x-mbx-used-weight-1m", said));

        Assert.Null((await AskedAsync(venue)).Used);
    }

    [Fact]
    public async Task ARetryAfterInSeconds_IsHowLongTheVenueAsksToBeLeftAlone()
    {
        using var venue = new Answering(HttpStatusCode.TooManyRequests, "{}", response => response.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(120)));

        Assert.Equal(TimeSpan.FromSeconds(120), (await AskedAsync(venue)).RetryAfter);
    }

    [Fact]
    public async Task ARetryAfterAsAMoment_IsHowLongItIsFromTheVenuesOwnDate()
    {
        using var venue = new Answering(HttpStatusCode.TooManyRequests, "{}", response =>
        {
            response.Headers.Date = VenueSaid;
            response.Headers.RetryAfter = new RetryConditionHeaderValue(VenueSaid.AddMinutes(3));
        });

        Assert.Equal(TimeSpan.FromMinutes(3), (await AskedAsync(venue)).RetryAfter);
    }

    [Fact]
    public async Task ARetryAfterAsAMoment_FromAVenueThatSaidNoDate_IsHowLongItIsFromNow()
    {
        var far = DateTimeOffset.UtcNow.AddYears(50);

        using var venue = new Answering(HttpStatusCode.TooManyRequests, "{}", response => response.Headers.RetryAfter = new RetryConditionHeaderValue(far));

        Assert.InRange((await AskedAsync(venue)).RetryAfter!.Value, TimeSpan.FromDays(365 * 49), TimeSpan.FromDays(365 * 51));
    }

    [Fact]
    public void TheAddresses_AreTheVenuesOwn_ASymbolEscaped_AndEveryPageAskedForTheMostItGives()
    {
        Assert.Equal("/api/v3/time", VenueWire.Time);
        Assert.Equal("/api/v3/exchangeInfo?symbol=BTCEUR", VenueWire.Information("BTCEUR"));
        Assert.Equal("/api/v3/exchangeInfo?symbol=BTC%20EUR%26x", VenueWire.Information("BTC EUR&x"));
        Assert.Equal(
            "/api/v3/klines?symbol=BTCEUR&interval=1d&startTime=1704067200000&endTime=1704153599999&limit=1000",
            VenueWire.Klines("BTCEUR", "1d", 1704067200000, 1704153599999));
    }

    [Fact]
    public void TheVenuesClock_IsTheMomentItsAnswerSays_ToTheMillisecond_InUniversalTime()
    {
        var moment = new DateTime(2026, 10, 6, 8, 0, 1, 123, DateTimeKind.Utc);

        var milliseconds = new DateTimeOffset(moment).ToUnixTimeMilliseconds();

        var read = VenueWire.ServerTime(Reply($$"""{"serverTime":{{milliseconds}} }"""));

        Assert.Equal(moment, read);
        Assert.Equal(DateTimeKind.Utc, read.Kind);
    }

    [Theory]
    [InlineData("not json", "it is not JSON")]
    [InlineData("", "it is not JSON")]
    [InlineData("[]", "serverTime")]
    [InlineData("{}", "serverTime")]
    [InlineData("""{"serverTime":"1"}""", "serverTime")]
    [InlineData("""{"serverTime":1.5}""", "serverTime")]
    [InlineData("""{"serverTime":9223372036854775807}""", "not one")]
    public void AnAnswerThatIsNotTheVenuesClock_IsRefused_SayingWhatItIs(string body, string said)
    {
        var refused = Assert.Throws<PageFormatException>(() => VenueWire.ServerTime(Reply(body)));

        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheWeightAnAddressMaySpendAMinute_IsTheOneLimitThatCountsRequestWeightByTheMinute()
    {
        var orders = """{"rateLimitType":"ORDERS","interval":"SECOND","intervalNum":10,"limit":100}""";
        var weight = """{"rateLimitType":"REQUEST_WEIGHT","interval":"MINUTE","intervalNum":1,"limit":1200}""";

        Assert.Equal(1200, VenueWire.WeightLimit(Reply(Limits(orders, weight))));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("""{"rateLimits":5}""")]
    [InlineData("""{"rateLimits":[]}""")]
    [InlineData("""{"rateLimits":[5,"x",null,[]]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"ORDERS","interval":"SECOND","intervalNum":10,"limit":100}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"SECOND","intervalNum":1,"limit":1200}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"MINUTE","intervalNum":2,"limit":1200}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"MINUTE","intervalNum":1,"limit":0}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"MINUTE","intervalNum":1,"limit":"1200"}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT","interval":"MINUTE","intervalNum":1.5,"limit":1200}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":5,"interval":null,"intervalNum":1,"limit":1200}]}""")]
    [InlineData("""{"rateLimits":[{"rateLimitType":"REQUEST_WEIGHT"}]}""")]
    [InlineData("not json")]
    [InlineData("")]
    public void AnAnswerThatNamesNoWeightPerMinute_SaysNothing_AndNeverThrows(string body)
    {
        Assert.Null(VenueWire.WeightLimit(Reply(body)));
    }

    // A venue that answers every request the way a test says, and remembers how the request looked.
    private sealed class Answering(HttpStatusCode status = HttpStatusCode.OK, string body = "{}", Action<HttpResponseMessage>? answer = null) : HttpMessageHandler
    {
        public string? Method { get; private set; }

        public string? Target { get; private set; }

        public string? UserAgent { get; private set; }

        public string? Accept { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellation)
        {
            Method = request.Method.Method;
            Target = request.RequestUri!.PathAndQuery;
            UserAgent = request.Headers.UserAgent.ToString();
            Accept = request.Headers.Accept.ToString();

            var response = new HttpResponseMessage(status) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(body)) };

            answer?.Invoke(response);

            return Task.FromResult(response);
        }
    }
}
