// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using Polly;
using Polly.Retry;
using Polly.Timeout;

namespace DeepSharp.Pipelines;

/// <summary>
/// One unit of work asked of the venue and carried to its end: paced, tried again as long as trying again can help, and
/// failed once.
/// </summary>
/// <remarks>
/// <para>
/// A unit is one request at one place — one page at one cursor — and it is the unit that is retried, never the walk: a
/// refusal that was met by skipping the unit and marching on is how one rejected call becomes a storm, so the same unit is
/// asked again behind a wait that rises, by as long as the venue says when it says, and only up to a ceiling. A venue that
/// asks for more than a landing waits has asked for a stop. What says stop — a firewall, a ban, a region — stops at once.
/// What the pace asks a try to wait for is no part of the try: the time a try may take begins when its request is sent.
/// What the venue refused in its own words is carried in them, and is not asked again.
/// </para>
/// <para>
/// The retries are Polly's, which is the retry and the timeout of a try; nothing of it is visible outside this package. It is
/// built on the clock a landing is handed, so a test moves time and waits for none.
/// </para>
/// </remarks>
/// <param name="wire">The venue's dialect.</param>
/// <param name="pace">What keeps the landing within its share of the budget.</param>
/// <param name="clock">The clock every wait and every timeout runs on.</param>
internal sealed class Courier(VenueWire wire, Pacer pace, TimeProvider clock)
{
    // The tries: the same unit is asked again behind a wait that rises, for as long as trying again can help.
    private readonly ResiliencePipeline _tries = new ResiliencePipelineBuilder { TimeProvider = clock }
        .AddRetry(new RetryStrategyOptions
        {
            MaxRetryAttempts = LandingPace.MostAttempts - 1,
            BackoffType = DelayBackoffType.Exponential,
            Delay = LandingPace.FirstBackoff,
            UseJitter = false,
            ShouldHandle = arguments => ValueTask.FromResult(IsWorthAnotherTry(arguments.Outcome.Exception)),
            DelayGenerator = arguments => ValueTask.FromResult<TimeSpan?>(WaitBefore(arguments.AttemptNumber, arguments.Outcome.Exception)),
        })
        .Build();

    // What one try may take: the venue's answer and nothing before it. The wait the pace asks for is not a try, so no
    // timeout runs while it lasts — a timed wait would spend a try for every ten seconds of it.
    private readonly ResiliencePipeline _answer = new ResiliencePipelineBuilder { TimeProvider = clock }
        .AddTimeout(LandingPace.AttemptTimeout)
        .Build();

    /// <summary>Asks the venue, and reads what it said, until it is answered or the unit is given up.</summary>
    /// <typeparam name="T">What is read from the answer.</typeparam>
    /// <param name="pathAndQuery">What is asked.</param>
    /// <param name="what">What the answer is meant to be, as a refusal names it.</param>
    /// <param name="read">Reads the answer; refuses what is not it with a <see cref="PageFormatException"/>.</param>
    /// <param name="cancellation">Ends the unit, a wait included.</param>
    /// <returns>What was read.</returns>
    /// <exception cref="BinanceException">The venue said stop, asked for more than a landing waits, refused in its own words, or could not be made to answer.</exception>
    public async Task<T> AskAsync<T>(string pathAndQuery, string what, Func<VenueReply, T> read, CancellationToken cancellation)
    {
        try
        {
            return await _tries.ExecuteAsync(
                async token =>
                {
                    await pace.BeforeAsync(token);

                    return await _answer.ExecuteAsync(
                        async attempt =>
                        {
                            var reply = await wire.GetAsync(pathAndQuery, attempt);

                            pace.Told(reply);

                            return reply.Status == HttpStatusCode.OK ? read(reply) : throw new VenueStatusException(reply.Status, reply.RetryAfter, reply.Words());
                        },
                        token);
                },
                cancellation);
        }
        catch (Exception failure) when (failure is VenueStatusException or PageFormatException or HttpRequestException or TimeoutRejectedException)
        {
            throw Refusal(failure, pathAndQuery, what);
        }
    }

    private static bool IsWorthAnotherTry(Exception? failure) => failure switch
    {
        VenueStatusException { Retryable: true } busy => busy.Wait is not { } wait || wait <= LandingPace.MostWait,
        PageFormatException or HttpRequestException or TimeoutRejectedException => true,
        _ => false,
    };

    // As long as the venue asks, when it asks; else the wait that doubles from one retry to the next.
    private static TimeSpan WaitBefore(int retry, Exception? failure)
    {
        var backoff = TimeSpan.FromTicks(LandingPace.FirstBackoff.Ticks << retry);

        return failure is VenueStatusException { Wait: { } asked } && asked > backoff ? asked : backoff;
    }

    private static BinanceException Refusal(Exception failure, string asked, string what) => failure switch
    {
        VenueStatusException { Status: (HttpStatusCode)418 } banned => new BinanceException(
            $"Binance has banned this address (418){(banned.Wait is { } ban ? $" for {LandingPace.Said(ban)}" : string.Empty)}: the landing stopped, and nothing more is sent. {Words(banned)}".TrimEnd(),
            banned.Status,
            banned.Words.Code),
        VenueStatusException { Stops: true } stopped => new BinanceException(
            $"Binance stopped the landing with {(int)stopped.Status} ({stopped.Status switch { HttpStatusCode.Forbidden => "its firewall, which can mean a rate limit was broken", _ => "it does not answer from where this is run" }}): the landing stopped, and nothing more is sent. {Words(stopped)}".TrimEnd(),
            stopped.Status,
            stopped.Words.Code),
        VenueStatusException { Wait: { } wait } waiting when wait > LandingPace.MostWait => new BinanceException(
            $"Binance asked this address to wait {LandingPace.Said(wait)} ({(int)waiting.Status}), which is more than the {LandingPace.Said(LandingPace.MostWait)} a landing waits: the landing stopped. {Words(waiting)}".TrimEnd(),
            waiting.Status,
            waiting.Words.Code),
        VenueStatusException { Retryable: true } tired => new BinanceException(
            $"Binance did not take {asked} in {LandingPace.MostAttempts} tries; its last answer was {(int)tired.Status}. {Words(tired)}".TrimEnd(),
            tired.Status,
            tired.Words.Code),
        VenueStatusException refused => new BinanceException(
            $"Binance refused {asked} ({(int)refused.Status}). {Words(refused)}".TrimEnd(),
            refused.Status,
            refused.Words.Code),
        PageFormatException odd => new BinanceException($"Binance answered {asked} with something that is not {what}, in {LandingPace.MostAttempts} tries: {odd.Message}.", HttpStatusCode.OK, null, odd),
        TimeoutRejectedException => new BinanceException($"Binance did not answer {asked} within {LandingPace.Said(LandingPace.AttemptTimeout)} in any of {LandingPace.MostAttempts} tries.", null, null),
        _ => new BinanceException($"No answer came to {asked} in {LandingPace.MostAttempts} tries: {failure.Message}", null, null, failure),
    };

    private static string Words(VenueStatusException failure) => failure.Words.Message is { } message
        ? $"Its words: {message}{(failure.Words.Code is { } code ? $" (code {code})" : string.Empty)}"
        : string.Empty;
}
