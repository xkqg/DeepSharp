// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using DeepSharp.Pipelines;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What keeps a landing within the share of a budget it has: a pace between requests, and a wait for the venue's next minute
/// once the address is past half of what a minute allows — the count the venue's own, taken from its answers, and the minute
/// the venue's own, taken from the <c>Date</c> it answered with.
/// </summary>
public sealed class PacerTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 30, TimeSpan.Zero);

    private readonly FakeTimeProvider _clock = new(Now);

    private static VenueReply Told(int? used, string? date = "Tue, 06 Oct 2026 08:00:30 GMT") =>
        new(HttpStatusCode.OK, [], used, RetryAfter: null, Date: date, Uuid: null);

    private static async Task Completes(Task waiting) => await waiting.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);

    [Fact]
    public async Task AnAddressPastHalfOfWhatAMinuteAllows_WaitsForTheVenuesNextMinute_AndASecondBeyondIt()
    {
        var pace = new Pacer(_clock);

        pace.Allowing(100);
        pace.Told(Told(used: 60));

        var waiting = pace.BeforeAsync(TestContext.Current.CancellationToken);

        Assert.False(waiting.IsCompleted);

        _clock.Advance(TimeSpan.FromSeconds(30));

        Assert.False(waiting.IsCompleted, "It was released before the venue's next minute had begun.");

        _clock.Advance(TimeSpan.FromSeconds(1));

        await Completes(waiting);
    }

    [Fact]
    public async Task AnAnswerThatSaysNothingOfWhatIsSpent_LeavesTheCountAsItWas_SoTheBrakeStillHolds()
    {
        var pace = new Pacer(_clock);

        pace.Allowing(100);
        pace.Told(Told(used: 60));
        pace.Told(Told(used: null));

        var waiting = pace.BeforeAsync(TestContext.Current.CancellationToken);

        Assert.False(waiting.IsCompleted);

        _clock.Advance(TimeSpan.FromSeconds(31));

        await Completes(waiting);
    }

    [Fact]
    public async Task AnAddressBelowHalfOfWhatAMinuteAllows_IsNotHeldBack()
    {
        var pace = new Pacer(_clock);

        pace.Allowing(100);
        pace.Told(Told(used: 49));

        await Completes(pace.BeforeAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task AnAnswerWithNoDate_LeavesTheVenuesClockAsItWas_SoThereIsNoMinuteToWaitFor()
    {
        var pace = new Pacer(_clock);

        pace.Allowing(100);
        pace.Told(Told(used: 90, date: null));

        await Completes(pace.BeforeAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task TwoRequests_AreNeverCloserThanThePace()
    {
        var pace = new Pacer(_clock);

        await Completes(pace.BeforeAsync(TestContext.Current.CancellationToken));

        var waiting = pace.BeforeAsync(TestContext.Current.CancellationToken);

        Assert.False(waiting.IsCompleted);

        _clock.Advance(LandingPace.PerRequest);

        await Completes(waiting);
    }
}
