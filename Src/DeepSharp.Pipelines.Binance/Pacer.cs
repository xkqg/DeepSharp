// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What keeps a landing from spending a budget it shares: no two requests closer than the fixed pace, and a wait for the
/// venue's next minute once the address is past half of what a minute allows, whoever spent it.
/// </summary>
/// <remarks>
/// <para>
/// The count is the venue's own, read from each answer, and the minute is the venue's too, taken from the <c>Date</c> it
/// answered with and moved on by this clock since; a landing that trusted its own clock would be wrong by the second or
/// two the two differ by. All waiting runs on the one clock a landing is handed, so a test holds it.
/// </para>
/// <para>
/// A landing owns one of these and nothing else holds it. It has no thread and no timer of its own: it waits by awaiting, and
/// a landing that ended has nothing of it left running.
/// </para>
/// </remarks>
/// <param name="clock">The clock every wait runs on.</param>
internal sealed class Pacer(TimeProvider clock)
{
    private DateTimeOffset? _lastAsked;
    private DateTimeOffset? _venueSaid;
    private DateTimeOffset _heardAt;
    private int _used;
    private int _limit = LandingPace.DocumentedLimit;

    /// <summary>Takes what an address may spend a minute, as the venue says it, in place of what its documentation gives.</summary>
    /// <param name="limit">The weight.</param>
    public void Allowing(int limit) => _limit = limit;

    /// <summary>Waits as long as the pace and the venue's budget ask, before a request is made.</summary>
    /// <param name="cancellation">Ends the wait.</param>
    /// <returns>When the request may be made.</returns>
    public async Task BeforeAsync(CancellationToken cancellation)
    {
        if (_used >= _limit * LandingPace.BrakeShare && _venueSaid is { } said)
        {
            var venueNow = said + (clock.GetUtcNow() - _heardAt);
            var nextMinute = DateTimeOffset.FromUnixTimeSeconds(((venueNow.ToUnixTimeSeconds() / 60) + 1) * 60);

            // A second past the minute, for the second the venue's own date was cut to.
            await Task.Delay(nextMinute + TimeSpan.FromSeconds(1) - venueNow, clock, cancellation);

            _used = 0;
        }

        if (_lastAsked is { } asked && asked + LandingPace.PerRequest - clock.GetUtcNow() is { Ticks: > 0 } wait)
        {
            await Task.Delay(wait, clock, cancellation);
        }

        _lastAsked = clock.GetUtcNow();
    }

    /// <summary>Takes what an answer says of what has been spent, and what the venue's own clock says it is.</summary>
    /// <param name="reply">The answer.</param>
    public void Told(VenueReply reply)
    {
        _used = reply.Used ?? _used;

        if (reply.DateValue is { } date)
        {
            _venueSaid = date;
            _heardAt = clock.GetUtcNow();
        }
    }
}
