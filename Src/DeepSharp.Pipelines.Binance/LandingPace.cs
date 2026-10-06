// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// How a landing keeps to a budget it shares: how often it asks, when it holds back, how long it waits and for how long it
/// is patient.
/// </summary>
/// <remarks>
/// <para>
/// The venue counts weight for an address, by the wall-clock minute, and other programs on the same address spend from the
/// same count. A landing is a one-off with no deadline, so it yields by construction: every request is paced, the venue's
/// own count of what has been spent is read from each answer, and the landing waits for the next minute once the address is
/// past half of what the minute allows, whoever spent it.
/// </para>
/// <para>
/// Each of these is named once, here, because a number written where it is used is a number somebody changes there.
/// </para>
/// </remarks>
internal static class LandingPace
{
    /// <summary>The least time between one request and the next, every try included: at most four per cent of a minute's budget for the pages.</summary>
    public static readonly TimeSpan PerRequest = TimeSpan.FromMilliseconds(500);

    /// <summary>The share of a minute's weight the address may have spent before the landing waits for the next minute.</summary>
    public const double BrakeShare = 0.5;

    /// <summary>What the venue allows an address a minute until it says otherwise: the figure its documentation gives.</summary>
    public const int DocumentedLimit = 6000;

    /// <summary>How long after a candle closes it is left to settle before it is taken.</summary>
    public static readonly TimeSpan SettleAfterClose = TimeSpan.FromSeconds(60);

    /// <summary>How long one try may take: the time the venue itself gives a request.</summary>
    public static readonly TimeSpan AttemptTimeout = TimeSpan.FromSeconds(10);

    /// <summary>How many times a unit of work is tried, the first included.</summary>
    public const int MostAttempts = 5;

    /// <summary>How long the first retry waits, when the venue says nothing; each retry waits twice as long as the one before.</summary>
    public static readonly TimeSpan FirstBackoff = TimeSpan.FromSeconds(1);

    /// <summary>The longest a landing waits because the venue asked it to: more is a stop, since a ban can last for days.</summary>
    public static readonly TimeSpan MostWait = TimeSpan.FromMinutes(2);

    /// <summary>A length of time in the words a person says it in.</summary>
    /// <param name="span">The length.</param>
    /// <returns>Such as <c>3 days</c>, <c>5 minutes</c> or <c>7 seconds</c>.</returns>
    public static string Said(TimeSpan span) => span switch
    {
        { TotalDays: >= 1 } => Counted(span.TotalDays, "day"),
        { TotalHours: >= 1 } => Counted(span.TotalHours, "hour"),
        { TotalMinutes: >= 1 } => Counted(span.TotalMinutes, "minute"),
        _ => Counted(span.TotalSeconds, "second"),
    };

    private static string Counted(double count, string unit)
    {
        var rounded = Math.Round(count);

        return string.Create(CultureInfo.InvariantCulture, $"{rounded} {unit}{(rounded == 1 ? string.Empty : "s")}");
    }
}
