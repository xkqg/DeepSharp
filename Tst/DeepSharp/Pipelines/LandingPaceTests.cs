// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The figures a landing keeps to, and the words it says a length of time in — the ones a refusal names a wait with, so a
/// person who reads it knows what was asked of them.
/// </summary>
public sealed class LandingPaceTests
{
    [Theory]
    [InlineData(1, "1 second")]
    [InlineData(7, "7 seconds")]
    [InlineData(59, "59 seconds")]
    [InlineData(60, "1 minute")]
    [InlineData(300, "5 minutes")]
    [InlineData(3600, "1 hour")]
    [InlineData(7200, "2 hours")]
    [InlineData(86400, "1 day")]
    [InlineData(259200, "3 days")]
    public void ALengthOfTime_IsSaidInTheLargestUnitItFills_AndPluralOnlyWhenItIsNotOne(int seconds, string said)
    {
        Assert.Equal(said, LandingPace.Said(TimeSpan.FromSeconds(seconds)));
    }

    [Fact]
    public void TheLandingYieldsByConstruction_ItNeverTakesMoreThanHalfOfWhatAMinuteAllows()
    {
        // Two requests of a second each would be four per cent of a minute's budget; the pace is the floor, the brake the ceiling.
        Assert.Equal(TimeSpan.FromMilliseconds(500), LandingPace.PerRequest);
        Assert.Equal(0.5, LandingPace.BrakeShare);
        Assert.Equal(6000, LandingPace.DocumentedLimit);
    }
}
