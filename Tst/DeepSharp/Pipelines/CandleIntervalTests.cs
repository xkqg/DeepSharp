// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Where the venue opens a candle, for each length it documents, and how a window of them is counted: a week on a Monday, a
/// month on the first — however many days it has — and everything else at a multiple of its length counted from the epoch.
/// </summary>
public sealed class CandleIntervalTests
{
    private static DateTime Utc(int year, int month, int day) => new(year, month, day, 0, 0, 0, DateTimeKind.Utc);

    private static CandleInterval Interval(string venue) => CandleInterval.Named(venue)!;

    [Theory]
    [InlineData("1h", 2024, 1, 1, 0, 2024, 1, 1, 1)]
    [InlineData("1d", 2024, 1, 31, 0, 2024, 2, 1, 0)]
    [InlineData("3d", 2024, 1, 2, 0, 2024, 1, 5, 0)]
    [InlineData("1w", 2024, 1, 1, 0, 2024, 1, 8, 0)]
    [InlineData("1w", 2024, 12, 30, 0, 2025, 1, 6, 0)]
    [InlineData("1M", 2024, 1, 1, 0, 2024, 2, 1, 0)]
    [InlineData("1M", 2024, 2, 1, 0, 2024, 3, 1, 0)]
    [InlineData("1M", 2023, 2, 1, 0, 2023, 3, 1, 0)]
    [InlineData("1M", 2024, 12, 1, 0, 2025, 1, 1, 0)]
    public void TheCandleAfter_OpensWhereTheNextOneDoes_WhateverTheLengthOfThisOne(string venue, int y, int m, int d, int h, int ny, int nm, int nd, int nh)
    {
        var open = Utc(y, m, d).AddHours(h);

        Assert.Equal(Utc(ny, nm, nd).AddHours(nh), Interval(venue).After(open));
    }

    [Theory]
    [InlineData("1m")]
    [InlineData("1h")]
    [InlineData("1d")]
    [InlineData("3d")]
    [InlineData("1w")]
    [InlineData("1M")]
    public void StepingFromOneOpeningToTheNext_CountsAsManyCandlesAsTheWindowHolds(string venue)
    {
        var interval = Interval(venue);
        var from = interval.Around(Utc(2023, 2, 20)).After;
        var to = from;

        for (var step = 0; step < 40; step++)
        {
            Assert.True(interval.Opens(to), $"{venue}: {to:O} is not where a candle opens");

            to = interval.After(to);
        }

        Assert.Equal(40, interval.Candles(from, to));
    }

    [Fact]
    public void AMonthIsSpelledOneThingToTheVenueAndAnotherToAFileSystem_ThatDoesNotTellTheCasesApart()
    {
        Assert.Equal("1M", Interval("1M").Venue);
        Assert.Equal("1mo", Interval("1M").File);
        Assert.Equal("1m", Interval("1m").File);
        Assert.Null(CandleInterval.Named("1D"));
        Assert.Equal(CandleInterval.All.Count, CandleInterval.All.Select(each => each.File.ToUpperInvariant()).Distinct().Count());
    }
}
