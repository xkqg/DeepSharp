// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What is asked of the venue: a symbol, an interval it names, and a window closed at both ends of the interval's own grid,
/// half open so that two windows side by side tile. A request that could not be answered as it was meant is refused where it
/// is written, and nothing of it is sent until it is.
/// </summary>
public sealed class BinanceCandlesTests
{
    private static readonly DateTime Epoch = DateTime.UnixEpoch;

    // The intervals the venue documents, each with how long a candle is — written here by hand, not read from the package.
    private static readonly (string Venue, TimeSpan Span)[] Fixed =
    [
        ("1s", TimeSpan.FromSeconds(1)), ("1m", TimeSpan.FromMinutes(1)), ("3m", TimeSpan.FromMinutes(3)), ("5m", TimeSpan.FromMinutes(5)),
        ("15m", TimeSpan.FromMinutes(15)), ("30m", TimeSpan.FromMinutes(30)), ("1h", TimeSpan.FromHours(1)), ("2h", TimeSpan.FromHours(2)),
        ("4h", TimeSpan.FromHours(4)), ("6h", TimeSpan.FromHours(6)), ("8h", TimeSpan.FromHours(8)), ("12h", TimeSpan.FromHours(12)),
        ("1d", TimeSpan.FromDays(1)), ("3d", TimeSpan.FromDays(3)),
    ];

    private static readonly string[] Documented = [.. Fixed.Select(each => each.Venue), "1w", "1M"];

    public static TheoryData<string> EveryDocumentedInterval() => [.. Documented];

    private static DateTime Utc(int year, int month, int day, int hour = 0, int minute = 0, int second = 0) =>
        new(year, month, day, hour, minute, second, DateTimeKind.Utc);

    // The last instant at or before the first of 2024 that stands on an interval's grid: a multiple of the candle counted
    // from the epoch, a Monday for a week, the first of a month for a month.
    private static DateTime OnTheGrid(string interval) => interval switch
    {
        "1w" => Utc(2024, 1, 1),
        "1M" => Utc(2024, 1, 1),
        _ => Epoch.AddTicks((Utc(2024, 1, 1) - Epoch).Ticks / Fixed.Single(each => each.Venue == interval).Span.Ticks * Fixed.Single(each => each.Venue == interval).Span.Ticks),
    };

    private static DateTime After(string interval, DateTime from, int candles) => interval switch
    {
        "1w" => from.AddDays(7 * candles),
        "1M" => from.AddMonths(candles),
        _ => from + (Fixed.Single(each => each.Venue == interval).Span * candles),
    };

    [Fact]
    public void AWindow_KeepsWhatItWasAskedFor_AsTheVenueSpellsIt()
    {
        var candles = new BinanceCandles("btceur", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));

        Assert.Equal("BTCEUR", candles.Symbol);
        Assert.Equal("1d", candles.Interval);
        Assert.Equal(Utc(2024, 1, 1), candles.From);
        Assert.Equal(Utc(2024, 2, 1), candles.To);
        Assert.Equal(DateTimeKind.Utc, candles.From.Kind);
        Assert.Equal(DateTimeKind.Utc, candles.To.Kind);
    }

    [Fact]
    public void AMonth_IsSpelledAsTheVenueSpellsIt_NeverLikeAMinute()
    {
        Assert.Equal("1M", new BinanceCandles("BTCEUR", "1M", Utc(2024, 1, 1), Utc(2024, 4, 1)).Interval);
        Assert.Equal("1m", new BinanceCandles("BTCEUR", "1m", Utc(2024, 1, 1), Utc(2024, 1, 2)).Interval);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("BTC/EUR")]
    [InlineData("BTC EUR")]
    [InlineData("BTCEUR&limit=1")]
    [InlineData("BTCEUR\n")]
    [InlineData("€UR")]
    [InlineData("ABCDEFGHIJKLMNOPQRSTU")]
    [InlineData("@")]
    [InlineData("[")]
    [InlineData("`")]
    [InlineData("{")]
    [InlineData(":")]
    [InlineData("A,B")]
    public void ASymbolTheVenueCouldNotMean_IsRefused_BeforeAnythingIsSent(string symbol)
    {
        var refused = Assert.Throws<ArgumentException>(() => new BinanceCandles(symbol, "1d", Utc(2024, 1, 1), Utc(2024, 2, 1)));

        Assert.Equal("symbol", refused.ParamName);
    }

    [Theory]
    [InlineData("btceur", "BTCEUR")]
    [InlineData("1000SATSUSDT", "1000SATSUSDT")]
    [InlineData("1inchbtc", "1INCHBTC")]
    [InlineData("A.B-C_D", "A.B-C_D")]
    [InlineData("abcdefghijklmnopqrst", "ABCDEFGHIJKLMNOPQRST")]
    public void ASymbolOfLettersDigitsDotsHyphensAndUnderscores_IsTaken_InCapitals(string symbol, string kept)
    {
        Assert.Equal(kept, new BinanceCandles(symbol, "1d", Utc(2024, 1, 1), Utc(2024, 2, 1)).Symbol);
    }

    [Fact]
    public void NoSymbol_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new BinanceCandles(null!, "1d", Utc(2024, 1, 1), Utc(2024, 2, 1)));
    }

    [Theory]
    [InlineData("")]
    [InlineData("1D")]
    [InlineData("2d")]
    [InlineData("1y")]
    [InlineData("60m")]
    [InlineData("1 m")]
    [InlineData("1mo")]
    [InlineData("1W")]
    [InlineData("1H")]
    public void AnIntervalTheVenueDoesNotDocument_IsRefused_NamingTheOnesItDoes(string interval)
    {
        var refused = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", interval, Utc(2024, 1, 1), Utc(2024, 2, 1)));

        Assert.Equal("interval", refused.ParamName);
        Assert.Contains("1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NoInterval_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new BinanceCandles("BTCEUR", null!, Utc(2024, 1, 1), Utc(2024, 2, 1)));
    }

    [Theory]
    [MemberData(nameof(EveryDocumentedInterval))]
    public void AWindowOnTheGridOfEveryInterval_IsTaken(string interval)
    {
        var from = OnTheGrid(interval);
        var to = After(interval, from, 3);
        var candles = new BinanceCandles("BTCEUR", interval, from, to);

        Assert.Equal(from, candles.From);
        Assert.Equal(to, candles.To);
    }

    [Theory]
    [MemberData(nameof(EveryDocumentedInterval))]
    public void AStartOffTheGrid_IsRefused_ForEveryInterval(string interval)
    {
        var from = OnTheGrid(interval);
        var refused = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", interval, from.AddMilliseconds(1), After(interval, from, 3)));

        Assert.Equal("from", refused.ParamName);
        Assert.Contains(interval, refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(EveryDocumentedInterval))]
    public void AnEndOffTheGrid_IsRefused_ForEveryInterval(string interval)
    {
        var from = OnTheGrid(interval);
        var refused = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", interval, from, After(interval, from, 3).AddMilliseconds(1)));

        Assert.Equal("to", refused.ParamName);
    }

    [Fact]
    public void ThreeDaysAreCountedFromTheEpoch_AWeekStartsOnAMonday_AMonthOnItsFirst()
    {
        // The venue opens a three-day candle every third day counted from the epoch, a week candle on a Monday and a month
        // candle on the first, so a start that is only a midnight is not enough for any of them.
        Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "3d", Utc(2024, 1, 1), Utc(2024, 1, 10)));
        Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "1w", Utc(2024, 1, 4), Utc(2024, 1, 11)));
        Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "1M", Utc(2024, 1, 2), Utc(2024, 2, 1)));

        Assert.Equal(Utc(2023, 12, 31), new BinanceCandles("BTCEUR", "3d", Utc(2023, 12, 31), Utc(2024, 1, 9)).From);
        Assert.Equal(Utc(2024, 1, 8), new BinanceCandles("BTCEUR", "1w", Utc(2024, 1, 8), Utc(2024, 1, 22)).From);
        Assert.Equal(Utc(2024, 2, 1), new BinanceCandles("BTCEUR", "1M", Utc(2024, 2, 1), Utc(2024, 3, 1)).From);
    }

    [Fact]
    public void AWindowThatEndsWhereItBegins_OrBefore_IsRefused()
    {
        var same = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 1, 1)));
        var backwards = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "1d", Utc(2024, 2, 1), Utc(2024, 1, 1)));

        Assert.Equal("to", same.ParamName);
        Assert.Equal("to", backwards.ParamName);
    }

    [Fact]
    public void AMomentThatSaysNothingOfItsZone_IsUniversal_NeverTheMachinesOwn()
    {
        // An unspecified moment is the same instant on every machine: universal, as a pipeline's own are.
        var unspecified = new DateTime(2024, 1, 1, 2, 0, 0, DateTimeKind.Unspecified);
        var candles = new BinanceCandles("BTCEUR", "1h", unspecified, new DateTime(2024, 1, 1, 5, 0, 0, DateTimeKind.Unspecified));

        Assert.Equal(Utc(2024, 1, 1, 2), candles.From);
        Assert.Equal(Utc(2024, 1, 1, 5), candles.To);
    }

    [Fact]
    public void ALocalMoment_IsTheInstantItStandsFor()
    {
        var instant = Utc(2024, 1, 1, 2);
        var candles = new BinanceCandles("BTCEUR", "1h", instant.ToLocalTime(), Utc(2024, 1, 1, 5).ToLocalTime());

        Assert.Equal(instant, candles.From);
        Assert.Equal(Utc(2024, 1, 1, 5), candles.To);
    }

    [Fact]
    public void AWindowOfMoreCandlesThanOneLandingMayAsk_IsRefused_NamingHowManyPagesItWouldTake()
    {
        // One second candles for a leap year are 31,623 pages of a thousand: hours of a budget other programs may be sharing.
        var refused = Assert.Throws<ArgumentException>(() => new BinanceCandles("BTCEUR", "1s", Utc(2024, 1, 1), Utc(2025, 1, 1)));

        Assert.Equal("to", refused.ParamName);
        Assert.Contains("31,623", refused.Message, StringComparison.Ordinal);
        Assert.Contains(BinanceCandles.MostPages.ToString("N0", CultureInfo.InvariantCulture), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AYearOfMinutes_IsWithinWhatOneLandingMayAsk()
    {
        var candles = new BinanceCandles("BTCEUR", "1m", Utc(2024, 1, 1), Utc(2025, 1, 1));

        Assert.Equal(Utc(2025, 1, 1), candles.To);
    }

    [Fact]
    public void TheWindowsOfTwoAdjacentLandings_ShareNoCandle_AndNameDifferentFiles()
    {
        // Half open: the first ends where the second begins, and a candle that opens at that instant is the second's.
        var first = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 1, 2));
        var second = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 2), Utc(2024, 1, 3));

        Assert.Equal(first.To, second.From);
        Assert.NotEqual(first.LandingName, second.LandingName);
    }

    [Fact]
    public void EveryDocumentedInterval_NamesADistinctFile_UnderOrdinalIgnoreCase()
    {
        // The venue spells a minute 1m and a month 1M, and a file system that folds case holds one file for both: a month
        // landing read as a minute landing. The name spells a month 1mo, as the venue's own archive does.
        var names = Documented
            .Select(interval => new BinanceCandles("BTCEUR", interval, OnTheGrid(interval), After(interval, OnTheGrid(interval), 3)).LandingName)
            .ToArray();

        Assert.Equal(16, names.Length);
        Assert.Equal(names.Length, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    [Fact]
    public void AMonthLandingIsNeverReadAsAMinuteLanding()
    {
        var minutes = new BinanceCandles("BTCEUR", "1m", Utc(2024, 1, 1), Utc(2024, 1, 2));
        var months = new BinanceCandles("BTCEUR", "1M", Utc(2024, 1, 1), Utc(2024, 2, 1));

        Assert.NotEqual(minutes.LandingName, months.LandingName, StringComparer.OrdinalIgnoreCase);
        Assert.Contains("-1mo-", months.LandingName, StringComparison.Ordinal);
    }

    [Fact]
    public void TheLandingIsNamedByWhatWasAsked_RelativeAndWithoutAColon()
    {
        var candles = new BinanceCandles("btceur", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));

        Assert.Equal("BTCEUR-1d-20240101T000000Z-20240201T000000Z.csv", candles.LandingName);
        Assert.Equal("BTCEUR-1d-20240101T000000Z-20240201T000000Z.manifest.json", candles.ManifestName);
        Assert.False(Path.IsPathRooted(candles.LandingName));
        Assert.DoesNotContain(':', candles.LandingName);
        Assert.Equal(candles.LandingName, Path.GetFileName(candles.LandingName));
        Assert.Empty(candles.LandingName.Intersect(Path.GetInvalidFileNameChars()));
    }

    [Fact]
    public void TwoWindowsOfOneSymbol_NameTwoFiles_AndTheSameWindowTheSameFileWhoeverAsks()
    {
        var one = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));
        var again = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1, 0), Utc(2024, 2, 1, 0));
        var other = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 3, 1));
        var elsewhere = new BinanceCandles("ETHEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));

        Assert.Equal(one.LandingName, again.LandingName);
        Assert.NotEqual(one.LandingName, other.LandingName);
        Assert.NotEqual(one.LandingName, elsewhere.LandingName);
    }

    [Fact]
    public void TheNameDoesNotDependOnTheMachinesCultureOrZone()
    {
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("nl-NL");

            var dutch = new BinanceCandles("BTCEUR", "1d", new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Unspecified), Utc(2024, 2, 1)).LandingName;

            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

            Assert.Equal(new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1)).LandingName, dutch);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public void AnAddress_ThatIsNotGiven_IsRefused()
    {
        var window = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));

        Assert.Throws<ArgumentNullException>(() => window.At(null!));
    }

    [Theory]
    [InlineData("/api/v3")]
    [InlineData("ftp://venue.test")]
    [InlineData("https://venue.test/api/v3")]
    [InlineData("https://venue.test/?symbol=BTCEUR")]
    [InlineData("https://venue.test/#top")]
    [InlineData("https://user:secret@venue.test")]
    public void AnAddressThatIsNotAHostAlone_IsRefused_AndNeverEchoesCredentials(string address)
    {
        var window = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1));

        var refused = Assert.Throws<ArgumentException>(() => window.At(new Uri(address, UriKind.RelativeOrAbsolute)));

        Assert.Equal("host", refused.ParamName);
        Assert.Contains("an http or https address with no path, no query and no credentials", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("secret", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("https://data-api.binance.vision", "https://data-api.binance.vision")]
    [InlineData("https://data-api.binance.vision/", "https://data-api.binance.vision")]
    [InlineData("http://127.0.0.1:8080", "http://127.0.0.1:8080")]
    public void AnAddressOfAHostAlone_IsTaken_AsTheHostTheWindowIsAskedOf(string address, string host)
    {
        var elsewhere = new BinanceCandles("BTCEUR", "1d", Utc(2024, 1, 1), Utc(2024, 2, 1)).At(new Uri(address));

        Assert.Equal(host, elsewhere.Host);
    }
}
