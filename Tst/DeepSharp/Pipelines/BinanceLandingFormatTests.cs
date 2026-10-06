// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a landing is, byte for byte: the column set and the spelling of every cell are a format other people's pipelines
/// are declared against the day it ships, and a row is known by every cell as it is spelled, so none of it is left to the
/// machine that writes it — not its culture, not its line endings, not its encoding's mark.
/// </summary>
public sealed class BinanceLandingFormatTests
{
    private static readonly Candle[] Pair =
    [
        new(1704067200000, "100.00000000", "110.00000000", "90.00000000", "105.50000000", "12.50000000", "1300.25000000", "42", "6.00000000", "640.00000000"),
        new(1704153600000, "105.50000000", "108.00000000", "101.25000000", "102.00000000", "0.00000000", "0.00000000", "0", "0.00000000", "0.00000000"),
    ];

    private static readonly string Written = string.Join(
        '\n',
        "timestamp,open,high,low,close,volume,quoteVolume,trades,takerBuyVolume,takerBuyQuoteVolume",
        "2024-01-01T00:00:00Z,100.00000000,110.00000000,90.00000000,105.50000000,12.50000000,1300.25000000,42,6.00000000,640.00000000",
        "2024-01-02T00:00:00Z,105.50000000,108.00000000,101.25000000,102.00000000,0.00000000,0.00000000,0,0.00000000,0.00000000") + "\n";

    [Fact]
    public void TheLandingIsWrittenAsTheseBytes()
    {
        var bytes = Pair.AsCsv();

        Assert.Equal(Encoding.UTF8.GetBytes(Written), bytes);
    }

    [Fact]
    public void TheBytesCarryNoMarkAndOnlyLineFeeds_AndEndInExactlyOne()
    {
        var bytes = Pair.AsCsv();

        Assert.False(bytes is [0xEF, 0xBB, 0xBF, ..], "A byte-order mark leads the file.");
        Assert.DoesNotContain((byte)'\r', bytes);
        Assert.Equal((byte)'\n', bytes[^1]);
        Assert.NotEqual((byte)'\n', bytes[^2]);
    }

    [Fact]
    public void TheBytesAreTheSame_WhateverCultureTheMachineWritesThemIn()
    {
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("nl-NL");

            var dutch = Pair.AsCsv();

            CultureInfo.CurrentCulture = new CultureInfo("ar-SA");

            var arabic = Pair.AsCsv();

            Assert.Equal(Encoding.UTF8.GetBytes(Written), dutch);
            Assert.Equal(dutch, arabic);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public void ADecimalIsWrittenAsTheVenueSpellsIt_NeverRespelled()
    {
        // 105.50000000 stays 105.50000000: a row is known by every cell as it is written, so a trailing nought dropped
        // would be another row to every split that ranks rows by what they say.
        var landed = Encoding.UTF8.GetString(Pair.AsCsv());

        Assert.Contains(",105.50000000,", landed, StringComparison.Ordinal);
        Assert.Contains(",0.00000000,", landed, StringComparison.Ordinal);
        Assert.DoesNotContain(",105.5,", landed, StringComparison.Ordinal);
    }

    [Fact]
    public void AMomentIsWrittenFromTheEpochTheVenueGaveItAs_InUniversalTimeToTheSecond()
    {
        var landed = Encoding.UTF8.GetString(new Candle[] { Pair[0] with { OpenTime = 1704153600000 } }.AsCsv()).Split('\n');

        Assert.Equal("2024-01-02T00:00:00Z", landed[1].Split(',')[0]);
    }

    [Fact]
    public void TheHeaderNamesTheColumnsAPipelineIsDeclaredAgainst()
    {
        var header = Encoding.UTF8.GetString(Pair.AsCsv()).Split('\n')[0].Split(',');

        Assert.Equal(10, header.Length);
        Assert.Contains("timestamp", header);
        Assert.Contains("close", header);
        Assert.Contains("trades", header);
        Assert.DoesNotContain("closeTime", header);
        Assert.DoesNotContain("ignore", header);
    }

    [Fact]
    public void TheLandingReadsBackAsTheRowsItWasWrittenFrom_ByTheReaderEveryPipelineUses()
    {
        var bytes = Pair.AsCsv();
        var source = new ReadCsvStep("BTCEUR.csv").Open(bytes, "BTCEUR.csv");

        Assert.Equal(CandleCsv.Header.Split(','), source.ColumnNames);
        Assert.Equal(2, source.Rows.Count());
        Assert.Equal(["2024-01-01T00:00:00Z", "100.00000000", "110.00000000", "90.00000000", "105.50000000", "12.50000000", "1300.25000000", "42", "6.00000000", "640.00000000"], source.Rows.First());
    }

    [Fact]
    public void TheCellsProposeTheKindsAPipelineDeclares_ATimestampNumbersAndAWholeNumber()
    {
        var proposal = KindProposal.Of(new ReadCsvStep("BTCEUR.csv").Open(Pair.AsCsv(), "BTCEUR.csv"));

        Assert.Equal(ColumnKind.Timestamp, proposal["timestamp"].Kind);
        Assert.Null(proposal["timestamp"].Format);
        Assert.Equal(ColumnKind.Number, proposal["close"].Kind);
        Assert.Equal(ColumnKind.Number, proposal["volume"].Kind);
        Assert.Equal(ColumnKind.Integer, proposal["trades"].Kind);
    }

    [Fact]
    public void NoCandles_AreAHeaderAlone()
    {
        Assert.Equal(Encoding.UTF8.GetBytes(CandleCsv.Header + "\n"), Array.Empty<Candle>().AsCsv());
    }

    [Theory]
    [InlineData("1,5")]
    [InlineData("1\"5")]
    [InlineData("1\r5")]
    [InlineData("1\n5")]
    public void ACellThatWouldNeedQuotes_IsRefused_NotQuotedAndNotGuessedAt(string cell)
    {
        // The venue answers decimals. A cell that a CSV would have to quote is not one, and a writer that quoted it would be
        // writing what it did not understand: it is refused, naming the column and the candle.
        var refused = Assert.Throws<FormatException>(() => new[] { Pair[0] with { Close = cell } }.AsCsv());

        Assert.Contains("close", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2024-01-01T00:00:00Z", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACellThatIsNothing_IsRefused()
    {
        Assert.Throws<FormatException>(() => new[] { Pair[0] with { Volume = string.Empty } }.AsCsv());
        Assert.Throws<FormatException>(() => new[] { Pair[0] with { Trades = null! } }.AsCsv());
    }

    [Fact]
    public void NoCandles_AreNotWrittenOutOfNothing()
    {
        Assert.Throws<ArgumentNullException>(() => ((IReadOnlyList<Candle>)null!).AsCsv());
    }

    [Theory]
    [InlineData(1704067200500)]
    [InlineData(1704067200001)]
    [InlineData(-1)]
    public void ACandleThatOpensBetweenTwoSeconds_IsRefused_AsTheVenuesUnitHavingChanged(long opens)
    {
        var refused = Assert.Throws<FormatException>(() => new[] { Pair[0] with { OpenTime = opens } }.AsCsv());

        Assert.Contains("not a whole second", refused.Message, StringComparison.Ordinal);
        Assert.Contains(opens.ToString(CultureInfo.InvariantCulture), refused.Message, StringComparison.Ordinal);
    }
}
