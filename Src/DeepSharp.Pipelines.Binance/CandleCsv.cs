// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// A landing as the bytes of a comma-separated file: the inverse of the one reader every pipeline reads such a file with.
/// </summary>
/// <remarks>
/// <para>
/// The column set and the spelling of every cell are a format other people's pipelines are declared against, and a row is
/// known by every cell as it is spelled, so nothing here is left to the machine: UTF-8 without a byte-order mark, a line
/// feed between lines and exactly one at the end, the moment as <c>2024-01-02T00:00:00Z</c> written from the epoch the venue
/// gave it as, and each number as the venue wrote it. The header names a <c>timestamp</c>, the prices, the volumes and the
/// <c>trades</c>.
/// </para>
/// <para>
/// It writes no quotes. The venue answers decimals, and a cell that would need quoting is not one: it is refused, naming the
/// column and the candle, rather than written as something this writer did not understand.
/// </para>
/// </remarks>
internal static class CandleCsv
{
    /// <summary>The names of the columns, in the order the cells follow.</summary>
    public const string Header = "timestamp,open,high,low,close,volume,quoteVolume,trades,takerBuyVolume,takerBuyQuoteVolume";

    extension(IReadOnlyList<Candle> candles)
    {
        /// <summary>The bytes of the landing these candles make.</summary>
        /// <returns>A header and a row each, as <see cref="CandleCsv"/> spells them.</returns>
        /// <exception cref="ArgumentNullException">There are no candles.</exception>
        /// <exception cref="FormatException">
        /// A candle opens at a moment that is not a whole second, or a cell is nothing or would need quoting.
        /// </exception>
        public byte[] AsCsv()
        {
            ArgumentNullException.ThrowIfNull(candles);

            var text = new StringBuilder(Header).Append('\n');

            foreach (var candle in candles)
            {
                var opens = Opens(candle);

                text.Append(opens)
                    .Append(',').Append(Cell("open", candle.Open, opens))
                    .Append(',').Append(Cell("high", candle.High, opens))
                    .Append(',').Append(Cell("low", candle.Low, opens))
                    .Append(',').Append(Cell("close", candle.Close, opens))
                    .Append(',').Append(Cell("volume", candle.Volume, opens))
                    .Append(',').Append(Cell("quoteVolume", candle.QuoteVolume, opens))
                    .Append(',').Append(Cell("trades", candle.Trades, opens))
                    .Append(',').Append(Cell("takerBuyVolume", candle.TakerBuyVolume, opens))
                    .Append(',').Append(Cell("takerBuyQuoteVolume", candle.TakerBuyQuoteVolume, opens))
                    .Append('\n');
            }

            return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(text.ToString());
        }
    }

    private static string Opens(Candle candle)
    {
        if (candle.OpenTime % 1000 != 0)
        {
            throw new FormatException($"A candle opens at {candle.OpenTime.ToString(CultureInfo.InvariantCulture)} milliseconds, which is not a whole second: the venue's unit has changed, and nothing is written.");
        }

        return DateTimeOffset.FromUnixTimeMilliseconds(candle.OpenTime).UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture);
    }

    private static string Cell(string column, string? value, string opens)
    {
        if (string.IsNullOrEmpty(value))
        {
            throw new FormatException($"The venue answered nothing for {column} of the candle that opens at {opens}, and a landing invents no cell.");
        }

        if (value.AsSpan().IndexOfAny(",\"\r\n") >= 0)
        {
            throw new FormatException($"The venue answered a {column} that would need quoting for the candle that opens at {opens}: a landing holds numbers as the venue wrote them, and writes no quotes.");
        }

        return value;
    }
}
