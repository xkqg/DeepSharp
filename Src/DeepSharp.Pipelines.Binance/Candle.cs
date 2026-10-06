// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// One candle as the venue answered with it, every number in the spelling the venue wrote it in.
/// </summary>
/// <remarks>
/// A row is known by every cell as it is written, so a decimal is never turned into a number and written again: the
/// landing holds the venue's own words, and a candle comes back from the file as the venue said it. Two of the venue's
/// twelve fields are not here — the moment a candle closes, which is the next one's opening less a millisecond, and the
/// field the venue's own documentation says it ignores.
/// </remarks>
/// <param name="OpenTime">When the candle opens, in milliseconds since the epoch, universal.</param>
/// <param name="Open">The price at which it opened.</param>
/// <param name="High">The highest price in it.</param>
/// <param name="Low">The lowest price in it.</param>
/// <param name="Close">The price at which it closed.</param>
/// <param name="Volume">What was traded in it, in the base asset.</param>
/// <param name="QuoteVolume">What was traded in it, in the quote asset.</param>
/// <param name="Trades">How many trades there were.</param>
/// <param name="TakerBuyVolume">How much of the volume was bought by takers, in the base asset.</param>
/// <param name="TakerBuyQuoteVolume">How much of the quote volume was bought by takers.</param>
internal readonly record struct Candle(
    long OpenTime,
    string Open,
    string High,
    string Low,
    string Close,
    string Volume,
    string QuoteVolume,
    string Trades,
    string TakerBuyVolume,
    string TakerBuyQuoteVolume);
