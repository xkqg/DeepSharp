// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// The small pair of matrices that sits beside a held kernel and is all of a large layer that learns: Keras's <c>enable_lora</c>,
/// its rank and its alpha.
/// </summary>
/// <remarks>
/// The layer answers with its kernel's product plus the product of the pair, scaled by alpha over the rank: <c>x W + (x A B) ·
/// alpha / rank</c>, A as many rows as the layer has inputs and the rank columns, B the rank rows and as many columns as outputs.
/// B starts at nothing, so the first step's answer is the kernel's own. An alpha left out is the rank, which scales by one, as
/// Keras leaves it.
/// </remarks>
public readonly record struct LowRank
{
    /// <summary>A pair of the given rank and alpha.</summary>
    /// <param name="rank">How many columns A has and rows B has: at least one.</param>
    /// <param name="alpha">What the pair\'s product is scaled by over the rank; the rank itself, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rank is below one, or the alpha is not a finite number above nothing.</exception>
    public LowRank(int rank, double? alpha = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rank, 1);

        if (alpha is { } given && !(double.IsFinite(given) && given > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "Alpha is a finite number above nothing.");
        }

        Rank = rank;
        Alpha = alpha ?? rank;
    }

    /// <summary>How many columns A has and rows B has.</summary>
    public int Rank { get; }

    /// <summary>What the pair\'s product is scaled by over the rank.</summary>
    public double Alpha { get; }

    /// <summary>What the pair\'s product is multiplied by: alpha over the rank.</summary>
    public double Scale => Alpha / Rank;
}
