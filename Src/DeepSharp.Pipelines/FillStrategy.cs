// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// How a gap is filled, as a value with a name.
/// </summary>
/// <param name="Name">The name this strategy is written under, in a file and at a call site.</param>
/// <remarks>
/// A named value rather than a flag, because <c>Fill("trades", true, false)</c> tells the next reader
/// nothing at all: it has to be read with the signature open beside it. Whatever a strategy eventually
/// computes is a learned value, fitted on the training rows alone.
/// </remarks>
public readonly record struct FillStrategy(string Name);

/// <summary>
/// The strategies a gap can be filled with.
/// </summary>
/// <remarks>
/// Named so that the call site reads as a sentence: <c>FillMissing("trades", With.Mean)</c>. There is no
/// state here, only names — each property hands back the same value, and nothing can be set.
/// </remarks>
public static class With
{
    /// <summary>The average of the column, over the training rows.</summary>
    public static FillStrategy Mean => new("mean");

    /// <summary>The middle value of the column, which a single extreme value cannot drag around.</summary>
    public static FillStrategy Median => new("median");

    /// <summary>Zero, which is a measurement and not an absence — the marking column says which it was.</summary>
    public static FillStrategy Zero => new("zero");

    /// <summary>The last value before the gap, for data that arrives in order.</summary>
    public static FillStrategy Previous => new("previous");
}
