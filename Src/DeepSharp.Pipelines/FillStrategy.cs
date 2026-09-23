// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// How a gap is filled: a name, and a number when the name needs one.
/// </summary>
/// <remarks>
/// A named value rather than a flag, because <c>Fill("trades", true, false)</c> tells the next reader
/// nothing at all: it has to be read with the signature open beside it. Whatever a strategy eventually
/// computes is a learned value, fitted on the training rows alone.
/// <para>
/// The number is here from the start although only one strategy uses it. Adding it later would change the
/// written form of a verb that had already shipped, and a file format is the one thing that cannot be
/// tidied up afterwards.
/// </para>
/// </remarks>
public readonly record struct FillStrategy
{
    /// <summary>A strategy by name, with the number it needs when it needs one.</summary>
    /// <param name="name">The name it is written under, at a call site and in a file.</param>
    /// <param name="value">The number, for a strategy that takes one; otherwise nothing.</param>
    /// <exception cref="ArgumentException">The name is empty or nothing but spaces.</exception>
    public FillStrategy(string name, double? value = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Name = name;
        Value = value;
    }

    /// <summary>The name this strategy is written under.</summary>
    public string Name { get; }

    /// <summary>The number this strategy fills with, for the one that takes a number.</summary>
    public double? Value { get; }

    /// <summary>The strategy as it is spoken: <c>mean</c>, or <c>constant(-1)</c>.</summary>
    /// <returns>The name, with the number when there is one.</returns>
    public override string ToString() =>
        Value is null ? Name ?? string.Empty : $"{Name}({Value})";
}

/// <summary>
/// The strategies a gap can be filled with.
/// </summary>
/// <remarks>
/// Named so that the call site reads as a sentence: <c>FillMissing("trades", With.Mean)</c>. There is no
/// state here, only names — each member hands back a fresh value, and nothing can be set. This is also the
/// list a file is checked against, so a misspelled strategy in a hand-written pipeline is refused rather
/// than carried to whatever an executor's default branch happens to do.
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

    /// <summary>Stop. For a column that is not supposed to have anything wrong with it.</summary>
    public static FillStrategy Refuse => new("refuse");

    /// <summary>A number you choose, for a column where absence has a meaning you already know.</summary>
    /// <param name="value">The number to put in every gap.</param>
    /// <returns>The strategy, carrying its number.</returns>
    public static FillStrategy Constant(double value) => new("constant", value);

    /// <summary>Whether a name is one of the strategies this library defines.</summary>
    /// <param name="name">The name read from a file, or written at a call site.</param>
    /// <returns><see langword="true"/> when the name is known here.</returns>
    public static bool Knows(string name) =>
        name is "mean" or "median" or "zero" or "previous" or "constant" or "refuse";

    /// <summary>Whether a strategy needs a number, and therefore whether one must be there.</summary>
    /// <param name="name">The name of the strategy.</param>
    /// <returns><see langword="true"/> when the strategy is written with a number beside it.</returns>
    public static bool TakesAValue(string name) => name is "constant";
}
