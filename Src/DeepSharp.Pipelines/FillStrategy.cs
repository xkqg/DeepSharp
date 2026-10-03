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

/// <summary>
/// The gaps of one line: which value goes in them, and the columns it goes into.
/// </summary>
/// <typeparam name="TLine">The line itself, so each kind hands back the line it was written on.</typeparam>
/// <remarks>
/// A value that was never there and a value arithmetic could not make are two different things with two different
/// verbs, and both choose from one vocabulary — so the kinds are written once here and each verb's line says only which
/// step it makes. A column with a limit above which filling is refused keeps the verb that takes one column: that limit
/// is a fact about that column, not about the group, and it has no default on purpose.
/// </remarks>
public abstract class FillLine<TLine> : IDeclaresSteps
    where TLine : FillLine<TLine>
{
    private readonly List<IPipelineStep> _steps = [];

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>Columns filled with the average of the column, over the training rows.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public TLine Mean(params string[] columns) => Add(columns, With.Mean);

    /// <summary>Columns filled with the middle value of the column, which a single extreme cannot drag around.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public TLine Median(params string[] columns) => Add(columns, With.Median);

    /// <summary>Columns filled with nought, which is a measurement and not an absence.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>The marking column beside each says which it was.</remarks>
    public TLine Zero(params string[] columns) => Add(columns, With.Zero);

    /// <summary>Columns where a gap stops the run.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>For a column that is not supposed to have anything wrong with it.</remarks>
    public TLine Refuse(params string[] columns) => Add(columns, With.Refuse);

    /// <summary>Columns filled with a number you choose.</summary>
    /// <param name="value">The number to put in every gap.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>For a column where absence has a meaning you already know.</remarks>
    public TLine Constant(double value, params string[] columns) => Add(columns, With.Constant(value));

    /// <summary>The step this line's verb makes for one column.</summary>
    /// <param name="column">The column.</param>
    /// <param name="strategy">What goes in its gaps.</param>
    /// <returns>The step.</returns>
    protected abstract IPipelineStep Step(string column, FillStrategy strategy);

    /// <summary>Takes the columns of one kind into this line.</summary>
    /// <param name="columns">The columns.</param>
    /// <param name="strategy">What goes in their gaps.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <exception cref="ArgumentNullException">There are no columns.</exception>
    protected TLine Add(string[] columns, FillStrategy strategy)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            _steps.Add(Step(column, strategy));
        }

        return (TLine)this;
    }
}

/// <summary>
/// The gaps of one line: values that were never there.
/// </summary>
public sealed class GapLine : FillLine<GapLine>
{
    /// <summary>Columns filled with the last value before the gap, for data that arrives in order.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>
    /// This kind belongs to the gaps alone: the row above a value that is not a number says nothing about what it
    /// should have been, so the verb that watches for those refuses it where it is read — and a line may not offer a
    /// kind its verb would throw on. It reads the rows in their order, so the order is declared above it.
    /// </remarks>
    public GapLine Previous(params string[] columns) => Add(columns, With.Previous);

    /// <inheritdoc />
    protected override IPipelineStep Step(string column, FillStrategy strategy) => FillMissingStep.Of(column, strategy);
}

/// <summary>
/// The values of one line that arithmetic could not make: a division by nought, almost always.
/// </summary>
public sealed class NotANumberLine : FillLine<NotANumberLine>
{
    /// <inheritdoc />
    protected override IPipelineStep Step(string column, FillStrategy strategy) => new FillNaNStep(column, strategy);
}
