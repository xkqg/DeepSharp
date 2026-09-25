// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>Which part of the data a row belongs to: what it is there for, not where it sits.</summary>
public enum Part
{
    /// <summary>The rows a model learns from, and the only rows anything is fitted on.</summary>
    Train,

    /// <summary>The rows used while choosing between models.</summary>
    Validation,

    /// <summary>The rows kept back until the end.</summary>
    Test,

    /// <summary>The rows held back further still, to predict on once the model is trained.</summary>
    Predict,

    /// <summary>Rows nothing has divided: a pipeline that never splits them.</summary>
    /// <remarks>
    /// Never training. Rows no split has divided are not the rows a model learns from, and calling them that
    /// is how a statistic over every row comes to look like one over the training rows alone.
    /// </remarks>
    Undivided,

    /// <summary>The last moments of a part, kept apart by a split in time: fitted on by nothing, handed to nothing.</summary>
    /// <remarks>
    /// An answer read some rows ahead is read from later rows, so without these the last training rows would learn
    /// their answers from the rows a model is measured on, and the last rows of all have no later row to answer
    /// them.
    /// </remarks>
    Gap,
}

/// <summary>
/// The shares a split divides the rows into.
/// </summary>
/// <param name="Train">The share the model learns from.</param>
/// <param name="Validation">The share used while choosing between models.</param>
/// <param name="Test">The share kept back until the end.</param>
/// <param name="Predict">The share held back to predict on; usually none.</param>
/// <remarks>
/// One type rather than four loose numbers, because they are only meaningful together: each has to be a
/// share, and together they have to make a whole. Written as the negation of what a share is, because every
/// comparison against a not-a-number is false and a range test lets NaN through.
/// </remarks>
public readonly record struct SplitShares(double Train, double Validation, double Test, double Predict = 0)
{
    /// <summary>The shares you write down; the one to be measured on is worked out.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless you say.</param>
    /// <param name="predict">The share held back to predict on; none, unless you say.</param>
    /// <returns>The shares, as fractions.</returns>
    /// <exception cref="ArgumentException">They ask for more than there is, or leave nothing to measure on.</exception>
    /// <exception cref="ArgumentOutOfRangeException">One of them is not a number.</exception>
    /// <remarks>
    /// The test share is never written down: it is whatever is left, so nothing can add up to more than
    /// everything there is. Percentages and fractions are told apart by what the written shares add up to —
    /// more than one means percentages — so <c>Of(80, 10)</c> and <c>Of(0.8, 0.1)</c> say the same thing,
    /// and <c>Of(0.7, 0.1, 10)</c>, which mixes the two, is refused rather than read as seven-tenths of a
    /// percent.
    /// </remarks>
    public static SplitShares Of(double train, double validation = 0, double predict = 0)
    {
        // A share that is not a finite number is refused before anything is added up: the sum of an
        // infinity is an infinity, and the message would then be about a whole rather than about the
        // number that is not one.
        ThrowIfNotFinite(train, nameof(train));
        ThrowIfNotFinite(validation, nameof(validation));
        ThrowIfNotFinite(predict, nameof(predict));

        // Percentages or fractions, decided by whether any one of them is bigger than a whole — not by
        // what they add up to, because 0.70 and 0.45 add up to more than one while plainly being written
        // as fractions, and reading them as percentages would quietly make training seven-tenths of one.
        var spoken = train + validation + predict;
        var whole = train > 1 || validation > 1 || predict > 1 ? 100 : 1;

        if (whole == 100 && new[] { train, validation, predict }.Any(share => share > 0 && share < 1))
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation} and {predict} are not written in the same units: one of them is a fraction among percentages."),
                nameof(train));
        }

        if (spoken > whole + 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation} and {predict} ask for {spoken:0.####} of {whole}, and a split cannot use more rows than there are."),
                nameof(train));
        }

        if (Math.Abs(spoken - whole) <= 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation} and {predict} are the whole of the data and leave nothing to be measured on."),
                nameof(train));
        }

        // Rounded, because a subtraction in one unit and the same subtraction in the other do not give
        // the same bits — 1 - 0.9 is not 10 / 100 — and two declarations that say the same thing have
        // to compare equal.
        var test = Math.Round(whole - spoken, 12);

        return new SplitShares(train / whole, validation / whole, test / whole, predict / whole);
    }

    /// <summary>Checks that these are shares that make a whole.</summary>
    /// <exception cref="ArgumentOutOfRangeException">One of them is not a share.</exception>
    /// <exception cref="ArgumentException">Together they are not a whole.</exception>
    public void Validate()
    {
        ThrowIfNotAShare(Train, nameof(Train));
        ThrowIfNotAShare(Test, nameof(Test));

        // Validation and predict may be nothing at all: a two-way division into training and test is an
        // ordinary way to work, and so is having no slice to predict on. A part that is simply not there
        // is different from one that was meant to exist and came out empty. Training and test may not be
        // nothing, because a model learns from one and is measured on the other.
        ThrowIfNotAShareOrNothing(Validation, nameof(Validation));
        ThrowIfNotAShareOrNothing(Predict, nameof(Predict));

        var total = Train + Validation + Test + Predict;

        if (Math.Abs(total - 1) > 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The shares add up to {total:0.####} and a split has to use every row."),
                nameof(Train));
        }
    }

    /// <summary>Hands out the parts for a given number of rows, in order.</summary>
    /// <param name="rows">How many rows there are.</param>
    /// <returns>Which part each position belongs to, training first and predict last.</returns>
    public Part[] Over(int rows)
    {
        var train = (int)Math.Round(rows * Train, MidpointRounding.AwayFromZero);
        var validation = (int)Math.Round(rows * Validation, MidpointRounding.AwayFromZero);
        var test = (int)Math.Round(rows * Test, MidpointRounding.AwayFromZero);

        var parts = new Part[rows];

        for (var at = 0; at < rows; at++)
        {
            parts[at] = at < train ? Part.Train
                : at < train + validation ? Part.Validation
                : at < train + validation + test || Predict == 0 ? Part.Test
                : Part.Predict;
        }

        return parts;
    }

    private static void ThrowIfNotFinite(double share, string name)
    {
        if (!double.IsFinite(share))
        {
            throw new ArgumentOutOfRangeException(name, share, "A share of the data is a number.");
        }
    }

    private static void ThrowIfNotAShare(double share, string name)
    {
        if (!(share > 0 && share <= 1))
        {
            throw new ArgumentOutOfRangeException(
                name, share, "A share of the data is more than none of it and at most all of it.");
        }
    }

    private static void ThrowIfNotAShareOrNothing(double share, string name)
    {
        if (!(share >= 0 && share <= 1))
        {
            throw new ArgumentOutOfRangeException(
                name, share, "A share of the data is at least none of it and at most all of it.");
        }
    }
}

/// <summary>
/// Split the rows at random, with a seed so it happens the same way every time.
/// </summary>
/// <remarks>
/// The right split for rows that do not depend on one another. The seed is part of the declaration rather
/// than something the fit discovers, because a split you cannot reproduce makes every number after it
/// unreproducible too.
/// </remarks>
public sealed record SplitAtRandomStep : ISplitStep, IPipelineStep<SplitAtRandomStep>, IDescribesColumns
{
    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that makes the shuffle repeatable: the same seed deals the same rows the same way.", 20260923);

    /// <summary>Declares a split at random.</summary>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    public SplitAtRandomStep(SplitShares shares, int seed)
    {
        Shares = SharesKey.Require(shares);
        Seed = SeedKey.Require(seed);
    }

    /// <inheritdoc />
    public static StepParameters<SplitAtRandomStep> Parameters { get; } = new StepParameters<SplitAtRandomStep>()
        .With(SharesKey, step => step.Shares)
        .With(SeedKey, step => step.Seed);

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <inheritdoc />
    public static string Name => "split.atRandom";

    /// <inheritdoc />
    /// <remarks>The second version places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public static string Purpose => "Divides the rows at random, the same way every time for the same seed.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>
    /// Each row is ranked by a digest of the seed and what the row says, so the same rows are dealt the same
    /// way whatever order they arrive in, and every copy of a repeated row lands where its first copy does.
    /// </remarks>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var ranked = table.Ranked(Enumerable.Range(0, table.RowCount), Seed);
        var parts = Shares.Over(table.RowCount).Placed(ranked);

        parts.KeptTogether(ranked, (one, other) => table.Identities[one].Key == table.Identities[other].Key);

        return parts;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static SplitAtRandomStep ReadFrom(JsonElement element) =>
        new(SharesKey.Read(element), SeedKey.Read(element));
}

/// <summary>
/// Split the rows at random, keeping the mixture of one column the same in every part.
/// </summary>
/// <remarks>
/// The right split when an answer is rare. A plain shuffle can leave a class almost absent from validation,
/// and a model measured there is measured on nothing much; this deals each group out separately so every
/// part looks like the whole.
/// </remarks>
public sealed record SplitStratifiedStep : ISplitStep, IPipelineStep<SplitStratifiedStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column whose mixture of values is kept the same in every part.", "class", ColumnKinds.Any);

    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that makes the shuffle repeatable: the same seed deals the same rows the same way.", 20260923);

    /// <summary>Declares a split that keeps the mixture of a column.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public SplitStratifiedStep(string column, SplitShares shares, int seed)
    {
        Column = ColumnKey.Require(column);
        Shares = SharesKey.Require(shares);
        Seed = SeedKey.Require(seed);
    }

    /// <inheritdoc />
    public static StepParameters<SplitStratifiedStep> Parameters { get; } = new StepParameters<SplitStratifiedStep>()
        .With(ColumnKey, step => step.Column)
        .With(SharesKey, step => step.Shares)
        .With(SeedKey, step => step.Seed);

    /// <summary>The column whose mixture is kept the same in every part.</summary>
    public string Column { get; }

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <inheritdoc />
    public static string Name => "split.stratified";

    /// <inheritdoc />
    /// <remarks>The second version places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public static string Purpose => "Divides the rows at random while keeping the mixture of one column the same in every part.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>
    /// Within each group, each row is ranked by a digest of the seed and what the row says, so the same rows
    /// are dealt the same way whatever order they arrive in, and every copy of a repeated row — which is in the
    /// same group, holding the same value — lands where its first copy does.
    /// </remarks>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var column = table[Column];
        var parts = new Part[table.RowCount];

        var groups = Enumerable.Range(0, table.RowCount)
            .GroupBy(row => column.TextAt(row) ?? "\u0000missing")
            .OrderBy(group => group.Key, StringComparer.Ordinal);

        foreach (var group in groups)
        {
            var ranked = table.Ranked(group, Seed);
            var share = Shares.Over(ranked.Length);

            for (var at = 0; at < ranked.Length; at++)
            {
                parts[ranked[at]] = share[at];
            }

            parts.KeptTogether(ranked, (one, other) => table.Identities[one].Key == table.Identities[other].Key);
        }

        return parts;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static SplitStratifiedStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), SharesKey.Read(element), SeedKey.Read(element));
}

/// <summary>
/// Putting a run of parts back onto the rows they belong to.
/// </summary>
internal static class SplitPlacement
{
    /// <summary>Rows in the order of a digest of the seed and what each row says.</summary>
    /// <param name="table">The rows' table, which knows each row's key.</param>
    /// <param name="rows">The rows to rank.</param>
    /// <param name="seed">The number that makes the ranking one of many possible ones, and the same one each time.</param>
    /// <returns>The rows, ranked; rows that say the same thing stand side by side.</returns>
    /// <remarks>
    /// SHA-256 rather than a generator of random numbers: a generator shuffles places, so the same rows in
    /// another order were dealt differently, while a digest of what a row says ranks it the same wherever it
    /// stands, on every machine.
    /// </remarks>
    internal static int[] Ranked(this Table table, IEnumerable<int> rows, int seed)
    {
        var ranked = rows.ToArray();
        var ranks = ranked.Select(row => table.Identities[row].Key.Ranked(seed)).ToArray();

        Array.Sort(ranks, ranked);

        return ranked;
    }

    /// <summary>Gives every row the part of the row before it, when the two belong together.</summary>
    /// <param name="parts">The part of each row, by row; changed in place.</param>
    /// <param name="ordered">The rows in the order the parts were handed out.</param>
    /// <param name="together">Whether a row belongs with the one before it: the same row twice, the same moment.</param>
    internal static void KeptTogether(this Part[] parts, int[] ordered, Func<int, int, bool> together)
    {
        for (var at = 1; at < ordered.Length; at++)
        {
            if (together(ordered[at - 1], ordered[at]))
            {
                parts[ordered[at]] = parts[ordered[at - 1]];
            }
        }
    }

    /// <summary>Keeps the last moments of every part apart: before each line, and at the end.</summary>
    /// <param name="parts">The part of each row, by row; changed in place.</param>
    /// <param name="ordered">The rows in the order the parts were handed out.</param>
    /// <param name="gap">How many moments of each part are kept apart.</param>
    /// <param name="together">Whether a row belongs with the one before it: the same moment.</param>
    /// <remarks>Moments rather than rows, so a moment is kept apart whole, as a split keeps it whole.</remarks>
    internal static void Gapped(this Part[] parts, int[] ordered, int gap, Func<int, int, bool> together)
    {
        var end = ordered.Length;

        while (end > 0)
        {
            var part = parts[ordered[end - 1]];
            var start = end - 1;

            while (start > 0 && parts[ordered[start - 1]] == part)
            {
                start--;
            }

            var moments = 0;

            for (var at = end - 1; at >= start && moments < gap; at--)
            {
                parts[ordered[at]] = Part.Gap;

                // The row that begins its moment completes one moment kept apart.
                if (at == start || !together(ordered[at - 1], ordered[at]))
                {
                    moments++;
                }
            }

            end = start;
        }
    }

    /// <summary>Lays a run of parts out over the rows in a given order.</summary>
    /// <param name="inOrder">The parts, training first.</param>
    /// <param name="rows">The rows, in the order the parts should be handed out.</param>
    /// <returns>One part per row, in row order.</returns>
    internal static Part[] Placed(this Part[] inOrder, int[] rows)
    {
        var parts = new Part[inOrder.Length];

        for (var at = 0; at < rows.Length; at++)
        {
            parts[rows[at]] = inOrder[at];
        }

        return parts;
    }
}
