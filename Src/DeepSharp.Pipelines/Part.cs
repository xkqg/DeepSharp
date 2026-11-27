// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

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

    /// <summary>The rows a split keeps apart from the training: fitted on by nothing, handed to nothing, counted in the fit.</summary>
    /// <remarks>
    /// A split in time keeps the last moments of every part here: an answer read some rows ahead is read from later rows,
    /// so without them the last training rows would learn their answers from the rows a model is measured on, and the last
    /// rows of all have no later row to answer them. A split in blocks of time keeps here the training rows within its
    /// embargo of a row a model is chosen, calibrated or measured on, which a model could otherwise interpolate from.
    /// </remarks>
    Gap,

    /// <summary>The rows an interval is calibrated on: fitted on by nothing, and chosen by nothing.</summary>
    /// <remarks>
    /// Not training, which fits on them; not validation, which chooses between models by them and so has already seen what it
    /// chose; and not test, which a number is reported on once everything else is settled. A split that sets none aside
    /// divides the rows exactly as it always did.
    /// </remarks>
    Calibration,
}

/// <summary>
/// What a part is for, said once: which rows a fit learns from, which a split deals again, and which parts a fit or a split
/// writes down.
/// </summary>
/// <remarks>
/// Every reader of a <see cref="Part"/> that has to decide a set of them asks here, so a part added later is decided in one place
/// rather than in each reader that happened to list the others. A split that WRITES parts is not a reader and stays where it is.
/// </remarks>
internal static class PartFacts
{
    extension(Part part)
    {
        /// <summary>Whether a fit learns from the rows of this part: training alone, the only rows anything is fitted on.</summary>
        internal bool IsLearnedFrom => part == Part.Train;

        /// <summary>Whether the rows of this part are rows that learn: the training rows, and the validation rows a model is chosen on.</summary>
        /// <remarks>What a split deals again into folds or origins, once test, calibration and predict are set aside.</remarks>
        internal bool Learns => part is Part.Train or Part.Validation;

        /// <summary>Whether a learner is handed the rows of this part: all of them but the rows a split keeps apart.</summary>
        internal bool IsHandedOver => part != Part.Gap;

        /// <summary>Whether a model is chosen, calibrated or measured on rows of this part: validation, calibration or test.</summary>
        /// <remarks>The rows an embargo keeps the training rows away from in time.</remarks>
        internal bool IsMeasuredOn => part is Part.Validation or Part.Calibration or Part.Test;
    }

    extension(Part)
    {
        /// <summary>The parts a fit always counts the rows of, in the order it writes them, empty or not.</summary>
        internal static IReadOnlyList<Part> Counted => [Part.Train, Part.Validation, Part.Test, Part.Predict];

        /// <summary>The parts a fit counts the rows of only when a split gave it any, so a file written before them is unchanged.</summary>
        internal static IReadOnlyList<Part> CountedWhenKept => [Part.Gap, Part.Calibration];

        /// <summary>The parts a split deals rows to, in the order a time split names the stretch of time each covers.</summary>
        internal static IReadOnlyList<Part> Dealt => [Part.Train, Part.Validation, Part.Calibration, Part.Test, Part.Predict];
    }

    extension(IReadOnlyList<Part> parts)
    {
        /// <summary>Whether a row is a training row.</summary>
        /// <param name="row">The row, by its place.</param>
        internal bool IsTraining(int row) => parts[row].IsLearnedFrom;

        /// <summary>Every training row, in row order: the rows a fit learns from.</summary>
        internal int[] TrainingRows() => [.. Enumerable.Range(0, parts.Count).Where(parts.IsTraining)];

        /// <summary>The rows that learn among those given, in the order given.</summary>
        /// <param name="among">The rows to choose from, by their place.</param>
        internal int[] LearningRows(IEnumerable<int> among) => [.. among.Where(row => parts[row].Learns)];
    }
}

/// <summary>
/// The shares a split divides the rows into.
/// </summary>
/// <param name="Train">The share the model learns from.</param>
/// <param name="Validation">The share used while choosing between models.</param>
/// <param name="Test">The share kept back until the end.</param>
/// <param name="Predict">The share held back to predict on; usually none.</param>
/// <param name="Calibration">The share an interval is calibrated on; usually none.</param>
/// <remarks>
/// One type rather than five loose numbers, because they are only meaningful together: each has to be a
/// share, and together they have to make a whole. Written as the negation of what a share is, because every
/// comparison against a not-a-number is false and a range test lets NaN through.
/// </remarks>
public readonly record struct SplitShares(double Train, double Validation, double Test, double Predict = 0, double Calibration = 0)
{
    /// <summary>Shares of four parts: training, validation, test and predict, none of them to calibrate an interval on.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <param name="predict">The share held back to predict on.</param>
    /// <remarks>The form the shares had before a share could be set aside to calibrate on, kept so that nothing written against it breaks.</remarks>
    public SplitShares(double train, double validation, double test, double predict)
        : this(train, validation, test, predict, 0)
    {
    }

    /// <summary>The four shares that were all there was before a share could be set aside to calibrate on.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="test">The share kept back until the end.</param>
    /// <param name="predict">The share held back to predict on.</param>
    public void Deconstruct(out double train, out double validation, out double test, out double predict)
    {
        train = Train;
        validation = Validation;
        test = Test;
        predict = Predict;
    }

    /// <summary>The shares you write down, none of them to calibrate an interval on; the one to be measured on is worked out.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless you say.</param>
    /// <param name="predict">The share held back to predict on; none, unless you say.</param>
    /// <returns>The shares, as fractions.</returns>
    /// <exception cref="ArgumentException">They ask for more than there is, or leave nothing to measure on.</exception>
    /// <exception cref="ArgumentOutOfRangeException">One of them is not a number.</exception>
    /// <remarks>The form this had before a share could be set aside to calibrate on, kept so that nothing written against it breaks.</remarks>
    public static SplitShares Of(double train, double validation = 0, double predict = 0) => Of(train, validation, predict, calibration: 0);

    /// <summary>The shares you write down; the one to be measured on is worked out.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models.</param>
    /// <param name="predict">The share held back to predict on.</param>
    /// <param name="calibration">The share an interval is calibrated on.</param>
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
    public static SplitShares Of(double train, double validation, double predict, double calibration)
    {
        // A share that is not a finite number is refused before anything is added up: the sum of an
        // infinity is an infinity, and the message would then be about a whole rather than about the
        // number that is not one.
        ThrowIfNotFinite(train, nameof(train));
        ThrowIfNotFinite(validation, nameof(validation));
        ThrowIfNotFinite(predict, nameof(predict));
        ThrowIfNotFinite(calibration, nameof(calibration));

        // Percentages or fractions, decided by whether any one of them is bigger than a whole — not by
        // what they add up to, because 0.70 and 0.45 add up to more than one while plainly being written
        // as fractions, and reading them as percentages would quietly make training seven-tenths of one.
        var spoken = train + validation + predict + calibration;
        var whole = train > 1 || validation > 1 || predict > 1 || calibration > 1 ? 100 : 1;

        if (whole == 100 && new[] { train, validation, predict, calibration }.Any(share => share > 0 && share < 1))
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation}, {predict} and {calibration} are not written in the same units: one of them is a fraction among percentages."),
                nameof(train));
        }

        if (spoken > whole + 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation}, {predict} and {calibration} ask for {spoken:0.####} of {whole}, and a split cannot use more rows than there are."),
                nameof(train));
        }

        if (Math.Abs(spoken - whole) <= 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{train}, {validation}, {predict} and {calibration} are the whole of the data and leave nothing to be measured on."),
                nameof(train));
        }

        // Rounded, because a subtraction in one unit and the same subtraction in the other do not give
        // the same bits — 1 - 0.9 is not 10 / 100 — and two declarations that say the same thing have
        // to compare equal.
        var test = Math.Round(whole - spoken, 12);

        return new SplitShares(train / whole, validation / whole, test / whole, predict / whole, calibration / whole);
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
        ThrowIfNotAShareOrNothing(Calibration, nameof(Calibration));

        var total = Train + Validation + Test + Predict + Calibration;

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
    /// <returns>Which part each position belongs to, training first and predict last; the calibration rows come before the test rows.</returns>
    public Part[] Over(int rows)
    {
        var train = (int)Math.Round(rows * Train, MidpointRounding.AwayFromZero);
        var validation = (int)Math.Round(rows * Validation, MidpointRounding.AwayFromZero);
        var calibration = (int)Math.Round(rows * Calibration, MidpointRounding.AwayFromZero);
        var test = (int)Math.Round(rows * Test, MidpointRounding.AwayFromZero);

        var parts = new Part[rows];

        for (var at = 0; at < rows; at++)
        {
            parts[at] = at < train ? Part.Train
                : at < train + validation ? Part.Validation
                : at < train + validation + calibration ? Part.Calibration
                : at < train + validation + calibration + test || Predict is 0 ? Part.Test
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
