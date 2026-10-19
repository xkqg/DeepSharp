// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What a run measured on each fold, laid beside itself: for every measure on every part, the value of each fold, their mean and how
/// far they spread.
/// </summary>
/// <remarks>
/// A small table measured on one deal is measured by that deal's luck, and the spread across folds says how much of the number it
/// was. The spread is the sample standard deviation of the folds' values, so two folds spread by their difference over the root of
/// two; it gives no verdict, since how wide a spread may be is for the person who reads it, as the paired comparison of two
/// pipelines gives none. It belongs to the pipeline as the measures do, so the run of any learner can be laid beside another's: a
/// network and a tree measured on the same folds are measured by the same report.
/// </remarks>
public sealed class MeasuresAcross
{
    private MeasuresAcross(int folds, IReadOnlyList<PartAcross> parts)
    {
        Folds = folds;
        Parts = parts;
    }

    /// <summary>How many folds were measured.</summary>
    public int Folds { get; }

    /// <summary>The parts the report measured, each with its measures across the folds, in the order the report names them.</summary>
    public IReadOnlyList<PartAcross> Parts { get; }

    /// <summary>Lays the measures of folds beside one another.</summary>
    /// <param name="folds">What each fold measured, in the order of the folds: at least two, of one report.</param>
    /// <returns>Every measure of every part across them.</returns>
    /// <exception cref="ArgumentException">There are fewer than two folds, or they were not measured by one report.</exception>
    public static MeasuresAcross Of(IReadOnlyList<Measures> folds)
    {
        ArgumentNullException.ThrowIfNull(folds);

        if (folds.Count < 2)
        {
            throw new ArgumentException($"A spread is of at least two folds, and {folds.Count} {(folds.Count == 1 ? "was" : "were")} given: one value spreads by nothing it can say.", nameof(folds));
        }

        var first = folds[0];

        for (var at = 1; at < folds.Count; at++)
        {
            RequireOneReport(first, folds[at], at);
        }

        return new MeasuresAcross(
            folds.Count,
            [.. first.Parts.Select((part, place) => new PartAcross(
                part.Part,
                [.. folds.Select(fold => fold.Parts[place].Rows)],
                [.. part.Values.Select((measured, which) => Across(measured.Metric, folds.Select(fold => fold.Parts[place].Values[which]).ToArray()))]))]);
    }

    // Folds measured by one report: the same measures of the same parts for the same answers, or they are not laid side by side.
    private static void RequireOneReport(Measures first, Measures other, int at)
    {
        string Said(Measures measures) =>
            string.Join("; ", measures.Parts.Select(part => $"{part.Part}: {string.Join(", ", part.Values.Select(value => value.Metric))}"));

        if (!first.Answers.SequenceEqual(other.Answers, StringComparer.Ordinal) || Said(first) != Said(other))
        {
            throw new ArgumentException(
                $"Fold {at + 1} was not measured by the report fold 1 was: fold 1 measured {Said(first)}, and fold {at + 1} {Said(other)}. Folds are laid beside one another when one report measured them.",
                "folds");
        }
    }

    private static MeasuredAcross Across(Metric metric, Measured[] each)
    {
        double[] values = [.. each.Select(measured => measured.Value)];
        var mean = values.Average();

        return new MeasuredAcross(
            metric,
            mean,
            Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / (values.Length - 1)),
            each.Average(measured => measured.Baseline),
            values);
    }
}

/// <summary>One part of a report, measured on every fold.</summary>
/// <param name="Part">Which part.</param>
/// <param name="Rows">How many rows the part held on each fold, in the order of the folds.</param>
/// <param name="Values">Each of the report's measures of it across the folds, in the order the report names them.</param>
public sealed record PartAcross(Part Part, IReadOnlyList<int> Rows, IReadOnlyList<MeasuredAcross> Values);

/// <summary>One measure of one part across the folds.</summary>
/// <param name="Metric">What was measured.</param>
/// <param name="Mean">The mean of the folds' values.</param>
/// <param name="Spread">The sample standard deviation of the folds' values.</param>
/// <param name="Baseline">The mean of the folds' baselines: the same measure of predicting, for every row, the average of the training rows' answers.</param>
/// <param name="Values">The value of each fold, in the order of the folds.</param>
public readonly record struct MeasuredAcross(Metric Metric, double Mean, double Spread, double Baseline, IReadOnlyList<double> Values);
