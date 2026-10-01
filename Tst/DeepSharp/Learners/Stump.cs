// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A learner of another kind than a network, written as a package outside the pipeline would write one: a tree of one split.
/// It states that it takes categories itself, splits a feature handed over as places by a set of its categories and any
/// other by a threshold at one of its training values, answers each side with the share of its training rows that answered
/// one, and names unfamiliar any place of a category no training row held.
/// </summary>
internal sealed class Stump
{
    private readonly IReadOnlyList<string> _features;
    private readonly IReadOnlyDictionary<string, IReadOnlyList<string>> _categories;
    private readonly Split _split;
    private readonly Sides _sides;

    private Stump(IReadOnlyList<string> features, IReadOnlyDictionary<string, IReadOnlyList<string>> categories, Split split, Sides sides)
    {
        _features = features;
        _categories = categories;
        _split = split;
        _sides = sides;
    }

    /// <summary>What this learner needs of the features it is handed: each category as its place, numbers of any size.</summary>
    public static Needs Needs => Needs.Categories;

    /// <summary>The feature it splits, and where.</summary>
    public string SplitOn => _features[_split.Feature];

    /// <summary>Learns the one split that leaves the training rows' answers least mixed on either side.</summary>
    /// <param name="train">The training part, handed over for <see cref="Needs"/>, with one answer of nought or one.</param>
    /// <returns>The tree.</returns>
    public static Stump Fit(Batch train)
    {
        var labels = train.Labels!;
        var marked = train.Categories!;
        Split? best = null;
        var least = double.PositiveInfinity;

        for (var feature = 0; feature < train.Width; feature++)
        {
            double[] values = [.. train.Features.Select(row => row[feature])];
            var candidates = marked.ContainsKey(train.FeatureNames[feature]) ? BySet(feature, values, labels) : ByThreshold(feature, values);

            foreach (var split in candidates)
            {
                var mixed = Mixed(split, values, labels);

                if (mixed < least)
                {
                    least = mixed;
                    best = split;
                }
            }
        }

        var chosen = best!;
        var left = Enumerable.Range(0, train.RowCount).Where(row => chosen.Left(train.Features[row][chosen.Feature])).ToArray();
        var right = Enumerable.Range(0, train.RowCount).Except(left).ToArray();

        return new Stump(train.FeatureNames, marked, chosen, new Sides(left.Average(row => labels[row]), right.Average(row => labels[row])));
    }

    /// <summary>What the tree predicts for the rows of a part, and which of their features it learned nothing about.</summary>
    /// <param name="batch">The part, handed over for <see cref="Needs"/> as the training part was.</param>
    /// <returns>The predictions, as the report measures them.</returns>
    public PartPredictions Predict(Batch batch)
    {
        Assert.Equal(_features, batch.FeatureNames);

        return new PartPredictions(batch, [.. batch.Features.Select(row => new[] { Answer(row) })])
        {
            Unfamiliar = [.. batch.Features.Select(Unfamiliar)],
        };
    }

    /// <summary>What the tree answers for one row of features.</summary>
    /// <param name="row">The row, in the order of the features it learned from.</param>
    /// <returns>The share of its side's training rows that answered one.</returns>
    public double Answer(double[] row) => _split.Left(row[_split.Feature]) ? _sides.Left : _sides.Right;

    /// <summary>The features of a row handed over as a place no training row held: at or past the count of their categories.</summary>
    /// <param name="row">The row.</param>
    /// <returns>Their names.</returns>
    public IReadOnlyList<string> Unfamiliar(double[] row) =>
        [.. Enumerable.Range(0, _features.Count).Where(at => _categories.TryGetValue(_features[at], out var held) && row[at] >= held.Count).Select(at => _features[at])];

    // Every threshold at a training value but the largest: a row goes left at or below it.
    private static IEnumerable<Split> ByThreshold(int feature, double[] values) =>
        values.Distinct().Order().SkipLast(1).Select(threshold => new Split(feature, threshold, null));

    // The places, ordered by the share of their rows that answered one, cut once anywhere in that order: the best set to
    // send left is always one of these.
    private static IEnumerable<Split> BySet(int feature, double[] values, IReadOnlyList<double> labels)
    {
        double[] places =
        [
            .. values.Distinct()
                .OrderBy(place => Enumerable.Range(0, values.Length).Where(row => values[row] == place).Average(row => labels[row]))
                .ThenBy(place => place),
        ];

        return Enumerable.Range(1, places.Length - 1).Select(count => new Split(feature, double.NaN, places.Take(count).ToHashSet()));
    }

    // How mixed the answers are on the two sides, each side's Gini impurity weighted by its rows.
    private static double Mixed(Split split, double[] values, IReadOnlyList<double> labels)
    {
        double[] counts = new double[4];

        for (var row = 0; row < values.Length; row++)
        {
            var side = split.Left(values[row]) ? 0 : 2;

            counts[side]++;
            counts[side + 1] += labels[row];
        }

        return Gini(counts[0], counts[1]) + Gini(counts[2], counts[3]);

        static double Gini(double rows, double ones) => rows == 0 ? 0 : 2 * ones * (rows - ones) / rows;
    }

    /// <summary>Where the tree splits: a feature, and a threshold or the set of places that go left.</summary>
    private sealed record Split(int Feature, double Threshold, IReadOnlySet<double>? Places)
    {
        public bool Left(double value) => Places is { } set ? set.Contains(value) : value <= Threshold;
    }

    /// <summary>What the tree answers on each side of its split.</summary>
    private readonly record struct Sides(double Left, double Right);
}
