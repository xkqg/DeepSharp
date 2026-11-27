// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

/// <summary>
/// One candidate tried: what it was, the network it declared, and how that network was judged on each deal.
/// </summary>
/// <param name="Number">The place of the trial in its study, from nought.</param>
/// <param name="Candidate">The values the sampler gave.</param>
/// <param name="Declared">
/// The network those values declared; a step a pipeline keeps, and trains again with <c>Train</c>. The first of the trial's
/// networks, where it has several. Nothing, for a trial a <see cref="Study{TModel}"/> tried, whose candidate is trained by a
/// function of the caller's rather than declared as a step.
/// </param>
/// <param name="Scores">
/// How the network was judged, once on each deal that finished; lower is better. None, for a trial a <see cref="Study.Pruner"/>
/// stopped mid-epoch, which has no deal it finished fairly; the real scores of however many deals did finish, for a trial a
/// <see cref="Study.DealStopper"/> stopped between deals, which has nothing untrustworthy about the ones it kept.
/// </param>
public sealed record Trial(int Number, Candidate Candidate, LearnNetworkStep? Declared, IReadOnlyList<double> Scores)
{
    /// <summary>Whether the study's pruner or deal stopper stopped the trial before its deals ran out; it ran to their end, unless said.</summary>
    public bool Pruned { get; init; }

    /// <summary>
    /// The validation loss of every epoch the trial's runs went through, one list for each deal it trained on — the last cut
    /// short where a pruner stopped it mid-epoch; nothing, unless said. Of the first network, where the trial has several. A
    /// deal stopper never cuts one of these short: it only ever stops a trial between two deals, each already whole.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<double>> Heard { get; init; } = [];

    /// <summary>
    /// The networks the trial declared, judged together: only <see cref="Declared"/>, unless said. A deal trains every one of them,
    /// each behind a pipeline of its own, and a judge of the study's says how they did together.
    /// </summary>
    public IReadOnlyList<LearnNetworkStep> Members
    {
        get => field ?? (Declared is { } one ? [one] : []);
        init;
    }

    /// <summary>
    /// How the networks were weighed on every objective, once on each deal that finished, in whatever units the caller's
    /// <see cref="Study.Objectives"/> chooses; empty, for a trial a <see cref="Study.Score"/> or a <see cref="Study.Judge"/>
    /// judges, whose <see cref="Scores"/> holds its numbers instead, and for one a pruner stopped mid-epoch.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<double>> ObjectiveScores { get; init; } = [];

    /// <summary>Whether every deal gave a number, or every one of a vector: a network that diverged, or a trial that was pruned, is a trial that did not finish.</summary>
    public bool Finished =>
        !Pruned
        && ((Scores.Count > 0 && Scores.All(double.IsFinite)) || (ObjectiveScores.Count > 0 && ObjectiveScores.All(objectives => objectives.All(double.IsFinite))));

    /// <summary>How the network was judged: the mean of its scores over the deals; no number for a trial that has none.</summary>
    public double Score => Scores.Mean();

    /// <summary>How the networks were weighed on every objective: the mean of each over the deals; none for a trial that has none.</summary>
    public IReadOnlyList<double> ObjectiveMean =>
        ObjectiveScores.Count == 0
            ? []
            : [.. Enumerable.Range(0, ObjectiveScores[0].Count).Select(at => ObjectiveScores.Average(objectives => objectives[at]))];

    /// <summary>How much the score moved from one deal to the next: the sample standard deviation, none for a single deal.</summary>
    public double Spread => Scores.SampleSpread();

    /// <summary>
    /// How this trial is weighed against another of its study: the mean of each objective, for a trial judged by several numbers,
    /// and its <see cref="Score"/> as a vector of one otherwise — the one verdict every ranking of trials reads.
    /// </summary>
    internal Verdict Verdict => new(ObjectiveScores.Count > 0 ? ObjectiveMean : [Score]);
}

/// <summary>
/// How a trial is weighed against another: the one number a <see cref="Study.Score"/> or a <see cref="Study.Judge"/> gives, as
/// a vector of one, or the several a <see cref="Study.Objectives"/> gives — the shape every trial of one study is weighed in.
/// </summary>
/// <param name="Values">The number, or numbers, lower being better on each.</param>
/// <remarks>
/// A vector of one value makes <see cref="Better"/> the plain <c>&lt;</c> a <see cref="Study.Score"/> or a <see cref="Study.Judge"/>
/// already chose a trial by, and <see cref="Dominates"/> moot — nothing else is weighed against it on a second value — so the one
/// comparison serves both a study's ordinary trials and its objectives alike, never a branch of its own for either. The order is
/// the one a study chooses its best by, the one <see cref="EnsembleExtensions.Top"/> hands on its best few by, and the one the
/// Pareto front is drawn against.
/// </remarks>
internal readonly record struct Verdict(IReadOnlyList<double> Values) : IComparable<Verdict>
{
    /// <summary>Where this verdict stands against another: the first value that differs deciding it, below nought where this one is better.</summary>
    /// <param name="other">The other verdict, of the same number of values.</param>
    /// <returns>Below nought where this one is better, above where it is worse, nought where every value is the same.</returns>
    public int CompareTo(Verdict other)
    {
        for (var at = 0; at < Values.Count; at++)
        {
            var compared = Values[at].CompareTo(other.Values[at]);

            if (compared != 0)
            {
                return compared;
            }
        }

        return 0;
    }

    /// <summary>Whether this verdict is better than another: the lexicographic least of the two, the first value that differs deciding it.</summary>
    /// <param name="than">The other verdict, of the same number of values.</param>
    public bool Better(Verdict than) => CompareTo(than) < 0;

    /// <summary>Whether this verdict dominates another under a Pareto order: no worse on any value, and better on at least one.</summary>
    /// <param name="than">The other verdict, of the same number of values.</param>
    public bool Dominates(Verdict than) =>
        Values.Zip(than.Values, (mine, theirs) => mine <= theirs).All(atLeastAsGood => atLeastAsGood)
        && Values.Zip(than.Values, (mine, theirs) => mine < theirs).Any(strictlyBetter => strictlyBetter);
}

// Whether two objective vectors hold the same values, in the same order: a plain SequenceEqual of IReadOnlyList<double>
// values would compare the lists by reference, never their numbers, since a list is not a value type.
internal sealed class VectorEquality : IEqualityComparer<IReadOnlyList<double>>
{
    public static readonly VectorEquality Instance = new();

    public bool Equals(IReadOnlyList<double>? first, IReadOnlyList<double>? second) =>
        first is null ? second is null : second is not null && first.SequenceEqual(second);

    public int GetHashCode(IReadOnlyList<double> vector) => vector.Aggregate(17, (hash, value) => (hash * 31) + value.GetHashCode());
}

/// <summary>What a list of numbers judged deal by deal says as a whole: its mean, and how much it moved from one deal to the next.</summary>
internal static class SampleStatistics
{
    extension(IReadOnlyList<double> values)
    {
        /// <summary>The mean of the numbers; no number, where there are none.</summary>
        internal double Mean() => values.Count == 0 ? double.NaN : values.Average();

        /// <summary>The sample standard deviation of the numbers, over one fewer than there are; none, where there are fewer than two.</summary>
        internal double SampleSpread()
        {
            if (values.Count < 2)
            {
                return 0;
            }

            var mean = values.Average();

            return Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / (values.Count - 1));
        }
    }
}
