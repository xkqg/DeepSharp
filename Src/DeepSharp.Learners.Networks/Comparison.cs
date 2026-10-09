// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// How far the second of two pipelines sits from the first, deal by deal.
/// </summary>
public sealed class PairedDifferences
{
    internal PairedDifferences(IReadOnlyList<double> firsts, IReadOnlyList<double> seconds)
    {
        Firsts = firsts;
        Seconds = seconds;
        Differences = [.. seconds.Zip(firsts, (second, first) => second - first)];
        Mean = Differences.Average();
        Spread = Math.Sqrt(Differences.Sum(difference => (difference - Mean) * (difference - Mean)) / (Differences.Count - 1));
    }

    /// <summary>How many deals were compared.</summary>
    public int Deals => Differences.Count;

    /// <summary>How the first pipeline's network was judged on each deal.</summary>
    public IReadOnlyList<double> Firsts { get; }

    /// <summary>How the second pipeline's network was judged on each deal.</summary>
    public IReadOnlyList<double> Seconds { get; }

    /// <summary>The second's score less the first's, on each deal: above nought where the second is worse by a score that is a loss.</summary>
    public IReadOnlyList<double> Differences { get; }

    /// <summary>The mean of the differences.</summary>
    public double Mean { get; }

    /// <summary>How much the difference moved from one deal to the next: the sample standard deviation.</summary>
    public double Spread { get; }

    /// <summary>How far the mean is itself uncertain: the spread over the root of the number of deals.</summary>
    public double StandardError => Spread / Math.Sqrt(Deals);
}

/// <summary>
/// Trains one network behind two pipelines on each of many deals, and says how far the second sits from the first.
/// </summary>
/// <remarks>
/// <para>
/// One division of the rows cannot say whether a change helped: the same network scores differently on two divisions of the
/// same rows by about as much as most changes move it. Here each deal prepares both pipelines, trains the same network behind
/// each, and judges both; a change is then read as the mean of the differences against their own spread, and a change that
/// helps on one deal and hurts on another shows as a spread that is larger than its mean.
/// </para>
/// <para>
/// The two pipelines may differ in anything — a feature left out or put in, a gap filled another way, a split at random or by
/// time — and they are given the same deal number, which is what makes the differences pairs. Comparing a split at random
/// with a split by time is one use; a feature on and off is another, and the two together show whether a feature helps
/// because it knows the future of a row it should not.
/// </para>
/// </remarks>
public sealed class Comparison
{
    private readonly Func<NetworkDeclaration, NetworkDeclaration>? _network;
    private readonly LearnNetworkStep? _step;

    /// <summary>Compares pipelines for a network written in the chain that declares one.</summary>
    /// <param name="network">Writes the network trained behind both pipelines.</param>
    /// <exception cref="ArgumentNullException">There is no way to declare the network.</exception>
    public Comparison(Func<NetworkDeclaration, NetworkDeclaration> network)
    {
        ArgumentNullException.ThrowIfNull(network);

        _network = network;
    }

    /// <summary>Compares pipelines for a network written out as a step.</summary>
    /// <param name="learner">The network trained behind both pipelines.</param>
    /// <exception cref="ArgumentNullException">There is no network.</exception>
    public Comparison(LearnNetworkStep learner)
    {
        ArgumentNullException.ThrowIfNull(learner);

        _step = learner;
    }

    /// <summary>How many deals are compared; ten, unless said.</summary>
    public int Deals
    {
        get;
        init => field = value >= 2 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A spread needs at least two deals to be a spread of anything.");
    } = 10;

    /// <summary>The engines the name in the network stands for; the light engine alone, unless given.</summary>
    public Engines? Engines { get; init; }

    /// <summary>The words the chain declares the network in; PyTorch's, unless said.</summary>
    public Vocabularies Words { get; init; } = Vocabularies.Torch;

    /// <summary>How a trained network is judged, lower being better; its validation loss, unless said.</summary>
    public Func<TrainedNetwork, double> Score
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(value));
    } = trained => trained.ValidationLoss();

    /// <summary>Compares the two pipelines.</summary>
    /// <param name="first">The pipeline for a deal, given its place from nought: the one the second is read against.</param>
    /// <param name="second">The pipeline for the same deal that differs from it.</param>
    /// <param name="cancellation">What stops the comparison before its next deal, and a network before its next batch; nothing stops it, unless said.</param>
    /// <returns>How each was judged on each deal, and how far the second sits from the first.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make a pipeline.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">A pipeline sets no rows aside as validation, or a network is judged by a number that is none.</exception>
    public PairedDifferences Run(Func<int, Pipeline> first, Func<int, Pipeline> second, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);

        var step = _step ?? _network!(new NetworkDeclaration(Words)).Step();
        var firsts = new double[Deals];
        var seconds = new double[Deals];

        for (var deal = 0; deal < Deals; deal++)
        {
            cancellation.ThrowIfCancellationRequested();

            firsts[deal] = Judged(first(deal), step, new Place(deal, "first"), cancellation);
            seconds[deal] = Judged(second(deal), step, new Place(deal, "second"), cancellation);
        }

        return new PairedDifferences(firsts, seconds);
    }

    private double Judged(Pipeline pipeline, LearnNetworkStep step, Place place, CancellationToken cancellation)
    {
        var (deal, side) = place;
        var score = Score((pipeline ?? throw new InvalidOperationException($"The {side} pipeline for deal {deal} is nothing.")).RunToBeJudged().Train(step, Engines, cancellation));

        return double.IsFinite(score)
            ? score
            : throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture, $"The network behind the {side} pipeline was judged by {score} on deal {deal}, which is no score to take a difference of."));
    }
}

// Which deal, and which of the two pipelines, a network is being judged for: what a refusal names.
internal readonly record struct Place(int Deal, string Side);
