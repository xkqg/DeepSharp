// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// One candidate tried: what it was, the network it declared, and how that network was judged on each deal.
/// </summary>
/// <param name="Number">The place of the trial in its study, from nought.</param>
/// <param name="Candidate">The values the sampler gave.</param>
/// <param name="Declared">The network those values declared; a step a pipeline keeps, and trains again with <c>Train</c>.</param>
/// <param name="Scores">How the network was judged, once on each deal; lower is better.</param>
public sealed record Trial(int Number, Candidate Candidate, LearnNetworkStep Declared, IReadOnlyList<double> Scores)
{
    /// <summary>Whether every deal gave a number: a network that diverged is a trial that did not finish.</summary>
    public bool Finished => Scores.Count > 0 && Scores.All(double.IsFinite);

    /// <summary>How the network was judged: the mean of its scores over the deals.</summary>
    public double Score => Scores.Average();

    /// <summary>How much the score moved from one deal to the next: the sample standard deviation, none for a single deal.</summary>
    public double Spread
    {
        get
        {
            if (Scores.Count < 2)
            {
                return 0;
            }

            var mean = Scores.Average();

            return Math.Sqrt(Scores.Sum(score => (score - mean) * (score - mean)) / (Scores.Count - 1));
        }
    }
}

/// <summary>
/// What a study found: every trial, the best of them, and the networks it trained.
/// </summary>
public sealed class StudyResult
{
    internal StudyResult(IReadOnlyList<Trial> trials, Trial best, IReadOnlyList<TrainedNetwork> winner)
    {
        Trials = trials;
        Best = best;
        Winner = winner;
    }

    /// <summary>Every trial, in the order they were tried.</summary>
    public IReadOnlyList<Trial> Trials { get; }

    /// <summary>The trial judged lowest, the earlier one when two were judged alike.</summary>
    public Trial Best { get; }

    /// <summary>
    /// The networks the best trial trained, one for each deal. They are the only networks a study keeps, and so the only ones
    /// whose report of the test rows anything can read: a trial reports nothing about them.
    /// </summary>
    public IReadOnlyList<TrainedNetwork> Winner { get; }
}

/// <summary>
/// Tries many candidates for one network, each trained on rows a pipeline prepared and judged by that pipeline's validation
/// rows, and keeps the one judged best.
/// </summary>
/// <remarks>
/// <para>
/// The pipeline is prepared once for each deal, and every candidate is trained on those rows: what the preparation learned is
/// learned from the training rows alone and replayed, whichever network is trained behind it. A deal is a different division
/// of the rows into training and validation; the rows to be measured on must be the same on every deal, which a split at
/// random does when it is given a <c>testSeed</c>, so a candidate that suits one deal by luck is not chosen by luck and then
/// measured on rows another deal learned from. A candidate is judged by the mean of its validation loss over the deals.
/// </para>
/// <para>
/// A study chooses between finished networks by the validation rows and by no others; it never takes part in the training
/// of any of them. It does not read a trial's test rows: only the network it keeps, <see cref="StudyResult.Winner"/>, is
/// kept, and with it the report's measures of the test rows. Trials run one after another and share nothing they change: a
/// candidate's network is declared afresh for each trial, and the same study run again is the same numbers.
/// </para>
/// <para>
/// The winner leaves the study as a declaration: <see cref="Trial.Declared"/> is the step a pipeline keeps in its file, and
/// <c>pipeline.Train()</c> trains it again from there.
/// </para>
/// </remarks>
public sealed class Study
{
    private readonly Func<Candidate, NetworkDeclaration, NetworkDeclaration>? _network;
    private readonly Func<Candidate, LearnNetworkStep>? _declare;

    /// <summary>Declares a study whose candidates are written in the chain that declares a network.</summary>
    /// <param name="space">What may vary.</param>
    /// <param name="network">Writes the network a candidate declares, from its values and a declaration to begin with.</param>
    /// <exception cref="ArgumentNullException">There is no space, or no way to declare the network.</exception>
    public Study(SearchSpace space, Func<Candidate, NetworkDeclaration, NetworkDeclaration> network)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(network);

        Space = space;
        _network = network;
    }

    /// <summary>Declares a study whose candidates are written out as steps.</summary>
    /// <param name="space">What may vary.</param>
    /// <param name="declare">Gives the step a candidate declares.</param>
    /// <exception cref="ArgumentNullException">There is no space, or no way to declare the network.</exception>
    public Study(SearchSpace space, Func<Candidate, LearnNetworkStep> declare)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(declare);

        Space = space;
        _declare = declare;
    }

    /// <summary>What may vary.</summary>
    public SearchSpace Space { get; }

    /// <summary>Which candidate each trial tries; random, drawn from a seed, unless said.</summary>
    public ISampler Sampler
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(value));
    } = new RandomSampler(20261009);

    /// <summary>How many candidates are tried.</summary>
    public int Trials
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A study tries at least one candidate.");
    } = 20;

    /// <summary>How many times the rows are divided again, each candidate being trained on every division.</summary>
    public int Deals
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A study uses at least one division of the rows.");
    } = 1;

    /// <summary>The engines the names in the declared networks stand for; the light engine alone, unless given.</summary>
    public Engines? Engines { get; init; }

    /// <summary>The words the chain declares a network in; PyTorch's, unless said.</summary>
    public Vocabularies Words { get; init; } = Vocabularies.Torch;

    /// <summary>How a trained network is judged, lower being better; its validation loss, unless said.</summary>
    public Func<TrainedNetwork, double> Score
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(value));
    } = trained => trained.ValidationLoss();

    /// <summary>
    /// What is handed every trial as it ends, in order, on the thread the study runs on, with how it was judged; nothing, unless said.
    /// </summary>
    /// <remarks>
    /// For a person watching a study that takes hours: it hears each trial once and changes no number, and one that throws
    /// abandons the study, its exception reaching the caller as it was thrown. It may cancel the token the study was run with.
    /// </remarks>
    public Action<Trial>? OnTrial { get; init; }

    /// <summary>Runs the study.</summary>
    /// <param name="pipelineFor">
    /// The pipeline for a deal, given its place from nought: the same steps every time, the split dealing the training and
    /// validation rows by a seed that moves with the deal and the rows to be measured on by one that does not.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the best of them, and the networks it trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned, and the trials already tried are not kept.</exception>
    /// <exception cref="InvalidOperationException">
    /// A pipeline sets no rows aside as validation; the deals do not share their test rows; a sampler gives a candidate that is
    /// not of the space; or no trial finished.
    /// </exception>
    public StudyResult Run(Func<int, Pipeline> pipelineFor, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);

        var prepared = Prepared(pipelineFor, cancellation);
        var trials = new List<Trial>();
        Trial? best = null;
        IReadOnlyList<TrainedNetwork> winner = [];

        for (var number = 0; number < Trials; number++)
        {
            cancellation.ThrowIfCancellationRequested();

            var candidate = Sampler.Suggest(Space, number, [.. trials]);

            ThrowIfNotOfTheSpace(candidate);

            var step = Declared(candidate);
            var networks = prepared.Select(each => each.Train(step, Engines, cancellation)).ToArray();
            var trial = new Trial(number, candidate, step, [.. networks.Select(Score)]);

            trials.Add(trial);
            OnTrial?.Invoke(trial);

            if (trial.Finished && (best is null || trial.Score < best.Score))
            {
                best = trial;
                winner = networks;
            }
        }

        return best is null
            ? throw new InvalidOperationException($"No trial finished: all {Trials} were judged by a number that is none, or infinite. Try smaller steps, or look at the candidates the space allows.")
            : new StudyResult(trials, best, winner);
    }

    // A step of its own for every trial: nothing a trial changes is shared with the next.
    private LearnNetworkStep Declared(Candidate candidate) =>
        _declare is { } declare ? declare(candidate) : _network!(candidate, new NetworkDeclaration(Words)).Step();

    private PreparedData[] Prepared(Func<int, Pipeline> pipelineFor, CancellationToken cancellation)
    {
        var prepared = new PreparedData[Deals];

        for (var deal = 0; deal < prepared.Length; deal++)
        {
            cancellation.ThrowIfCancellationRequested();

            prepared[deal] = (pipelineFor(deal) ?? throw new InvalidOperationException($"The pipeline for deal {deal} is nothing.")).RunToBeJudged();
        }

        var first = prepared[0].Batch(Part.Test, TrainedNetwork.FeatureNeeds).Keys!.ToHashSet();

        for (var deal = 1; deal < prepared.Length; deal++)
        {
            if (!first.SetEquals(prepared[deal].Batch(Part.Test, TrainedNetwork.FeatureNeeds).Keys!))
            {
                throw new InvalidOperationException(
                    $"Deal {deal} measures on other rows than deal 0 does: a candidate would be chosen on rows another deal learned from. Give the split a test seed — SplitAtRandom(train, validation, seed, testSeed) — so the rows to be measured on are the same on every deal.");
            }
        }

        return prepared;
    }

    private void ThrowIfNotOfTheSpace(Candidate candidate)
    {
        if (!candidate.Settings.Select(each => each.Name).SequenceEqual(Space.Select(each => each.Name), StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                $"The sampler gave a candidate of other dimensions than the space has: the space has {string.Join(", ", Space.Select(each => $"'{each.Name}'"))}, "
                + $"and the candidate {string.Join(", ", candidate.Settings.Select(each => $"'{each.Name}'"))}.");
        }
    }
}
