// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using static DeepSharp.Learners.Networks.TrainExtensions;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// One candidate tried: what it was, the network it declared, and how that network was judged on each deal.
/// </summary>
/// <param name="Number">The place of the trial in its study, from nought.</param>
/// <param name="Candidate">The values the sampler gave.</param>
/// <param name="Declared">The network those values declared; a step a pipeline keeps, and trains again with <c>Train</c>.</param>
/// <param name="Scores">How the network was judged, once on each deal; lower is better. None, for a trial that was pruned.</param>
public sealed record Trial(int Number, Candidate Candidate, LearnNetworkStep Declared, IReadOnlyList<double> Scores)
{
    /// <summary>Whether the study's pruner stopped the trial before its runs ended; it ran to their end, unless said.</summary>
    public bool Pruned { get; init; }

    /// <summary>
    /// The validation loss of every epoch the trial's runs went through, one list for each deal it trained on — the last cut
    /// short where it was pruned; nothing, unless said.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<double>> Heard { get; init; } = [];

    /// <summary>Whether every deal gave a number: a network that diverged, or a trial that was pruned, is a trial that did not finish.</summary>
    public bool Finished => !Pruned && Scores.Count > 0 && Scores.All(double.IsFinite);

    /// <summary>How the network was judged: the mean of its scores over the deals; no number for a trial that has none.</summary>
    public double Score => Scores.Count == 0 ? double.NaN : Scores.Average();

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
/// of any of them, though a <see cref="Pruner"/> may stop one. It does not read a trial's test rows: only the network it keeps,
/// <see cref="StudyResult.Winner"/>, is kept, and with it the report's measures of the test rows. Trials run one after another
/// and share nothing they change: a candidate's network is declared afresh for each trial, and the same study run again is the
/// same numbers.
/// </para>
/// <para>
/// The winner leaves the study as a declaration: <see cref="Trial.Declared"/> is the step a pipeline keeps in its file, and
/// <c>pipeline.Train()</c> trains it again from there. A study that took hours is handed the trials it already tried, which
/// the caller keeps, and goes on from them as the study that never stopped.
/// </para>
/// </remarks>
public sealed class Study
{
    private readonly Func<Candidate, NetworkDeclaration, NetworkDeclaration>? _network;
    private readonly Func<Candidate, LearnNetworkStep>? _declare;
    private bool _scoredItsOwnWay;

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
        init
        {
            field = value ?? throw new ArgumentNullException(nameof(value));
            _scoredItsOwnWay = true;
        }
    } = trained => trained.ValidationLoss();

    /// <summary>
    /// What is handed every trial as it ends, in order, on the thread the study runs on, with how it was judged; nothing, unless said.
    /// </summary>
    /// <remarks>
    /// For a person watching a study that takes hours: it hears each trial once and changes no number, and one that throws
    /// abandons the study, its exception reaching the caller as it was thrown. It may cancel the token the study was run with.
    /// </remarks>
    public Action<Trial>? OnTrial { get; init; }

    /// <summary>
    /// What hears every epoch of every deal a trial trains on, and says whether to stop the trial there; nothing, unless said,
    /// and then every trial runs to its end. <see cref="MedianPruner.Prunes"/> is Optuna's median rule.
    /// </summary>
    /// <remarks>
    /// <para>
    /// It hears where the run stands — the deal, the validation loss of every epoch of it so far, and the trials before this
    /// one — on the thread the study runs on, once each epoch has been judged. Saying yes stops the run before its next batch;
    /// the trial is then <see cref="Trial.Pruned"/>, keeps what it heard, is judged by nothing and is never the best. Saying no
    /// changes no number, so a pruner that never prunes is the study without one.
    /// </para>
    /// <para>
    /// A loss halfway through a run is not the loss it ends at, and a run that starts slowly can end best; a pruner trades that
    /// for trials, and a rule that waits — for trials to finish, and for a trial's first epochs — loses less. The pruner hears
    /// the validation loss, which is what a study judges by unless it is told a <see cref="Score"/> of its own; told one, a
    /// pruner would stop trials by a number they are not judged by, and the study is refused.
    /// </para>
    /// </remarks>
    public Func<Progress, bool>? Pruner { get; init; }

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
    /// not of the space; a pruner is given beside a score of the study's own; or no trial finished.
    /// </exception>
    public StudyResult Run(Func<int, Pipeline> pipelineFor, CancellationToken cancellation = default) => Run(pipelineFor, [], cancellation);

    /// <summary>Runs the study on from the trials it already tried.</summary>
    /// <param name="pipelineFor">The pipeline for a deal, given its place from nought, as the study that tried them was handed it.</param>
    /// <param name="earlier">
    /// The trials already tried, as the study that tried them handed them back: numbered from nought without a gap, each of this
    /// study's space and declaring the network this study declares for its candidate. The study goes on from the next number up
    /// to <see cref="Trials"/>, and its sampler is handed them as it was then.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the networks the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">
    /// Anything <see cref="Run(Func{int, Pipeline}, CancellationToken)"/> refuses; the earlier trials skip a number, are not of
    /// the space or declare another network; or the best of them is an earlier one whose network, trained again on every deal,
    /// is judged otherwise than it was.
    /// </exception>
    /// <remarks>
    /// The networks of a trial are not kept, so where the best is an earlier trial, its network is trained again on every deal
    /// — the same declaration on the same rows, and so, on one engine, the same numbers — and those networks are the winner.
    /// </remarks>
    public StudyResult Run(Func<int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);
        ArgumentNullException.ThrowIfNull(earlier);

        if (Pruner is not null && _scoredItsOwnWay)
        {
            throw new InvalidOperationException(
                $"A {nameof(Pruner)} hears each epoch's validation loss, and this study judges its trials by a {nameof(Score)} of its own: a pruner would stop trials by a number they are not judged by. Leave one of the two out.");
        }

        ThrowIfNotThisStudys(earlier);

        var prepared = Prepared(pipelineFor, cancellation);
        var trials = new List<Trial>(earlier);
        var best = earlier.Where(trial => trial.Finished).MinBy(trial => trial.Score);
        IReadOnlyList<TrainedNetwork> winner = [];

        for (var number = earlier.Count; number < Trials; number++)
        {
            cancellation.ThrowIfCancellationRequested();

            var candidate = Sampler.Suggest(Space, number, [.. trials]);

            ThrowIfNotOfTheSpace(candidate);

            var (trial, networks) = Tried(candidate, prepared, trials, cancellation);

            trials.Add(trial);
            OnTrial?.Invoke(trial);

            if (trial.Finished && (best is null || trial.Score < best.Score))
            {
                best = trial;
                winner = networks;
            }
        }

        if (best is null)
        {
            throw new InvalidOperationException($"No trial finished: all {trials.Count} were pruned, or judged by a number that is none, or infinite. Try smaller steps, or look at the candidates the space allows.");
        }

        return new StudyResult(trials, best, best.Number < earlier.Count ? TrainedAgain(best, prepared, cancellation) : winner);
    }

    // One trial, numbered after the trials before it: its network trained on every deal, each epoch handed to the pruner, which may stop it.
    private TriedTrial Tried(Candidate candidate, PreparedData[] prepared, List<Trial> before, CancellationToken cancellation)
    {
        var step = Declared(candidate);
        var networks = new List<TrainedNetwork>();
        var heard = new List<IReadOnlyList<double>>();
        var pruned = false;

        for (var deal = 0; deal < prepared.Length && !pruned; deal++)
        {
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            var soFar = new List<double>();
            var at = deal;

            void Heard(Epoch epoch)
            {
                soFar.Add(epoch.ValidationLoss!.Value);

                if (Pruner?.Invoke(new Progress(at, [.. soFar], [.. before])) is true)
                {
                    pruned = true;
                    stop.Cancel();
                }
            }

            try
            {
                var network = prepared[deal].Train(step, new RunControls(Engines, Pruner is null ? null : Heard, stop.Token));

                if (!pruned)
                {
                    networks.Add(network);
                }
            }
            catch (OperationCanceledException) when (pruned && !cancellation.IsCancellationRequested)
            {
                // The pruner stopped the run: the trial ends here, with what it heard.
            }

            heard.Add(Pruner is null ? [.. networks[^1].History!.Epochs.Select(epoch => epoch.ValidationLoss!.Value)] : soFar);
        }

        var trial = new Trial(before.Count, candidate, step, pruned ? [] : [.. networks.Select(Score)]) { Pruned = pruned, Heard = heard };

        return new TriedTrial(trial, networks);
    }

    // The best trial was one the caller handed back, whose networks were not kept: trained again, they must be judged as it was.
    private TrainedNetwork[] TrainedAgain(Trial best, PreparedData[] prepared, CancellationToken cancellation)
    {
        var networks = prepared.Select(each => each.Train(best.Declared, new RunControls(Engines, null, cancellation))).ToArray();

        if (!networks.Select(Score).SequenceEqual(best.Scores))
        {
            throw new InvalidOperationException(
                $"The best of the earlier trials, trial {best.Number}, was trained again on every deal and judged otherwise than it was handed back: these are not the rows, "
                + "the engine or the trials this study tried. Hand it the pipelines and the engines the trials were tried with.");
        }

        return networks;
    }

    // Earlier trials are this study's own: numbered from nought without a gap, of its space, declaring what it declares.
    private void ThrowIfNotThisStudys(IReadOnlyList<Trial> earlier)
    {
        for (var at = 0; at < earlier.Count; at++)
        {
            var trial = earlier[at] ?? throw new ArgumentException($"Earlier trial {at} is nothing.", nameof(earlier));

            if (trial.Number != at)
            {
                throw new InvalidOperationException($"The earlier trials are numbered from nought without a gap, and the one at place {at} is trial {trial.Number}.");
            }

            ThrowIfNotOfTheSpace(trial.Candidate);

            if (trial.Declared != Declared(trial.Candidate))
            {
                throw new InvalidOperationException($"Earlier trial {at} declares another network than this study declares for its candidate: it was tried by another study.");
            }
        }
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

    // A candidate holds a setting for every dimension that is there for it, in the space's order, and for no other.
    private void ThrowIfNotOfTheSpace(Candidate candidate)
    {
        var there = Space.Where(dimension => dimension.IsThereAmong(candidate.Settings)).Select(dimension => dimension.Name).ToArray();

        if (!candidate.Settings.Select(each => each.Name).SequenceEqual(there, StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                $"The sampler gave a candidate of other dimensions than the space has for it: the space has {string.Join(", ", there.Select(name => $"'{name}'"))}, "
                + $"and the candidate {string.Join(", ", candidate.Settings.Select(each => $"'{each.Name}'"))}.");
        }
    }

    // A trial and the networks it trained, which only the best trial's are kept of.
    private readonly record struct TriedTrial(Trial Trial, IReadOnlyList<TrainedNetwork> Networks);
}
