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
/// <param name="Declared">The network those values declared; a step a pipeline keeps, and trains again with <c>Train</c>. The first of the trial's networks, where it has several.</param>
/// <param name="Scores">
/// How the network was judged, once on each deal that finished; lower is better. None, for a trial a <see cref="Study.Pruner"/>
/// stopped mid-epoch, which has no deal it finished fairly; the real scores of however many deals did finish, for a trial a
/// <see cref="Study.DealStopper"/> stopped between deals, which has nothing untrustworthy about the ones it kept.
/// </param>
public sealed record Trial(int Number, Candidate Candidate, LearnNetworkStep Declared, IReadOnlyList<double> Scores)
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
        get => field ?? [Declared];
        init;
    }

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
    internal StudyResult(IReadOnlyList<Trial> trials, Trial best, IReadOnlyList<IReadOnlyList<TrainedNetwork>> members)
    {
        Trials = trials;
        Best = best;
        Members = members;
        Winner = [.. members.Select(deal => deal[0])];
    }

    /// <summary>Every trial, in the order they were tried.</summary>
    public IReadOnlyList<Trial> Trials { get; }

    /// <summary>The trial judged lowest, the earlier one when two were judged alike.</summary>
    public Trial Best { get; }

    /// <summary>
    /// The networks the best trial trained, one for each deal: the first of its networks, where it has several. They are the only
    /// networks a study keeps, and so the only ones whose report of the test rows anything can read: a trial reports nothing about them.
    /// </summary>
    public IReadOnlyList<TrainedNetwork> Winner { get; }

    /// <summary>The networks the best trial trained, every one of them, for each deal in order and in the order the trial declared them.</summary>
    public IReadOnlyList<IReadOnlyList<TrainedNetwork>> Members { get; }
}

/// <summary>
/// Tries many candidates for one network, each trained on rows a pipeline prepared and judged by that pipeline's validation
/// rows, and keeps the one judged best.
/// </summary>
/// <remarks>
/// <para>
/// Handed one pipeline for each deal, the study prepares it once for each deal, and every candidate is trained on those rows: what
/// the preparation learned is learned from the training rows alone and replayed, whichever network is trained behind it. A deal is
/// a different division of the rows into training and validation; the rows to be measured on must be the same on every deal, which
/// a split at random does when it is given a <c>testSeed</c>, so a candidate that suits one deal by luck is not chosen by luck and
/// then measured on rows another deal learned from. A candidate is judged by the mean of its validation loss over the deals.
/// </para>
/// <para>
/// Handed a pipeline for each candidate and deal, which a feature, a window or a way of filling a gap the candidate chooses asks
/// for, the study prepares the pipeline for every trial and deal, and holds each to the rows the first one prepared were judged on:
/// the same validation rows on a deal for every candidate, and the same rows to be measured on throughout. Scores are chosen between
/// only where the rows they were measured on are the same, and a loss is in the units of the answers the pipeline hands over, so
/// pipelines that hand over the answers in other units are judged by a <see cref="Score"/> of the caller's, in the unit the caller
/// chooses.
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
/// <para>
/// A model of several networks — a mean and a shape, say — is declared as several steps, each trained on every deal behind a
/// pipeline of its own, and judged by a <see cref="Judge"/> that gives the one number for the networks a deal trained, in whatever unit
/// the caller means: a trial is chosen as the one model it is.
/// </para>
/// </remarks>
public sealed class Study
{
    private readonly Func<Candidate, NetworkDeclaration, NetworkDeclaration>? _network;
    private readonly Func<Candidate, LearnNetworkStep>? _declare;
    private readonly Func<Candidate, IReadOnlyList<LearnNetworkStep>>? _declareMembers;
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

    /// <summary>Declares a study whose candidates are several networks, written out as steps and judged together.</summary>
    /// <param name="space">What may vary.</param>
    /// <param name="declare">Gives the steps a candidate declares, one for each network, in the order their pipelines are handed in.</param>
    /// <exception cref="ArgumentNullException">There is no space, or no way to declare the networks.</exception>
    /// <remarks>A candidate that declares more than one network needs a <see cref="Judge"/>.</remarks>
    public Study(SearchSpace space, Func<Candidate, IReadOnlyList<LearnNetworkStep>> declare)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(declare);

        Space = space;
        _declareMembers = declare;
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
    /// How the networks one deal trained are judged together, lower being better: the one number for them, in whatever unit the caller
    /// chooses; nothing, unless said, and then a trial's network is judged by the <see cref="Score"/>.
    /// </summary>
    /// <remarks>
    /// A trial that declares several networks needs one, since no score of one network says how they do together. It is handed the
    /// networks of a deal in the order the trial declared them, and is asked once for each deal of each trial. It is refused beside a
    /// <see cref="Score"/> or a <see cref="Pruner"/>: the trial would be judged by two numbers, or stopped by one it is not judged by.
    /// </remarks>
    public Func<IReadOnlyList<TrainedNetwork>, double>? Judge
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(value));
    }

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
    /// the validation loss, which is what a study judges by unless it is told a <see cref="Score"/> or a <see cref="Judge"/> of its own; told
    /// one, a pruner would stop trials by a number they are not judged by, and the study is refused.
    /// </para>
    /// </remarks>
    public Func<Progress, bool>? Pruner { get; init; }

    /// <summary>
    /// What hears every deal a trial finishes, judged by the same number the trial itself is judged by, and says whether to stop
    /// the trial there without running its remaining deals; nothing, unless said, and then every trial runs every deal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// It hears where the run stands — every deal's score so far, judged exactly as <see cref="Judge"/>, a <see cref="Score"/> of
    /// the caller's, or the default validation loss already judges a finished deal, and the trials before this one — on the
    /// thread the study runs on, once each deal has finished, except the last: nothing is saved by stopping once every deal has
    /// already trained. Saying yes stops the trial before its next deal; it is then <see cref="Trial.Pruned"/> and never the
    /// best, but keeps the real score of every deal that did finish, since each is as trustworthy as a trial that ran them all.
    /// Saying no changes no number, so a deal stopper that never stops is the study without one.
    /// </para>
    /// <para>
    /// Unlike a <see cref="Pruner"/>, which hears a single network's own raw validation loss mid-training and is refused beside
    /// a <see cref="Judge"/> for that reason, a deal stopper hears the same number the trial is judged by and so never breaks
    /// the rule that one number judges a trial: it may stand beside a <see cref="Judge"/>, a <see cref="Score"/>, or a
    /// <see cref="Pruner"/> alike.
    /// </para>
    /// </remarks>
    public Func<DealsJudged, bool>? DealStopper { get; init; }

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
    /// not of the space; a pruner is given beside a score or a judge of the study's own; or no trial finished.
    /// </exception>
    /// <remarks>The pipeline is prepared once for each deal, whatever the candidates.</remarks>
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

        return Run(earlier, () =>
        {
            var prepared = Prepared(pipelineFor, cancellation);

            return (_, _, deal) => [prepared[deal]];
        }, cancellation);
    }

    /// <summary>Runs the study with a pipeline for each candidate and each deal.</summary>
    /// <param name="pipelineFor">
    /// The pipeline for a candidate on a deal, given the candidate and the deal's place from nought: the steps the candidate chooses, and
    /// the split dealing the validation rows and the rows to be measured on the same way for every candidate.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the best of them, and the networks it trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">
    /// Anything <see cref="Run(Func{int, Pipeline}, CancellationToken)"/> refuses; or a candidate's pipeline sets other rows aside as
    /// validation or to be measured on than the first one prepared did.
    /// </exception>
    /// <remarks>The pipeline is prepared for every trial and deal; the study that is handed one for each deal prepares it once for each.</remarks>
    public StudyResult Run(Func<Candidate, int, Pipeline> pipelineFor, CancellationToken cancellation = default) => Run(pipelineFor, [], cancellation);

    /// <summary>Runs the study on from the trials it already tried, with a pipeline for each candidate and each deal.</summary>
    /// <param name="pipelineFor">The pipeline for a candidate on a deal, as the study that tried the earlier trials was handed it.</param>
    /// <param name="earlier">The trials already tried, as <see cref="Run(Func{int, Pipeline}, IReadOnlyList{Trial}, CancellationToken)"/> takes them.</param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the networks the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">Anything the other forms of <c>Run</c> refuse.</exception>
    public StudyResult Run(Func<Candidate, int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);

        return Run((candidate, deal) => [pipelineFor(candidate, deal)], earlier, cancellation);
    }

    /// <summary>Runs the study of a model of several networks, a pipeline behind each network of each candidate on each deal.</summary>
    /// <param name="pipelinesFor">
    /// The pipelines for a candidate on a deal, given the candidate and the deal's place from nought: one for each network the candidate
    /// declares, in the order it declares them, all dealing the validation rows and the rows to be measured on the same way.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the best of them, and the networks it trained, every one of them for each deal.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">
    /// Anything the other forms of <c>Run</c> refuse; a candidate declares several networks and the study has no <see cref="Judge"/>; or
    /// a candidate's pipelines are not as many as its networks.
    /// </exception>
    public StudyResult Run(Func<Candidate, int, IReadOnlyList<Pipeline>> pipelinesFor, CancellationToken cancellation = default) => Run(pipelinesFor, [], cancellation);

    /// <summary>Runs the study of a model of several networks on from the trials it already tried.</summary>
    /// <param name="pipelinesFor">The pipelines for a candidate on a deal, as the study that tried the earlier trials was handed them.</param>
    /// <param name="earlier">The trials already tried, as <see cref="Run(Func{int, Pipeline}, IReadOnlyList{Trial}, CancellationToken)"/> takes them.</param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the networks the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">Anything the other forms of <c>Run</c> refuse.</exception>
    public StudyResult Run(Func<Candidate, int, IReadOnlyList<Pipeline>> pipelinesFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelinesFor);

        return Run(earlier, () => new Preparing(pipelinesFor, cancellation).Prepared, cancellation);
    }

    // The study, once it is known how a candidate's pipelines are prepared: begun after the earlier trials are held to this study, so a
    // study that is refused for them prepares nothing.
    private StudyResult Run(IReadOnlyList<Trial> earlier, Func<Preparation> begin, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(earlier);

        if (Judge is not null && (_scoredItsOwnWay || Pruner is not null))
        {
            throw new InvalidOperationException(
                $"A {nameof(Judge)} judges a trial by what its networks do together, and this study also has a {(_scoredItsOwnWay ? nameof(Score) : nameof(Pruner))}: "
                + "the trial would be judged by two numbers, or stopped by one it is not judged by. Leave one of the two out.");
        }

        if (Pruner is not null && _scoredItsOwnWay)
        {
            throw new InvalidOperationException(
                $"A {nameof(Pruner)} hears each epoch's validation loss, and this study judges its trials by a {nameof(Score)} of its own: a pruner would stop trials by a number they are not judged by. Leave one of the two out.");
        }

        ThrowIfNotThisStudys(earlier);

        var prepare = begin();
        var trials = new List<Trial>(earlier);
        var best = earlier.Where(trial => trial.Finished).MinBy(trial => trial.Score);
        IReadOnlyList<IReadOnlyList<TrainedNetwork>> winner = [];

        for (var number = earlier.Count; number < Trials; number++)
        {
            cancellation.ThrowIfCancellationRequested();

            var candidate = Sampler.Suggest(Space, number, [.. trials]);

            ThrowIfNotOfTheSpace(candidate);

            var (trial, networks) = Tried(candidate, prepare, trials, cancellation);

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

        return new StudyResult(trials, best, best.Number < earlier.Count ? TrainedAgain(best, prepare, cancellation) : winner);
    }

    // One trial, numbered after the trials before it: its networks trained on every deal, each epoch handed to the pruner, and
    // each finished deal's score handed to the deal stopper, either of which may stop it.
    private TriedTrial Tried(Candidate candidate, Preparation prepare, List<Trial> before, CancellationToken cancellation)
    {
        var steps = DeclaredBy(candidate);

        if (steps.Count > 1 && Judge is null)
        {
            throw new InvalidOperationException($"Trial {before.Count} declares {steps.Count} networks, and no score of one of them says how they do together: give the study a {nameof(Judge)}.");
        }

        var networks = new List<IReadOnlyList<TrainedNetwork>>();
        var heard = new List<IReadOnlyList<double>>();
        var dealScores = new List<double>();
        var pruned = false;
        var stoppedBetweenDeals = false;
        var deals = Deals;

        for (var deal = 0; deal < deals && !pruned && !stoppedBetweenDeals; deal++)
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
                var trained = TrainedOn(prepare(candidate, before.Count, deal), steps, new RunControls(Engines, Pruner is null ? null : Heard, stop.Token));

                if (!pruned)
                {
                    networks.Add(trained);
                    dealScores.Add(Judged(trained));

                    if (deal < deals - 1 && DealStopper?.Invoke(new DealsJudged([.. dealScores], [.. before])) is true)
                    {
                        stoppedBetweenDeals = true;
                    }
                }
            }
            catch (OperationCanceledException) when (pruned && !cancellation.IsCancellationRequested)
            {
                // The pruner stopped the run: the trial ends here, with what it heard.
            }

            heard.Add(Pruner is null ? [.. networks[^1][0].History!.Epochs.Select(epoch => epoch.ValidationLoss!.Value)] : soFar);
        }

        var trial = new Trial(before.Count, candidate, steps[0], pruned ? [] : dealScores) { Pruned = pruned || stoppedBetweenDeals, Heard = heard };

        return new TriedTrial(steps.Count > 1 ? trial with { Members = steps } : trial, networks);
    }

    // The networks a trial declared, each trained on the rows its own pipeline prepared for the deal.
    private static IReadOnlyList<TrainedNetwork> TrainedOn(IReadOnlyList<PreparedData> prepared, IReadOnlyList<LearnNetworkStep> steps, RunControls controls)
    {
        if (prepared.Count != steps.Count)
        {
            throw new InvalidOperationException($"A trial declares {steps.Count} networks and was handed {prepared.Count} pipelines: one for each network.");
        }

        return [.. prepared.Select((rows, at) => rows.Train(steps[at], controls))];
    }

    // How the networks of one deal are judged: by the judge, which looks at all of them, or by the score of the one there is.
    private double Judged(IReadOnlyList<TrainedNetwork> networks) => Judge is { } judge ? judge(networks) : Score(networks[0]);

    // The best trial was one the caller handed back, whose networks were not kept: trained again, they must be judged as it was.
    private IReadOnlyList<IReadOnlyList<TrainedNetwork>> TrainedAgain(Trial best, Preparation prepare, CancellationToken cancellation)
    {
        var networks = Enumerable.Range(0, Deals)
            .Select(deal => TrainedOn(prepare(best.Candidate, best.Number, deal), best.Members, new RunControls(Engines, null, cancellation)))
            .ToArray();

        if (!networks.Select(Judged).SequenceEqual(best.Scores))
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

            if (!trial.Members.SequenceEqual(DeclaredBy(trial.Candidate)))
            {
                throw new InvalidOperationException($"Earlier trial {at} declares another network than this study declares for its candidate: it was tried by another study.");
            }
        }
    }

    // The steps of its own for every trial: nothing a trial changes is shared with the next.
    private IReadOnlyList<LearnNetworkStep> DeclaredBy(Candidate candidate)
    {
        if (_declareMembers is { } declare)
        {
            var steps = declare(candidate) ?? throw new InvalidOperationException("The study declared no networks for a candidate.");

            return steps.Count > 0 ? steps : throw new InvalidOperationException("A candidate declares at least one network.");
        }

        return [_declare is { } one ? one(candidate) : _network!(candidate, new NetworkDeclaration(Words)).Step()];
    }

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

        // A value is held to the dimension it is for: a sampler is the caller's, and a network is not to learn from what it gave by mistake.
        foreach (var setting in candidate.Settings)
        {
            if (Space.Single(dimension => dimension.Name == setting.Name).Refusal(setting) is { } refusal)
            {
                throw new InvalidOperationException($"The sampler gave a setting the space cannot take: '{setting.Name}' {refusal}.");
            }
        }
    }

    // How the pipelines of a candidate on a deal are prepared: the rows of each, for a trial and a deal.
    private delegate IReadOnlyList<PreparedData> Preparation(Candidate candidate, int trial, int deal);

    // A trial and the networks it trained, which only the best trial's are kept of: for each deal, those of every one of its networks.
    private readonly record struct TriedTrial(Trial Trial, IReadOnlyList<IReadOnlyList<TrainedNetwork>> Networks);

    // Pipelines handed for each candidate: every one is run for its trial and deal, and held to the rows the first one prepared were judged on.
    private sealed class Preparing(Func<Candidate, int, IReadOnlyList<Pipeline>> pipelinesFor, CancellationToken cancellation)
    {
        private HashSet<RowKey>? _test;
        private readonly Dictionary<int, HashSet<RowKey>> _validation = [];

        public Preparation Prepared => Prepare;

        private IReadOnlyList<PreparedData> Prepare(Candidate candidate, int trial, int deal)
        {
            cancellation.ThrowIfCancellationRequested();

            var pipelines = pipelinesFor(candidate, deal) ?? throw new InvalidOperationException($"The pipelines for trial {trial}, deal {deal} are nothing.");
            var prepared = new List<PreparedData>(pipelines.Count);

            foreach (var pipeline in pipelines)
            {
                var rows = (pipeline ?? throw new InvalidOperationException($"The pipeline for trial {trial}, deal {deal} is nothing.")).RunToBeJudged();

                Hold(rows, trial, deal);
                prepared.Add(rows);
            }

            return prepared;
        }

        // The rows a trial is judged on and those it is measured on are the same for every candidate: scores are chosen between only so.
        private void Hold(PreparedData rows, int trial, int deal)
        {
            var test = rows.Batch(Part.Test, TrainedNetwork.FeatureNeeds).Keys!.ToHashSet();
            var validation = rows.Batch(Part.Validation, TrainedNetwork.FeatureNeeds).Keys!.ToHashSet();

            if (_test is null)
            {
                _test = test;
            }
            else if (!_test.SetEquals(test))
            {
                throw new InvalidOperationException(
                    $"The pipeline of trial {trial} on deal {deal} measures on other test rows than the first one prepared did: a candidate would be chosen on rows another learned from. Give the split a test seed — SplitAtRandom(train, validation, seed, testSeed) — and move it by nothing a candidate chooses.");
            }

            if (!_validation.TryAdd(deal, validation) && !_validation[deal].SetEquals(validation))
            {
                throw new InvalidOperationException(
                    $"The pipeline of trial {trial} on deal {deal} sets other rows aside as validation than the first trial's does: its score would be compared with scores measured on other rows. Deal the validation rows by nothing a candidate chooses.");
            }
        }
    }
}
