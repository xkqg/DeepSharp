// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using static DeepSharp.Learners.Networks.TrainExtensions;

namespace DeepSharp.Learners.Networks;

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
/// Deals that measure on other rows by design — a split at random and a split in time — are told apart as <see cref="Kinds"/>,
/// each held to its own test rows, and a candidate is judged by each kind's mean, weighed.
/// </para>
/// <para>
/// Handed a pipeline for each candidate and deal, which a feature, a window or a way of filling a gap the candidate chooses asks
/// for, the study prepares the pipeline for every trial and deal, and holds each to the rows the first one prepared were judged on:
/// the same validation rows on a deal for every candidate, and the same rows to be measured on throughout a kind. Scores are chosen between
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
    private readonly StudySettings _settings = new();
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
        get => _settings.Sampler;
        init => _settings = _settings with { Sampler = value };
    }

    /// <summary>How many candidates are tried.</summary>
    public int Trials
    {
        get => _settings.Trials;
        init => _settings = _settings with { Trials = value };
    }

    /// <summary>
    /// How many times the rows are divided again, each candidate being trained on every division: as many as the
    /// <see cref="Kinds"/> hold, where they are said.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The deals are below one.</exception>
    /// <exception cref="ArgumentException">The <see cref="Kinds"/> are said too, and hold another number of deals.</exception>
    public int Deals
    {
        get => _settings.Deals;
        init => _settings = _settings with { Deals = value };
    }

    /// <summary>
    /// The kinds the deals are of — five dealt at random and one split in time, say — laid out kind after kind, each with a
    /// weight in a trial's score; unless said, one kind of every deal, weighing one, which judges a trial exactly as plain deals do.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A pipeline for a deal is handed the deal's place over all the kinds, from nought, and reads which kind it is with
    /// <c>Kinds.KindOf(deal)</c>. A trial is judged by each kind's mean, weighed: Σ weight × mean ÷ Σ weight, its spread
    /// alike, and the best, <see cref="EnsembleExtensions.Top"/> and the Pareto front all rank by it.
    /// </para>
    /// <para>
    /// The promise is made per kind. Every deal of one kind measures on the same test rows, so a candidate is chosen on no row
    /// its own kind measures on; a row one kind measures on may be learned from by another kind, so a kind's test number is
    /// honest only within that kind.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">The kinds are nothing.</exception>
    /// <exception cref="ArgumentException">
    /// There are none; one is a kind of no deals; two have one name; or <see cref="Deals"/> is said too, and the kinds hold another number of deals.
    /// </exception>
    public IReadOnlyList<DealKind> Kinds
    {
        get => _settings.Kinds;
        init => _settings = _settings with { Kinds = value };
    }

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
    /// How the networks one deal trained are weighed against more than one trade-off, lower being better on each: the several
    /// numbers for them, in whatever units the caller chooses; nothing, unless said, and then a trial is judged by the one
    /// number a <see cref="Score"/> or a <see cref="Judge"/> gives.
    /// </summary>
    /// <remarks>
    /// Where two or more trade-offs are really being weighed against each other, no one number chooses between candidates
    /// honestly, so a study with objectives keeps every candidate none of the others betters on every one of them —
    /// <see cref="EnsembleExtensions.ParetoFront"/> — rather than a single <see cref="StudyResult.Best"/> alone; <see cref="StudyResult.Best"/>
    /// still answers, as the lexicographic least of the mean objective vector, the earlier trial on a tie, which is always a
    /// trial the front holds. It is refused beside a <see cref="Score"/> of the caller's own, a <see cref="Judge"/>, a
    /// <see cref="Pruner"/> or a <see cref="DealStopper"/>: a <see cref="Pruner"/> hears one raw number an epoch, which does not
    /// exist where judging is a vector, and a <see cref="DealStopper"/> hears the same shape a trial is judged by, which here
    /// is several numbers, not the one <see cref="DealsJudged.Scores"/> gives.
    /// </remarks>
    public Func<IReadOnlyList<TrainedNetwork>, IReadOnlyList<double>>? Objectives
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
    public Action<Trial>? OnTrial
    {
        get => _settings.OnTrial;
        init => _settings = _settings with { OnTrial = value };
    }

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
    public Func<Progress, bool>? Pruner
    {
        get => _settings.Pruner;
        init => _settings = _settings with { Pruner = value };
    }

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
    /// Saying no changes no number, so a deal stopper that never stops is the study without one. With deals of several
    /// <see cref="Kinds"/>, <see cref="DealsJudged.Score"/> is the number the trial is judged by so far: each kind it reached,
    /// its mean weighed.
    /// </para>
    /// <para>
    /// Unlike a <see cref="Pruner"/>, which hears a single network's own raw validation loss mid-training and is refused beside
    /// a <see cref="Judge"/> for that reason, a deal stopper hears the same number the trial is judged by and so never breaks
    /// the rule that one number judges a trial: it may stand beside a <see cref="Judge"/>, a <see cref="Score"/>, or a
    /// <see cref="Pruner"/> alike.
    /// </para>
    /// </remarks>
    public Func<DealsJudged, bool>? DealStopper
    {
        get => _settings.DealStopper;
        init => _settings = _settings with { DealStopper = value };
    }

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
    /// A pipeline sets no rows aside as validation; the deals of a kind do not share their test rows; a sampler gives a candidate that is
    /// not of the space; a pruner is given beside a score or a judge of the study's own; or no trial finished.
    /// </exception>
    /// <remarks>The pipeline is prepared once for each deal, whatever the candidates.</remarks>
    public StudyResult Run(Func<int, Pipeline> pipelineFor, CancellationToken cancellation = default) => Run(pipelineFor, [], cancellation);

    /// <summary>Runs the study on from the trials it already tried.</summary>
    /// <param name="pipelineFor">The pipeline for a deal, given its place from nought, as the study that tried them was handed it.</param>
    /// <param name="earlier">
    /// The trials already tried, as the study that tried them handed them back: numbered from nought without a gap, each of this
    /// study's space, declaring the network this study declares for its candidate, and dealt in this study's <see cref="Kinds"/>
    /// on as many deals as they hold. The study goes on from the next number up to <see cref="Trials"/>, and its sampler is
    /// handed them as it was then.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next batch; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the networks the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">
    /// Anything <see cref="Run(Func{int, Pipeline}, CancellationToken)"/> refuses; the earlier trials skip a number, are not of
    /// the space, declare another network, or were dealt in other kinds or on another number of deals; or the best of them is
    /// an earlier one whose network, trained again on every deal, is judged otherwise than it was.
    /// </exception>
    /// <remarks>
    /// The networks of a trial are not kept, so where the best is an earlier trial, its network is trained again on every deal
    /// — the same declaration on the same rows, and so, on one engine, the same numbers — and those networks are the winner.
    /// </remarks>
    public StudyResult Run(Func<int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);

        return Run(earlier, needs => DealPreparation.PerDeal(pipelineFor, Kinds, needs, cancellation), cancellation);
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

        return Run(earlier, needs => DealPreparation.PerCandidate(pipelinesFor, Kinds, needs, cancellation), cancellation);
    }

    // The study, once it is known how a candidate's pipelines are prepared: the one loop every study runs, with the networks
    // a candidate declares trained and judged on each deal.
    private StudyResult Run(IReadOnlyList<Trial> earlier, Func<Needs, DealPreparation> begin, CancellationToken cancellation)
    {
        var found = new StudyLoop<IReadOnlyList<TrainedNetwork>>(Space, _settings, new NetworksOfADeal(this)).Run(earlier, begin, cancellation);

        return new StudyResult(found.Trials, found.Best, found.Winner);
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

    // How a deal of a study of networks is trained and judged: the networks a candidate declares, each behind the pipeline
    // prepared for it, judged by the study's score, its judge, or its objectives.
    private sealed class NetworksOfADeal(Study study) : IDealTrainer<IReadOnlyList<TrainedNetwork>>
    {
        public Needs Needs => TrainedNetwork.FeatureNeeds;

        // A trial is judged by one shape of number, and stopped by nothing it is not judged by.
        public void RefuseHowItIsSet()
        {
            if (study.Judge is not null && (study._scoredItsOwnWay || study.Pruner is not null))
            {
                throw new InvalidOperationException(
                    $"A {nameof(Judge)} judges a trial by what its networks do together, and this study also has a {(study._scoredItsOwnWay ? nameof(Score) : nameof(Pruner))}: "
                    + "the trial would be judged by two numbers, or stopped by one it is not judged by. Leave one of the two out.");
            }

            if (study.Pruner is not null && study._scoredItsOwnWay)
            {
                throw new InvalidOperationException(
                    $"A {nameof(Pruner)} hears each epoch's validation loss, and this study judges its trials by a {nameof(Score)} of its own: a pruner would stop trials by a number they are not judged by. Leave one of the two out.");
            }

            if (study.Objectives is not null && (study._scoredItsOwnWay ? nameof(Score) : study.Judge is not null ? nameof(Judge) : study.Pruner is not null ? nameof(Pruner) : study.DealStopper is not null ? nameof(DealStopper) : null) is { } against)
            {
                throw new InvalidOperationException(
                    $"{nameof(Objectives)} judges a trial by several numbers, and this study also has a {against}: the trial would be judged by two shapes of number, "
                    + "or stopped by one it is not judged by that way. Leave one of the two out.");
            }
        }

        // An earlier trial declares the networks this study declares for its candidate.
        public void RequireItsOwn(Trial earlier)
        {
            if (earlier.Declared is null)
            {
                throw new InvalidOperationException($"Earlier trial {earlier.Number} declares no network, as a trial a caller-trained study tried does: it was tried by another kind of study.");
            }

            if (!earlier.Members.SequenceEqual(study.DeclaredBy(earlier.Candidate)))
            {
                throw new InvalidOperationException($"Earlier trial {earlier.Number} declares another network than this study declares for its candidate: it was tried by another study.");
            }
        }

        public IReadOnlyList<LearnNetworkStep> Declares(Candidate candidate, int trial)
        {
            var steps = study.DeclaredBy(candidate);

            if (steps.Count > 1 && study.Judge is null && study.Objectives is null)
            {
                throw new InvalidOperationException(
                    $"Trial {trial} declares {steps.Count} networks, and no score of one of them says how they do together: give the study a {nameof(Judge)} or an {nameof(Objectives)}.");
            }

            return steps;
        }

        // The networks a trial declared, each trained on the rows its own pipeline prepared for the deal, and judged together.
        public JudgedDeal<IReadOnlyList<TrainedNetwork>> Trained(DealToTrain deal)
        {
            var steps = deal.Trial.Steps;

            if (deal.Rows.Count != steps.Count)
            {
                throw new InvalidOperationException($"A trial declares {steps.Count} networks and was handed {deal.Rows.Count} pipelines: one for each network.");
            }

            var controls = new RunControls(study.Engines, deal.Heard is { } heard ? epoch => heard(epoch.ValidationLoss!.Value) : null, deal.Cancellation);
            IReadOnlyList<TrainedNetwork> networks = [.. deal.Rows.Select((rows, at) => rows.Train(steps[at], controls))];

            // Where no pruner heard the epochs, the first network's own history says what the deal heard.
            IReadOnlyList<double>? ofItself = deal.Heard is null ? [.. networks[0].History!.Epochs.Select(epoch => epoch.ValidationLoss!.Value)] : null;

            return study.Objectives is { } objectives
                ? new(networks, double.NaN) { Objectives = objectives(networks), Heard = ofItself }
                : new(networks, study.Judge is { } judge ? judge(networks) : study.Score(networks[0])) { Heard = ofItself };
        }

        public InvalidOperationException JudgedOtherwise(Trial best) => new(
            $"The best of the earlier trials, trial {best.Number}, was trained again on every deal and judged otherwise than it was handed back: these are not the rows, "
            + "the engine or the trials this study tried. Hand it the pipelines and the engines the trials were tried with.");
    }
}
