// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// What every study is set by, whatever trains its candidates: which candidates it tries, how many, on how many deals, and
/// what hears and may stop its trials. Each guard and each default is said here once, for both faces of a study.
/// </summary>
internal sealed record StudySettings
{
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

    private readonly int? _deals;
    private readonly IReadOnlyList<DealKind>? _kinds;

    /// <summary>
    /// How many times the rows are divided again, each candidate being trained on every division: as many as the kinds hold,
    /// where they are said, and one, unless either is said.
    /// </summary>
    public int Deals
    {
        get => _deals ?? _kinds?.Sum(kind => kind.Deals) ?? 1;
        init
        {
            if (value < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "A study uses at least one division of the rows.");
            }

            _deals = _kinds is { } kinds ? Agreeing(value, kinds) : value;
        }
    }

    /// <summary>The kinds the deals are of, kind after kind; one kind of every deal, weighing one, unless said.</summary>
    public IReadOnlyList<DealKind> Kinds
    {
        get => _kinds ?? [new DealKind("deals", Deals, 1)];
        init
        {
            ArgumentNullException.ThrowIfNull(value, nameof(Kinds));

            if (value.Count == 0)
            {
                throw new ArgumentException("A study's deals are of at least one kind.", nameof(Kinds));
            }

            if (value.Any(kind => kind.Name is null))
            {
                throw new ArgumentException("A kind of no deals was given: a kind is declared with new DealKind(name, deals, weight).", nameof(Kinds));
            }

            if (value.GroupBy(kind => kind.Name, StringComparer.Ordinal).FirstOrDefault(named => named.Count() > 1) is { } twice)
            {
                throw new ArgumentException($"The kind '{twice.Key}' is named twice: a refusal names a kind, and two of one name cannot be told apart.", nameof(Kinds));
            }

            _kinds = [.. value];

            if (_deals is { } deals)
            {
                Agreeing(deals, _kinds);
            }
        }
    }

    // The deals, where the kinds hold as many; refused, naming both, where they do not.
    private static int Agreeing(int deals, IReadOnlyList<DealKind> kinds) =>
        kinds.Sum(kind => kind.Deals) == deals
            ? deals
            : throw new ArgumentException(
                $"{nameof(Deals)} says {deals} deals and {nameof(Kinds)} holds {kinds.Sum(kind => kind.Deals)}: say one of the two, or both alike.",
                nameof(Kinds));

    /// <summary>What hears each loss a deal reports as it trains, and says whether to stop the trial there.</summary>
    public Func<Progress, bool>? Pruner { get; init; }

    /// <summary>What hears every deal a trial finishes, and says whether to stop the trial before its next deal.</summary>
    public Func<DealsJudged, bool>? DealStopper { get; init; }

    /// <summary>What is handed every trial as it ends.</summary>
    public Action<Trial>? OnTrial { get; init; }
}

/// <summary>
/// What trains and judges one deal of one candidate, for one face of a study: the loop the faces share asks it, and nothing
/// else about a trial is the face's own.
/// </summary>
/// <typeparam name="TTrained">What one deal trains: the networks of a candidate, or a model of the caller's.</typeparam>
internal interface IDealTrainer<TTrained>
{
    /// <summary>What the pipelines are run for: what the thing trained behind them needs of their features.</summary>
    Needs Needs { get; }

    /// <summary>Refuses what this face is set to that cannot judge a trial honestly, before anything is prepared.</summary>
    void RefuseHowItIsSet();

    /// <summary>Refuses an earlier trial another kind of study tried, or one that declares otherwise than this one would.</summary>
    /// <param name="earlier">The earlier trial, already held to its number and the space.</param>
    void RequireItsOwn(Trial earlier);

    /// <summary>What a candidate's trial declares, once for the trial: its networks, or none for a model of the caller's.</summary>
    /// <param name="candidate">The candidate.</param>
    /// <param name="trial">The trial's number.</param>
    /// <returns>The declarations, the first of them the trial's own.</returns>
    IReadOnlyList<LearnNetworkStep> Declares(Candidate candidate, int trial);

    /// <summary>Trains one deal of one trial, and judges it.</summary>
    /// <param name="deal">The trial, the rows its deal prepared, what hears it train, and what stops it.</param>
    /// <returns>What the deal trained, and how it was judged.</returns>
    JudgedDeal<TTrained> Trained(DealToTrain deal);

    /// <summary>The refusal of an earlier best trial that, trained again, was judged otherwise than it was handed back.</summary>
    /// <param name="best">The earlier trial.</param>
    /// <returns>The refusal, naming what to hand the study.</returns>
    InvalidOperationException JudgedOtherwise(Trial best);
}

/// <summary>What a trial is: its candidate, and the networks it declared, none for a model of the caller's.</summary>
/// <param name="Candidate">The values the sampler gave.</param>
/// <param name="Steps">The declarations, once for the trial.</param>
internal readonly record struct DeclaredTrial(Candidate Candidate, IReadOnlyList<LearnNetworkStep> Steps);

/// <summary>One deal of one trial, as the face trains it.</summary>
/// <param name="Trial">The trial.</param>
/// <param name="Rows">The rows the deal prepared, one for each thing the trial trains.</param>
/// <param name="Heard">What a loss is reported to as the deal trains; nothing, where the study has no pruner.</param>
/// <param name="Cancellation">What stops the deal: a pruner that pruned it, or the token the study was run with.</param>
internal readonly record struct DealToTrain(DeclaredTrial Trial, IReadOnlyList<PreparedData> Rows, Action<double>? Heard, CancellationToken Cancellation);

/// <summary>What one deal trained, and the number it is judged by.</summary>
/// <param name="Trained">What the deal trained.</param>
/// <param name="Score">The one number the deal is judged by; no number for a deal weighed on objectives instead.</param>
internal readonly record struct JudgedDeal<TTrained>(TTrained Trained, double Score)
{
    /// <summary>The numbers the deal is weighed by on every objective; nothing, for a deal judged by its one score.</summary>
    public IReadOnlyList<double>? Objectives { get; init; }

    /// <summary>The losses the deal heard of itself as it trained, where no pruner was there to hear them; nothing otherwise.</summary>
    public IReadOnlyList<double>? Heard { get; init; }
}

/// <summary>What a study found, before its face puts it in its own shape: every trial, the best, and what the best trained on each deal.</summary>
/// <param name="Trials">Every trial, in the order they were tried.</param>
/// <param name="Best">The trial judged best.</param>
/// <param name="Winner">What the best trial trained, one for each deal.</param>
internal readonly record struct StudyOutcome<TTrained>(IReadOnlyList<Trial> Trials, Trial Best, IReadOnlyList<TTrained> Winner);

/// <summary>
/// The one loop every study runs: earlier trials held to the study, a candidate drawn and held to the space, its deals trained
/// one after another while a pruner and a deal stopper may stop it, every trial handed on as it ends, and the best kept by the
/// one verdict trials are ranked by.
/// </summary>
/// <typeparam name="TTrained">What one deal trains.</typeparam>
/// <remarks>
/// The faces of a study differ in what trains a candidate and how a deal is judged, and in nothing else; they hand that to
/// the loop as an <see cref="IDealTrainer{TTrained}"/>, and the loop does the rest the same way for both.
/// </remarks>
internal sealed class StudyLoop<TTrained>(SearchSpace space, StudySettings settings, IDealTrainer<TTrained> trainer)
{
    // The kinds the deals are of, read once: every trial carries them, and every earlier trial is held to them.
    private readonly IReadOnlyList<DealKind> _kinds = settings.Kinds;

    /// <summary>Runs the study on from its earlier trials.</summary>
    /// <param name="earlier">The trials already tried, as the study that tried them handed them back.</param>
    /// <param name="begin">
    /// Prepares the pipelines for what the trainer needs of their features: begun only once the earlier trials are held to this
    /// study, so a study refused for them prepares nothing.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next step.</param>
    /// <returns>Every trial, the best of them, and what the best trained on each deal.</returns>
    public StudyOutcome<TTrained> Run(IReadOnlyList<Trial> earlier, Func<Needs, DealPreparation> begin, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(earlier);

        trainer.RefuseHowItIsSet();
        RequireItsOwn(earlier);

        var prepare = begin(trainer.Needs);
        var trials = new List<Trial>(earlier);
        var best = earlier.Aggregate((Trial?)null, (sofar, trial) => Beats(trial, sofar) ? trial : sofar);
        IReadOnlyList<TTrained> winner = [];

        for (var number = earlier.Count; number < settings.Trials; number++)
        {
            cancellation.ThrowIfCancellationRequested();

            var candidate = settings.Sampler.Suggest(space, number, [.. trials]);

            space.Require(candidate);

            var tried = Tried(new DeclaredTrial(candidate, trainer.Declares(candidate, number)), prepare, trials, cancellation);

            trials.Add(tried.Trial);
            settings.OnTrial?.Invoke(tried.Trial);

            if (Beats(tried.Trial, best))
            {
                best = tried.Trial;
                winner = tried.Trained;
            }
        }

        if (best is null)
        {
            throw new InvalidOperationException($"No trial finished: all {trials.Count} were pruned, or judged by a number that is none, or infinite. Try smaller steps, or look at the candidates the space allows.");
        }

        return new StudyOutcome<TTrained>(trials, best, best.Number < earlier.Count ? TrainedAgain(best, prepare, cancellation) : winner);
    }

    // Whether a trial is better than the best so far: it finished, and its verdict is better; the earlier one stays on a tie.
    private static bool Beats(Trial trial, Trial? best) => trial.Finished && (best is null || trial.Verdict.Better(best.Verdict));

    // One trial, numbered after the trials before it: trained on every deal, each reported loss handed to the pruner and each
    // finished deal's score to the deal stopper, either of which may stop it.
    private TriedTrial Tried(DeclaredTrial declared, DealPreparation prepare, List<Trial> before, CancellationToken cancellation)
    {
        var trained = new List<TTrained>();
        var heard = new List<IReadOnlyList<double>>();
        var dealScores = new List<double>();
        var objectiveScores = new List<IReadOnlyList<double>>();
        var pruned = false;
        var stoppedBetweenDeals = false;
        var deals = settings.Deals;

        for (var deal = 0; deal < deals && !pruned && !stoppedBetweenDeals; deal++)
        {
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            var soFar = new List<double>();
            var at = deal;
            IReadOnlyList<double>? heardOfItself = null;

            void Heard(double loss)
            {
                soFar.Add(loss);

                if (settings.Pruner!(new Progress(at, [.. soFar], [.. before])))
                {
                    pruned = true;
                    stop.Cancel();
                }
            }

            try
            {
                var rows = prepare.Of(declared.Candidate, before.Count, deal);
                var judged = trainer.Trained(new DealToTrain(declared, rows, settings.Pruner is null ? null : Heard, stop.Token));

                heardOfItself = judged.Heard;

                if (!pruned)
                {
                    trained.Add(judged.Trained);

                    if (judged.Objectives is { } objectives)
                    {
                        objectiveScores.Add(objectives);
                    }
                    else
                    {
                        dealScores.Add(judged.Score);
                    }

                    if (deal < deals - 1 && settings.DealStopper?.Invoke(new DealsJudged([.. dealScores], [.. before]) { Kinds = _kinds }) is true)
                    {
                        stoppedBetweenDeals = true;
                    }
                }
            }
            catch (OperationCanceledException) when (pruned && !cancellation.IsCancellationRequested)
            {
                // The pruner stopped the deal: the trial ends here, with what it heard.
            }

            heard.Add(heardOfItself ?? soFar);
        }

        var steps = declared.Steps;
        var trial = new Trial(before.Count, declared.Candidate, steps.Count > 0 ? steps[0] : null, pruned ? [] : dealScores)
        {
            Pruned = pruned || stoppedBetweenDeals,
            Heard = heard,
            ObjectiveScores = pruned ? [] : objectiveScores,
            Kinds = _kinds,
        };

        return new TriedTrial(steps.Count > 1 ? trial with { Members = steps } : trial, trained);
    }

    // The best trial was one the caller handed back, whose deals were not kept: trained again, it must be judged as it was.
    private TTrained[] TrainedAgain(Trial best, DealPreparation prepare, CancellationToken cancellation)
    {
        var declared = new DeclaredTrial(best.Candidate, best.Members);
        var again = Enumerable.Range(0, settings.Deals)
            .Select(deal => trainer.Trained(new DealToTrain(declared, prepare.Of(best.Candidate, best.Number, deal), null, cancellation)))
            .ToArray();

        var matches = best.ObjectiveScores.Count > 0
            ? again.Select(judged => judged.Objectives!).SequenceEqual(best.ObjectiveScores, VectorEquality.Instance)
            : again.Select(judged => judged.Score).SequenceEqual(best.Scores);

        return matches ? [.. again.Select(judged => judged.Trained)] : throw trainer.JudgedOtherwise(best);
    }

    // Earlier trials are this study's own: numbered from nought without a gap, of its space, of the face's own kind, and dealt
    // as this study deals.
    private void RequireItsOwn(IReadOnlyList<Trial> earlier)
    {
        for (var at = 0; at < earlier.Count; at++)
        {
            var trial = earlier[at] ?? throw new ArgumentException($"Earlier trial {at} is nothing.", nameof(earlier));

            if (trial.Number != at)
            {
                throw new InvalidOperationException($"The earlier trials are numbered from nought without a gap, and the one at place {at} is trial {trial.Number}.");
            }

            space.Require(trial.Candidate);
            trainer.RequireItsOwn(trial);
            RequireDealtAlike(trial);
        }
    }

    // An earlier trial was dealt in this study's kinds — or in none, where this study's deals are of one kind, as every trial's
    // were before kinds were said — and judged on every deal they hold, or on fewer where it was stopped: its score is then
    // weighed as this study weighs a trial of its own.
    private void RequireDealtAlike(Trial earlier)
    {
        if (earlier.Kinds.Count == 0 ? _kinds.Count != 1 : !earlier.Kinds.SequenceEqual(_kinds))
        {
            throw new InvalidOperationException(
                $"Earlier trial {earlier.Number} was dealt in {Described(earlier.Kinds)}, and this study deals in {Described(_kinds)}: it was tried by a study of other deals. Hand it the kinds the trials were tried with.");
        }

        var judged = Math.Max(earlier.Scores.Count, earlier.ObjectiveScores.Count);
        var deals = settings.Deals;

        if (judged > deals || (!earlier.Pruned && judged != deals))
        {
            throw new InvalidOperationException(
                $"Earlier trial {earlier.Number} was judged on {judged} deals, and this study deals {deals}{(earlier.Pruned ? ", and no trial is judged on more" : ", and a trial that was not stopped is judged on every one")}: it was tried by a study of other deals.");
        }
    }

    // Kinds as a refusal names them.
    private static string Described(IReadOnlyList<DealKind> kinds) =>
        kinds.Count == 0 ? "no kinds" : $"the kinds {string.Join(", ", kinds.Select(kind => kind.Described))}";

    // A trial and what it trained, which only the best trial's are kept of: one for each deal it finished.
    private readonly record struct TriedTrial(Trial Trial, IReadOnlyList<TTrained> Trained);
}
