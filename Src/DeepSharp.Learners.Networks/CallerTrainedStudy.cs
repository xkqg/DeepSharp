// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>What one deal of a caller-trained candidate gave back: the number it is judged by, and the model it trained.</summary>
/// <param name="Score">How the model was judged, lower being better: the same number a trial's <see cref="Trial.Scores"/> keeps.</param>
/// <param name="Model">The model trained for this deal.</param>
public readonly record struct Trained<TModel>(double Score, TModel Model);

/// <summary>What a caller's trainer is handed to train one deal of one trial.</summary>
/// <param name="Candidate">The values the sampler gave.</param>
/// <param name="Rows">The pipeline's rows, prepared for this deal: training, validation and whatever else the pipeline sets aside.</param>
/// <param name="Heard">
/// What the trainer reports a raw loss to, once for every step it judges progress by; nothing, where the study has no
/// <see cref="Study{TModel}.Pruner"/>. The trainer checks <see cref="Cancellation"/> after reporting, and stops there — a
/// pruner that said yes has already requested it.
/// </param>
/// <param name="Cancellation">What the trainer checks to stop early: a pruner that pruned this deal, or the token the run was given.</param>
public readonly record struct TrainingDeal(Candidate Candidate, PreparedData Rows, Action<double>? Heard, CancellationToken Cancellation);

/// <summary>What a caller-trained study found: every trial, the best of them, and the model trained for each of its deals.</summary>
public sealed class StudyResult<TModel>
{
    internal StudyResult(IReadOnlyList<Trial> trials, Trial best, IReadOnlyList<TModel> winner)
    {
        Trials = trials;
        Best = best;
        Winner = winner;
    }

    /// <summary>Every trial, in the order they were tried.</summary>
    public IReadOnlyList<Trial> Trials { get; }

    /// <summary>The trial judged lowest, the earlier one when two were judged alike.</summary>
    public Trial Best { get; }

    /// <summary>The models the best trial trained, one for each deal. They are the only models a study keeps.</summary>
    public IReadOnlyList<TModel> Winner { get; }
}

/// <summary>
/// Tries many candidates for one model, each trained by a function of the caller's rather than declared as a network step,
/// judged by the number that function gives back, and keeps the one judged best.
/// </summary>
/// <remarks>
/// <para>
/// Everything a <see cref="Study"/> of networks keeps is kept here too: the sampler, the deals, a pruner heard once a step
/// at a time, a deal stopper heard once a deal at a time, <see cref="OnTrial"/>, and resuming from the trials an earlier
/// run already tried. Only what trains, and the model trained, is the caller's: handed a <see cref="TrainingDeal"/>, the
/// trainer returns the number its trial is judged by and the model it trained, in one call.
/// </para>
/// <para>
/// There is no <see cref="Study.Score"/>, <see cref="Study.Judge"/> or <see cref="Study.Objectives"/> here: the trainer's
/// own returned score is the one number a trial is judged by, so none of the three means anything to a study whose
/// candidate is not a network — a trial here never declares more than one model.
/// </para>
/// </remarks>
public sealed class Study<TModel>
{
    private readonly Func<TrainingDeal, Trained<TModel>> _train;

    /// <summary>Declares a study whose candidates are trained by a function of the caller's.</summary>
    /// <param name="space">What may vary.</param>
    /// <param name="train">Trains one deal of one candidate, and returns the score it is judged by and the model it trained.</param>
    /// <exception cref="ArgumentNullException">There is no space, or no way to train a candidate.</exception>
    public Study(SearchSpace space, Func<TrainingDeal, Trained<TModel>> train)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentNullException.ThrowIfNull(train);

        Space = space;
        _train = train;
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

    /// <summary>
    /// What hears a raw loss the trainer reports, once a step, and says whether to stop the deal there; nothing, unless said,
    /// and then the trainer's <see cref="TrainingDeal.Heard"/> is nothing, and no deal is ever stopped this way.
    /// </summary>
    /// <remarks>It hears the deal, the raw losses so far, and the trials before this one, on the thread the study runs on.</remarks>
    public Func<Progress, bool>? Pruner { get; init; }

    /// <summary>
    /// What hears every deal a trial finishes, judged by the number the trainer returned, and says whether to stop the trial
    /// there without running its remaining deals; nothing, unless said, and then every trial runs every deal.
    /// </summary>
    public Func<DealsJudged, bool>? DealStopper { get; init; }

    /// <summary>
    /// What is handed every trial as it ends, in order, on the thread the study runs on, with how it was judged; nothing, unless said.
    /// </summary>
    public Action<Trial>? OnTrial { get; init; }

    /// <summary>Runs the study.</summary>
    /// <param name="pipelineFor">The pipeline for a deal, given its place from nought.</param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next deal; nothing stops it, unless said.</param>
    /// <returns>Every trial, the best of them, and the models it trained.</returns>
    public StudyResult<TModel> Run(Func<int, Pipeline> pipelineFor, CancellationToken cancellation = default) => Run(pipelineFor, [], cancellation);

    /// <summary>Runs the study on from the trials it already tried.</summary>
    /// <param name="pipelineFor">The pipeline for a deal, as the study that tried them was handed it.</param>
    /// <param name="earlier">The trials already tried, as the study that tried them handed them back.</param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next deal; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the models the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="InvalidOperationException">
    /// A pipeline sets no rows aside as validation; the deals do not share their test rows; a sampler gives a candidate that
    /// is not of the space; an earlier trial declares a network, which this study's candidates never do; or no trial finished.
    /// </exception>
    public StudyResult<TModel> Run(Func<int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);
        ArgumentNullException.ThrowIfNull(earlier);

        ThrowIfNotThisStudys(earlier);

        var prepared = Prepared(pipelineFor, cancellation);
        var trials = new List<Trial>(earlier);
        var best = BestOf(earlier.Where(trial => trial.Finished));
        IReadOnlyList<TModel> winner = [];

        for (var number = earlier.Count; number < Trials; number++)
        {
            cancellation.ThrowIfCancellationRequested();

            var candidate = Sampler.Suggest(Space, number, [.. trials]);

            ThrowIfNotOfTheSpace(candidate);

            var (trial, models) = Tried(candidate, prepared, trials, cancellation);

            trials.Add(trial);
            OnTrial?.Invoke(trial);

            if (trial.Finished && (best is null || VerdictOf(trial).Better(VerdictOf(best))))
            {
                best = trial;
                winner = models;
            }
        }

        if (best is null)
        {
            throw new InvalidOperationException($"No trial finished: all {trials.Count} were pruned, or judged by a number that is none, or infinite. Try smaller steps, or look at the candidates the space allows.");
        }

        return new StudyResult<TModel>(trials, best, best.Number < earlier.Count ? TrainedAgain(best, prepared, cancellation) : winner);
    }

    // One trial, numbered after the trials before it: its model trained on every deal, each report handed to the pruner, and
    // each finished deal's score handed to the deal stopper, either of which may stop it.
    private (Trial Trial, IReadOnlyList<TModel> Models) Tried(Candidate candidate, PreparedData[] prepared, List<Trial> before, CancellationToken cancellation)
    {
        var models = new List<TModel>();
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

            void Heard(double loss)
            {
                soFar.Add(loss);

                if (Pruner?.Invoke(new Progress(at, [.. soFar], [.. before])) is true)
                {
                    pruned = true;
                    stop.Cancel();
                }
            }

            try
            {
                var trained = _train(new TrainingDeal(candidate, prepared[deal], Pruner is null ? null : Heard, stop.Token));

                if (!pruned)
                {
                    models.Add(trained.Model);
                    dealScores.Add(trained.Score);

                    if (deal < deals - 1 && DealStopper?.Invoke(new DealsJudged([.. dealScores], [.. before])) is true)
                    {
                        stoppedBetweenDeals = true;
                    }
                }
            }
            catch (OperationCanceledException) when (pruned && !cancellation.IsCancellationRequested)
            {
                // The pruner stopped the deal: the trial ends here, with what it heard.
            }

            heard.Add(soFar);
        }

        var trial = new Trial(before.Count, candidate, Declared: null, pruned ? [] : dealScores)
        {
            Pruned = pruned || stoppedBetweenDeals,
            Heard = heard,
        };

        return (trial, models);
    }

    // The verdict a finished trial stands for: its score, as a vector of one.
    private static Verdict VerdictOf(Trial trial) => new([trial.Score]);

    // The finished trial whose verdict is best, the earlier one on a tie; none, where there is no finished trial.
    private static Trial? BestOf(IEnumerable<Trial> finished)
    {
        Trial? best = null;

        foreach (var trial in finished)
        {
            if (best is null || VerdictOf(trial).Better(VerdictOf(best)))
            {
                best = trial;
            }
        }

        return best;
    }

    // The best trial was one the caller handed back, whose models were not kept: trained again through the handed-in trainer,
    // it must be judged as it was.
    private IReadOnlyList<TModel> TrainedAgain(Trial best, PreparedData[] prepared, CancellationToken cancellation)
    {
        Trained<TModel>[] trained = [.. Enumerable.Range(0, Deals).Select(deal => _train(new TrainingDeal(best.Candidate, prepared[deal], null, cancellation)))];

        if (!trained.Select(each => each.Score).SequenceEqual(best.Scores))
        {
            throw new InvalidOperationException(
                $"The best of the earlier trials, trial {best.Number}, was trained again on every deal through the handed-in trainer and judged otherwise than it was handed back: "
                + "this is not the trainer, or not the rows, this study tried it with. Hand it the same pipelines and the same trainer the trials were tried with.");
        }

        return [.. trained.Select(each => each.Model)];
    }

    // Earlier trials are this study's own: numbered from nought without a gap, of its space, and trained by a function of the
    // caller's rather than declaring a network.
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

            if (trial.Declared is not null)
            {
                throw new InvalidOperationException($"Earlier trial {at} declares a network, and this study's candidates are trained by a function of the caller's: it was tried by another kind of study.");
            }
        }
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

        foreach (var setting in candidate.Settings)
        {
            if (Space.Single(dimension => dimension.Name == setting.Name).Refusal(setting) is { } refusal)
            {
                throw new InvalidOperationException($"The sampler gave a setting the space cannot take: '{setting.Name}' {refusal}.");
            }
        }
    }

    /// <summary>Trains several candidates on the rows a run of a pipeline prepared, through this study's own trainer, and puts them together as a <see cref="ModelEnsemble{TModel}"/>.</summary>
    /// <param name="rows">The rows every candidate trains on: one fit, so the members answer rows replayed once for all of them.</param>
    /// <param name="candidates">The candidates to train: at least two.</param>
    /// <param name="answer">What a model answers for rows of features, in the shape its pipeline hands them over.</param>
    /// <param name="options">How their answers become one, and what stops the run; averaged and unstoppable, unless said.</param>
    /// <returns>The ensemble.</returns>
    /// <exception cref="ArgumentNullException">There are no rows, no candidates, or no way to answer with a model.</exception>
    /// <exception cref="ArgumentException">There are fewer than two candidates, or the options name an engine.</exception>
    /// <remarks>Every candidate trains on the very same rows, so the members are trained behind one fit by construction: no digest is compared.</remarks>
    public ModelEnsemble<TModel> TrainEnsemble(PreparedData rows, IReadOnlyList<Candidate> candidates, Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> answer, EnsembleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(answer);

        if (options?.Engines is not null)
        {
            throw new ArgumentException(
                "An ensemble's members are trained by a function of the caller's, and the engines a declaration's names stand for mean nothing to it: leave Engines out.",
                nameof(options));
        }

        if (candidates.Count < 2)
        {
            throw new ArgumentException(
                $"An ensemble is of at least two candidates, and {candidates.Count} {(candidates.Count == 1 ? "was" : "were")} given: a single model predicts for itself.",
                nameof(candidates));
        }

        var cancellation = options?.Cancellation ?? default;
        TModel[] trained = [.. candidates.Select(candidate => _train(new TrainingDeal(candidate, rows, null, cancellation)).Model)];

        return new ModelEnsemble<TModel>(trained, answer, rows, options?.Blend ?? Blend.Average);
    }

    private PreparedData[] Prepared(Func<int, Pipeline> pipelineFor, CancellationToken cancellation)
    {
        var prepared = new PreparedData[Deals];

        for (var deal = 0; deal < prepared.Length; deal++)
        {
            cancellation.ThrowIfCancellationRequested();

            prepared[deal] = (pipelineFor(deal) ?? throw new InvalidOperationException($"The pipeline for deal {deal} is nothing.")).RunToBeJudged();
        }

        var first = prepared[0].Batch(Part.Test, Needs.Numbers).Keys!.ToHashSet();

        for (var deal = 1; deal < prepared.Length; deal++)
        {
            if (!first.SetEquals(prepared[deal].Batch(Part.Test, Needs.Numbers).Keys!))
            {
                throw new InvalidOperationException(
                    $"Deal {deal} measures on other rows than deal 0 does: a candidate would be chosen on rows another deal learned from. Give the split a test seed — SplitAtRandom(train, validation, seed, testSeed) — so the rows to be measured on are the same on every deal.");
            }
        }

        return prepared;
    }
}
