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

/// <summary>
/// Tries many candidates for one model, each trained by a function of the caller's rather than declared as a network step,
/// judged by the number that function gives back, and keeps the one judged best.
/// </summary>
/// <remarks>
/// <para>
/// Everything a <see cref="Study"/> of networks keeps is kept here too: the sampler, the deals, a pruner heard once a step
/// at a time, a deal stopper heard once a deal at a time, <see cref="OnTrial"/>, a pipeline for each candidate and deal,
/// and resuming from the trials an earlier run already tried — the two run one and the same loop. Only what trains, and the
/// model trained, is the caller's: handed a <see cref="TrainingDeal"/>, the trainer returns the number its trial is judged by
/// and the model it trained, in one call.
/// </para>
/// <para>
/// The pipelines are run for a learner of numbers, every step as declared (<see cref="Needs.Numbers"/>): the caller's model
/// takes the rows as it chooses, and no feature has to be put on one scale, as a network's would.
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
    private readonly StudySettings _settings = new();

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
        get => _settings.Sampler;
        init => _settings = _settings with { Sampler = value };
    }

    /// <summary>How many candidates are tried.</summary>
    public int Trials
    {
        get => _settings.Trials;
        init => _settings = _settings with { Trials = value };
    }

    /// <summary>How many times the rows are divided again, each candidate being trained on every division.</summary>
    public int Deals
    {
        get => _settings.Deals;
        init => _settings = _settings with { Deals = value };
    }

    /// <summary>
    /// What hears a raw loss the trainer reports, once a step, and says whether to stop the deal there; nothing, unless said,
    /// and then the trainer's <see cref="TrainingDeal.Heard"/> is nothing, and no deal is ever stopped this way.
    /// </summary>
    /// <remarks>It hears the deal, the raw losses so far, and the trials before this one, on the thread the study runs on.</remarks>
    public Func<Progress, bool>? Pruner
    {
        get => _settings.Pruner;
        init => _settings = _settings with { Pruner = value };
    }

    /// <summary>
    /// What hears every deal a trial finishes, judged by the number the trainer returned, and says whether to stop the trial
    /// there without running its remaining deals; nothing, unless said, and then every trial runs every deal.
    /// </summary>
    public Func<DealsJudged, bool>? DealStopper
    {
        get => _settings.DealStopper;
        init => _settings = _settings with { DealStopper = value };
    }

    /// <summary>
    /// What is handed every trial as it ends, in order, on the thread the study runs on, with how it was judged; nothing, unless said.
    /// </summary>
    public Action<Trial>? OnTrial
    {
        get => _settings.OnTrial;
        init => _settings = _settings with { OnTrial = value };
    }

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

        return Run(earlier, needs => DealPreparation.PerDeal(pipelineFor, Deals, needs, cancellation), cancellation);
    }

    /// <summary>Runs the study with a pipeline for each candidate and each deal.</summary>
    /// <param name="pipelineFor">
    /// The pipeline for a candidate on a deal, given the candidate and the deal's place from nought: the steps the candidate chooses —
    /// a feature, a window, a way of filling a gap — and the split dealing the validation rows and the rows to be measured on the
    /// same way for every candidate.
    /// </param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next deal; nothing stops it, unless said.</param>
    /// <returns>Every trial, the best of them, and the models it trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">
    /// Anything <see cref="Run(Func{int, Pipeline}, IReadOnlyList{Trial}, CancellationToken)"/> refuses; or a candidate's pipeline
    /// sets other rows aside as validation on a deal, or to be measured on, than the first one prepared did.
    /// </exception>
    /// <remarks>
    /// The pipeline is prepared for every trial and deal, and the trainer is handed the rows the candidate's own pipeline prepared;
    /// the study that is handed one for each deal prepares it once for each. Scores are chosen between only where the rows they
    /// were measured on are the same, so every candidate is held to the validation rows the first trial was judged on for that deal,
    /// and every deal to the rows the first one prepared to be measured on.
    /// </remarks>
    public StudyResult<TModel> Run(Func<Candidate, int, Pipeline> pipelineFor, CancellationToken cancellation = default) => Run(pipelineFor, [], cancellation);

    /// <summary>Runs the study on from the trials it already tried, with a pipeline for each candidate and each deal.</summary>
    /// <param name="pipelineFor">The pipeline for a candidate on a deal, as the study that tried the earlier trials was handed it.</param>
    /// <param name="earlier">The trials already tried, as <see cref="Run(Func{int, Pipeline}, IReadOnlyList{Trial}, CancellationToken)"/> takes them.</param>
    /// <param name="cancellation">What stops the study before its next trial, and a trial before its next deal; nothing stops it, unless said.</param>
    /// <returns>Every trial, the earlier ones first, the best of them, and the models the best trained.</returns>
    /// <exception cref="ArgumentNullException">There is no way to make the pipelines, or no earlier trials.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled: nothing is returned.</exception>
    /// <exception cref="InvalidOperationException">Anything the other forms of <c>Run</c> refuse.</exception>
    public StudyResult<TModel> Run(Func<Candidate, int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);

        return Run(earlier, needs => DealPreparation.PerCandidate((candidate, deal) => [pipelineFor(candidate, deal)], needs, cancellation), cancellation);
    }

    // The study, once it is known how a candidate's pipelines are prepared: the one loop every study runs, with the model the
    // caller's trainer gives back for each deal.
    private StudyResult<TModel> Run(IReadOnlyList<Trial> earlier, Func<Needs, DealPreparation> begin, CancellationToken cancellation)
    {
        var found = new StudyLoop<TModel>(Space, _settings, new ModelOfADeal(_train)).Run(earlier, begin, cancellation);

        return new StudyResult<TModel>(found.Trials, found.Best, found.Winner);
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

    // How a deal of a caller-trained study is trained and judged: by the caller's trainer, handed the rows of the deal, whose
    // returned score is the one number the trial is judged by.
    private sealed class ModelOfADeal(Func<TrainingDeal, Trained<TModel>> train) : IDealTrainer<TModel>
    {
        // A learner of numbers: every step as declared, and no scale asked of any feature, since the caller's model asks its own.
        public Needs Needs => Needs.Numbers;

        // The caller's trainer gives one number a deal, and nothing else is there to judge or stop a trial by another.
        public void RefuseHowItIsSet()
        {
        }

        public void RequireItsOwn(Trial earlier)
        {
            if (earlier.Declared is not null)
            {
                throw new InvalidOperationException($"Earlier trial {earlier.Number} declares a network, and this study's candidates are trained by a function of the caller's: it was tried by another kind of study.");
            }
        }

        public IReadOnlyList<LearnNetworkStep> Declares(Candidate candidate, int trial) => [];

        public JudgedDeal<TModel> Trained(DealToTrain deal)
        {
            var trained = train(new TrainingDeal(deal.Trial.Candidate, deal.Rows[0], deal.Heard, deal.Cancellation));

            return new(trained.Model, trained.Score);
        }

        public InvalidOperationException JudgedOtherwise(Trial best) => new(
            $"The best of the earlier trials, trial {best.Number}, was trained again on every deal through the handed-in trainer and judged otherwise than it was handed back: "
            + "this is not the trainer, or not the rows, this study tried it with. Hand it the same pipelines and the same trainer the trials were tried with.");
    }
}
