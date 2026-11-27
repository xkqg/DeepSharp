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
/// The pipelines are run for what the caller's model says it needs of its features, <see cref="Needs"/>: a learner of numbers,
/// every step as declared, unless said — the caller's model takes the rows as it chooses, and no feature has to be put on one
/// scale, as a network's would.
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

    /// <summary>
    /// What the caller's model needs of the features it is handed: every pipeline is run for it, every deal's rows handed over for it,
    /// and an ensemble's members served so; a learner of numbers, every step as declared, unless said.
    /// </summary>
    /// <remarks>A model that takes each category as its place says <see cref="Needs.Categories"/>, and is served each category as its place.</remarks>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by the value.</exception>
    public Needs Needs
    {
        get;
        init => field = value.Named();
    }

    = Needs.Numbers;

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
    /// The kinds the deals are of, laid out kind after kind, each with a weight in a trial's score; unless said, one kind of
    /// every deal, weighing one. They work as a <see cref="Study.Kinds"/> does: a trial is judged by each kind's mean, weighed.
    /// </summary>
    /// <remarks>
    /// The promise is made per kind: a candidate is chosen on no row its own kind measures on; a row one kind measures on may be
    /// learned from by another kind, so a kind's test number is honest only within that kind.
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
    /// A pipeline sets no rows aside as validation; the deals of a kind do not share their test rows; a sampler gives a candidate that
    /// is not of the space; an earlier trial declares a network, which this study's candidates never do, or was dealt in other
    /// kinds or on another number of deals; or no trial finished.
    /// </exception>
    public StudyResult<TModel> Run(Func<int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier, CancellationToken cancellation = default)
    {
        ArgumentNullException.ThrowIfNull(pipelineFor);

        return Run(earlier, needs => DealPreparation.PerDeal(pipelineFor, Kinds, needs, cancellation), cancellation);
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
    /// and every deal to the rows the first one of its kind prepared to be measured on.
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

        return Run(earlier, needs => DealPreparation.PerCandidate((candidate, deal) => [pipelineFor(candidate, deal)], Kinds, needs, cancellation), cancellation);
    }

    // The study, once it is known how a candidate's pipelines are prepared: the one loop every study runs, with the model the
    // caller's trainer gives back for each deal.
    private StudyResult<TModel> Run(IReadOnlyList<Trial> earlier, Func<Needs, DealPreparation> begin, CancellationToken cancellation)
    {
        var found = new StudyLoop<TModel>(Space, _settings, new ModelOfADeal(_train, Needs)).Run(earlier, begin, cancellation);

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
    /// <exception cref="InvalidOperationException">
    /// The training rows are refused as a learner that needs what <see cref="Needs"/> says is refused them: the ensemble serves rows so;
    /// or the blend is a vote, and the answers are amounts.
    /// </exception>
    /// <remarks>
    /// The form of <see cref="TrainEnsemble(IReadOnlyList{Candidate}, Func{Candidate, PreparedData}, Func{TModel, IReadOnlyList{double[]}, IReadOnlyList{double[]}}, EnsembleOptions?)"/>
    /// that hands every candidate the very same rows, so the members are trained behind one pipeline, which is the ensemble's
    /// <see cref="ModelEnsemble{TModel}.Prepared"/>.
    /// </remarks>
    public ModelEnsemble<TModel> TrainEnsemble(PreparedData rows, IReadOnlyList<Candidate> candidates, Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> answer, EnsembleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rows);

        return TrainEnsemble(candidates, _ => rows, answer, options);
    }

    /// <summary>
    /// Trains several candidates, each on the rows a pipeline of its own prepared, through this study's own trainer, and puts them
    /// together as a <see cref="ModelEnsemble{TModel}"/>: an ensemble of whole models, each with its own steps.
    /// </summary>
    /// <param name="candidates">The candidates to train: at least two.</param>
    /// <param name="rowsFor">The run of the pipeline a candidate trains on: the same for every candidate, or one of its own each.</param>
    /// <param name="answer">What a model answers for rows of features, in the shape its pipeline hands them over.</param>
    /// <param name="options">How their answers become one, and what stops the run; averaged and unstoppable, unless said.</param>
    /// <returns>The ensemble.</returns>
    /// <exception cref="ArgumentNullException">There are no candidates, no way to find their rows, or no way to answer with a model.</exception>
    /// <exception cref="ArgumentException">
    /// There are fewer than two candidates, the options name an engine, or a member answers another question than the first: other
    /// answers, or the same named by another output.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The rows for a candidate are nothing, a member's training rows are refused as a learner that needs what <see cref="Needs"/> says
    /// is refused them, or the blend is a vote and the answers are amounts.
    /// </exception>
    /// <remarks>
    /// Each member is served rows later through its own pipeline, as <see cref="Needs"/> says, and comes back through that pipeline's
    /// own way back, so the members' answers meet in the answer's own units however their pipelines shape it on the way in. The
    /// features a member's training rows hand over are what it says it learned from, so a row its pipeline serves as other features is
    /// refused, as it is of a network. The ensemble is measured and calibrated on rows every member set aside alike.
    /// </remarks>
    public ModelEnsemble<TModel> TrainEnsemble(IReadOnlyList<Candidate> candidates, Func<Candidate, PreparedData> rowsFor, Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> answer, EnsembleOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(rowsFor);
        ArgumentNullException.ThrowIfNull(answer);

        if (options?.Engines is not null)
        {
            throw new ArgumentException(
                "An ensemble's members are trained by a function of the caller's, and the engines a declaration's names stand for mean nothing to it: leave Engines out.",
                nameof(options));
        }

        candidates.ThrowIfFewerThanTwo("candidates", "a single model predicts for itself", nameof(candidates));

        var cancellation = options?.Cancellation ?? default;
        var needs = Needs;
        ModelMember<TModel>[] members =
        [
            .. candidates.Select((candidate, at) =>
            {
                var rows = rowsFor(candidate) ?? throw new InvalidOperationException(
                    $"The rows for candidate {at + 1} are nothing: rowsFor hands each candidate the run of the pipeline it trains on.");
                var model = _train(new TrainingDeal(candidate, rows, null, cancellation)).Model;

                // The member learned from the training rows as they are handed over for what it needs, which is how it is served rows.
                return new ModelMember<TModel>(model, new ServedAnswer(rows, needs, new LearnedFrom("ensemble", rows.Batch(Part.Train, needs).FeatureNames)), answer);
            }),
        ];

        return new ModelEnsemble<TModel>([.. members.Select(member => member.Model)], new Ensemble(members, options?.Blend ?? Blend.Average));
    }

    // How a deal of a caller-trained study is trained and judged: by the caller's trainer, handed the rows of the deal, whose
    // returned score is the one number the trial is judged by.
    private sealed class ModelOfADeal(Func<TrainingDeal, Trained<TModel>> train, Needs needs) : IDealTrainer<TModel>
    {
        // What the study says the caller's model needs: a learner of numbers, every step as declared, unless said.
        public Needs Needs => needs;

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
