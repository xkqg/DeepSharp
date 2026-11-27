// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>Which deal a pipeline is prepared for, and for which trial or which side of a comparison: what a refusal names.</summary>
/// <param name="Deal">The deal, from nought.</param>
internal readonly record struct DealPlace(int Deal)
{
    /// <summary>The trial the pipeline is prepared for; nothing, for a pipeline prepared once for every trial of its deal.</summary>
    public int? Trial { get; init; }

    /// <summary>Which of a comparison's two pipelines it is; nothing, for a study's.</summary>
    public string? Side { get; init; }

    /// <summary>The pipeline this place names, as a refusal begins with it.</summary>
    public string Pipeline =>
        Side is { } side ? $"The {side} pipeline for deal {Deal}"
        : Trial is { } trial ? $"The pipeline for trial {trial}, deal {Deal}"
        : $"The pipeline for deal {Deal}";
}

/// <summary>
/// How a study's pipelines are prepared: once for each deal, whatever the candidates, or for each candidate on each deal, and
/// in either form held to the rows the first of a kind was measured on and the first of a deal chosen on.
/// </summary>
/// <remarks>
/// Both faces of a study prepare their rows here, so what a pipeline is refused for, and the rows a candidate is held to, are the
/// same whether a network or a model of the caller's trains behind it.
/// </remarks>
internal abstract class DealPreparation
{
    private readonly MeasuredRows _measured;
    private readonly Needs _needs;

    private DealPreparation(IReadOnlyList<DealKind> kinds, Needs needs, CancellationToken cancellation)
    {
        _measured = new MeasuredRows(kinds);
        _needs = needs;
        Cancellation = cancellation;
    }

    /// <summary>What stops the preparation before its next pipeline.</summary>
    private protected CancellationToken Cancellation { get; }

    /// <summary>Prepares one pipeline for each deal, at once, before any trial: every candidate trains on those rows.</summary>
    /// <param name="pipelineFor">The pipeline for a deal, given its place from nought.</param>
    /// <param name="kinds">The kinds the deals are of, kind after kind: as many deals as they hold.</param>
    /// <param name="needs">What trains behind the pipelines needs of their features.</param>
    /// <param name="cancellation">What stops the preparation before its next deal.</param>
    /// <returns>The rows of every deal, held to the rows the first deal of its kind measures on.</returns>
    public static DealPreparation PerDeal(Func<int, Pipeline> pipelineFor, IReadOnlyList<DealKind> kinds, Needs needs, CancellationToken cancellation) =>
        new OncePerDeal(pipelineFor, kinds, needs, cancellation);

    /// <summary>Prepares the pipelines a candidate chooses for every trial and deal, as each trial comes to it.</summary>
    /// <param name="pipelinesFor">The pipelines for a candidate on a deal: one for each thing the candidate trains behind one.</param>
    /// <param name="kinds">The kinds the deals are of, kind after kind.</param>
    /// <param name="needs">What trains behind the pipelines needs of their features.</param>
    /// <param name="cancellation">What stops the preparation before its next trial's deal.</param>
    /// <returns>A preparation that holds every candidate's rows to those the first prepared were measured and chosen on.</returns>
    public static DealPreparation PerCandidate(Func<Candidate, int, IReadOnlyList<Pipeline>> pipelinesFor, IReadOnlyList<DealKind> kinds, Needs needs, CancellationToken cancellation) =>
        new EveryTrial(pipelinesFor, kinds, needs, cancellation);

    /// <summary>The rows a candidate's trial trains on for a deal: one prepared pipeline for each thing it trains.</summary>
    /// <param name="candidate">The candidate.</param>
    /// <param name="trial">The trial's number.</param>
    /// <param name="deal">The deal, from nought.</param>
    /// <returns>The prepared rows.</returns>
    public abstract IReadOnlyList<PreparedData> Of(Candidate candidate, int trial, int deal);

    // A pipeline run to be judged, for what trains behind it, and held to the rows measured and chosen on before it.
    private protected PreparedData Prepared(Pipeline? pipeline, DealPlace place)
    {
        var rows = pipeline.RunToBeJudged(place, _needs);

        _measured.Hold(rows, place);

        return rows;
    }

    // Every deal prepared once, in order, so the first deal of a kind holds the test rows every other deal of it is held to.
    private sealed class OncePerDeal : DealPreparation
    {
        private readonly PreparedData[] _prepared;

        public OncePerDeal(Func<int, Pipeline> pipelineFor, IReadOnlyList<DealKind> kinds, Needs needs, CancellationToken cancellation)
            : base(kinds, needs, cancellation)
        {
            _prepared = new PreparedData[kinds.Sum(kind => kind.Deals)];

            for (var deal = 0; deal < _prepared.Length; deal++)
            {
                Cancellation.ThrowIfCancellationRequested();

                _prepared[deal] = Prepared(pipelineFor(deal), new DealPlace(deal));
            }
        }

        public override IReadOnlyList<PreparedData> Of(Candidate candidate, int trial, int deal) => [_prepared[deal]];
    }

    // Every trial's pipelines prepared for its deal, each held to what the first trial prepared.
    private sealed class EveryTrial(Func<Candidate, int, IReadOnlyList<Pipeline>> pipelinesFor, IReadOnlyList<DealKind> kinds, Needs needs, CancellationToken cancellation)
        : DealPreparation(kinds, needs, cancellation)
    {
        public override IReadOnlyList<PreparedData> Of(Candidate candidate, int trial, int deal)
        {
            Cancellation.ThrowIfCancellationRequested();

            var place = new DealPlace(deal) { Trial = trial };
            var pipelines = pipelinesFor(candidate, deal) ?? throw new InvalidOperationException($"The pipelines for trial {trial}, deal {deal} are nothing.");

            return [.. pipelines.Select(pipeline => Prepared(pipeline, place))];
        }
    }
}
