// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Several models trained behind one fit of a pipeline, by a function of the caller's, answering as one: what it predicts
/// for rows served later is its members' answers combined, in the answer's own units.
/// </summary>
/// <remarks>
/// The sibling of <see cref="TrainedEnsemble"/> for a model this library did not train: the members share the pipeline they
/// were trained behind, since <see cref="Study{TModel}.TrainEnsemble"/> trains every one of them on the very same rows, and
/// their answers meet only once each has come back into the answer's own units.
/// </remarks>
public sealed class ModelEnsemble<TModel>
{
    private readonly Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> _answer;

    internal ModelEnsemble(IReadOnlyList<TModel> members, Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> answer, PreparedData prepared, Blend blend)
    {
        Members = members;
        _answer = answer;
        Prepared = prepared;
        Blend = blend;
    }

    /// <summary>The models, in the order they were trained.</summary>
    public IReadOnlyList<TModel> Members { get; }

    /// <summary>How the members' answers become one.</summary>
    public Blend Blend { get; }

    /// <summary>The pipeline every member was trained behind: what serves it rows, and what brings its answers back into their own units.</summary>
    public PreparedData Prepared { get; }

    /// <summary>
    /// How the pipeline's report measured the ensemble beside each of its members, on the same rows; nothing when the pipeline
    /// declares no report that measures coverage or width.
    /// </summary>
    public EnsembleMeasures? Measures { get; private init; }

    /// <summary>
    /// The width the ensemble's answers are worth: the margin the errors of its combined answer on the calibration rows allow.
    /// Nothing until <see cref="WithInterval"/> gives it one.
    /// </summary>
    public ConformalInterval? Interval { get; private init; }

    /// <summary>What the ensemble predicts for rows that arrived after training, in the answer's own units.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the member's own function reads, or a row is refused as a training row would be.</exception>
    /// <remarks>The rows are replayed and named once for all the members, each member answers them through <c>answer</c>, and the answers are brought back through one walk of the rows and combined in the answer's own units.</remarks>
    public Predictions Predict(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var served = Prepared.Served(rows);
        IReadOnlyList<double[]>[] said = [.. Members.Select(member => _answer(member, served.Features))];
        var answers = Prepared.BackToOriginal(said, served, rows, Blend);
        var ends = Interval?.Around(answers);

        return new Predictions(Prepared.Declaration.Output!.Answers, answers, served.HandedInAt)
        {
            Ids = served.Ids,
            Lower = ends?.Lower,
            Upper = ends?.Upper,
        };
    }

    /// <summary>An ensemble of the same members with the width its combined answers are worth, worked out on the rows the pipeline set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>The same members behind the same pipeline, answering with a lower and an upper end.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">The pipeline sets no rows aside to calibrate on, they are too few for the level, or the answers are classes.</exception>
    public ModelEnsemble<TModel> WithInterval(double alpha)
    {
        var calibration = Prepared.Batch(Part.Calibration, Needs.Numbers);
        var interval = Prepared.IntervalOf([.. Members.Select(member => Predicted(member, calibration))], Blend, alpha);

        return new ModelEnsemble<TModel>(Members, _answer, Prepared, Blend) { Interval = interval, Measures = Report(interval) };
    }

    // What one member answers for a batch's rows.
    private PartPredictions Predicted(TModel member, Batch batch) => new(batch, _answer(member, batch.Features));

    // The report measured again, with the interval's margins, when it names the coverage or the width of one; as it was otherwise.
    private EnsembleMeasures? Report(ConformalInterval interval)
    {
        if (Prepared.Declaration.Report is not { } report || !report.Metrics.Any(metric => metric is Metric.Coverage or Metric.Width))
        {
            return Measures;
        }

        var parts = report.Parts.Distinct().ToArray();
        IReadOnlyList<IReadOnlyList<PartPredictions>> said = [.. Members.Select(member => parts.Select(part => Predicted(member, Prepared.Batch(part, Needs.Numbers))).ToArray())];

        return Prepared.MeasureEnsemble(said, Blend, interval);
    }
}
