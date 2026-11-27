// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Several models trained by a function of the caller's, answering as one: what it predicts for rows served later is its members'
/// answers combined, in the answer's own units.
/// </summary>
/// <remarks>
/// <para>
/// The sibling of <see cref="TrainedEnsemble"/> for a model this library did not train. The members may share the pipeline they
/// were trained behind, or each be a whole model behind a pipeline of its own: each member is served the rows through its own
/// pipeline, as it needs its features, and comes back through its own way back, and their answers meet only then, in the answer's
/// own units.
/// </para>
/// <para>
/// It has no file of its own: the models are the caller's, and so is keeping them. Each member's pipeline is kept as any run of a
/// pipeline is, by its own file.
/// </para>
/// </remarks>
public sealed class ModelEnsemble<TModel>
{
    private readonly Ensemble _core;

    // The members answer as the core's members do: each behind its own pipeline, as it needs its features.
    internal ModelEnsemble(IReadOnlyList<TModel> members, Ensemble core)
    {
        Members = members;
        _core = core;
    }

    /// <summary>The models, in the order they were trained.</summary>
    public IReadOnlyList<TModel> Members { get; }

    /// <summary>The run of the pipeline each member was trained behind, in the order of <see cref="Members"/>: what serves it rows, and what brings its answers back.</summary>
    public IReadOnlyList<PreparedData> MembersPrepared => [.. _core.Members.Select(member => member.Serving.Prepared)];

    /// <summary>How the members' answers become one.</summary>
    public Blend Blend => _core.Blend;

    /// <summary>The pipeline every member was trained behind: what serves it rows, and what brings its answers back into their own units.</summary>
    /// <exception cref="InvalidOperationException">
    /// The members were trained behind pipelines of their own, so no one pipeline serves the ensemble: each member's own stands in
    /// <see cref="MembersPrepared"/>.
    /// </exception>
    public PreparedData Prepared => _core.Shared("each member answers behind its own, MembersPrepared[i]");

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
    /// <exception cref="InvalidOperationException">
    /// A pipeline hands over other features than its members were trained on, a row is refused as a training row would be, or a row
    /// handed in is answered by one member and not by another, whose pipeline left it out.
    /// </exception>
    /// <remarks>
    /// The rows are replayed and named once for every pipeline the members were trained behind, each member answers them through
    /// <c>answer</c>, and each member's answers come back through its own pipeline's way back before they are combined in the answer's
    /// own units, in the order the first member's pipeline served the rows.
    /// </remarks>
    public Predictions Predict(IRowSource rows) => Predictions.Of(_core.Answered(rows, default, Interval), _core.Answers, unfamiliar: null);

    /// <summary>An ensemble of the same members with the width its combined answers are worth, worked out on the rows every member set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>The same members behind the same pipelines, answering with a lower and an upper end.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">
    /// The pipelines set no rows aside to calibrate on, they are too few for the level, the answers are classes, or the members' pipelines
    /// set other rows aside — a member that learned from a row another calibrates on named first.
    /// </exception>
    public ModelEnsemble<TModel> WithInterval(double alpha)
    {
        var interval = _core.IntervalOf(alpha, default);

        return new ModelEnsemble<TModel>(Members, _core) { Interval = interval, Measures = _core.Remeasured(default, interval, Measures) };
    }
}
