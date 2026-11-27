// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>What a model learned from, as a refusal names it: what the model is, and the features it was handed.</summary>
/// <param name="Model">What the model is, as a refusal calls it: a network, an ensemble, a boosted pair, a model.</param>
/// <param name="Features">The features it learned from, in the order it was handed them.</param>
internal readonly record struct LearnedFrom(string Model, IReadOnlyList<string> Features);

/// <summary>What a model answered for rows served to it, and what was served.</summary>
/// <param name="Answers">For each served row, in the order the rows were served, a number for each answer in the answer's own units.</param>
/// <param name="Served">What the pipeline handed the model for the rows: their features, and where each stood among the rows handed in.</param>
internal readonly record struct ServedAnswers(IReadOnlyList<double[]> Answers, ServedBatch Served)
{
    /// <summary>The ends of the interval around each answer, in the order of <see cref="Answers"/>; nothing when the model was given none.</summary>
    public IntervalEnds? Ends { get; init; }
}

/// <summary>
/// The one way a model trained behind a pipeline answers rows that arrive after training: the rows are served, refused when the
/// pipeline hands over other features than the model learned from, answered by the model's own function, brought back into the
/// answer's own units — the members' answers combined there, for an ensemble — and given the ends of the model's interval.
/// </summary>
/// <remarks>
/// A network, an ensemble of networks or of a caller's models, a boosted pair and a trainer from ML.NET each supply only how they
/// answer the rows the pipeline hands over; the rest is this, so every one of them serves, refuses and comes back alike. The answers
/// stand in the order the rows were served, which a pipeline that orders its rows makes another order than the one they were handed
/// in: <see cref="ServedOrderExtensions"/> is the one way back to that.
/// </remarks>
internal sealed class ServedAnswer
{
    private readonly Needs _needs;
    private readonly LearnedFrom _learned;

    /// <summary>The way a model answers behind a pipeline.</summary>
    /// <param name="prepared">The pipeline the model was trained behind.</param>
    /// <param name="needs">What the model needs of the features it is handed.</param>
    /// <param name="learned">What the model is, and the features it learned from.</param>
    internal ServedAnswer(PreparedData prepared, Needs needs, LearnedFrom learned)
    {
        Prepared = prepared;
        _needs = needs;
        _learned = learned;
    }

    /// <summary>The pipeline the model was trained behind.</summary>
    internal PreparedData Prepared { get; }

    /// <summary>What one model answers for rows served to it.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="answer">How the model answers the rows the pipeline hands over, one row of answers for each, in their order and in the units they were handed over in.</param>
    /// <param name="interval">The model's interval; nothing, for a model given none.</param>
    /// <returns>The answers in their own units, in the order the rows were served, and what was served.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the model learned from, or a row is refused as a training row would be.</exception>
    internal ServedAnswers Answered(IRowSource rows, Func<ServedBatch, IReadOnlyList<double[]>> answer, ConformalInterval? interval)
    {
        var served = Served(rows);

        return Ended(new ServedAnswers(Prepared.BackToOriginal(answer(served), served, rows), served), interval);
    }

    /// <summary>What several members answer together for rows served to them, each brought back into the answer's own units and combined there.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="members">How the members answer the rows the pipeline hands over: for each member, one row of answers for each row, in the units they were handed over in.</param>
    /// <param name="blend">How the members' answers become one.</param>
    /// <param name="interval">The ensemble's interval; nothing, for one given none.</param>
    /// <returns>The combined answers in their own units, in the order the rows were served, and what was served.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the members learned from, or a row is refused as a training row would be.</exception>
    internal ServedAnswers Blended(IRowSource rows, Func<ServedBatch, IReadOnlyList<IReadOnlyList<double[]>>> members, Blend blend, ConformalInterval? interval)
    {
        var served = Served(rows);

        return Ended(new ServedAnswers(Prepared.BackToOriginal(members(served), served, rows, blend), served), interval);
    }

    // The rows served, refused when the pipeline hands over other features than the model learned from: a model fed other
    // numbers than it learned from answers something nobody asked.
    private ServedBatch Served(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var served = Prepared.Served(rows, _needs);

        return served.FeatureNames.SequenceEqual(_learned.Features)
            ? served
            : throw new InvalidOperationException(
                $"This {_learned.Model} was trained on {_learned.Features.Quoted()}, and its pipeline hands over {served.FeatureNames.Quoted()}: a model answers only what it learned from.");
    }

    private static ServedAnswers Ended(ServedAnswers answered, ConformalInterval? interval) =>
        answered with { Ends = interval?.Around(answered.Answers) };
}

/// <summary>From the order rows were served in back to the order they were handed in.</summary>
/// <remarks>
/// A pipeline that orders its rows — <c>order.by</c> — serves them in its own order, whatever order they were handed in, so the
/// answer that stands at a row's place among the served rows is another row's. Every answer a caller is handed row for row, as
/// <see cref="Answering"/> promises them, comes back through this, and nothing else reads an answer by the place it was served at.
/// </remarks>
internal static class ServedOrderExtensions
{
    extension(IReadOnlyList<double[]> served)
    {
        /// <summary>The answers of served rows, in the order the rows were handed in.</summary>
        /// <param name="handedInAt">For each served row, in the order of the answers, where it stood among the rows handed in, from nought.</param>
        /// <returns>The same answers, the one for the row handed in first first; a row the pipeline's steps left out has none.</returns>
        internal IReadOnlyList<double[]> InHandedInOrder(IReadOnlyList<int> handedInAt) =>
            [.. Enumerable.Range(0, served.Count).OrderBy(at => handedInAt[at]).Select(at => served[at])];
    }
}
