// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>One member of an ensemble: a model behind the pipeline it was trained behind, and how it answers the rows that pipeline hands over.</summary>
/// <remarks>
/// A member carries its own pipeline and what it needs of its features, so every member of one ensemble may have been trained
/// behind another pipeline — a whole model, with its own steps — and is handed rows, served and brought back through its own.
/// </remarks>
internal interface IEnsembleMember
{
    /// <summary>The way the member answers behind its pipeline: the pipeline, what it needs of its features, and the features it learned from.</summary>
    ServedAnswer Serving { get; }

    /// <summary>The digest of the fit of the pipeline it was trained behind: members of one digest are served by one replay.</summary>
    string Behind { get; }

    /// <summary>What the member answers for the rows of a part its pipeline handed over, as the report measures them.</summary>
    /// <param name="batch">The rows, as the member's pipeline handed them over for what it needs.</param>
    /// <param name="chunking">Where the arithmetic runs, and how many rows a pass takes; a member that answers by a function of its own reads neither.</param>
    /// <returns>Its predictions for the rows, in the units they were handed over in.</returns>
    PartPredictions Predicted(Batch batch, Chunking chunking);

    /// <summary>What the member answers for served rows, in the units its pipeline handed them over in.</summary>
    /// <param name="served">The rows, as its pipeline served them.</param>
    /// <param name="chunking">Where the arithmetic runs, the shape of one example among them; read as by <see cref="Predicted"/>.</param>
    /// <returns>One row of answers for each served row.</returns>
    IReadOnlyList<double[]> Answered(ServedBatch served, Chunking chunking);
}

/// <summary>A trained network as a member of an ensemble.</summary>
/// <param name="network">The network, behind the pipeline it was trained behind.</param>
internal sealed class NetworkMember(TrainedNetwork network) : IEnsembleMember
{
    /// <inheritdoc />
    public ServedAnswer Serving { get; } = new(network.Prepared, TrainedNetwork.FeatureNeeds, new LearnedFrom("ensemble", network.TrainedOn.Features));

    /// <inheritdoc />
    public string Behind => network.TrainedOn.TrainedBehind;

    /// <inheritdoc />
    public PartPredictions Predicted(Batch batch, Chunking chunking) => network.Predicted(batch, chunking);

    /// <inheritdoc />
    public IReadOnlyList<double[]> Answered(ServedBatch served, Chunking chunking) =>
        TrainedNetwork.Answered(network.Network, network.Loss, served.Features, chunking with { Example = served.ExampleShape });
}

/// <summary>A model of the caller's as a member of an ensemble: the model, the way it answers behind its pipeline, and the function that answers with it.</summary>
/// <param name="model">The model.</param>
/// <param name="serving">The pipeline it was trained behind, what it needs of its features, and the features it learned from.</param>
/// <param name="answer">What the model answers for rows of features, in the shape its pipeline hands them over.</param>
internal sealed class ModelMember<TModel>(TModel model, ServedAnswer serving, Func<TModel, IReadOnlyList<double[]>, IReadOnlyList<double[]>> answer) : IEnsembleMember
{
    private string? _behind;

    /// <summary>The model.</summary>
    public TModel Model => model;

    /// <inheritdoc />
    public ServedAnswer Serving => serving;

    /// <inheritdoc />
    public string Behind => _behind ??= PipelineText.Of(serving.Prepared).Digest;

    /// <inheritdoc />
    public PartPredictions Predicted(Batch batch, Chunking chunking) => new(batch, answer(model, batch.Features));

    /// <inheritdoc />
    public IReadOnlyList<double[]> Answered(ServedBatch served, Chunking chunking) => answer(model, served.Features);
}

/// <summary>
/// What every ensemble is, whatever its members are: members that may each answer behind a pipeline of their own, how their answers
/// become one, and the one way they are measured, given an interval and asked about rows.
/// </summary>
/// <remarks>
/// <see cref="TrainedEnsemble"/> and <see cref="ModelEnsemble{TModel}"/> are its two faces: of networks, and of a caller's models.
/// Each member is handed the rows its own pipeline prepared and serves, and comes back through that pipeline's own way back; the
/// answers meet only then, in the answer's own units, through <c>CombinedBack</c>. Members of one fit are served by one replay, as
/// an ensemble behind one pipeline always was. What the members are measured and calibrated on is held to rows every member set
/// aside alike.
/// </remarks>
internal sealed class Ensemble
{
    /// <summary>Puts members together as one ensemble.</summary>
    /// <param name="members">The members, at least two: the faces refuse fewer, each in its own nouns.</param>
    /// <param name="blend">How their answers become one.</param>
    /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
    /// <exception cref="ArgumentException">A member answers another question than the first: other answers, or the same named by another output.</exception>
    /// <exception cref="InvalidOperationException">The blend is a vote, and the answers are amounts.</exception>
    internal Ensemble(IReadOnlyList<IEnsembleMember> members, Blend blend)
    {
        blend.ThrowIfUndefined();

        if (Otherwise(members) is { } otherwise)
        {
            throw new ArgumentException(otherwise, nameof(members));
        }

        blend.ThrowIfVotedOnAmounts([.. members.Select(member => member.Serving.Prepared)]);

        Members = members;
        Blend = blend;
    }

    /// <summary>The members, in the order they were given.</summary>
    internal IReadOnlyList<IEnsembleMember> Members { get; }

    /// <summary>How the members' answers become one.</summary>
    internal Blend Blend { get; }

    /// <summary>What each number of a row's answers stands for, as every member answers it.</summary>
    internal IReadOnlyList<string> Answers => First.Declaration.Output!.Answers;

    /// <summary>Whether every member was trained behind one fit of one pipeline.</summary>
    internal bool SharesOnePipeline => Members.All(member => member.Behind == Members[0].Behind);

    private PreparedData First => Members[0].Serving.Prepared;

    /// <summary>What is said of members that answer another question than the first; nothing when they all answer one.</summary>
    /// <param name="members">The members.</param>
    /// <returns>The sentence, or nothing.</returns>
    internal static string? Otherwise(IReadOnlyList<IEnsembleMember> members)
    {
        var first = Question(members[0]);

        return Enumerable.Range(1, members.Count - 1).FirstOrDefault(at => Question(members[at]) != first, -1) is var other and > 0
            ? $"Member {other + 1} answers {Question(members[other])}, and member 1 answers {first}: the members of an ensemble answer one question, so their answers can meet in its units."
            : null;
    }

    /// <summary>The one pipeline every member was trained behind.</summary>
    /// <param name="remedy">Where each member's own pipeline is read instead, as the face calls it.</param>
    /// <returns>The pipeline.</returns>
    /// <exception cref="InvalidOperationException">The members were trained behind pipelines of their own.</exception>
    internal PreparedData Shared(string remedy) =>
        SharesOnePipeline
            ? First
            : throw new InvalidOperationException(
                $"The members of this ensemble were trained behind {Members.Select(member => member.Behind).Distinct().Count()} pipelines of their own, so no one pipeline serves it: {remedy}.");

    /// <summary>How the first member's pipeline's report measures the ensemble beside each member; nothing when it declares no report.</summary>
    /// <param name="chunking">Where each member answers, and how many rows a pass takes, by its place among the members.</param>
    /// <param name="interval">The ensemble's interval, so the report's coverage and width are measured of it; nothing, and they are left out.</param>
    /// <returns>The measures, or nothing.</returns>
    /// <exception cref="InvalidOperationException">Members set the rows a part the report measures apart, or as their reports refuse.</exception>
    internal EnsembleMeasures? Measured(Func<int, Chunking> chunking, ConformalInterval? interval)
    {
        if (First.Declaration.Report is not { } report)
        {
            return null;
        }

        var parts = report.Parts.Distinct().ToArray();
        IReadOnlyList<MemberPredictions> said =
        [
            .. Members.Select((member, at) => new MemberPredictions(
                member.Serving.Prepared,
                [.. parts.Select(part => member.Predicted(member.Serving.Prepared.Batch(part, member.Serving.Needs), chunking(at)))])),
        ];

        return Measurement.OfEnsemble(said, Blend, interval);
    }

    /// <summary>The measures again, with the interval's margins, when the report names the coverage or the width of one; as they were otherwise.</summary>
    /// <param name="chunking">Where every member answers.</param>
    /// <param name="interval">The interval the ensemble was just given.</param>
    /// <param name="measures">The measures it had.</param>
    /// <returns>The measures.</returns>
    internal EnsembleMeasures? Remeasured(Chunking chunking, ConformalInterval interval, EnsembleMeasures? measures) =>
        First.Declaration.Report is { } report && report.NamesAnInterval() ? Measured(_ => chunking, interval) : measures;

    /// <summary>The width the ensemble's combined answers are worth, worked out on the rows every member set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin.</param>
    /// <param name="chunking">Where every member answers.</param>
    /// <returns>The interval.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">
    /// No rows are set aside to calibrate on, they are too few for the level, the answers are classes, or members set them aside apart.
    /// </exception>
    internal ConformalInterval IntervalOf(double alpha, Chunking chunking) =>
        Conformal.Of(
            [
                .. Members.Select(member => new MemberPredictions(
                    member.Serving.Prepared,
                    [member.Predicted(member.Serving.Prepared.Batch(Part.Calibration, member.Serving.Needs), chunking)])),
            ],
            Blend,
            alpha);

    /// <summary>What the ensemble answers for rows that arrived after training: each member served through its own pipeline, and the answers combined.</summary>
    /// <param name="rows">The rows, without the answer.</param>
    /// <param name="chunking">Where every member answers.</param>
    /// <param name="interval">The ensemble's interval; nothing, for one given none.</param>
    /// <returns>The combined answers, in the order the first member's pipeline served the rows, and what it served.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="InvalidOperationException">
    /// A pipeline hands over other features than its members learned from, a row is refused as a training row would be, or a row
    /// handed in is answered by one member and not by another.
    /// </exception>
    /// <remarks>Members behind one fit are replayed, named and walked back once for all of them.</remarks>
    internal ServedAnswers Answered(IRowSource rows, Chunking chunking, ConformalInterval? interval)
    {
        var each = new ServedAnswers[Members.Count];

        var groups = Enumerable.Range(0, Members.Count)
            .GroupBy(at => new ServedTogether(Members[at].Behind, Members[at].Serving.Needs))
            .Select(together => together.ToArray());

        foreach (var places in groups)
        {
            var answered = Members[places[0]].Serving.Each(
                rows,
                [.. places.Select(at => (Func<ServedBatch, IReadOnlyList<double[]>>)(served => Members[at].Answered(served, chunking)))]);

            for (var at = 0; at < places.Length; at++)
            {
                each[places[at]] = answered[at];
            }
        }

        return ServedAnswer.Blended(each, Blend, First.Declaration.Output!.Ones, interval);
    }

    // The question a member answers, as a refusal names it: its answers and the output that names them. A member was trained, so
    // its pipeline names an output.
    private static string Question(IEnsembleMember member)
    {
        var output = member.Serving.Prepared.Declaration.Output!;

        return $"{output.Answers.Quoted()} as '{output.Verb}'";
    }

    /// <summary>Members that one replay serves: one fit of one pipeline, handing its features over for one need.</summary>
    /// <param name="Behind">The digest of the fit.</param>
    /// <param name="Needs">What the members need of their features.</param>
    private readonly record struct ServedTogether(string Behind, Needs Needs);
}
