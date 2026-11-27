// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Several trained networks answering as one: what it predicts for rows served later is its members' answers combined, in the
/// answer's own units.
/// </summary>
/// <remarks>
/// <para>
/// The members may share the pipeline they were trained behind — one fit, so the same rows are replayed, named and served once for
/// all of them, and they differ in what they were declared to be or in the seed they started from — or each be a whole model
/// behind a pipeline of its own, with its own steps: then every member is served the rows through its own pipeline and comes back
/// through its own way back. Either way their answers meet only once each has come back into the answer's own units, since a way
/// back is not a straight line: the mean of exponentials is not the exponential of the mean.
/// </para>
/// <para>
/// How many members an ensemble holds, and which, are chosen on the validation rows like everything else is; the test rows report
/// what the members and the ensemble measured and choose nothing. An ensemble is measured and calibrated on rows every member set
/// aside alike, so a member never answers, through the blend, a row it learned from.
/// </para>
/// </remarks>
public sealed class TrainedEnsemble
{
    /// <summary>
    /// The version of the one file this library reads an ensemble's file up to: an ensemble behind one pipeline is written as the
    /// first, its networks beside that pipeline; one whose members each have a pipeline of their own as the second, each member
    /// whole, as a single network's file holds it.
    /// </summary>
    public const int Version = 2;

    /// <summary>
    /// The most parameters, over all its members, an ensemble's file is written for: five trained networks of 5.3 million parameters
    /// between them were measured at 25.6 characters a parameter, and a text is no longer than <see cref="MostText"/> characters,
    /// which is 41.9 million parameters at that rate. The text itself is held to its length as well, whatever its numbers spell.
    /// </summary>
    internal const long MostParameters = 40_000_000;

    /// <summary>The longest text .NET makes, in characters.</summary>
    internal const long MostText = 1_073_741_791;

    internal const string BlendKey = "blend";

    internal const string MembersKey = "members";

    // The version an ensemble behind one pipeline is written as, as it always was.
    private const int BehindOnePipeline = 1;

    private readonly Ensemble _core;

    internal TrainedEnsemble(IReadOnlyList<TrainedNetwork> members, Blend blend)
        : this(members, new Ensemble([.. members.Select(member => new NetworkMember(member))], blend))
    {
    }

    private TrainedEnsemble(IReadOnlyList<TrainedNetwork> members, Ensemble core)
    {
        Members = members;
        _core = core;
    }

    /// <summary>The networks, in the order they were given, each behind the pipeline it was trained behind.</summary>
    public IReadOnlyList<TrainedNetwork> Members { get; }

    /// <summary>How the members' answers become one.</summary>
    public Blend Blend => _core.Blend;

    /// <summary>The pipeline every member was trained behind: what serves it rows, and what brings its answers back into their own units.</summary>
    /// <exception cref="InvalidOperationException">
    /// The members were trained behind pipelines of their own, so no one pipeline serves the ensemble: each member's own is its
    /// <see cref="TrainedNetwork.Prepared"/>.
    /// </exception>
    public PreparedData Prepared => _core.Shared("each member answers behind its own, Members[i].Prepared");

    /// <summary>
    /// How the pipeline's report measured the ensemble beside each of its members, on the same rows; nothing when the pipeline
    /// declares no report, or for an ensemble put together from networks trained apart.
    /// </summary>
    public EnsembleMeasures? Measures { get; internal init; }

    /// <summary>
    /// The width the ensemble\'s answers are worth: the margin the errors of its combined answer on the calibration rows allow.
    /// Nothing until <see cref="WithInterval(double)"/> gives it one.
    /// </summary>
    public ConformalInterval? Interval { get; internal init; }

    /// <summary>Puts trained networks together as an ensemble.</summary>
    /// <param name="members">
    /// The networks: at least two, all answering one question — the same answers, named by the same output — each behind the
    /// pipeline it was trained behind, one for all of them or one each.
    /// </param>
    /// <param name="blend">How their answers become one; averaged, unless said.</param>
    /// <returns>The ensemble.</returns>
    /// <exception cref="ArgumentException">There are fewer than two networks, or one answers another question than the first.</exception>
    /// <exception cref="ArgumentNullException">There are no members.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
    /// <exception cref="InvalidOperationException">The blend is a vote, and the answers are amounts: the members' answers to an amount are averaged.</exception>
    public static TrainedEnsemble Of(IReadOnlyList<TrainedNetwork> members, Blend blend = Blend.Average)
    {
        ArgumentNullException.ThrowIfNull(members);

        members.ThrowIfFewerThanTwo("networks", "a single network predicts for itself", nameof(members));

        return new TrainedEnsemble([.. members], blend);
    }

    /// <summary>
    /// The one file: how the members' answers become one, and the members — every network beside the one pipeline they were trained
    /// behind, or each member whole with its own.
    /// </summary>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    /// <exception cref="InvalidOperationException">The members hold more parameters between them than a text can hold.</exception>
    /// <remarks>
    /// <para>
    /// An ensemble behind one pipeline is written as it always was, the first version of the file: its networks, then the one
    /// pipeline. One whose members were trained behind pipelines of their own is written as the second, each member as a single
    /// network's file holds it, with its own pipeline; a DeepSharp that reads only the first refuses it whole, naming the version.
    /// </para>
    /// <para>
    /// The file is one text, and .NET makes no text longer than 1,073,741,791 characters. A network's numbers take up to 24
    /// characters a parameter — measured at 25.6 for the members of a trained ensemble — so the members of an ensemble hold about
    /// 40 million parameters between them at the most. An ensemble beyond that is refused by name, before anything is written, and
    /// so is one whose numbers spell a text that is too long all the same; there is no way to write it in pieces.
    /// </para>
    /// </remarks>
    public string ToJson()
    {
        RequireWritable(Members.Sum(member => member.Network.Parameters().Sum(parameter => (long)parameter.Value.Shape.Count)));

        if (!_core.SharesOnePipeline)
        {
            return new ModelFileFrame(Version, null, Interval) { LengthChecked = RequireTextFits }.Written(writer =>
            {
                writer.WriteString(BlendKey, Blend.Word());
                writer.WriteStartArray(MembersKey);

                foreach (var member in Members)
                {
                    writer.WriteRawValue(member.ToJson(), skipInputValidation: true);
                }

                writer.WriteEndArray();
            });
        }

        return new ModelFileFrame(BehindOnePipeline, Members[0].Carried, Interval) { LengthChecked = RequireTextFits }.Written(writer =>
        {
            writer.WriteString(BlendKey, Blend.Word());
            writer.WriteStartArray(NetworkDocument.NetworksKey);

            foreach (var member in Members)
            {
                NetworkDocument.WriteNetwork(writer, member.Network, member.Loss, member.TrainedOn);
            }

            writer.WriteEndArray();
        });
    }

    /// <summary>Reads an ensemble back from its file, of either version.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its networks may be made of.</param>
    /// <param name="steps">The verbs its pipelines may use.</param>
    /// <returns>The ensemble, each member behind its pipeline, ready to predict.</returns>
    /// <exception cref="NetworkFileException">
    /// The file is not one this library reads as an ensemble's — a single network's file is refused by name — holds fewer than
    /// two networks, names no blend it knows, holds a network trained behind another fit of the pipeline beside it or for other
    /// answers, holds the parts of the other version than the one it names, or holds members that answer another question than the
    /// first: every fault at its line and column, a fault within a member placed where it stands in the whole file.
    /// </exception>
    /// <exception cref="PipelineFileException">Anything in a pipeline is wrong, at its line and column in the file.</exception>
    /// <exception cref="InvalidOperationException">The file's blend is a vote, and its output says the answers are amounts.</exception>
    public static TrainedEnsemble FromJson(string json, NetworkCatalog networks, StepCatalog steps)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(steps);

        var file = new FileText(json, FileKind.Ensemble);
        var members = file.WrittenAgainst > BehindOnePipeline ? Whole(file, networks, steps) : BehindOne(file, networks, steps);
        var words = string.Join(", ", Vocabulary<Blend>.Words);

        if (file.Blend is not { } word)
        {
            throw file.Refused(BlendKey, $"This file does not say how its members' answers become one: an ensemble's file names it as text under '{BlendKey}', one of {words}.");
        }

        if (!Vocabulary<Blend>.Words.Contains(word, StringComparer.OrdinalIgnoreCase))
        {
            throw file.Refused(BlendKey, $"'{word}' is none of the ways an ensemble's members' answers become one: {words}.");
        }

        var interval = file.IntervalHeldTo(members[0].TrainedOn.Answers.Count);

        return new TrainedEnsemble(members, Vocabulary<Blend>.Read(word, BlendKey)) { Interval = interval };
    }

    // The members of a file of the first version: every network, held to the one pipeline beside them.
    private static IReadOnlyList<TrainedNetwork> BehindOne(FileText file, NetworkCatalog networks, StepCatalog steps)
    {
        if (file.Holds(MembersKey))
        {
            throw file.Refused(
                MembersKey,
                $"'{MembersKey}' holds an ensemble's members each with its own pipeline, as the second version of the file does, and this file names the first: its networks stand under '{NetworkDocument.NetworksKey}', behind the one pipeline beside them.");
        }

        var saved = NetworkDocument.ReadNetworks(file.Text, NetworkDocument.NetworksKey, networks);
        var prepared = PreparedData.FromJson(file.Text, steps, TrainedNetwork.PipelineKey);
        var carried = file.Pipeline();

        if (saved.FewerThanTwo("networks", "a single network predicts for itself") is { } fewer)
        {
            throw file.Refused(NetworkDocument.NetworksKey, fewer);
        }

        var members = new List<TrainedNetwork>();

        for (var at = 0; at < saved.Count; at++)
        {
            var trainedOn = file.TrainedOnOf(saved[at], at);

            file.HeldTo(trainedOn, prepared, at);

            members.Add(new TrainedNetwork(saved[at].Network, saved[at].Loss, prepared, trainedOn) { Carried = carried });
        }

        return members;
    }

    // The members of a file of the second version: each read whole, as a single network's file is, with its own pipeline.
    private static IReadOnlyList<TrainedNetwork> Whole(FileText file, NetworkCatalog networks, StepCatalog steps)
    {
        if (new[] { NetworkDocument.NetworksKey, TrainedNetwork.PipelineKey }.FirstOrDefault(file.Holds) is { } stray)
        {
            throw file.Refused(
                stray,
                $"'{stray}' is a part of the first version of an ensemble's file, and this one names the second: each member stands whole under '{MembersKey}', with its own pipeline.");
        }

        var texts = file.MemberTexts() ?? throw file.Refused(
            MembersKey,
            $"An ensemble's file of the second version holds its members as a list under '{MembersKey}', each as a single network's file holds it.");

        TrainedNetwork[] members = [.. texts.Select((text, at) => file.Read(text, at, member => TrainedNetwork.FromJson(member, networks, steps)))];

        if (members.FewerThanTwo("networks", "a single network predicts for itself") is { } fewer)
        {
            throw file.Refused(MembersKey, fewer);
        }

        if (Ensemble.Otherwise([.. members.Select(member => new NetworkMember(member))]) is { } otherwise)
        {
            throw file.Refused(MembersKey, otherwise);
        }

        return members;
    }

    /// <summary>Refuses an ensemble holding more parameters than a text can hold.</summary>
    /// <param name="parameters">The parameters of all the members together.</param>
    /// <exception cref="InvalidOperationException">There are more than <see cref="MostParameters"/>.</exception>
    internal static void RequireWritable(long parameters)
    {
        if (parameters > MostParameters)
        {
            throw new InvalidOperationException(
                string.Create(
                    System.Globalization.CultureInfo.InvariantCulture,
                    $"This ensemble holds {parameters:N0} parameters between its members, and its file is one text that holds about 40 million at the most ({MostParameters:N0}): .NET makes no text longer than {MostText:N0} characters. Keep fewer members, or smaller ones."));
        }
    }

    /// <summary>Refuses a text that .NET could not hold as a string.</summary>
    /// <param name="length">The text, in bytes written; the file is ASCII but for names, so its characters are no more.</param>
    /// <exception cref="InvalidOperationException">The text is longer than <see cref="MostText"/>.</exception>
    internal static void RequireTextFits(long length)
    {
        if (length > MostText)
        {
            throw new InvalidOperationException(
                string.Create(
                    System.Globalization.CultureInfo.InvariantCulture,
                    $"This ensemble's file would be {length:N0} characters long, and .NET makes no text longer than {MostText:N0}: keep fewer members, or smaller ones."));
        }
    }

    /// <summary>The same ensemble, measured by the first member's pipeline's report beside each of its members, as each answers on the engine its chunking names.</summary>
    internal TrainedEnsemble Measured(Func<int, Chunking> chunking) => new(Members, _core) { Measures = _core.Measured(chunking, null), Interval = Interval };

    /// <summary>An ensemble of the same members with the width its combined answers are worth, worked out on the rows every member set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>The same members behind the same pipelines, answering with a lower and an upper end.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">
    /// The pipelines set no rows aside to calibrate on, they are too few for the level, the answers are classes, or the members' pipelines
    /// set other rows aside — a member that learned from a row another calibrates on named first.
    /// </exception>
    /// <remarks>
    /// The errors are those of the combined answer, in the answer\'s own units, so the margin is the ensemble\'s and no member\'s: it is
    /// the same split conformal prediction on a point answer, of whatever answers the ensemble gives.
    /// </remarks>
    public TrainedEnsemble WithInterval(double alpha) => WithInterval(alpha, new CpuBackend());

    /// <summary>An ensemble of the same members with the width its combined answers are worth, answered for the calibration rows on the engine handed in.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>The same members behind the same pipelines, with an interval.</returns>
    /// <exception cref="ArgumentNullException">There is no engine.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">As <see cref="WithInterval(double)"/> refuses.</exception>
    public TrainedEnsemble WithInterval(double alpha, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize);
        var interval = _core.IntervalOf(alpha, chunking);

        return new TrainedEnsemble(Members, _core) { Interval = interval, Measures = _core.Remeasured(chunking, interval, Measures) };
    }

    /// <summary>What the ensemble predicts for rows that arrived after training, in the answer's own units, on a light engine of its own.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">
    /// A pipeline hands over other features than its members were trained on, a row is refused as a training row would be, or a row
    /// handed in is answered by one member and not by another.
    /// </exception>
    public Predictions Predict(IRowSource rows) => Predict(rows, new CpuBackend());

    /// <summary>What the ensemble predicts for rows that arrived after training, in the answer's own units, on the engine handed in.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">As <see cref="Predict(IRowSource)"/> refuses.</exception>
    /// <remarks>
    /// The rows are replayed and named once for every pipeline the members were trained behind — once for all of them, when they
    /// share one — each member answers them on the engine, and each member's answers come back through its own pipeline's way back
    /// before they are combined in the answer's own units. They stand in the order the first member's pipeline served the rows, each
    /// other member's found by where the row was handed in; a row one member's pipeline left out is answered by no member, and refused
    /// by where it was handed in. What the rows move away from is said as the first member learned it.
    /// </remarks>
    public Predictions Predict(IRowSource rows, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        var answered = _core.Answered(rows, new Chunking(backend, TrainedNetwork.ServingBatchSize), Interval);
        var first = Members[0];

        return Predictions.Of(answered, first.TrainedOn.Answers, TrainedNetwork.Unfamiliar(first.TrainedOn, answered.Served.Features));
    }
}

/// <summary>How an ensemble is trained, beyond the declarations of its members.</summary>
public sealed class EnsembleOptions
{
    /// <summary>How the members' answers become one; averaged, unless said.</summary>
    public Blend Blend { get; init; } = Blend.Average;

    /// <summary>The engines the names in the members' declarations stand for; the light engine alone, unless given.</summary>
    public Engines? Engines { get; init; }

    /// <summary>What stops the run before the next batch or the next look at the validation rows of any member; nothing stops it, unless said.</summary>
    public CancellationToken Cancellation { get; init; }
}

/// <summary>Training several networks behind one fit, and choosing them from a study.</summary>
public static partial class EnsembleExtensions
{
    extension(PreparedData prepared)
    {
        /// <summary>Trains several declared networks on the rows a run of a pipeline prepared, and puts them together as one ensemble.</summary>
        /// <param name="members">The networks to train: at least two, each its layers, what moves them and what judges them, and its own seed.</param>
        /// <param name="options">How their answers become one, the engines the names in the declarations stand for, and what stops the run; averaged, on the light engine alone and unstoppable, unless said.</param>
        /// <returns>The ensemble, with the pipeline's report measured of it beside each member, when the pipeline declares one.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared, or no members.</exception>
        /// <exception cref="ArgumentException">There are fewer than two members, or two are the same declaration under the same seed.</exception>
        /// <exception cref="InvalidOperationException">As training one network refuses; or the blend is a vote, and the answers are amounts.</exception>
        /// <remarks>
        /// Two members that are the same declaration under the same seed train the same network twice, which is no second opinion:
        /// they are refused, naming both. To train one declaration several times, give each a seed — <c>step with { Seed = 2 }</c>.
        /// Networks trained behind pipelines of their own are put together with <see cref="TrainedEnsemble.Of"/>.
        /// </remarks>
        public TrainedEnsemble TrainEnsemble(IReadOnlyList<LearnNetworkStep> members, EnsembleOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(members);

            var blend = options?.Blend ?? Blend.Average;
            var engines = options?.Engines;
            var cancellation = options?.Cancellation ?? default;

            members.ThrowIfFewerThanTwo("networks", "train one with Train", nameof(members));

            for (var later = 1; later < members.Count; later++)
            {
                for (var earlier = 0; earlier < later; earlier++)
                {
                    if (members[earlier].Equals(members[later]))
                    {
                        throw new ArgumentException(
                            $"Members {earlier + 1} and {later + 1} are the same declaration under the same seed, and would train the same network twice: give each its own seed.",
                            nameof(members));
                    }
                }
            }

            TrainedNetwork[] trained = [.. members.Select(step => prepared.Train(step, engines, cancellation))];

            return TrainedEnsemble.Of(trained, blend).Measured(at => new Chunking((engines ?? new Engines()).Named(members[at].Engine), members[at].Batch));
        }
    }
}
