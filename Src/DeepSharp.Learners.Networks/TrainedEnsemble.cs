// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Several networks trained behind one fit of a pipeline, answering as one: what it predicts for rows served later is its members'
/// answers combined, in the answer's own units.
/// </summary>
/// <remarks>
/// The members share the pipeline they were trained behind — one fit, so the same rows are replayed, named and served once for all
/// of them — and differ in what they were declared to be or in the seed they started from. Their answers meet only once each has
/// come back into the answer's own units, since a way back is not a straight line: the mean of exponentials is not the exponential
/// of the mean. How many members an ensemble holds, and which, are chosen on the validation rows like everything else is; the test
/// rows report what the members and the ensemble measured and choose nothing.
/// </remarks>
public sealed class TrainedEnsemble
{
    /// <summary>The version of the one file this library writes an ensemble as, and reads up to.</summary>
    public const int Version = 1;

    /// <summary>
    /// The most parameters, over all its members, an ensemble's file is written for: five trained networks of 5.3 million parameters
    /// between them were measured at 25.6 characters a parameter, and a text is no longer than <see cref="MostText"/> characters,
    /// which is 41.9 million parameters at that rate. The text itself is held to its length as well, whatever its numbers spell.
    /// </summary>
    internal const long MostParameters = 40_000_000;

    /// <summary>The longest text .NET makes, in characters.</summary>
    internal const long MostText = 1_073_741_791;

    internal const string BlendKey = "blend";

    internal TrainedEnsemble(IReadOnlyList<TrainedNetwork> members, Blend blend)
    {
        Members = members;
        Blend = blend;
    }

    /// <summary>The networks, in the order they were given.</summary>
    public IReadOnlyList<TrainedNetwork> Members { get; }

    /// <summary>How the members' answers become one.</summary>
    public Blend Blend { get; }

    /// <summary>The pipeline every member was trained behind: what serves it rows, and what brings its answers back into their own units.</summary>
    public PreparedData Prepared => Members[0].Prepared;

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

    /// <summary>Puts networks that were trained behind one fit together as an ensemble.</summary>
    /// <param name="members">The networks: at least two, all trained behind the same fit of a pipeline.</param>
    /// <param name="blend">How their answers become one; averaged, unless said.</param>
    /// <returns>The ensemble.</returns>
    /// <exception cref="ArgumentException">There are fewer than two networks, or one was trained behind another fit than the first.</exception>
    /// <exception cref="ArgumentNullException">There are no members.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The blend is none of the blends.</exception>
    public static TrainedEnsemble Of(IReadOnlyList<TrainedNetwork> members, Blend blend = Blend.Average)
    {
        ArgumentNullException.ThrowIfNull(members);

        if (!Enum.IsDefined(blend))
        {
            throw new ArgumentOutOfRangeException(nameof(blend), blend, "An ensemble's answers are averaged or voted for.");
        }

        if (members.Count < 2)
        {
            throw new ArgumentException(
                $"An ensemble is of at least two networks, and {members.Count} {(members.Count == 1 ? "was" : "were")} given: a single network predicts for itself.",
                nameof(members));
        }

        for (var at = 1; at < members.Count; at++)
        {
            if (members[at].TrainedOn.TrainedBehind != members[0].TrainedOn.TrainedBehind)
            {
                throw new ArgumentException(
                    $"Network {at + 1} was trained behind another fit of its pipeline than network 1 was: {members[at].TrainedOn.TrainedBehind}, not {members[0].TrainedOn.TrainedBehind}. "
                    + "Members of one ensemble answer rows replayed once, so they are trained behind one fit.",
                    nameof(members));
            }
        }

        return new TrainedEnsemble([.. members], blend);
    }

    /// <summary>The one file: how the members' answers become one, every member's network, and the pipeline they were all trained behind.</summary>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    /// <exception cref="InvalidOperationException">The members hold more parameters between them than a text can hold.</exception>
    /// <remarks>
    /// The file is one text, and .NET makes no text longer than 1,073,741,791 characters. A network's numbers take up to 24
    /// characters a parameter — measured at 25.6 for the members of a trained ensemble — so the members of an ensemble hold about
    /// 40 million parameters between them at the most. An ensemble beyond that is refused by name, before anything is written, and
    /// so is one whose numbers spell a text that is too long all the same; there is no way to write it in pieces.
    /// </remarks>
    public string ToJson()
    {
        RequireWritable(Members.Sum(member => member.Network.Parameters().Sum(parameter => (long)parameter.Value.Shape.Count)));

        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(TrainedNetwork.VersionKey, Version);
            writer.WriteString(BlendKey, Blend.ToString().ToLowerInvariant());
            writer.WriteStartArray(NetworkDocument.NetworksKey);

            foreach (var member in Members)
            {
                NetworkDocument.WriteNetwork(writer, member.Network, member.Loss, member.TrainedOn);
            }

            writer.WriteEndArray();
            writer.WritePropertyName(TrainedNetwork.PipelineKey);
            writer.WriteRawValue(Members[0].Carried.Text);

            if (Interval is { } interval)
            {
                writer.WritePropertyName(TrainedNetwork.IntervalKey);
                interval.WriteTo(writer);
            }

            writer.WriteEndObject();
        }

        RequireTextFits(stream.Length);

        return stream.TextWithLineFeeds();
    }

    /// <summary>Reads an ensemble back from its file.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its networks may be made of.</param>
    /// <param name="steps">The verbs its pipeline may use.</param>
    /// <returns>The ensemble behind its pipeline, ready to predict.</returns>
    /// <exception cref="NetworkFileException">
    /// The file is not one this library reads as an ensemble's — a single network's file is refused by name — holds fewer than
    /// two networks, names no blend it knows, or holds a network trained behind another fit of the pipeline beside it or for
    /// other answers: every fault at its line and column.
    /// </exception>
    /// <exception cref="PipelineFileException">Anything in the pipeline is wrong, at its line and column in the file.</exception>
    public static TrainedEnsemble FromJson(string json, NetworkCatalog networks, StepCatalog steps)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(steps);

        var file = new FileText(json, ensemble: true);
        var saved = NetworkDocument.ReadNetworks(json, NetworkDocument.NetworksKey, networks);
        var prepared = PreparedData.FromJson(json, steps, TrainedNetwork.PipelineKey);
        var carried = file.Pipeline();
        var words = string.Join(", ", Enum.GetNames<Blend>().Select(name => name.ToLowerInvariant()));

        if (file.Blend is not { } word)
        {
            throw file.Refused(BlendKey, $"This file does not say how its members' answers become one: an ensemble's file names it as text under '{BlendKey}', one of {words}.");
        }

        if (!Enum.TryParse<Blend>(word, ignoreCase: true, out var blend) || !Enum.IsDefined(blend))
        {
            throw file.Refused(BlendKey, $"'{word}' is none of the ways an ensemble's members' answers become one: {words}.");
        }

        if (saved.Count < 2)
        {
            throw file.Refused(NetworkDocument.NetworksKey, $"An ensemble's file holds at least two networks, and this one holds {saved.Count}.");
        }

        var members = new List<TrainedNetwork>();

        for (var at = 0; at < saved.Count; at++)
        {
            var trainedOn = saved[at].TrainedOn
                ?? throw file.Refused(NetworkDocument.NetworksKey, $"Network {at + 1} names nothing it was trained on: the file of a trained network says its features, its answers and the fit of the pipeline it was trained behind.");

            if ((TrainedNetwork.Mismatch(trainedOn, prepared) ?? TrainedNetwork.Behind(trainedOn, carried)) is { } mismatch)
            {
                throw file.Refused(NetworkDocument.NetworksKey, $"Network {at + 1}: {mismatch.Message}");
            }

            members.Add(new TrainedNetwork(saved[at].Network, saved[at].Loss, prepared, trainedOn) { Carried = carried });
        }

        var interval = file.Interval();

        if (interval is not null && interval.Margins.Count != members[0].TrainedOn.Answers.Count)
        {
            throw file.Refused(TrainedNetwork.IntervalKey, $"The interval names {interval.Margins.Count} margins, and the networks answer {members[0].TrainedOn.Answers.Count}: it has one margin for each answer.");
        }

        return new TrainedEnsemble(members, blend) { Interval = interval };
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

    /// <summary>An ensemble of the same members with the width its combined answers are worth, worked out on the rows the pipeline set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>The same members behind the same pipeline, answering with a lower and an upper end.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">The pipeline sets no rows aside to calibrate on, they are too few for the level, or the answers are classes.</exception>
    /// <remarks>
    /// The errors are those of the combined answer, in the answer\'s own units, so the margin is the ensemble\'s and no member\'s: it is
    /// the same split conformal prediction on a point answer, of whatever answers the ensemble gives.
    /// </remarks>
    public TrainedEnsemble WithInterval(double alpha) => WithInterval(alpha, new CpuBackend());

    /// <summary>An ensemble of the same members with the width its combined answers are worth, answered for the calibration rows on the engine handed in.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>The same members behind the same pipeline, with an interval.</returns>
    /// <exception cref="ArgumentNullException">There is no engine.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">The pipeline sets no rows aside to calibrate on, they are too few for the level, or the answers are classes.</exception>
    public TrainedEnsemble WithInterval(double alpha, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize);
        var calibration = Prepared.Batch(Part.Calibration, TrainedNetwork.FeatureNeeds);
        var interval = Prepared.IntervalOf([.. Members.Select(member => member.Predicted(calibration, chunking))], Blend, alpha);

        return new TrainedEnsemble(Members, Blend) { Interval = interval, Measures = Report(chunking, interval) };
    }

    // The report measured again, with the interval\'s margins, when it names the coverage or the width of one; as it was otherwise.
    private EnsembleMeasures? Report(Chunking chunking, ConformalInterval interval)
    {
        if (Prepared.Declaration.Report is not { } report || !report.Metrics.Any(metric => metric is Metric.Coverage or Metric.Width))
        {
            return Measures;
        }

        var parts = report.Parts.Distinct().ToArray();
        IReadOnlyList<IReadOnlyList<PartPredictions>> said =
        [
            .. Members.Select(member => (IReadOnlyList<PartPredictions>)[.. parts.Select(part => member.Predicted(Prepared.Batch(part, TrainedNetwork.FeatureNeeds), chunking))]),
        ];

        return Prepared.MeasureEnsemble(said, Blend, interval);
    }

    /// <summary>What the ensemble predicts for rows that arrived after training, in the answer's own units, on a light engine of its own.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the members were trained on, or a row is refused as a training row would be.</exception>
    public Predictions Predict(IRowSource rows) => Predict(rows, new CpuBackend());

    /// <summary>What the ensemble predicts for rows that arrived after training, in the answer's own units, on the engine handed in.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the members were trained on, or a row is refused as a training row would be.</exception>
    /// <remarks>
    /// The rows are replayed and named once for all the members, each member answers them on the engine, and the answers are
    /// brought back through one walk of the rows and combined in the answer's own units. The features a row moves away from the
    /// one value every training row held are the same for every member, since they were all trained on the same rows.
    /// </remarks>
    public Predictions Predict(IRowSource rows, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(backend);

        var first = Members[0];
        var served = Prepared.Served(rows, TrainedNetwork.FeatureNeeds);

        if (!served.FeatureNames.SequenceEqual(first.TrainedOn.Features))
        {
            throw new InvalidOperationException(
                $"These networks were trained on {string.Join(", ", first.TrainedOn.Features.Select(name => $"'{name}'"))}, and their pipeline hands over {string.Join(", ", served.FeatureNames.Select(name => $"'{name}'"))}: a network answers only what it learned from.");
        }

        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize) { Example = served.ExampleShape };
        IReadOnlyList<double[]>[] said = [.. Members.Select(member => (IReadOnlyList<double[]>)TrainedNetwork.Answered(member.Network, member.Loss, served.Features, chunking))];

        var answers = Prepared.BackToOriginal(said, served, rows, Blend);
        var ends = Interval?.Around(answers);

        return new Predictions(first.TrainedOn.Answers, answers, served.HandedInAt)
        {
            Unfamiliar = TrainedNetwork.Unfamiliar(first.TrainedOn, served.Features),
            Ids = served.Ids,
            Lower = ends?.Lower,
            Upper = ends?.Upper,
        };
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
public static class EnsembleExtensions
{
    extension(PreparedData prepared)
    {
        /// <summary>Trains several declared networks on the rows a run of a pipeline prepared, and puts them together as one ensemble.</summary>
        /// <param name="members">The networks to train: at least two, each its layers, what moves them and what judges them, and its own seed.</param>
        /// <param name="options">How their answers become one, the engines the names in the declarations stand for, and what stops the run; averaged, on the light engine alone and unstoppable, unless said.</param>
        /// <returns>The ensemble, with the pipeline's report measured of it beside each member, when the pipeline declares one.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared, or no members.</exception>
        /// <exception cref="ArgumentException">There are fewer than two members, or two are the same declaration under the same seed.</exception>
        /// <exception cref="InvalidOperationException">As training one network refuses.</exception>
        /// <remarks>
        /// Two members that are the same declaration under the same seed train the same network twice, which is no second opinion:
        /// they are refused, naming both. To train one declaration several times, give each a seed — <c>step with { Seed = 2 }</c>.
        /// </remarks>
        public TrainedEnsemble TrainEnsemble(IReadOnlyList<LearnNetworkStep> members, EnsembleOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(members);

            var blend = options?.Blend ?? Blend.Average;
            var engines = options?.Engines;
            var cancellation = options?.Cancellation ?? default;

            if (members.Count < 2)
            {
                throw new ArgumentException(
                    $"An ensemble is of at least two networks, and {members.Count} {(members.Count == 1 ? "was" : "were")} given: train one with Train.",
                    nameof(members));
            }

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
            var ensemble = TrainedEnsemble.Of(trained, blend);

            if (prepared.Declaration.Report is not { } report)
            {
                return ensemble;
            }

            var named = report.Parts.Distinct().ToArray();
            IReadOnlyList<IReadOnlyList<PartPredictions>> said =
            [
                .. trained.Select((network, at) =>
                {
                    var chunking = new Chunking((engines ?? new Engines()).Named(members[at].Engine), members[at].Batch);

                    return (IReadOnlyList<PartPredictions>)[.. named.Select(part => network.Predicted(prepared.Batch(part, TrainedNetwork.FeatureNeeds), chunking))];
                }),
            ];

            return new TrainedEnsemble(ensemble.Members, blend) { Measures = prepared.MeasureEnsemble(said, blend) };
        }
    }

    extension(StudyResult result)
    {
        /// <summary>The networks of the best few trials, best first, as declarations a pipeline trains again.</summary>
        /// <param name="count">How many: at least one, and no more than the trials that finished.</param>
        /// <returns>The declarations, the trial judged lowest first, the earlier of two judged alike first.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The count is below one, or above the number of trials that finished.</exception>
        /// <remarks>
        /// A study chooses by the validation rows alone, so the few it hands on are chosen by them alone; train them with
        /// <c>prepared.TrainEnsemble(result.Top(3))</c>. Trials that were pruned, or judged by a number that is none, are not among them.
        /// </remarks>
        public IReadOnlyList<LearnNetworkStep> Top(int count)
        {
            ArgumentNullException.ThrowIfNull(result);

            Trial[] finished = [.. result.Trials.Where(trial => trial.Finished).OrderBy(trial => trial.Score)];

            if (count < 1 || count > finished.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    count,
                    $"A study hands on the best of the trials that finished, and {finished.Length} did: ask for between one and {finished.Length}.");
            }

            return [.. finished.Take(count).Select(trial => trial.Declared)];
        }
    }
}
