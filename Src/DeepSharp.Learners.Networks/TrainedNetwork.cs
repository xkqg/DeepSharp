// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// A network behind the pipeline it was trained behind: what it predicts for rows served later, in the answer's own units,
/// and the one file they are kept in together.
/// </summary>
/// <remarks>
/// The file holds the network — its layers, every slot's numbers, its loss and what it was trained on — and the pipeline
/// as its own file holds it, declaration and fit. A network trained behind one fit of a pipeline answers nothing behind
/// another, however alike their columns' names are, so the file refuses to be read beside any other fit. That is judged on
/// the pipeline's text as the file carries it, never on the pipeline written again by the library reading the file: a file
/// stays readable when the pipeline file's version moves on, and when another writer spaces it or escapes it otherwise.
/// What the run did and how the report measured it are output of the run, and are not in the file.
/// </remarks>
public sealed class TrainedNetwork
{
    /// <summary>The version of the one file this library writes and reads up to.</summary>
    public const int Version = 1;

    internal const string VersionKey = "version";
    internal const string NetworkKey = "network";
    internal const string PipelineKey = "pipeline";
    internal const string TrainingKey = "training";

    // How many rows a serving pass takes at once, unless a caller says otherwise: Keras's own default for `model.predict`,
    // the same default FitOptions.BatchSize holds for a run unless it is told another.
    private const int ServingBatchSize = 32;

    /// <summary>
    /// What a network needs of the features it is handed, stated once for every place it is handed them: every feature on one
    /// scale, between minus one and one, so it is handed only a run of every step, or one made for it.
    /// </summary>
    internal const Needs FeatureNeeds = Needs.OneScale;

    private PipelineText? _carried;

    internal TrainedNetwork(Network network, Loss loss, PreparedData prepared, TrainedOn trainedOn)
    {
        Network = network;
        Loss = loss;
        Prepared = prepared;
        TrainedOn = trainedOn;
    }

    /// <summary>The network, holding the numbers it learned.</summary>
    public Network Network { get; }

    /// <summary>What it was trained to bring down, whose output activation every prediction goes through.</summary>
    public Loss Loss { get; }

    /// <summary>The pipeline it was trained behind: what serves it rows, and what brings its answers back into their own units.</summary>
    public PreparedData Prepared { get; }

    /// <summary>
    /// What it was trained on: its features, its answers, the fit of the pipeline, the seed and the epoch — and, unless it was
    /// read from a file of 0.4.0's, each feature that held one value on every training row.
    /// </summary>
    public TrainedOn TrainedOn { get; }

    /// <summary>What the run did, epoch by epoch; nothing for a network read from its file.</summary>
    public History? History { get; internal init; }

    /// <summary>How the pipeline's report measured it; nothing when the pipeline declares no report, or for a network read from its file.</summary>
    public Measures? Measures { get; internal init; }

    /// <summary>
    /// The pipeline's text as the one file carries it: as the file held it, for a network read from one; as this library
    /// writes the pipeline, for a network trained here.
    /// </summary>
    internal PipelineText Carried
    {
        get => _carried ??= PipelineText.Of(Prepared);
        init => _carried = value;
    }

    /// <summary>What the network predicts for rows that arrived after training, in the answer's own units, on a light engine of its own.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline hands over other features than the network was trained on; or a row is refused as a training row
    /// would be.
    /// </exception>
    /// <remarks>
    /// The light engine is the one that ships with DeepSharp and needs nothing installed — the one a run takes when its
    /// options name none. <see cref="Predict(IRowSource, ITensorBackend)"/> serves on another, and answers alike.
    /// </remarks>
    public Predictions Predict(IRowSource rows) => Predict(rows, new CpuBackend());

    /// <summary>What the network predicts for rows that arrived after training, in the answer's own units, on the engine handed in.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="backend">
    /// Where the arithmetic runs, for this call alone. The engine is the caller's and owned by nothing: the network holds
    /// none and its file names none, so it serves on whichever engine its caller hands it, whichever one it was trained on.
    /// </param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline hands over other features than the network was trained on; or a row is refused as a training row
    /// would be.
    /// </exception>
    /// <remarks>
    /// Each row is replayed with the numbers the training rows produced, answered on the engine by the evaluation the
    /// report's measures are taken with, through the loss's output activation — thirty-two rows a pass, Keras's own default
    /// for a model's <c>predict</c>, which moves no answer — and brought back: a value or
    /// the chance of one for a target, a share of a whole as a count, a return as a price. Each row is also named with the
    /// features it moves away from the one value every training row held them at, which the network learned nothing about.
    /// </remarks>
    public Predictions Predict(IRowSource rows, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(backend);

        var served = Prepared.Served(rows, FeatureNeeds);

        if (!served.FeatureNames.SequenceEqual(TrainedOn.Features))
        {
            throw new InvalidOperationException(
                $"This network was trained on {Listed(TrainedOn.Features)}, and its pipeline hands over {Listed(served.FeatureNames)}: a network answers only what it learned from.");
        }

        return new Predictions(TrainedOn.Answers, Prepared.BackToOriginal(Answered(Network, Loss, served.Features, new Chunking(backend, ServingBatchSize)), served, rows), served.HandedInAt)
        {
            Unfamiliar = Unfamiliar(TrainedOn, served.Features),
        };
    }

    /// <summary>The one file: the network, what it was trained on, and the pipeline it was trained behind.</summary>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    /// <exception cref="OutOfMemoryException">
    /// The file would be longer than .NET makes a text: a network of about 45 million parameters or more.
    /// </exception>
    /// <remarks>
    /// <para>
    /// A network read from its file writes the pipeline back as that file carried it, not as this library would write the
    /// pipeline now, so what the network recorded of the fit it was trained behind still holds of the file written again.
    /// </para>
    /// <para>
    /// The file is one text, and .NET makes no text longer than 1,073,741,791 characters. A network's numbers take between
    /// 23.6 and 24 characters a parameter, so the largest network the file holds has about 45 million parameters, on every
    /// system alike.
    /// </para>
    /// </remarks>
    public string ToJson() => Json(compiled: null, checkpoint: null);

    /// <summary>Reads a trained network back from its file — or from a checkpoint, as it stood at the end of that epoch.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its network may be made of.</param>
    /// <param name="steps">The verbs its pipeline may use.</param>
    /// <returns>The network behind its pipeline, ready to predict.</returns>
    /// <exception cref="NetworkFileException">
    /// The file is not one this library reads, its network names nothing it was trained on, or it was trained behind
    /// another fit of the pipeline beside it, or for other answers — every fault at its line and column.
    /// </exception>
    /// <exception cref="PipelineFileException">Anything in the pipeline is wrong, at its line and column in the file.</exception>
    /// <remarks>
    /// The network is held to the pipeline's text as the file carries it: the SHA-256 it recorded is of that text, laid out as
    /// a pipeline file is, and nothing else. A file written when the pipeline file named another version, or spaced, broken
    /// into lines or escaped by another writer since, is read as it was written.
    /// </remarks>
    public static TrainedNetwork FromJson(string json, NetworkCatalog networks, StepCatalog steps)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(steps);

        var file = new FileText(json);
        var saved = NetworkDocument.ReadNetwork(json, NetworkKey, networks);
        var prepared = PreparedData.FromJson(json, steps, PipelineKey);
        var trainedOn = saved.TrainedOn
            ?? throw file.Refused(NetworkKey, "This file's network names nothing it was trained on: the file of a trained network says its features, its answers and the fit of the pipeline it was trained behind.");
        var carried = file.Pipeline();

        if ((Mismatch(trainedOn, prepared) ?? Behind(trainedOn, carried)) is { } mismatch)
        {
            throw file.Refused(mismatch.Key, mismatch.Message);
        }

        return new TrainedNetwork(saved.Network, saved.Loss, prepared, trainedOn) { Carried = carried };
    }

    /// <summary>A network behind a pipeline, as it stands at the start of a run: whatever its numbers are, trained on nothing yet.</summary>
    internal static TrainedNetwork Of(Network network, Loss loss, PreparedData prepared, long seed) =>
        new(network, loss, prepared, TrainedOnOf(prepared, prepared.Batch(Part.Train, FeatureNeeds), seed, epoch: 0));

    /// <summary>What a network trained on a pipeline's training rows was trained on.</summary>
    internal static TrainedOn TrainedOnOf(PreparedData prepared, Batch train, long seed, int epoch) => new()
    {
        Features = train.FeatureNames,
        Answers = train.AnswerNames!,
        Output = prepared.Declaration.Output!.Verb,
        TrainedBehind = PipelineText.Of(prepared).Digest,
        Seed = seed,
        Epoch = epoch,
        Unvaried = Unvaried(train),
    };

    /// <summary>
    /// For each of these rows, in their order, the features it moves away from the one value every training row held them
    /// at, in the order the network takes them — the one way a network here names what it learned nothing about, for the
    /// report and for serving alike; nothing when what it was trained on does not say which features held one value.
    /// </summary>
    /// <remarks>The rows hand over the features the network was trained on, in its order.</remarks>
    internal static IReadOnlyList<IReadOnlyList<string>>? Unfamiliar(TrainedOn trainedOn, IReadOnlyList<double[]> rows)
    {
        if (trainedOn.Unvaried is not { } unvaried)
        {
            return null;
        }

        var features = trainedOn.Features;
        int[] held = [.. Enumerable.Range(0, features.Count).Where(at => unvaried.ContainsKey(features[at]))];

        return
        [
            .. rows.Select(row => (IReadOnlyList<string>)[.. held.Where(at => row[at] != unvaried[features[at]]).Select(at => features[at])]),
        ];
    }

    /// <summary>
    /// What the network answers for rows of features, as a pipeline hands them over: <see cref="Network.Predict"/> on the
    /// engine <paramref name="chunking"/> names, so many rows at once — the one way a network here answers rows, for the
    /// report's measures and for serving alike.
    /// </summary>
    internal static double[][] Answered(Network network, Loss loss, IReadOnlyList<double[]> features, Chunking chunking)
    {
        if (features.Count == 0)
        {
            return [];
        }

        var width = features[0].Length;
        var answers = new List<double[]>(features.Count);

        for (var start = 0; start < features.Count; start += chunking.BatchSize)
        {
            var count = Math.Min(chunking.BatchSize, features.Count - start);
            var outputs = network.Predict(features.Skip(start).Take(count).ToArray().Floats(width), loss, chunking.Backend);
            var values = outputs.Values.ToArray();
            var columns = outputs.Shape[1];

            answers.AddRange(Enumerable.Range(0, count).Select(row => values.Skip(row * columns).Take(columns).Select(value => (double)value).ToArray()));
        }

        return [.. answers];
    }

    /// <summary>
    /// What a network was trained to answer, beside a pipeline it is to answer behind: nothing when they agree; otherwise the
    /// part of the file that disagrees, and how.
    /// </summary>
    internal static FileFault? Mismatch(TrainedOn trainedOn, PreparedData prepared)
    {
        var output = prepared.Declaration.Output;

        if (output?.Verb != trainedOn.Output)
        {
            return new FileFault(NetworkKey, $"The network was trained for the answers '{trainedOn.Output}' names, and its pipeline names them with '{output?.Verb}'.");
        }

        return output.Answers.SequenceEqual(trainedOn.Answers)
            ? null
            : new FileFault(NetworkKey, $"The network was trained to answer {Listed(trainedOn.Answers)}, and its pipeline answers {Listed(output.Answers)}.");
    }

    /// <summary>
    /// A network beside the pipeline its file carries: nothing when that text is the one the network recorded it was trained
    /// behind; otherwise the part of the file that disagrees, and how.
    /// </summary>
    internal static FileFault? Behind(TrainedOn trainedOn, PipelineText carried) =>
        carried.Digest == trainedOn.TrainedBehind ? null : new FileFault(PipelineKey, AnotherFit(trainedOn, carried));

    /// <summary>What is said of a network beside another fit of its pipeline than the one it was trained behind.</summary>
    internal static string AnotherFit(TrainedOn trainedOn, PipelineText beside) =>
        $"The network was trained behind another fit of its pipeline: it was trained behind {trainedOn.TrainedBehind}, and the pipeline beside it is {beside.Digest}. "
        + "The same names are not the same fit: its numbers were learned from rows prepared by the one it was trained behind.";

    /// <summary>The one file, and — for a checkpoint — what the run needs to go on.</summary>
    internal string Json(CompiledNetwork? compiled, Checkpoint? checkpoint)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, Version);
            writer.WritePropertyName(NetworkKey);
            NetworkDocument.WriteNetwork(writer, Network, Loss, TrainedOn);
            writer.WritePropertyName(PipelineKey);
            writer.WriteRawValue(Carried.Text);

            if (checkpoint is not null)
            {
                writer.WritePropertyName(TrainingKey);
                NetworkDocument.WriteTraining(writer, compiled!, checkpoint);
            }

            writer.WriteEndObject();
        }

        // The file is the same file everywhere, a line feed between lines as in the pipeline's own file it carries, and so
        // is the largest one a machine can write.
        return stream.TextWithLineFeeds();
    }

    private static string Listed(IEnumerable<string> names) => string.Join(", ", names.Select(name => $"'{name}'"));

    // Each feature that held one value on every training row, as the rows were handed over, with that value: a network
    // learns nothing about how its answer moves with it. A feature is looked at until it shows a second value, and none held
    // one where there are no rows.
    private static Dictionary<string, double> Unvaried(Batch train)
    {
        var unvaried = new Dictionary<string, double>(StringComparer.Ordinal);

        for (var at = 0; at < train.Width; at++)
        {
            if (train.Features.Select(row => row[at]).Distinct().Take(2).ToArray() is [var only])
            {
                unvaried[train.FeatureNames[at]] = only;
            }
        }

        return unvaried;
    }
}

/// <summary>What a trained network predicted for rows served to it, in its answers' own units.</summary>
/// <param name="AnswerNames">What each number of a row's answers stands for, in the order the output names them.</param>
/// <param name="Answers">
/// For each served row, a number for each answer: a value or the chance of one for a target, one chance for each label,
/// a count for each share of a whole, a price for a return — as the pipeline's way back gives it; never a class label.
/// </param>
/// <param name="HandedInAt">For each served row, where it stood among the rows handed in, from nought.</param>
public readonly record struct Predictions(IReadOnlyList<string> AnswerNames, IReadOnlyList<double[]> Answers, IReadOnlyList<int> HandedInAt)
{
    /// <summary>
    /// For each served row, in the order of <see cref="Answers"/>, the features it moves away from the one value every
    /// training row held them at, in the order the network takes them: empty for a row that moves none. Nothing when the
    /// network's file does not say which features held one value — no file written by 0.4.0 does — so nothing is said of
    /// any row either way.
    /// </summary>
    /// <remarks>
    /// A network learns nothing about how its answer moves with a feature its training rows never varied, so a row named
    /// here is answered by what no training row taught it: a category the training rows never held, or a gap where they held
    /// none, reaches weights that are still as the random start drew them, and the answer is that start's rather than
    /// anything learned. A pipeline that should answer no category its training rows never held declares
    /// <see cref="Unseen.Refuse"/>, and refuses it where it is read.
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<string>>? Unfamiliar { get; init; }
}

/// <summary>A fault in the one file: the part it is in, and what is wrong.</summary>
/// <param name="Key">The part of the file, by its key.</param>
/// <param name="Message">What is wrong there.</param>
internal readonly record struct FileFault(string Key, string Message);

/// <summary>Where an evaluation pass runs, and how many rows it takes at once.</summary>
/// <param name="Backend">Where the arithmetic runs.</param>
/// <param name="BatchSize">How many rows one evaluation pass takes at once, the last of the rows holding what is left.</param>
/// <remarks>
/// Serving and the report answer rows a chunk at a time rather than all of them in one pass, however many are handed in: the
/// report's chunks are the run's own <see cref="FitOptions.BatchSize"/>, so measuring costs the engine no more than training
/// did; serving's are thirty-two, Keras's own default for a model's <c>predict</c>. Chunking changes nothing an evaluation
/// pass computes — a row's answer never depends on which other rows share its pass — only how many passes carry it.
/// </remarks>
internal readonly record struct Chunking(ITensorBackend Backend, int BatchSize);
