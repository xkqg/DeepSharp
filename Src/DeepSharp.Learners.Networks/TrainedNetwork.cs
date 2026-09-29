// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;
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
/// another, however alike their columns' names are, so the file refuses to be read beside any other fit. What the run did
/// and how the report measured it are output of the run, and are not in the file.
/// </remarks>
public sealed class TrainedNetwork
{
    /// <summary>The version of the one file this library writes and reads up to.</summary>
    public const int Version = 1;

    internal const string VersionKey = "version";
    internal const string NetworkKey = "network";
    internal const string PipelineKey = "pipeline";
    internal const string TrainingKey = "training";

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

    /// <summary>What it was trained on: its features, its answers, the fit of the pipeline, the seed and the epoch.</summary>
    public TrainedOn TrainedOn { get; }

    /// <summary>What the run did, epoch by epoch; nothing for a network read from its file.</summary>
    public History? History { get; internal init; }

    /// <summary>How the pipeline's report measured it; nothing when the pipeline declares no report, or for a network read from its file.</summary>
    public Measures? Measures { get; internal init; }

    /// <summary>What the network predicts for rows that arrived after training, in the answer's own units.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline hands over other features than the network was trained on; or a row is refused as a training row
    /// would be.
    /// </exception>
    /// <remarks>
    /// Each row is replayed with the numbers the training rows produced, answered by an evaluation pass through the loss's
    /// output activation, and brought back: a value or the chance of one for a target, a share of a whole as a count, a
    /// return as a price.
    /// </remarks>
    public Predictions Predict(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var served = Prepared.Served(rows, Needs.OneScale);

        if (!served.FeatureNames.SequenceEqual(TrainedOn.Features))
        {
            throw new InvalidOperationException(
                $"This network was trained on {Listed(TrainedOn.Features)}, and its pipeline hands over {Listed(served.FeatureNames)}: a network answers only what it learned from.");
        }

        return new Predictions(TrainedOn.Answers, Prepared.BackToOriginal(Answered(Network, Loss, served.Features), served, rows), served.HandedInAt);
    }

    /// <summary>The one file: the network, what it was trained on, and the pipeline it was trained behind.</summary>
    /// <returns>The file, as JSON.</returns>
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

        if (Mismatch(trainedOn, prepared) is { } mismatch)
        {
            throw file.Refused(mismatch.Key, mismatch.Message);
        }

        return new TrainedNetwork(saved.Network, saved.Loss, prepared, trainedOn);
    }

    /// <summary>A network behind a pipeline, as it stands at the start of a run: whatever its numbers are, trained on nothing yet.</summary>
    internal static TrainedNetwork Of(Network network, Loss loss, PreparedData prepared, long seed) =>
        new(network, loss, prepared, TrainedOnOf(prepared, prepared.Batch(Part.Train, Needs.OneScale), seed, epoch: 0));

    /// <summary>What a network trained on a pipeline's training rows was trained on.</summary>
    internal static TrainedOn TrainedOnOf(PreparedData prepared, Batch train, long seed, int epoch) => new()
    {
        Features = train.FeatureNames,
        Answers = train.AnswerNames!,
        Output = prepared.Declaration.Output!.Verb,
        TrainedBehind = Digest(prepared),
        Seed = seed,
        Epoch = epoch,
    };

    /// <summary>What the network answers for rows of features: an evaluation pass, through the loss's output activation.</summary>
    internal static double[][] Answered(Network network, Loss loss, IReadOnlyList<double[]> features)
    {
        if (features.Count == 0)
        {
            return [];
        }

        var backend = new CpuBackend();
        var input = Tensor.From(new Shape(features.Count, features[0].Length), [.. features.SelectMany(row => row.Select(value => (float)value))]);
        var outputs = loss.Predictions(network.Forward(input, Pass.Evaluation(backend)), backend);
        var answers = outputs.Values.ToArray();
        var columns = outputs.Shape[1];

        return [.. Enumerable.Range(0, features.Count).Select(row => answers.Skip(row * columns).Take(columns).Select(value => (double)value).ToArray())];
    }

    /// <summary>
    /// What a network was trained on, beside a pipeline it is to answer behind: nothing when they agree; otherwise the part
    /// of the file that disagrees, and how.
    /// </summary>
    internal static FileFault? Mismatch(TrainedOn trainedOn, PreparedData prepared)
    {
        var output = prepared.Declaration.Output;

        if (output?.Verb != trainedOn.Output)
        {
            return new FileFault(NetworkKey, $"The network was trained for the answers '{trainedOn.Output}' names, and its pipeline names them with '{output?.Verb}'.");
        }

        if (!output.Answers.SequenceEqual(trainedOn.Answers))
        {
            return new FileFault(NetworkKey, $"The network was trained to answer {Listed(trainedOn.Answers)}, and its pipeline answers {Listed(output.Answers)}.");
        }

        return Digest(prepared) == trainedOn.TrainedBehind
            ? null
            : new FileFault(
                PipelineKey,
                $"The network was trained behind another fit of its pipeline: it was trained behind {trainedOn.TrainedBehind}, and the pipeline beside it is {Digest(prepared)}. "
                + "The same names are not the same fit: its numbers were learned from rows prepared by the one it was trained behind.");
    }

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
            writer.WriteRawValue(Prepared.ToJson());

            if (checkpoint is not null)
            {
                writer.WritePropertyName(TrainingKey);
                NetworkDocument.WriteTraining(writer, compiled!, checkpoint);
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    // The digest of a pipeline as fitted: of its own file, declaration and fit.
    private static string Digest(PreparedData prepared) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(prepared.ToJson()))).ToLowerInvariant();

    private static string Listed(IEnumerable<string> names) => string.Join(", ", names.Select(name => $"'{name}'"));
}

/// <summary>What a trained network predicted for rows served to it, in its answers' own units.</summary>
/// <param name="AnswerNames">What each number of a row's answers stands for, in the order the output names them.</param>
/// <param name="Answers">
/// For each served row, a number for each answer: a value or the chance of one for a target, one chance for each label,
/// a count for each share of a whole, a price for a return — as the pipeline's way back gives it; never a class label.
/// </param>
/// <param name="HandedInAt">For each served row, where it stood among the rows handed in, from nought.</param>
public readonly record struct Predictions(IReadOnlyList<string> AnswerNames, IReadOnlyList<double[]> Answers, IReadOnlyList<int> HandedInAt);

/// <summary>A fault in the one file: the part it is in, and what is wrong.</summary>
/// <param name="Key">The part of the file, by its key.</param>
/// <param name="Message">What is wrong there.</param>
internal readonly record struct FileFault(string Key, string Message);
