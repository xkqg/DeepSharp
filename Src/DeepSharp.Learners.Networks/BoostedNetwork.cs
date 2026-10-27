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
/// A basis and the network trained in sequence after it, on what the basis left unexplained: one answerer, its answer the
/// two networks' raw answers summed and brought back into the answer's own units once.
/// </summary>
/// <remarks>
/// Not an ensemble of networks answering together, which <see cref="TrainedEnsemble"/> already is: there, every member is
/// trained apart and their answers meet only once each has come back into the answer's own units. Here the correction is
/// fitted on the basis's own frozen answer, taken on the training rows alone and never refit, subtracted from the truth —
/// the same training-split-only, replay-unchanged rule the rest of the library is built on. The two meet before either
/// comes back: their raw answers are summed first, and the single way back is walked once, since the mean of exponentials
/// is not the exponential of the mean and the sum of two is not the sum of their two ways back either.
/// </remarks>
public sealed class BoostedNetwork
{
    /// <summary>The version of the one file this library writes a boosted pair as, and reads up to.</summary>
    public const int Version = 1;

    internal const string BasisKey = "basis";
    internal const string CorrectionKey = "correction";
    internal const string BasisDigestKey = "basisDigest";

    private BoostedNetwork(TrainedNetwork basis, TrainedNetwork correction, Measures? measures)
    {
        Basis = basis;
        Correction = correction;
        Measures = measures;
    }

    /// <summary>The network trained first, on the rows a pipeline prepared.</summary>
    public TrainedNetwork Basis { get; }

    /// <summary>
    /// The network trained on what <see cref="Basis"/> left unexplained. Not public: a correction answers nothing on its
    /// own, for it was never trained to, so it is reached only behind the pair that trained it.
    /// </summary>
    internal TrainedNetwork Correction { get; }

    /// <summary>The pipeline both networks were trained behind: what serves them rows, and what brings their answers back into their own units.</summary>
    public PreparedData Prepared => Basis.Prepared;

    /// <summary>How the pipeline's report measured the pair, on the rows its combined answer gives; nothing when the pipeline declares no report.</summary>
    public Measures? Measures { get; }

    /// <summary>Puts a basis and a correction together as a boosted pair.</summary>
    /// <param name="basis">The network trained first.</param>
    /// <param name="correction">The network trained on what the basis left unexplained, by <see cref="BoostedExtensions.TrainBoosted"/>.</param>
    /// <param name="basisDigest">The digest of the basis the correction was fitted against.</param>
    /// <param name="measures">How the pipeline's report measured the pair; nothing when the pipeline declares no report.</param>
    /// <returns>The pair.</returns>
    /// <exception cref="ArgumentException">
    /// The two were trained behind different fits of a pipeline, or the correction was fitted against another basis than this one.
    /// </exception>
    internal static BoostedNetwork Of(TrainedNetwork basis, TrainedNetwork correction, string basisDigest, Measures? measures = null)
    {
        if (correction.TrainedOn.TrainedBehind != basis.TrainedOn.TrainedBehind)
        {
            throw new ArgumentException(
                $"The correction was trained behind another fit of its pipeline than the basis was: {correction.TrainedOn.TrainedBehind}, not {basis.TrainedOn.TrainedBehind}. "
                + "A boosted pair's two networks answer rows replayed once, so they are trained behind one fit.",
                nameof(correction));
        }

        if (Digest(basis) != basisDigest)
        {
            throw new ArgumentException(
                "The correction was fitted against a different basis than this one: its own frozen answer, not this basis's, was subtracted from the truth it learned from.",
                nameof(basis));
        }

        return new BoostedNetwork(basis, correction, measures);
    }

    /// <summary>The digest of a trained network's own numbers: what a correction's file records it was fitted against.</summary>
    internal static string Digest(TrainedNetwork network) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(network.ToJson()))).ToLowerInvariant();

    /// <summary>What the pair predicts for rows that arrived after training, in the answer's own units, on a light engine of its own.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    public Predictions Predict(IRowSource rows) => Predict(rows, new CpuBackend());

    /// <summary>What the pair predicts for rows that arrived after training, in the answer's own units, on the engine handed in.</summary>
    /// <param name="rows">The rows, without the answer: it is what is being asked.</param>
    /// <param name="backend">Where the arithmetic runs, for this call alone.</param>
    /// <returns>For each row, a number for each answer, and where the row stood among those handed in.</returns>
    public Predictions Predict(IRowSource rows, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(backend);

        var served = Prepared.Served(rows, TrainedNetwork.FeatureNeeds);

        if (!served.FeatureNames.SequenceEqual(Basis.TrainedOn.Features))
        {
            throw new InvalidOperationException(
                $"This boosted pair was trained on {string.Join(", ", Basis.TrainedOn.Features.Select(name => $"'{name}'"))}, and its pipeline hands over "
                + $"{string.Join(", ", served.FeatureNames.Select(name => $"'{name}'"))}: a network answers only what it learned from.");
        }

        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize) { Example = served.ExampleShape };
        var combined = Combined(served.Features, chunking);
        var answers = Prepared.BackToOriginal(combined, served, rows);

        return new Predictions(Basis.TrainedOn.Answers, answers, served.HandedInAt)
        {
            Unfamiliar = TrainedNetwork.Unfamiliar(Basis.TrainedOn, served.Features),
            Ids = served.Ids,
        };
    }

    // The two networks' raw answers, summed before either comes back into the answer's own units.
    private double[][] Combined(IReadOnlyList<double[]> features, Chunking chunking)
    {
        var basisAnswered = TrainedNetwork.Answered(Basis.Network, Basis.Loss, features, chunking);
        var correctionAnswered = TrainedNetwork.Answered(Correction.Network, Correction.Loss, features, chunking);

        return Summed(basisAnswered, correctionAnswered);
    }

    // One row for another, each number of one added to the matching number of the other.
    internal static double[][] Summed(IReadOnlyList<double[]> first, IReadOnlyList<double[]> second)
    {
        var summed = new double[first.Count][];

        for (var row = 0; row < first.Count; row++)
        {
            summed[row] = new double[first[row].Length];

            for (var column = 0; column < first[row].Length; column++)
            {
                summed[row][column] = first[row][column] + second[row][column];
            }
        }

        return summed;
    }

    /// <summary>The one file: the basis, the correction, the digest it was fitted against, and the pipeline they were trained behind.</summary>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    public string ToJson()
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(TrainedNetwork.VersionKey, Version);
            writer.WritePropertyName(BasisKey);
            NetworkDocument.WriteNetwork(writer, Basis.Network, Basis.Loss, Basis.TrainedOn);
            writer.WritePropertyName(CorrectionKey);
            NetworkDocument.WriteNetwork(writer, Correction.Network, Correction.Loss, Correction.TrainedOn);
            writer.WriteString(BasisDigestKey, Digest(Basis));
            writer.WritePropertyName(TrainedNetwork.PipelineKey);
            writer.WriteRawValue(Basis.Carried.Text);
            writer.WriteEndObject();
        }

        return stream.TextWithLineFeeds();
    }

    /// <summary>Reads a boosted pair back from its file.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its networks may be made of.</param>
    /// <param name="steps">The verbs its pipeline may use.</param>
    /// <returns>The pair behind its pipeline, ready to predict.</returns>
    /// <exception cref="NetworkFileException">
    /// The file is not one this library reads as a boosted pair's, or holds a network trained behind another fit of the
    /// pipeline beside it, for other answers, or fitted against another basis than the one beside it in the file.
    /// </exception>
    /// <exception cref="PipelineFileException">Anything in the pipeline is wrong, at its line and column in the file.</exception>
    public static BoostedNetwork FromJson(string json, NetworkCatalog networks, StepCatalog steps)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(steps);

        var file = new FileText(json, FileKind.Boosted);
        var basisSaved = NetworkDocument.ReadNetwork(json, BasisKey, networks);
        var correctionSaved = NetworkDocument.ReadNetwork(json, CorrectionKey, networks);
        var prepared = PreparedData.FromJson(json, steps, TrainedNetwork.PipelineKey);
        var carried = file.Pipeline();

        var basisTrainedOn = basisSaved.TrainedOn
            ?? throw file.Refused(BasisKey, "This file's basis names nothing it was trained on: the file of a trained network says its features, its answers and the fit of the pipeline it was trained behind.");
        var correctionTrainedOn = correctionSaved.TrainedOn
            ?? throw file.Refused(CorrectionKey, "This file's correction names nothing it was trained on: the file of a trained network says its features, its answers and the fit of the pipeline it was trained behind.");

        if ((TrainedNetwork.Mismatch(basisTrainedOn, prepared) ?? TrainedNetwork.Behind(basisTrainedOn, carried)) is { } basisMismatch)
        {
            throw file.Refused(BasisKey, basisMismatch.Message);
        }

        var basis = new TrainedNetwork(basisSaved.Network, basisSaved.Loss, prepared, basisTrainedOn) { Carried = carried };
        var correction = new TrainedNetwork(correctionSaved.Network, correctionSaved.Loss, prepared, correctionTrainedOn) { Carried = carried };
        var basisDigest = file.BasisDigest
            ?? throw file.Refused(BasisDigestKey, "This file names no digest the correction was fitted against, under 'basisDigest'.");

        return Of(basis, correction, basisDigest);
    }
}

/// <summary>How a boosted pair is trained, beyond the declarations of its basis and its correction.</summary>
public sealed class BoostedOptions
{
    /// <summary>The engines the names in the basis's and the correction's declarations stand for; the light engine alone, unless given.</summary>
    public Engines? Engines { get; init; }

    /// <summary>What stops either network's run before its next batch or its next look at the validation rows; nothing stops it, unless said.</summary>
    public CancellationToken Cancellation { get; init; }
}

/// <summary>Training one network on what another already answered.</summary>
public static class BoostedExtensions
{
    extension(PreparedData prepared)
    {
        /// <summary>
        /// Trains a basis on the rows a pipeline prepared, then trains a correction on what the basis left unexplained: the
        /// training rows' own answers less the basis's own frozen answer on them, replayed unchanged on the validation rows the
        /// correction is judged by.
        /// </summary>
        /// <param name="basis">The network to train first: its layers, what moves them, what judges it, and its own seed.</param>
        /// <param name="correction">The network to train on what the basis left unexplained.</param>
        /// <param name="options">The engines the names in the declarations stand for, and what stops either run; the light engine alone and unstoppable, unless said.</param>
        /// <returns>The pair.</returns>
        /// <exception cref="ArgumentNullException">There is nothing prepared, or no basis or no correction.</exception>
        /// <exception cref="InvalidOperationException">
        /// Either network's loss is not the identity, or either answers a quantile — the sum of two quantiles is not generally the
        /// quantile of the sum — or anything training one network behind a pipeline refuses.
        /// </exception>
        /// <remarks>
        /// Only a loss whose output activation is the identity composes this way: the correction's target is a bare subtraction,
        /// which means something only for a number a network can literally subtract — not a chance, a share of a whole, or a
        /// margin. A quantile is refused by name even though its own activation is the identity, since the sum of two
        /// independently-fitted quantile predictions is not in general the same quantile of the summed distribution.
        /// </remarks>
        public BoostedNetwork TrainBoosted(LearnNetworkStep basis, LearnNetworkStep correction, BoostedOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(basis);
            ArgumentNullException.ThrowIfNull(correction);

            var settings = options ?? new BoostedOptions();
            var output = prepared.Declaration.Output;

            RequireComposable(basis.Loss, output, "basis");
            RequireComposable(correction.Loss, output, "correction");

            var basisTrained = prepared.Train(basis, settings.Engines, settings.Cancellation);
            var correctionTrained = Corrected(prepared, basisTrained, correction, settings);
            var measures = Measured(prepared, basisTrained, correctionTrained, settings);

            return BoostedNetwork.Of(basisTrained, correctionTrained, BoostedNetwork.Digest(basisTrained), measures);
        }
    }

    // Refuses a declared part's loss unless its output activation is the identity, by name — a correction's target is a
    // bare subtraction, which means something only for a number a network can literally subtract — and a quantile beside,
    // whose own activation is the identity but whose sum is not in general the quantile of the summed distribution.
    private static void RequireComposable(PartDeclaration loss, INamesTheAnswer? output, string which)
    {
        var judged = TrainExtensions.JudgedFor(loss, output);

        if (judged is Quantile)
        {
            throw new InvalidOperationException(
                $"The {which} answers a Quantile, and a boosted pair's two networks are summed: the sum of two independently-fitted quantile predictions is not in "
                + "general the same quantile of the summed distribution, so a Quantile is refused here by name.");
        }

        if (judged.Activation != OutputActivation.Identity)
        {
            throw new InvalidOperationException(
                $"The {which}'s loss '{judged.GetType().Name}' answers through an output activation that is not the Identity, and a boosted pair's correction is a bare "
                + "subtraction of the basis's own raw answer from the truth: only a loss whose activation is the Identity composes that way.");
        }
    }

    // The correction, trained on the training rows' own answers less the basis's own frozen answer on them, and judged —
    // when the pipeline sets validation rows aside — by the same subtraction replayed on those rows, never refit on them.
    private static TrainedNetwork Corrected(PreparedData prepared, TrainedNetwork basisTrained, LearnNetworkStep correction, BoostedOptions settings)
    {
        var controls = new TrainExtensions.RunControls(settings.Engines, null, settings.Cancellation);
        var backend = (settings.Engines ?? new Engines()).Named(correction.Engine);
        var trainBatch = prepared.Batch(Part.Train, TrainedNetwork.FeatureNeeds);
        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize) { Example = trainBatch.ExampleShape };
        var trainingData = ResidualOf(trainBatch, basisTrained, chunking);

        var validationBatch = prepared.Batch(Part.Validation, TrainedNetwork.FeatureNeeds);
        var validationData = validationBatch.RowCount > 0 ? ResidualOf(validationBatch, basisTrained, chunking with { Example = validationBatch.ExampleShape }) : null;

        var options = TrainExtensions.OptionsFor(correction, controls);
        var compiled = correction.Layers.Described(prepared).Compile(correction.Optimizer.Moved(), TrainExtensions.JudgedFor(correction.Loss, prepared.Declaration.Output), correction.Schedule.Scheduled());
        var history = compiled.Fit(trainingData, validationData, options);
        var epoch = options.EarlyStopping is { RestoreBest: true } && history.BestEpoch is { } best ? best : history.Epochs[^1].Number;
        var trainedOn = TrainedNetwork.TrainedOnOf(prepared, trainBatch, correction.Seed, epoch);

        return new TrainedNetwork(compiled.Network, compiled.Loss, prepared, trainedOn) { History = history };
    }

    // The rows of a part, their answers replaced by what the basis left unexplained of them: the truth less the basis's
    // own frozen answer, worked out on the training rows alone and replayed here unchanged, never refit.
    private static TrainingData ResidualOf(Batch batch, TrainedNetwork basisTrained, Chunking chunking)
    {
        var basisAnswered = TrainedNetwork.Answered(basisTrained.Network, basisTrained.Loss, batch.Features, chunking);
        var residual = Subtracted(batch.Answers!, basisAnswered);

        return new TrainingData(batch.Features.Floats(batch.Width, batch.ExampleShape), residual.Floats(batch.AnswerNames!.Count));
    }

    // One row's answers for another, each number of the second taken from the matching number of the first.
    private static double[][] Subtracted(IReadOnlyList<double[]> truth, IReadOnlyList<double[]> basisAnswered)
    {
        var residual = new double[truth.Count][];

        for (var row = 0; row < truth.Count; row++)
        {
            residual[row] = new double[truth[row].Length];

            for (var column = 0; column < truth[row].Length; column++)
            {
                residual[row][column] = truth[row][column] - basisAnswered[row][column];
            }
        }

        return residual;
    }

    // The pipeline's report, measured on the pair's combined answer, part by part; nothing when the pipeline declares no report.
    private static Measures? Measured(PreparedData prepared, TrainedNetwork basisTrained, TrainedNetwork correctionTrained, BoostedOptions settings)
    {
        if (prepared.Declaration.Report is not { } report)
        {
            return null;
        }

        var backend = (settings.Engines ?? new Engines()).Named(LearnNetworkStep.LightEngine);
        var chunking = new Chunking(backend, TrainedNetwork.ServingBatchSize);

        IReadOnlyList<PartPredictions> said =
        [
            .. report.Parts.Distinct().Select(part =>
            {
                var batch = prepared.Batch(part, TrainedNetwork.FeatureNeeds);
                var basisAnswered = TrainedNetwork.Answered(basisTrained.Network, basisTrained.Loss, batch.Features, chunking with { Example = batch.ExampleShape });
                var correctionAnswered = TrainedNetwork.Answered(correctionTrained.Network, correctionTrained.Loss, batch.Features, chunking with { Example = batch.ExampleShape });

                return new PartPredictions(batch, BoostedNetwork.Summed(basisAnswered, correctionAnswered))
                {
                    Unfamiliar = TrainedNetwork.Unfamiliar(basisTrained.TrainedOn, batch.Features),
                };
            }),
        ];

        return prepared.Measure(said);
    }
}
