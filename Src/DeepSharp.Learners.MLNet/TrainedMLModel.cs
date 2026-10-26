// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.InteropServices;
using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using Microsoft.ML;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// A trainer from ML.NET, trained behind the pipeline that prepared its rows.
/// </summary>
/// <remarks>
/// What a run of a pipeline that names a trainer hands back: the trainer as the declaration named it, the rows it
/// learned from, and what it predicts for any part of the same run. It is held to the pipeline it was trained behind —
/// the same way a network is — so a prediction is never asked of a model beside another fit of its own steps. The
/// fitting and the predicting are <see cref="FittedMLModel"/>'s, held here behind the pipeline that gave it its rows.
/// </remarks>
public sealed class TrainedMLModel
{
    private readonly FittedMLModel _fitted;

    internal TrainedMLModel(FittedMLModel fitted, PreparedData prepared, TrainedOnRows trainedOn)
    {
        _fitted = fitted;
        Prepared = prepared;
        TrainedOn = trainedOn;
        Archive = new Lazy<byte[]>(Saved, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    /// <summary>The run this model was trained behind: its steps, its parts and everything they learned.</summary>
    public PreparedData Prepared { get; }

    /// <summary>The trainer as the declaration named it, with the settings it was given.</summary>
    public PartDeclaration Trainer => _fitted.Trainer;

    /// <summary>The number the trainer's random draws were worked out from.</summary>
    public int Seed => _fitted.Seed;

    /// <summary>What the model was trained on: how many rows, and the features and answer it was handed.</summary>
    public TrainedOnRows TrainedOn { get; }

    /// <summary>
    /// The width its answers are worth: the margin the errors on the calibration rows allow, at the level it was given. Nothing until
    /// <see cref="WithInterval(double)"/> gives it one.
    /// </summary>
    public ConformalInterval? Interval { get; private init; }

    // ML.NET's archive of the model, saved once: it stamps its entries with the time it is saved, so a model saved at every call would
    // write a different file each time it is asked, and the file a digest binds is not the one written next.
    private Lazy<byte[]> Archive { get; init; }

    /// <summary>How the pipeline's own report measures this model, on the parts the report names.</summary>
    /// <returns>The measures, or nothing when the pipeline declares no report.</returns>
    /// <remarks>
    /// The pipeline's measures, not ML.NET's: the same metrics on the same rows a network would have been measured on,
    /// which is the only reason a tree and a network can honestly be compared. Its own evaluators are left alone — one of
    /// them reads its logarithm in base two, which would be a different number under the same name.
    /// </remarks>
    public Measures? Measures =>
        Prepared.Declaration.Steps.OfType<INamesTheMeasures>().FirstOrDefault() is { } report
            ? Prepared.Measure([.. report.Parts.Select(part => PredictedFor(part) with { Margins = Interval?.Margins })])
            : null;

    /// <summary>The same model with the width its answers are worth, worked out on the rows the pipeline set aside to calibrate on.</summary>
    /// <param name="alpha">The share of rows allowed to land outside the margin: above nothing and below one. Required.</param>
    /// <returns>The same model behind the same pipeline, its report measured with the coverage and width of the interval when it names them.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">The pipeline sets no rows aside to calibrate on, they are too few for the level, or the answer is a class.</exception>
    /// <remarks>
    /// The same split conformal prediction a network is given: the margin is the ⌈(n+1)(1−α)⌉-th smallest of this model's errors on the
    /// calibration rows, in the answer's own units. It is kept in the model's file.
    /// </remarks>
    public TrainedMLModel WithInterval(double alpha) =>
        new(_fitted, Prepared, TrainedOn) { Interval = Prepared.IntervalOf(PredictedFor(Part.Calibration), alpha), Archive = Archive };

    // What the model predicts for a part, as the pipeline measures predictions: one answer a row.
    private PartPredictions PredictedFor(Part part) =>
        new(Prepared.Batch(part, LearnMLStep.FeatureNeeds), [.. Predict(part).Select(prediction => new[] { prediction })]);

    /// <summary>This model and the pipeline it was trained behind, as one file.</summary>
    /// <returns>The file's text.</returns>
    /// <remarks>
    /// It names the version of ML.NET that wrote the model and the processor it was written on, because the model itself
    /// is ML.NET's own archive: a reader that cannot open it can then say which package and which version would.
    /// </remarks>
    public string ToJson()
    {
        return (MLModelFile.Of(PipelineText.Of(Prepared), Trainer, Archive.Value) with
        {
            Seed = Seed,
            Library = typeof(MLContext).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion,
            Processor = RuntimeInformation.ProcessArchitecture.ToString(),
            Answer = TrainedOn.Answer,
            Classes = TrainedOn.Classes,
            Features = TrainedOn.Features,
            Rows = TrainedOn.Rows,
            Interval = Interval,
        }).ToJson();
    }

    private byte[] Saved()
    {
        using var saved = new MemoryStream();

        _fitted.Context.Model.Save(_fitted.Model, inputSchema: null, saved);

        return saved.ToArray();
    }

    /// <summary>What this model predicts for rows that arrived after training, in the answer's own units.</summary>
    /// <param name="rows">The rows, with the columns the pipeline reads; the answer's own column needs not be among them.</param>
    /// <returns>For each row, in the order they were handed in, the answer.</returns>
    /// <exception cref="ArgumentNullException">There are no rows.</exception>
    /// <exception cref="InvalidOperationException">The pipeline hands over other features than the model was trained on.</exception>
    /// <remarks>Each row is replayed with the numbers the training rows produced, scored by the model, and brought back through the pipeline's way back.</remarks>
    public IReadOnlyList<double[]> Predict(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var served = Prepared.Served(rows, LearnMLStep.FeatureNeeds);

        if (!served.FeatureNames.SequenceEqual(TrainedOn.Features))
        {
            throw new InvalidOperationException(
                $"This model was trained on {string.Join(", ", TrainedOn.Features.Select(name => $"'{name}'"))}, and its pipeline hands over {string.Join(", ", served.FeatureNames.Select(name => $"'{name}'"))}: a model answers only what it learned from.");
        }

        var scored = _fitted.Predict(FeaturesOf(served.Features));

        return Prepared.BackToOriginal([.. AnswersOf(scored).Select(answer => new[] { answer })], served, rows);
    }

    /// <summary>What this model predicts for every row of a part of the run it was trained behind.</summary>
    /// <param name="part">The part.</param>
    /// <returns>One prediction a row, in the order the handover gives them.</returns>
    /// <exception cref="ArgumentException">The part asked for is the gap a split keeps apart.</exception>
    public IReadOnlyList<double> Predict(Part part)
    {
        var batch = Prepared.Batch(part, LearnMLStep.FeatureNeeds);

        return AnswersOf(_fitted.Predict(FeaturesOf(batch.Features)));
    }

    // The rows of a pipeline's own numbers, as the Tensor FittedMLModel.Predict reads.
    private static Tensor FeaturesOf(IReadOnlyList<double[]> rows)
    {
        var width = rows.Count == 0 ? 0 : rows[0].Length;

        return Tensor.From(new Shape([rows.Count, width]), [.. rows.SelectMany(row => row.Select(value => (float)value))]);
    }

    // The one answer a row FittedMLModel.Predict hands back, read out of its width-one tensor.
    private static double[] AnswersOf(Tensor predicted) => [.. predicted.Values.ToArray().Select(value => (double)value)];
}

/// <summary>What a model was trained on: the rows it learned from, and what each of them held.</summary>
/// <param name="Rows">How many training rows it learned from.</param>
/// <param name="Features">The features it was handed, in their order.</param>
/// <param name="Answer">The column it was asked to predict.</param>
/// <param name="Classes">Whether that answer was one of two classes rather than a number, decided from the training rows.</param>
public readonly record struct TrainedOnRows(int Rows, IReadOnlyList<string> Features, string Answer, bool Classes);
