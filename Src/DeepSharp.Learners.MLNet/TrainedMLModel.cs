// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.InteropServices;
using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;
using Microsoft.ML;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// A trainer from ML.NET, trained behind the pipeline that prepared its rows.
/// </summary>
/// <remarks>
/// What a run of a pipeline that names a trainer hands back: the trainer as the declaration named it, the rows it
/// learned from, and what it predicts for any part of the same run. It is held to the pipeline it was trained behind —
/// the same way a network is — so a prediction is never asked of a model beside another fit of its own steps.
/// </remarks>
public sealed class TrainedMLModel
{
    private readonly ITransformer _model;
    private readonly MLContext _context;

    internal TrainedMLModel(ITransformer model, MLContext context, PreparedData prepared, TrainedOnRows trainedOn)
    {
        var step = (LearnMLStep)prepared.Declaration.Learner!;

        _model = model;
        _context = context;
        Prepared = prepared;
        Trainer = step.Trainer;
        Seed = step.Seed;
        TrainedOn = trainedOn;
    }

    /// <summary>The run this model was trained behind: its steps, its parts and everything they learned.</summary>
    public PreparedData Prepared { get; }

    /// <summary>The trainer as the declaration named it, with the settings it was given.</summary>
    public PartDeclaration Trainer { get; }

    /// <summary>The number the trainer's random draws were worked out from.</summary>
    public int Seed { get; }

    /// <summary>What the model was trained on: how many rows, and the features and answer it was handed.</summary>
    public TrainedOnRows TrainedOn { get; }

    /// <summary>How the pipeline's own report measures this model, on the parts the report names.</summary>
    /// <returns>The measures, or nothing when the pipeline declares no report.</returns>
    /// <remarks>
    /// The pipeline's measures, not ML.NET's: the same metrics on the same rows a network would have been measured on,
    /// which is the only reason a tree and a network can honestly be compared. Its own evaluators are left alone — one of
    /// them reads its logarithm in base two, which would be a different number under the same name.
    /// </remarks>
    public Measures? Measures =>
        Prepared.Declaration.Steps.OfType<INamesTheMeasures>().FirstOrDefault() is { } report
            ? Prepared.Measure([.. report.Parts.Select(part => new PartPredictions(
                Prepared.Batch(part, LearnMLStep.FeatureNeeds),
                [.. Predict(part).Select(prediction => new[] { prediction })]))])
            : null;

    /// <summary>This model and the pipeline it was trained behind, as one file.</summary>
    /// <returns>The file's text.</returns>
    /// <remarks>
    /// It names the version of ML.NET that wrote the model and the processor it was written on, because the model itself
    /// is ML.NET's own archive: a reader that cannot open it can then say which package and which version would.
    /// </remarks>
    public string ToJson()
    {
        using var saved = new MemoryStream();

        _context.Model.Save(_model, inputSchema: null, saved);

        return (MLModelFile.Of(PipelineText.Of(Prepared), Trainer, saved.ToArray()) with
        {
            Seed = Seed,
            Library = typeof(MLContext).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion,
            Processor = RuntimeInformation.ProcessArchitecture.ToString(),
            Answer = TrainedOn.Answer,
            Classes = TrainedOn.Classes,
            Features = TrainedOn.Features,
            Rows = TrainedOn.Rows,
        }).ToJson();
    }

    /// <summary>What this model predicts for every row of a part of the run it was trained behind.</summary>
    /// <param name="part">The part.</param>
    /// <returns>One prediction a row, in the order the handover gives them.</returns>
    /// <exception cref="ArgumentException">The part asked for is the gap a split keeps apart.</exception>
    public IReadOnlyList<double> Predict(Part part)
    {
        var batch = Prepared.Batch(part, LearnMLStep.FeatureNeeds);
        var scored = _model.Transform(HandedRows.Of(_context, batch, TrainedOn.Classes, answers: false));

        return [.. _context.Data.CreateEnumerable<ScoredRow>(scored, reuseRowObject: false).Select(row => (double)row.Score)];
    }
}

/// <summary>What a model was trained on: the rows it learned from, and what each of them held.</summary>
/// <param name="Rows">How many training rows it learned from.</param>
/// <param name="Features">The features it was handed, in their order.</param>
/// <param name="Answer">The column it was asked to predict.</param>
/// <param name="Classes">Whether that answer was one of two classes rather than a number, decided from the training rows.</param>
public readonly record struct TrainedOnRows(int Rows, IReadOnlyList<string> Features, string Answer, bool Classes);

/// <summary>One row as ML.NET hands a prediction back.</summary>
internal sealed class ScoredRow
{
    /// <summary>The number the model answers with.</summary>
    public float Score { get; init; }
}
