// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>A measure a trained model's answers are held to, taken in the answer's own units.</summary>
/// <remarks>
/// The measures of amounts and of classes are scikit-learn's: for an answer of several columns an amount is measured column
/// by column and the columns averaged, each counting alike, and a class is read from each column as nought or one. The
/// measures of shares — the earth mover's distance, the Kullback–Leibler divergence and the ranked probability score — follow
/// scipy's definitions and Weigel's: each row of answers and of predictions is divided by its own total and the two are
/// compared as shares, row by row, and the rows averaged. Each is taken of the outputs <see cref="MetricExtensions.Family"/>
/// says it is for, and refused where a report is written against any other.
/// </remarks>
public enum Metric
{
    /// <summary>The root of the mean of the squared errors.</summary>
    Rmse,

    /// <summary>The mean of the errors' sizes.</summary>
    Mae,

    /// <summary>
    /// One less the squared errors over the squared distances of the answers from their own mean in the part: one for
    /// predictions that are every answer, nought for predicting that mean.
    /// </summary>
    R2,

    /// <summary>The share of rows whose predicted classes are all the classes they hold.</summary>
    Accuracy,

    /// <summary>Of the rows predicted to hold a class, the share that hold it: for each class, averaged over the classes.</summary>
    Precision,

    /// <summary>Of the rows that hold a class, the share predicted to hold it: for each class, averaged over the classes.</summary>
    Recall,

    /// <summary>How many rows of each class held were predicted as each class.</summary>
    ConfusionMatrix,

    /// <summary>
    /// The earth mover's distance between each row's shares and the shares predicted for it, in bands: how far, band by band,
    /// the predicted shares have to move to be the answer's — the Wasserstein distance scipy measures over the bands' places.
    /// </summary>
    /// <remarks>
    /// Taken along the order of the bands, so only of an output whose answers are shares in an order; a band's width is not in
    /// the declaration, so a distance of one is one band. Each row is compared as shares of its own total, so a prediction
    /// whose every band is too large by the same proportion is not told apart from the answer; what is left of a whole, when
    /// the output makes it, stands outside the order and is left out. New in the eighth version of the file.
    /// </remarks>
    Emd,

    /// <summary>
    /// The Kullback–Leibler divergence of each row's answer from the shares predicted for it, in nats: the answer's shares
    /// times the logarithm of how much more of each the answer holds than the prediction, summed — scipy's relative entropy.
    /// </summary>
    /// <remarks>
    /// Taken of every answer of an output whose answers are shares of a whole, what is left of it among them. No number is added
    /// to a share of nought: a band the answer holds and the prediction gives nothing is a divergence without end, reported as
    /// infinite, and one the answer does not hold counts nought. New in the eighth version of the file.
    /// </remarks>
    Kl,

    /// <summary>
    /// The ranked probability score of the shares predicted for each row: the squared distance between the answer's shares
    /// and the prediction's, added up band by band from the first, summed over the bands as Weigel and xskillscore sum it.
    /// </summary>
    /// <remarks>
    /// Taken along the order of the bands, so only of an output whose answers are shares in an order, leaving out what is left of
    /// a whole as the earth mover's distance does. New in the eighth version of the file.
    /// </remarks>
    Rps,
}

/// <summary>What a measure is taken of, which decides the outputs a report may name it against.</summary>
public enum MetricFamily
{
    /// <summary>Amounts, column by column: rmse, mae and r2, taken of any output.</summary>
    Amounts,

    /// <summary>Classes, each answer nought or one: accuracy, precision, recall and the confusion matrix, of an output whose answers can be classes.</summary>
    Classes,

    /// <summary>Shares of a whole, row by row: kl, of a distribution.</summary>
    Shares,

    /// <summary>Shares of a whole in an order, row by row along it: emd and rps, of an output whose answers are in an order.</summary>
    OrderedShares,
}

/// <summary>
/// A step that names the measures a trained model's answers are held to, the parts they are taken on, and how they
/// are shown.
/// </summary>
/// <remarks>
/// Named with the pipeline, before any number exists, so every run is measured the same way and a report cannot shrink
/// to whatever happened to look good. Naming the measures does nothing to the rows, so the run acts on nothing for such
/// a step, as for an output that only names its answer; the measures are taken once a model has predicted, by
/// <see cref="PreparedData.Measure"/>. A pipeline has one at most, below the output whose answers it measures.
/// </remarks>
public interface INamesTheMeasures : IPipelineStep
{
    /// <summary>The measures, in the order they are shown.</summary>
    IReadOnlyList<Metric> Metrics { get; }

    /// <summary>The parts they are taken on, in the order they are shown: training, validation and test, or some of them.</summary>
    IReadOnlyList<Part> Parts { get; }

    /// <summary>How they are shown: drawn, as the numbers themselves, or both.</summary>
    IReadOnlyList<Shown> Shown { get; }
}

/// <summary>
/// Names the measures a trained model is held to, the parts they are taken on, and how they are shown.
/// </summary>
/// <remarks>
/// Every measure is taken in the answer's own units: a prediction and the answer it is compared with both come back
/// through the pipeline's way back, because in normalised units every error is small and every model looks excellent.
/// Each is shown beside the same measure of a model that predicts, for every row, the average answer of the training
/// rows, so a number always has something to stand beside. A measure that counts classes — accuracy, precision, recall,
/// the confusion matrix — is refused where it is written against an output whose answers are amounts, such as a
/// distribution's shares or a return; against an output whose answers can be classes, a row whose answer is neither
/// nought nor one is refused when it is measured. A measure of shares is refused where it is written against an output
/// whose answers are no shares of a whole, and one that follows an order against an output that has none.
/// </remarks>
public sealed record ReportStep : IPipelineStep<ReportStep>, INamesTheMeasures, IDescribesColumns
{
    private static readonly SeveralOfParameter<Metric> MetricsKey = new(
        "metrics",
        "The measures, in the order they are shown: rmse, mae and r2 for amounts; accuracy, precision, recall and the confusion matrix for classes; kl for shares of a whole; emd and rps for shares in an order.",
        [Metric.Rmse]);

    private static readonly SeveralOfParameter<Part> PartsKey = new(
        "parts",
        "The parts they are taken on, side by side: the rows a model learns from, the rows it is chosen on and the rows it is tested on. "
        + "The rows held back to predict on, and the rows a split keeps apart, are measured on by nothing.",
        [Part.Validation, Part.Test],
        [Part.Train, Part.Validation, Part.Test]);

    private static readonly SeveralOfParameter<Shown> ShownKey = new(
        "shown", "How they are shown: drawn, as the numbers themselves, or both.", [Pipelines.Shown.Numbers]);

    /// <summary>Declares the measures a trained model is held to, the parts they are taken on, and how they are shown.</summary>
    /// <param name="metrics">The measures, in the order they are shown.</param>
    /// <param name="parts">The parts they are taken on: training, validation and test, or some of them.</param>
    /// <param name="shown">How they are shown.</param>
    /// <exception cref="ArgumentException">It names no measure, no part, or no way of showing them.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// It names a part other than training, validation and test — the rows held back to predict on, the rows a split keeps
    /// apart — or a value that is none of the words.
    /// </exception>
    public ReportStep(IEnumerable<Metric> metrics, IEnumerable<Part> parts, IEnumerable<Shown> shown)
    {
        ArgumentNullException.ThrowIfNull(metrics);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(shown);

        Metrics = MetricsKey.Require([.. metrics]);
        Parts = PartsKey.Require([.. parts]);
        Shown = ShownKey.Require([.. shown]);
    }

    /// <inheritdoc />
    public IReadOnlyList<Metric> Metrics { get; }

    /// <inheritdoc />
    public IReadOnlyList<Part> Parts { get; }

    /// <inheritdoc />
    public IReadOnlyList<Shown> Shown { get; }

    /// <inheritdoc />
    public static string Name => "evidence.report";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the measures a trained model is held to, the parts they are taken on and how they are shown, each in the answer's own units.";

    /// <inheritdoc />
    /// <remarks>New in the third version of the file.</remarks>
    public static int Since => 3;

    /// <inheritdoc />
    public static StepParameters<ReportStep> Parameters { get; } = new StepParameters<ReportStep>()
        .With(MetricsKey, step => step.Metrics)
        .With(PartsKey, step => step.Parts)
        .With(ShownKey, step => step.Shown);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(ReportStep? other) =>
        other is not null && Metrics.SequenceEqual(other.Metrics) && Parts.SequenceEqual(other.Parts) && Shown.SequenceEqual(other.Shown);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var metric in Metrics)
        {
            hash.Add(metric);
        }

        foreach (var part in Parts)
        {
            hash.Add(part);
        }

        foreach (var shown in Shown)
        {
            hash.Add(shown);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ReportStep ReadFrom(JsonElement element) => new(MetricsKey.Read(element), PartsKey.Read(element), ShownKey.Read(element));
}

/// <summary>
/// A report being written in a chain: which measures, on which parts, shown how.
/// </summary>
/// <remarks>
/// Each word adds to what was said before it, so a report can be said in one breath or in several. What it says is
/// checked where the chain takes the report, by the same rules a file's report is held to.
/// </remarks>
public sealed class ReportBuilder
{
    private readonly List<Metric> _metrics = [];
    private readonly List<Part> _parts = [];
    private readonly List<Shown> _shown = [];

    /// <summary>A report that says nothing yet.</summary>
    public ReportBuilder()
    {
    }

    /// <summary>The measures said so far, in the order they were said.</summary>
    public IReadOnlyList<Metric> Metrics => _metrics;

    /// <summary>The parts said so far, in the order they were said.</summary>
    public IReadOnlyList<Part> Parts => _parts;

    /// <summary>How the measures are shown, as said so far.</summary>
    public IReadOnlyList<Shown> Shown => _shown;

    /// <summary>The measures a model's answers are held to.</summary>
    /// <param name="metrics">The measures, in the order they are shown.</param>
    /// <returns>This report, so the next word can be written after it.</returns>
    public ReportBuilder Measure(params Metric[] metrics)
    {
        ArgumentNullException.ThrowIfNull(metrics);

        _metrics.AddRange(metrics);

        return this;
    }

    /// <summary>The parts the measures are taken on.</summary>
    /// <param name="parts">Training, validation and test, or some of them, in the order they are shown.</param>
    /// <returns>This report, so the next word can be written after it.</returns>
    public ReportBuilder On(params Part[] parts)
    {
        ArgumentNullException.ThrowIfNull(parts);

        _parts.AddRange(parts);

        return this;
    }

    /// <summary>How the measures are shown.</summary>
    /// <param name="shown">Drawn, as the numbers themselves, or both.</param>
    /// <returns>This report, so the next word can be written after it.</returns>
    public ReportBuilder As(params Shown[] shown)
    {
        ArgumentNullException.ThrowIfNull(shown);

        _shown.AddRange(shown);

        return this;
    }
}

/// <summary>What each measure is taken of.</summary>
/// <remarks>
/// Said once, here, for every reader: the rule that refuses a measure where a report is written, the measures themselves, and
/// whatever draws them.
/// </remarks>
public static class MetricExtensions
{
    extension(Metric metric)
    {
        /// <summary>What a measure is taken of.</summary>
        /// <returns>Its family.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The value is none of the measures: a number cast to the set.</exception>
        public MetricFamily Family() => metric switch
        {
            Metric.Rmse or Metric.Mae or Metric.R2 => MetricFamily.Amounts,
            Metric.Accuracy or Metric.Precision or Metric.Recall or Metric.ConfusionMatrix => MetricFamily.Classes,
            Metric.Kl => MetricFamily.Shares,
            Metric.Emd or Metric.Rps => MetricFamily.OrderedShares,
            _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, "This is none of the measures a report can name."),
        };

        /// <summary>Whether a measure counts classes, which only answers that can be classes have.</summary>
        /// <returns><see langword="true"/> for accuracy, precision, recall and the confusion matrix.</returns>
        internal bool CountsClasses() => metric.Family() == MetricFamily.Classes;

        /// <summary>Whether a measure compares shares of a whole, row by row.</summary>
        /// <returns><see langword="true"/> for emd, kl and rps.</returns>
        internal bool ComparesShares() => metric.Family() is MetricFamily.Shares or MetricFamily.OrderedShares;
    }

    extension(IEnumerable<Metric> metrics)
    {
        /// <summary>Measures as a file writes them, each once, in a phrase: "accuracy", "accuracy and recall", "accuracy, precision and recall".</summary>
        /// <returns>The phrase.</returns>
        internal string Listed()
        {
            var words = metrics.Distinct().Select(metric => Vocabulary<Metric>.WordFor(metric, "metrics")).ToArray();

            return words.Length == 1 ? words[0] : $"{string.Join(", ", words[..^1])} and {words[^1]}";
        }
    }
}
