// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a model predicted for the rows of one part, with the batch those rows were handed over in.
/// </summary>
/// <param name="Batch">
/// The part as <see cref="Handover.Batch(PreparedData, Part, Needs)"/> handed it over: which part it is, and the key of each
/// row in its order.
/// </param>
/// <param name="Predictions">
/// One row of predictions for each row of the batch, in the batch's order, each with a number for every answer the output
/// names, in its order — in the units the answers were handed over in.
/// </param>
/// <remarks>
/// The batch is how a prediction finds the row it was made for. Predictions are measured only when the batch's keys are
/// the part's rows in the order the part hands them over: in any other order each would be measured against another
/// row's answer, and nothing about the numbers would say so.
/// </remarks>
public readonly record struct PartPredictions(Batch Batch, IReadOnlyList<double[]> Predictions)
{
    /// <summary>
    /// What the model says of each row, in the batch's order: the features it was handed a value of that it learned nothing
    /// about, named — empty for a row it knows. Nothing when the model says nothing of it.
    /// </summary>
    /// <remarks>
    /// The model is the one that knows: a network learns nothing about a feature its training rows held at one value, and
    /// says so of a row that moves it. The report counts the rows of each part that name any, beside its measures.
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<string>>? Unfamiliar { get; init; }
}

/// <summary>One measure of one part, beside the same measure of predicting the training rows' average answer.</summary>
/// <param name="Metric">The measure.</param>
/// <param name="Value">What the model's predictions measured.</param>
/// <param name="Baseline">
/// What predicting, for every row, the average of the training rows' answers measured: scikit-learn's dummy model that
/// predicts the mean, and for classes the one that predicts each class's share; for shares, the training rows' average
/// shares.
/// </param>
public readonly record struct Measured(Metric Metric, double Value, double Baseline);

/// <summary>How the rows of one part fall between the classes they hold and the classes predicted for them.</summary>
/// <remarks>
/// A row of the matrix is a class held and a column a class predicted, each cell counting the rows. When a row holds
/// exactly one of the answers there is one matrix across them, its classes the answers' names; otherwise one for each
/// answer, its classes nought and one.
/// </remarks>
public sealed class Confusion
{
    internal Confusion(string? answer, IReadOnlyList<string> classes, IReadOnlyList<int[]> counts, IReadOnlyList<int[]> baseline)
    {
        Answer = answer;
        Classes = classes;
        Counts = counts;
        Baseline = baseline;
    }

    /// <summary>The answer a matrix of nought and one counts; nothing for a matrix whose classes are the answers themselves.</summary>
    public string? Answer { get; }

    /// <summary>The classes, in order: the rows of the matrix and its columns alike.</summary>
    public IReadOnlyList<string> Classes { get; }

    /// <summary>For each class held, a row, how many of the rows holding it were predicted as each class, a column.</summary>
    public IReadOnlyList<int[]> Counts { get; }

    /// <summary>The same counts for predicting, for every row, the average of the training rows' answers.</summary>
    public IReadOnlyList<int[]> Baseline { get; }
}

/// <summary>What one part measured, measure by measure, and the rows it was measured on.</summary>
public sealed class PartMeasures
{
    internal PartMeasures(Part part, Answered rows, IReadOnlyList<Measured> values, IReadOnlyList<Confusion> confusions)
    {
        Part = part;
        Actual = rows.Actual;
        Predicted = rows.Predicted;
        Values = values;
        Confusions = confusions;
    }

    /// <summary>The part.</summary>
    public Part Part { get; }

    /// <summary>How many rows it was measured on.</summary>
    public int Rows => Actual.Count;

    /// <summary>
    /// How many of them the model said it was handed a value of that it learned nothing about, which it answered by nothing
    /// it learned; nothing when the predictions measured said nothing of it.
    /// </summary>
    public int? UnfamiliarRows { get; internal init; }

    /// <summary>
    /// Every measure the report names that is a number, in the order it names them, each beside the same measure of
    /// predicting the training rows' average answer.
    /// </summary>
    public IReadOnlyList<Measured> Values { get; }

    /// <summary>The confusion matrices, when the report names the confusion matrix; none otherwise.</summary>
    public IReadOnlyList<Confusion> Confusions { get; }

    /// <summary>Each row's answers in their own units, in the order the part hands its rows over.</summary>
    public IReadOnlyList<double[]> Actual { get; }

    /// <summary>What the model predicted for each row, brought back into the same units.</summary>
    public IReadOnlyList<double[]> Predicted { get; }
}

/// <summary>
/// The measures a trained model's predictions were held to, as its pipeline's report declared them: part by part, each in
/// the answer's own units and beside the same measure of predicting the training rows' average answer.
/// </summary>
/// <remarks>
/// Output, like every piece of evidence: kept with whatever measured it, and never written into the pipeline's file,
/// since the file is what is replayed. The numbers are all here, the rows' answers and predictions among them, for
/// whatever shows them to draw from; and what they were measured from crosses, as text, to wherever the same pipeline
/// runs (<see cref="PredictionsToJson"/>).
/// </remarks>
public sealed class Measures : Evidence
{
    // A report stands below an output, so the pipeline it was measured on names the answers: the declaration refuses one that
    // does not.
    internal Measures(INamesTheMeasures report, PreparedData measuredOn, IReadOnlyList<PartMeasures> parts, IReadOnlyList<WrittenPart> predicted)
    {
        Metrics = report.Metrics;
        Shown = report.Shown;
        Answers = measuredOn.Declaration.Output!.Answers;
        Parts = parts;
        Predicted = predicted;
        MeasuredOn = measuredOn;
    }

    /// <summary>The measures, in the order the report names them.</summary>
    public IReadOnlyList<Metric> Metrics { get; }

    /// <summary>How the report says they are shown.</summary>
    public IReadOnlyList<Shown> Shown { get; }

    /// <summary>The answers measured, in the order the output names them: what each number of a row's values stands for.</summary>
    public IReadOnlyList<string> Answers { get; }

    /// <summary>Each part measured, in the order the report names them.</summary>
    public IReadOnlyList<PartMeasures> Parts { get; }

    // What they were measured from: for each part the report names, once and in its order, the keys of its rows, what was
    // predicted for each and what the model said of each — none of the rows' numbers, which the text never carries.
    internal IReadOnlyList<WrittenPart> Predicted { get; }

    // The run they were measured on: the fit the predictions were made behind, which their text names.
    private PreparedData MeasuredOn { get; }

    /// <summary>
    /// What these measures were taken from, as text: the fit the predictions were made behind; and for each part the report
    /// names, the key of each of its rows, what was predicted for each in the units the answers were handed over in, and —
    /// when the model said it — the features of each row it learned nothing about.
    /// </summary>
    /// <returns>The text, as one JSON object, which <see cref="PreparedData.MeasureAgain"/> measures again.</returns>
    /// <remarks>
    /// The way predictions cross to wherever the same pipeline runs, where the types of this library are other types even when
    /// their names are the same: a notebook's C# cell that trains a model hands it back to the notebook with
    /// <c>Variables.Set("deepsharp.predictions", trained.Measures!.PredictionsToJson())</c>, and the notebook's report block
    /// measures it on the notebook's own run and draws it. Only what the model gave crosses, with the fit it was made behind
    /// — its pipeline's <see cref="PipelineText.FitDigest"/>, the version of the pipeline file that names and, for a run made
    /// for a learner that left steps out, which it left out — so it is measured only beside that very fit, by the rule a
    /// network's file is held to its pipeline by, or on a run of every step of the same declaration; what it is measured
    /// against, and the way back to the answers' own units, are the run's that measures it.
    /// </remarks>
    public string PredictionsToJson() => new WrittenPredictions(WrittenFit.Of(MeasuredOn), Answers, Predicted).ToJson();

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>The answers of a part's rows in their own units, and what was predicted for each, in the same units.</summary>
/// <param name="actual">Each row's answers.</param>
/// <param name="predicted">What was predicted for each row, as many numbers as it has answers.</param>
/// <remarks>
/// The measures of amounts and of classes are scikit-learn's, an amount of several columns measured column by column and
/// averaged, each column counting alike. The measures of shares follow scipy's definitions and Weigel's: each row divided by its
/// own total, compared row by row, and the rows averaged.
/// </remarks>
internal sealed class Answered(IReadOnlyList<double[]> actual, IReadOnlyList<double[]> predicted)
{
    /// <summary>Each row's answers.</summary>
    public IReadOnlyList<double[]> Actual { get; } = actual;

    /// <summary>What was predicted for each row.</summary>
    public IReadOnlyList<double[]> Predicted { get; } = predicted;

    /// <summary>How many of the answers, from the first, stand in the output's order; every one, unless said.</summary>
    public int? InTheOrder { get; init; }

    private int Width => Actual[0].Length;

    /// <summary>A measure that is a number.</summary>
    /// <param name="metric">The measure: any but the confusion matrix.</param>
    /// <param name="ones">How many answers a row holds as one, when they are classes; nought for any number.</param>
    /// <returns>Its value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The measure is the confusion matrix, or none of the measures.</exception>
    public double Of(Metric metric, int ones) => metric switch
    {
        Metric.Rmse => Averaged(RootMeanSquared),
        Metric.Mae => Averaged(MeanAbsolute),
        Metric.R2 => Averaged(Explained),
        Metric.Accuracy or Metric.Precision or Metric.Recall => Classes(ones).Of(metric),
        Metric.Emd => RowByRow(InTheOrder ?? Width, Moved),
        Metric.Kl => RowByRow(Width, Divergence),
        Metric.Rps => RowByRow(InTheOrder ?? Width, Ranked),
        _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, "A measure that is a number is one of the measures, and the confusion matrix is none."),
    };

    /// <summary>The classes the rows hold and the classes the predictions name.</summary>
    /// <param name="ones">How many answers a row holds as one; nought for any number.</param>
    /// <returns>Each answer of each row, nought or one, held and predicted.</returns>
    public Classed Classes(int ones) =>
        new([.. Actual.Select(row => row.Select(value => value > 0.5).ToArray())], [.. Predicted.Select(row => Named(row, ones))]);

    // The classes a row of predictions names: each answer at a half or more when a row holds any number of them; the ones
    // likeliest when it holds so many, an earlier answer first among equals.
    private static bool[] Named(double[] predicted, int ones)
    {
        if (ones == 0)
        {
            return [.. predicted.Select(value => value >= 0.5)];
        }

        var named = new bool[predicted.Length];

        foreach (var column in Enumerable.Range(0, predicted.Length).OrderByDescending(column => predicted[column]).Take(ones))
        {
            named[column] = true;
        }

        return named;
    }

    // Column by column, averaged over the columns, each counting alike: scikit-learn's multioutput='uniform_average'.
    private double Averaged(Func<int, double> column) => Enumerable.Range(0, Width).Average(column);

    private double RootMeanSquared(int column) => Math.Sqrt(Rows().Average(row => Squared(Actual[row][column] - Predicted[row][column])));

    private double MeanAbsolute(int column) => Rows().Average(row => Math.Abs(Actual[row][column] - Predicted[row][column]));

    // One less the squared errors over the squared distances from the part's own mean; one where nothing is left over, and
    // nought where the answers never vary and something is, as scikit-learn keeps it finite.
    private double Explained(int column)
    {
        var mean = Actual.Average(row => row[column]);
        var left = Rows().Sum(row => Squared(Actual[row][column] - Predicted[row][column]));
        var spread = Actual.Sum(row => Squared(row[column] - mean));

        return left == 0 ? 1 : spread == 0 ? 0 : 1 - (left / spread);
    }

    private IEnumerable<int> Rows() => Enumerable.Range(0, Actual.Count);

    private static double Squared(double value) => value * value;

    // Row by row, the first answers of each row and of its prediction each divided by their own total, compared as shares,
    // and the rows averaged.
    private double RowByRow(int answers, Func<double[], double[], double> compared) =>
        Rows().Average(row => compared(Shares(Actual[row], answers), Shares(Predicted[row], answers)));

    private static double[] Shares(double[] row, int answers)
    {
        var total = row.Take(answers).Sum();

        return [.. row.Take(answers).Select(value => value / total)];
    }

    // How far the predicted shares have to move to be the answer's: the distance between the two, added up band by band, at
    // every threshold but the last, where both are one.
    private static double Moved(double[] held, double[] said) => Below(held, said).Sum(Math.Abs);

    // The squared distance between the two added up band by band, at every threshold but the last.
    private static double Ranked(double[] held, double[] said) => Below(held, said).Sum(Squared);

    private static IEnumerable<double> Below(double[] held, double[] said)
    {
        var apart = 0.0;

        for (var band = 0; band < held.Length - 1; band++)
        {
            apart += held[band] - said[band];

            yield return apart;
        }
    }

    // The answer's shares times the logarithm of each over the prediction's: nought where the answer holds nothing, and
    // without end where it holds something the prediction gives nothing.
    private static double Divergence(double[] held, double[] said) =>
        Enumerable.Range(0, held.Length).Sum(at => held[at] is 0 ? 0 : said[at] is 0 ? double.PositiveInfinity : held[at] * Math.Log(held[at] / said[at]));
}

/// <summary>Each answer of a part's rows as a class, nought or one: the class held, and the class predicted.</summary>
/// <param name="held">Each row's answers as held.</param>
/// <param name="predicted">Each row's answers as predicted.</param>
internal sealed class Classed(bool[][] held, bool[][] predicted)
{
    private bool[][] Held { get; } = held;

    private bool[][] Predicted { get; } = predicted;

    private int Width => Held[0].Length;

    /// <summary>A measure that counts classes.</summary>
    /// <param name="metric">Accuracy, precision or recall.</param>
    /// <returns>Its value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The measure is none of the three.</exception>
    /// <remarks>
    /// Accuracy asks every answer of a row to be right, scikit-learn's accuracy of labels; precision and recall are taken
    /// for each answer and averaged over the answers, each counting alike — for one answer, its own.
    /// </remarks>
    public double Of(Metric metric) => metric switch
    {
        Metric.Accuracy => Rows().Count(row => Held[row].AsSpan().SequenceEqual(Predicted[row])) / (double)Held.Length,
        Metric.Precision => Averaged(column => Share(Both(column), Predicted.Count(row => row[column]))),
        Metric.Recall => Averaged(column => Share(Both(column), Held.Count(row => row[column]))),
        _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, "A measure that counts classes is accuracy, precision or recall."),
    };

    /// <summary>The confusion matrices of these classes, beside those of another prediction of the same rows.</summary>
    /// <param name="baseline">The classes predicting the training rows' average names.</param>
    /// <param name="answers">The answers' names.</param>
    /// <param name="ones">How many answers a row holds as one; nought for any number.</param>
    /// <returns>One matrix across the answers when a row holds exactly one; one for each answer otherwise.</returns>
    public IReadOnlyList<Confusion> Confusions(Classed baseline, IReadOnlyList<string> answers, int ones) =>
        ones == 1
            ? [new Confusion(null, answers, Across(), baseline.Across())]
            : [.. Enumerable.Range(0, answers.Count).Select(column => new Confusion(answers[column], ["0", "1"], Within(column), baseline.Within(column)))];

    // One matrix across the answers: the answer a row holds against the answer predicted for it.
    private int[][] Across()
    {
        var counts = Enumerable.Range(0, Width).Select(_ => new int[Width]).ToArray();

        foreach (var row in Rows())
        {
            counts[Array.IndexOf(Held[row], true)][Array.IndexOf(Predicted[row], true)]++;
        }

        return counts;
    }

    // One matrix for one answer: nought or one held against nought or one predicted.
    private int[][] Within(int column)
    {
        int[][] counts = [new int[2], new int[2]];

        foreach (var row in Rows())
        {
            counts[Held[row][column] ? 1 : 0][Predicted[row][column] ? 1 : 0]++;
        }

        return counts;
    }

    private int Both(int column) => Rows().Count(row => Held[row][column] && Predicted[row][column]);

    // A share of none is nought, as scikit-learn counts a precision or a recall with nothing to count.
    private static double Share(int part, int whole) => whole == 0 ? 0 : part / (double)whole;

    private double Averaged(Func<int, double> column) => Enumerable.Range(0, Width).Average(column);

    private IEnumerable<int> Rows() => Enumerable.Range(0, Held.Length);
}

/// <summary>
/// Measuring one run's predictions by its report: each part it names, brought back into the answer's units and measured
/// beside the training rows' average answer.
/// </summary>
internal sealed class Measurement
{
    // An answer within a millionth of nought or one is that class: a way back is held to leading back within as much.
    private const double Tolerance = 1e-6;

    private readonly PreparedData _prepared;
    private readonly INamesTheMeasures _report;
    private readonly INamesTheAnswer _output;
    private readonly double[] _average;

    private Measurement(PreparedData prepared, INamesTheMeasures report, INamesTheAnswer output, double[] average)
    {
        _prepared = prepared;
        _report = report;
        _output = output;
        _average = average;
    }

    /// <summary>The measures of these predictions, as the pipeline's report declares them.</summary>
    /// <param name="prepared">The run.</param>
    /// <param name="predictions">What a model predicted, for each part the report names.</param>
    /// <returns>The measures.</returns>
    public static Measures Of(PreparedData prepared, IReadOnlyList<PartPredictions> predictions)
    {
        var report = prepared.Declaration.Report ?? throw new InvalidOperationException(
            "This pipeline declares no report, so nothing says what a model is measured by: declare one below its output, with evidence.report.");

        prepared.RequireRows();

        var given = Given(report, predictions);

        // A report stands below an output: the declaration refuses one that does not.
        var output = prepared.Declaration.Output!;
        var measurement = new Measurement(prepared, report, output, Average(prepared));

        return new Measures(
            report,
            prepared,
            [.. report.Parts.Select(part => measurement.On(part, given[part]))],
            [.. report.Parts.Distinct().Select(part => WrittenPart.Of(part, given[part]))]);
    }

    // The predictions by the part they were made for: one set for every part the report names, and for none else.
    private static Dictionary<Part, PartPredictions> Given(INamesTheMeasures report, IReadOnlyList<PartPredictions> predictions)
    {
        var given = new Dictionary<Part, PartPredictions>();

        foreach (var each in predictions)
        {
            if (each.Batch is not { Part: { } part, Keys: not null })
            {
                throw new ArgumentException(
                    "These predictions were made for a batch the pipeline did not hand over: it names no part, or no row by its key. Predict for what Batch hands over.",
                    nameof(predictions));
            }

            if (!report.Parts.Contains(part))
            {
                throw new ArgumentException(
                    $"These predictions are for '{Word(part)}', which the report does not measure: it measures {Words(report.Parts)}.", nameof(predictions));
            }

            if (!given.TryAdd(part, each))
            {
                throw new ArgumentException($"There are two sets of predictions for '{Word(part)}', and a part is measured once.", nameof(predictions));
            }
        }

        if (report.Parts.Where(part => !given.ContainsKey(part)).Distinct().ToArray() is [_, ..] missing)
        {
            throw new ArgumentException(
                $"The report measures {Words(missing)}, and there are no predictions for it: predict for what Batch hands over.", nameof(predictions));
        }

        return given;
    }

    // The average of the training rows' answers as they were handed over: the prediction a measure stands beside. Only the
    // answers are read, never the features, whatever a learner was handed of them.
    private static double[] Average(PreparedData prepared)
    {
        var train = Handover.AnswersOf(prepared, Part.Train);

        if (train.Keys.Count == 0)
        {
            throw new InvalidOperationException(
                "The training part holds no rows, so there is no average of its answers for a model's measures to stand beside.");
        }

        var average = new double[train.Names!.Count];

        foreach (var answers in train.Answers!)
        {
            for (var column = 0; column < average.Length; column++)
            {
                average[column] += answers[column];
            }
        }

        return [.. average.Select(total => total / train.Keys.Count)];
    }

    // The measures of one part: its rows' answers and keys, checked against the rows the predictions were made for, both
    // brought back into the answer's units, and measured beside the training rows' average brought back the same way.
    private PartMeasures On(Part part, PartPredictions given)
    {
        var handed = Handover.AnswersOf(_prepared, part);
        var rows = handed.Keys.Count;

        if (rows == 0)
        {
            throw new InvalidOperationException($"The report measures '{Word(part)}', and it holds no rows: there is nothing to measure there.");
        }

        ThrowIfOtherRows(part, handed.Keys, given.Batch.Keys!);

        if (given.Unfamiliar is { } said && said.Count != rows)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The predictions for '{Word(part)}' say what is unfamiliar of {said.Count} rows, and the part holds {rows}: say it of every row Batch hands over, in its order."),
                "predictions");
        }

        int[] readAt = [.. Enumerable.Range(0, _prepared.Table.RowCount).Where(row => _prepared.Parts[row] == part).Select(row => _prepared.Table.Identities[row].ReadAt)];
        var inTheOrder = _output.InTheOrder();
        var model = new Answered(_prepared.BackToOriginal(handed.Answers!, part), _prepared.BackToOriginal(given.Predictions, part)) { InTheOrder = inTheOrder };
        var average = new Answered(model.Actual, _prepared.BackToOriginal([.. Enumerable.Repeat(_average, rows)], part)) { InTheOrder = inTheOrder };

        ThrowIfNotFinite(model.Predicted, readAt);

        if (_report.Metrics.Any(MetricExtensions.CountsClasses))
        {
            ThrowIfNotClasses(model.Actual, readAt);
        }

        if (_report.Metrics.Any(MetricExtensions.ComparesShares))
        {
            ThrowIfNoShares(model.Actual, "answers", readAt);
            ThrowIfNoShares(model.Predicted, "prediction", readAt);
        }

        if (_report.Metrics.Contains(Metric.R2) && rows < 2)
        {
            throw new InvalidOperationException(
                $"R² compares a part's answers with their own mean, and '{Word(part)}' holds one row, whose mean is its answer: measure it with rmse or mae, or divide more rows into it.");
        }

        var ones = _output.Ones;
        Measured[] values =
        [
            .. _report.Metrics
                .Where(metric => metric != Metric.ConfusionMatrix)
                .Select(metric => new Measured(metric, model.Of(metric, ones), average.Of(metric, ones))),
        ];

        return new PartMeasures(
            part,
            model,
            values,
            _report.Metrics.Contains(Metric.ConfusionMatrix) ? model.Classes(ones).Confusions(average.Classes(ones), _output.Answers, ones) : [])
        {
            UnfamiliarRows = given.Unfamiliar?.Count(names => names.Count > 0),
        };
    }

    // The part's rows in the order the part hands them over, known by their keys: anything else is measured against other rows.
    private static void ThrowIfOtherRows(Part part, IReadOnlyList<RowKey> keys, IReadOnlyList<RowKey> given)
    {
        if (given.Count != keys.Count)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The predictions for '{Word(part)}' were made for a batch of {given.Count} rows, and the part holds {keys.Count}: predict for every row Batch hands over, in its order."),
                "predictions");
        }

        var other = Enumerable.Range(0, keys.Count).FirstOrDefault(row => keys[row] != given[row], -1);

        if (other >= 0)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The predictions for '{Word(part)}' were made for other rows than the part's, or in another order: row {other + 1} of their batch is not row {other + 1} of the part. Predict for the rows in the order Batch hands them over, or each is measured against another row's answer."),
                "predictions");
        }
    }

    private void ThrowIfNotFinite(IReadOnlyList<double[]> predicted, int[] readAt)
    {
        for (var row = 0; row < predicted.Count; row++)
        {
            for (var column = 0; column < predicted[row].Length; column++)
            {
                if (!double.IsFinite(predicted[row][column]))
                {
                    throw new ArgumentException(
                        string.Create(
                            CultureInfo.InvariantCulture,
                            $"Row {readAt[row] + 1}: the prediction for '{_output.Answers[column]}' comes back as {predicted[row][column]}, which is not a finite number, and nothing measures it."),
                        "predictions");
                }
            }
        }
    }

    // Where the report compares shares, a row is shares of its own total: nothing below nought, and something to divide by —
    // among the answers in the output's order too, for a measure that follows it.
    private void ThrowIfNoShares(IReadOnlyList<double[]> rows, string what, int[] readAt)
    {
        var measures = _report.Metrics.Where(MetricExtensions.ComparesShares).Listed();
        var ordered = _report.Metrics.Any(metric => metric.Family() == MetricFamily.OrderedShares);

        for (var row = 0; row < rows.Count; row++)
        {
            if (rows[row].FirstOrDefault(value => value < 0) is var below and < 0)
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {readAt[row] + 1}: its {what} hold {below}, below nought, and {measures} compare shares, none of which is below nought."));
            }

            if (rows[row].Sum() is 0 || (ordered && rows[row].Take(_output.InTheOrder()).Sum() is 0))
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {readAt[row] + 1}: its {what} add up to nought{(ordered ? " in the order" : string.Empty)}, so they are no shares of anything, and {measures} compare shares."));
            }
        }
    }

    // Where the report counts classes, every answer is nought or one, and a row holds as many ones as its output says.
    private void ThrowIfNotClasses(IReadOnlyList<double[]> actual, int[] readAt)
    {
        for (var row = 0; row < actual.Count; row++)
        {
            for (var column = 0; column < actual[row].Length; column++)
            {
                var value = actual[row][column];

                if (Math.Abs(value) > Tolerance && Math.Abs(value - 1) > Tolerance)
                {
                    throw new InvalidOperationException(string.Create(
                        CultureInfo.InvariantCulture,
                        $"Row {readAt[row] + 1} of '{_output.Answers[column]}' is {value}, and the report counts classes, each nought or one, with {_report.Metrics.Where(MetricExtensions.CountsClasses).Listed()}: an amount is measured with rmse, mae or r2."));
                }
            }

            var held = actual[row].Count(value => value > 0.5);

            if (_output.Ones > 0 && held != _output.Ones)
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {readAt[row] + 1} holds {held} of {string.Join(", ", _output.Answers.Select(answer => $"'{answer}'"))} as one, and '{_output.Verb}' says a row holds {_output.Ones}."));
            }
        }
    }

    private static string Word(Part part) => Vocabulary<Part>.WordFor(part, "parts");

    // The parts as a file writes them, each quoted: 'train', 'validation' and 'test'.
    private static string Words(IEnumerable<Part> parts)
    {
        var words = parts.Select(part => $"'{Word(part)}'").ToArray();

        return words.Length == 1 ? words[0] : $"{string.Join(", ", words[..^1])} and {words[^1]}";
    }
}
