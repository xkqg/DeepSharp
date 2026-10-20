// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models;
using MatPlotLibNet.Models.Series;
using MatPlotLibNet.Rendering.TickFormatters;
using MatPlotLibNet.Rendering.TickLocators;
using MatPlotLibNet.Styling;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Charts;

/// <summary>
/// The charts of a trained model's measures, as its pipeline's report declared them, drawn from the measures themselves:
/// everything in the answer's own units, as the measures were taken.
/// </summary>
/// <remarks>
/// Each chart puts the parts side by side — the same measure on the rows a model learned from, was chosen on and was tested
/// on — because the gap between them is what a run has to show: a small error on training and a large one on test is a
/// model that learned its training rows by heart, seen at a glance rather than worked out.
/// </remarks>
public static class MeasureCharts
{
    private const int Side = 360;

    extension(Measures measures)
    {
        /// <summary>
        /// Every confusion matrix as a heatmap of counts: a row to each class the rows held, a column to each class predicted.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The measures hold no confusion matrix: the report names none.</exception>
        public string ConfusionMatrices() => ConfusionFigure(measures).ToSvg();

        /// <summary>What was predicted for each row against what the row held, part by part, beside the line where the two agree.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        public string PredictedAgainstActual() => PredictedFigure(measures).ToSvg();

        /// <summary>What was left over — each row's answer less what was predicted for it — against what was predicted, part by part.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <remarks>Residuals scattered evenly about nought are what a model that learned what there was to learn leaves behind.</remarks>
        public string Residuals() => ResidualFigure(measures).ToSvg();

        /// <summary>
        /// The reliability curve of every answer's chances, part by part: the rows sorted by the chance said, in bins of as many
        /// rows each, the mean chance said in a bin against the share of its rows that held the class — beside the line where the
        /// two agree, which a model whose seventy per cent is right seven times in ten lies on.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <remarks>
        /// Ten bins, or as many as a part has rows when it has fewer, each a run of the sorted rows as numpy's <c>array_split</c>
        /// deals them — scikit-learn's <c>calibration_curve</c> with <c>strategy='quantile'</c> in spirit, every bin holding rows, so
        /// no point stands on two rows while another stands on two hundred, the bias of equal-width bins Roelofs et al. measured.
        /// </remarks>
        public string Reliability() => ReliabilityFigure(measures).ToSvg();

        /// <summary>Every measure that is a number as bars: each part's, beside predicting the training rows' average.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The measures hold no number: the report names the confusion matrix alone.</exception>
        /// <remarks>A measure without end — a divergence from a prediction of nought — stands at nothing, its panel saying whose it is and where.</remarks>
        public string Bars() => BarFigure(measures).ToSvg();

        /// <summary>
        /// The report: the measures as their pipeline's report says they are shown — the numbers, these charts, or both — as one
        /// value a notebook shows as HTML.
        /// </summary>
        /// <returns>The report; <see cref="MeasuresReport.ToHtml"/> gives its HTML.</returns>
        /// <exception cref="ArgumentNullException">There are no measures.</exception>
        /// <remarks>
        /// A C# cell of a notebook that ends with <c>trained.Measures!.Report()</c> shows the report there, and a notebook's report
        /// block draws the same rendering: the report is rendered here, once, wherever it is shown.
        /// </remarks>
        public MeasuresReport Report()
        {
            ArgumentNullException.ThrowIfNull(measures);

            return new MeasuresReport(measures);
        }
    }

    extension(EnsembleMeasures ensemble)
    {
        /// <summary>
        /// Every measure that is a number as bars: the ensemble's bar beside each member's and beside predicting the training rows'
        /// average, part by part.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The measures hold no number: the report names the confusion matrix alone.</exception>
        /// <remarks>What combining bought is the gap between the ensemble's bar and the members'; a measure without end stands at nothing, as in <see cref="Bars"/>.</remarks>
        public string BarsWithMembers() => EnsembleBarFigure(ensemble).ToSvg();
    }

    extension(MeasuresAcross across)
    {
        /// <summary>
        /// What the folds measured, every measure that is a number as a panel: each part's mean over the folds with its spread as a
        /// bar about it, beside the mean of the folds' baselines.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The report names the confusion matrix alone: there is no number to draw.</exception>
        /// <remarks>
        /// A model that beats its baseline by less than its folds differ by is seen as a bar that reaches across the baseline. A
        /// mean without end — a divergence from a prediction of nought on every fold — stands at nothing, as it does in
        /// <see cref="Bars"/>.
        /// </remarks>
        public string AcrossFolds() => AcrossFoldsFigure(across).ToSvg();
    }

    /// <summary>The figure <see cref="BarsWithMembers"/> draws: a panel to each measure, the parts along it.</summary>
    internal static Figure EnsembleBarFigure(EnsembleMeasures ensemble)
    {
        ArgumentNullException.ThrowIfNull(ensemble);

        var whole = ensemble.Ensemble;
        var count = whole.Parts[0].Values.Count;

        if (count == 0)
        {
            throw new ArgumentException("These measures hold no number to draw as bars: the report names the confusion matrix alone.", nameof(ensemble));
        }

        string[] parts = [.. whole.Parts.Select(part => part.Part.Word())];
        var builder = new FigureBuilder().WithSize(Side * (double)count, Side);

        for (var metric = 0; metric < count; metric++)
        {
            var at = metric;
            double[] together = [.. whole.Parts.Select(part => part.Values[at].Value)];
            double[] average = [.. whole.Parts.Select(part => part.Values[at].Baseline)];
            BarGroup[] groups =
            [
                new("ensemble", [.. together.Select(Drawable)]),
                .. ensemble.Members.Select((member, number) => new BarGroup($"member {number + 1}", [.. member.Parts.Select(part => Drawable(part.Values[at].Value))])),
                new("average", [.. average.Select(Drawable)]),
            ];

            string[] infinite = [.. new[] { Infinite("ensemble's", together, parts), Infinite("average's", average, parts) }.Where(said => said.Length > 0)];
            var title = whole.Parts[0].Values[at].Metric.Word() + (infinite.Length > 0 ? $": {string.Join("; ", infinite)}" : string.Empty);

            builder.AddSubPlot(1, count, metric + 1, axes => axes
                .WithTitle(title)
                .GroupedBar(parts, groups)
                .WithLegend());
        }

        return builder.Build();
    }

    /// <summary>The figure <see cref="AcrossFolds"/> draws: a panel to each measure, the parts along it.</summary>
    internal static Figure AcrossFoldsFigure(MeasuresAcross across)
    {
        ArgumentNullException.ThrowIfNull(across);

        var count = across.Parts[0].Values.Count;

        if (count == 0)
        {
            throw new ArgumentException("These folds hold no number to draw: the report names the confusion matrix alone.", nameof(across));
        }

        string[] parts = [.. across.Parts.Select(part => part.Part.Word())];
        double[] places = [.. Enumerable.Range(0, parts.Length).Select(place => (double)place)];
        var builder = new FigureBuilder().WithSize(Side * (double)count, Side);

        for (var metric = 0; metric < count; metric++)
        {
            var at = metric;
            MeasuredAcross[] measured = [.. across.Parts.Select(part => part.Values[at])];
            double[] spread = [.. measured.Select(value => Drawable(value.Spread))];
            var title = $"{measured[0].Metric.Word()}, the mean of {across.Folds} folds";

            builder.AddSubPlot(1, count, metric + 1, axes => axes
                .WithTitle(title)
                .ErrorBar(places, [.. measured.Select(value => Drawable(value.Mean))], spread, spread, bars => bars.Label = "mean, and its spread")
                .Scatter(places, [.. measured.Select(value => Drawable(value.Baseline))], points => points.Label = "average")
                .SetXTickLocator(new FixedLocator(places))
                .SetXTickFormatter(new CategoryFormatter(parts))
                .WithLegend());
        }

        return builder.Build();
    }

    /// <summary>The figure <see cref="ConfusionMatrices"/> draws: the parts across, each part's matrices down.</summary>
    internal static Figure ConfusionFigure(Measures measures)
    {
        ArgumentNullException.ThrowIfNull(measures);

        var down = measures.Parts[0].Confusions.Count;

        if (down == 0)
        {
            throw new ArgumentException("These measures hold no confusion matrix: the report names none.", nameof(measures));
        }

        var across = measures.Parts.Count;
        var builder = new FigureBuilder().WithSize(Side * across, Side * down);

        for (var row = 0; row < down; row++)
        {
            for (var column = 0; column < across; column++)
            {
                var part = measures.Parts[column];
                var confusion = part.Confusions[row];
                var title = confusion.Answer is { } answer ? $"{part.Part.Word()}: {answer}" : part.Part.Word();

                builder.AddSubPlot(down, across, (row * across) + column + 1, axes => Counted(axes.WithTitle(title), confusion));
            }
        }

        return builder.Build();
    }

    /// <summary>The figure <see cref="PredictedAgainstActual"/> draws: the parts across, the answers down.</summary>
    internal static Figure PredictedFigure(Measures measures) =>
        EachPartAndAnswer(measures, (axes, part, answer) =>
        {
            var name = measures.Answers[answer];
            double[] actual = [.. part.Actual.Select(row => row[answer])];
            double[] predicted = [.. part.Predicted.Select(row => row[answer])];
            double[] ends = [Math.Min(actual.Min(), predicted.Min()), Math.Max(actual.Max(), predicted.Max())];

            axes.SetXLabel($"actual {name}").SetYLabel($"predicted {name}")
                .Scatter(actual, predicted, points => points.Label = "rows")
                .Plot(ends, ends, line =>
                {
                    line.Label = "predicted = actual";
                    line.LineStyle = LineStyle.Dashed;
                });
        });

    /// <summary>The figure <see cref="Residuals"/> draws: the parts across, the answers down.</summary>
    internal static Figure ResidualFigure(Measures measures) =>
        EachPartAndAnswer(measures, (axes, part, answer) =>
        {
            double[] predicted = [.. part.Predicted.Select(row => row[answer])];
            double[] left = [.. part.Actual.Select((row, at) => row[answer] - predicted[at])];

            axes.SetXLabel($"predicted {measures.Answers[answer]}").SetYLabel("actual less predicted")
                .Scatter(predicted, left, points => points.Label = "rows")
                .AxHLine(0);
        });

    /// <summary>The figure <see cref="Reliability"/> draws: the parts across, the answers down.</summary>
    internal static Figure ReliabilityFigure(Measures measures) =>
        EachPartAndAnswer(measures, (axes, part, answer) =>
        {
            var name = measures.Answers[answer];
            var bins = Binned(part, answer);

            axes.SetXLabel($"chance of {name} said").SetYLabel($"share that held {name}").SetXLim(0, 1).SetYLim(0, 1)
                .Plot([0, 1], [0, 1], line =>
                {
                    line.Label = "said = held";
                    line.LineStyle = LineStyle.Dashed;
                })
                .Plot(bins.Said, bins.Held, line =>
                {
                    line.Label = "rows, in bins of as many";
                    line.Marker = MarkerStyle.Circle;
                });
        });

    // A part's rows sorted by the chance said of one answer and dealt into ten runs — fewer when there are fewer rows — the first
    // runs a row longer where they do not divide evenly: each run's mean chance said, and the share of its rows that held it.
    internal static Bins Binned(PartMeasures part, int answer)
    {
        var order = Enumerable.Range(0, part.Rows).OrderBy(row => part.Predicted[row][answer]).ToArray();
        var count = Math.Min(10, order.Length);
        var said = new double[count];
        var held = new double[count];
        var start = 0;

        for (var bin = 0; bin < count; bin++)
        {
            var length = (order.Length / count) + (bin < order.Length % count ? 1 : 0);
            var rows = order[start..(start + length)];

            said[bin] = rows.Average(row => part.Predicted[row][answer]);
            held[bin] = rows.Average(row => part.Actual[row][answer] > 0.5 ? 1.0 : 0.0);
            start += length;
        }

        return new Bins(said, held);
    }

    /// <summary>The figure <see cref="Bars"/> draws: a panel to each measure, the parts along it.</summary>
    internal static Figure BarFigure(Measures measures)
    {
        ArgumentNullException.ThrowIfNull(measures);

        var count = measures.Parts[0].Values.Count;

        if (count == 0)
        {
            throw new ArgumentException("These measures hold no number to draw as bars: the report names the confusion matrix alone.", nameof(measures));
        }

        string[] parts = [.. measures.Parts.Select(part => part.Part.Word())];
        var builder = new FigureBuilder().WithSize(Side * (double)count, Side);

        for (var metric = 0; metric < count; metric++)
        {
            var at = metric;
            double[] model = [.. measures.Parts.Select(part => part.Values[at].Value)];
            double[] average = [.. measures.Parts.Select(part => part.Values[at].Baseline)];
            BarGroup[] groups =
            [
                new("model", [.. model.Select(Drawable)]),
                new("average", [.. average.Select(Drawable)]),
            ];

            string[] infinite = [.. new[] { Infinite("model's", model, parts), Infinite("average's", average, parts) }.Where(said => said.Length > 0)];
            var title = measures.Parts[0].Values[at].Metric.Word() + (infinite.Length > 0 ? $": {string.Join("; ", infinite)}" : string.Empty);

            builder.AddSubPlot(1, count, metric + 1, axes => axes
                .WithTitle(title)
                .GroupedBar(parts, groups)
                .WithLegend());
        }

        return builder.Build();
    }

    // A measure without end — a divergence from a prediction of nought where the answer holds something — has no height to
    // be drawn at: its bar stands at nothing, and the panel's title says whose it is and where.
    private static double Drawable(double value) => double.IsFinite(value) ? value : 0;

    private static string Infinite(string whose, double[] values, string[] parts)
    {
        string[] where = [.. parts.Where((_, at) => !double.IsFinite(values[at]))];

        return where switch
        {
            [] => string.Empty,
            [var only] => $"the {whose} is infinite on {only}",
            _ => $"the {whose} is infinite on {string.Join(", ", where[..^1])} and {where[^1]}",
        };
    }

    // A panel to each answer of each part, the parts across and the answers down, each drawn by the given hand.
    private static Figure EachPartAndAnswer(Measures measures, Action<AxesBuilder, PartMeasures, int> draw)
    {
        ArgumentNullException.ThrowIfNull(measures);

        var across = measures.Parts.Count;
        var down = measures.Answers.Count;
        var builder = new FigureBuilder().WithSize(Side * across, Side * down);

        for (var answer = 0; answer < down; answer++)
        {
            for (var column = 0; column < across; column++)
            {
                var part = measures.Parts[column];
                var at = answer;
                var title = down == 1 ? part.Part.Word() : $"{part.Part.Word()}: {measures.Answers[at]}";

                builder.AddSubPlot(down, across, (answer * across) + column + 1, axes => draw(axes.WithTitle(title).WithLegend(), part, at));
            }
        }

        return builder.Build();
    }

    // One confusion matrix: a class held to a row, a class predicted to a column, each cell its count.
    private static void Counted(AxesBuilder axes, Confusion confusion)
    {
        var size = confusion.Classes.Count;
        var counts = new double[size, size];
        var positions = Enumerable.Range(0, size).Select(position => (double)position).ToArray();
        string[] classes = [.. confusion.Classes];

        for (var held = 0; held < size; held++)
        {
            for (var predicted = 0; predicted < size; predicted++)
            {
                counts[held, predicted] = confusion.Counts[held][predicted];
            }
        }

        axes.SetXLabel("predicted").SetYLabel("actual")
            .Heatmap(counts, series =>
            {
                series.ColorMap = ColorMaps.Blues;
                series.ShowLabels = true;
                series.LabelFormat = "0";
            })
            .SetXTickLocator(new FixedLocator(positions))
            .SetXTickFormatter(new CategoryFormatter(classes))
            .SetYTickLocator(new FixedLocator(positions))
            .SetYTickFormatter(new CategoryFormatter(classes, reversed: true));
    }

    /// <summary>The points of a reliability curve: each bin's mean chance said, and the share of its rows that held the class.</summary>
    /// <param name="Said">Each bin's mean chance said, rising.</param>
    /// <param name="Held">The share of each bin's rows that held the class.</param>
    internal readonly record struct Bins(double[] Said, double[] Held);
}
