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

    /// <summary>
    /// Every confusion matrix as a heatmap of counts: a row to each class the rows held, a column to each class predicted.
    /// </summary>
    /// <param name="measures">The measures.</param>
    /// <returns>The chart, as the text of an SVG.</returns>
    /// <exception cref="ArgumentException">The measures hold no confusion matrix: the report names none.</exception>
    public static string ConfusionMatrices(this Measures measures) => ConfusionFigure(measures).ToSvg();

    /// <summary>What was predicted for each row against what the row held, part by part, beside the line where the two agree.</summary>
    /// <param name="measures">The measures.</param>
    /// <returns>The chart, as the text of an SVG.</returns>
    public static string PredictedAgainstActual(this Measures measures) => PredictedFigure(measures).ToSvg();

    /// <summary>What was left over — each row's answer less what was predicted for it — against what was predicted, part by part.</summary>
    /// <param name="measures">The measures.</param>
    /// <returns>The chart, as the text of an SVG.</returns>
    /// <remarks>Residuals scattered evenly about nought are what a model that learned what there was to learn leaves behind.</remarks>
    public static string Residuals(this Measures measures) => ResidualFigure(measures).ToSvg();

    /// <summary>Every measure that is a number as bars: each part's, beside predicting the training rows' average.</summary>
    /// <param name="measures">The measures.</param>
    /// <returns>The chart, as the text of an SVG.</returns>
    /// <exception cref="ArgumentException">The measures hold no number: the report names the confusion matrix alone.</exception>
    public static string Bars(this Measures measures) => BarFigure(measures).ToSvg();

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
        var builder = new FigureBuilder().WithSize(Side * count, Side);

        for (var metric = 0; metric < count; metric++)
        {
            var at = metric;
            BarGroup[] groups =
            [
                new("model", [.. measures.Parts.Select(part => part.Values[at].Value)]),
                new("average", [.. measures.Parts.Select(part => part.Values[at].Baseline)]),
            ];

            builder.AddSubPlot(1, count, metric + 1, axes => axes
                .WithTitle(measures.Parts[0].Values[at].Metric.Word())
                .GroupedBar(parts, groups)
                .WithLegend());
        }

        return builder.Build();
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
}
