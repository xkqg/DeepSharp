// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Styling;
using MatPlotLibNet.Models;

namespace DeepSharp.Charts;

/// <summary>
/// Why a model answers as it does, drawn from what the questions gave: the columns the answers lean on, what they do as one column
/// moves, and what one row's answer is made of.
/// </summary>
/// <remarks>
/// Drawn from the results themselves, in the answer's own units, and nothing is asked of the model again.
/// </remarks>
public static class ExplanationCharts
{
    private const int Side = 360;

    extension(Reliance reliance)
    {
        /// <summary>Each column as a bar of how far the answers move when its values are shuffled, with its spread over the shuffles, the column leaned on most first.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The reliance holds no column.</exception>
        public string Leaning() => LeaningFigure(reliance).ToSvg();
    }

    extension(Dependence dependence)
    {
        /// <summary>What the answers do as the column moves: a line through a number's range, bars over a category's values, one to each answer.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        public string Curve() => CurveFigure(dependence).ToSvg();
    }

    extension(RowMakeUp makeUp)
    {
        /// <summary>What each column adds to the row's answer, as bars, a panel to each answer.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        public string Parts() => MakeUpFigure(makeUp).ToSvg();
    }

    /// <summary>The figure <see cref="Leaning"/> draws.</summary>
    internal static Figure LeaningFigure(Reliance reliance)
    {
        ArgumentNullException.ThrowIfNull(reliance);

        if (reliance.Columns.Count == 0)
        {
            throw new ArgumentException("There is no column to draw: the rows held none that the answers could be asked of.", nameof(reliance));
        }

        string[] columns = [.. reliance.Columns.Select(column => column.Column)];
        double[] means = [.. reliance.Columns.Select(column => column.Mean)];
        double[] spread = [.. reliance.Columns.Select(column => column.Spread)];
        double[] places = [.. Enumerable.Range(0, columns.Length).Select(place => (double)place)];
        var title = string.Create(CultureInfo.InvariantCulture, $"how far {string.Join(", ", reliance.Answers)} moves, over {reliance.Repeats} shuffles");

        return new FigureBuilder()
            .WithSize(Side * 2, Side)
            .AddSubPlot(1, 1, 1, axes => axes
                .WithTitle(title)
                .SetYLabel("mean change of the answer")
                .Bar(columns, means)
                .ErrorBar(places, means, spread, spread))
            .Build();
    }

    /// <summary>The figure <see cref="Curve"/> draws.</summary>
    internal static Figure CurveFigure(Dependence dependence)
    {
        ArgumentNullException.ThrowIfNull(dependence);

        string[] values = [.. dependence.Values];

        return new FigureBuilder()
            .WithSize(Side * 2, Side)
            .AddSubPlot(1, 1, 1, axes =>
            {
                axes.WithTitle($"the answers as {dependence.Column} moves").SetXLabel(dependence.Column).SetYLabel("mean answer");

                for (var answer = 0; answer < dependence.AnswerNames.Count; answer++)
                {
                    var which = answer;
                    double[] said = [.. dependence.Answers.Select(row => row[which])];

                    if (dependence.Points is { } points)
                    {
                        axes.Plot([.. points], said, line =>
                        {
                            line.Label = dependence.AnswerNames[which];
                            line.Marker = MarkerStyle.Circle;
                        });
                    }
                    else
                    {
                        axes.Bar(values, said, bars => bars.Label = dependence.AnswerNames[which]);
                    }
                }

                axes.WithLegend();
            })
            .Build();
    }

    /// <summary>The figure <see cref="Parts"/> draws.</summary>
    internal static Figure MakeUpFigure(RowMakeUp makeUp)
    {
        ArgumentNullException.ThrowIfNull(makeUp);

        var count = makeUp.AnswerNames.Count;
        string[] columns = [.. makeUp.Columns.Select(column => column.Column)];
        var builder = new FigureBuilder().WithSize(Side * 2, Side * count);

        for (var answer = 0; answer < count; answer++)
        {
            var which = answer;
            double[] added = [.. makeUp.Columns.Select(column => column.Contributions[which])];
            var title = string.Create(
                CultureInfo.InvariantCulture,
                $"{makeUp.AnswerNames[which]}: from {makeUp.Background[which]:0.###} to {makeUp.Answer[which]:0.###}");

            builder.AddSubPlot(count, 1, answer + 1, axes => axes.WithTitle(title).SetYLabel("what the column adds").Bar(columns, added).AxHLine(0));
        }

        return builder.Build();
    }
}
