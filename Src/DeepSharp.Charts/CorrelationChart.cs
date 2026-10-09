// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models;
using MatPlotLibNet.Rendering.TickFormatters;
using MatPlotLibNet.Rendering.TickLocators;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Charts;

/// <summary>A correlation between columns, drawn.</summary>
public static class CorrelationChart
{
    extension(CorrelationInput correlation)
    {
        /// <summary>
        /// The correlation of the columns as a heatmap: every coefficient on the whole of its scale, from minus one to one, blue to
        /// red, each cell labelled with its value. The coefficient drawn is the one the correlation was declared with.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">Fewer than two rows were kept, which correlate nothing.</exception>
        /// <remarks>
        /// The coefficients are the pipeline's, worked out once from the rows it kept
        /// (<see cref="CorrelationInput.Correlate"/>); the drawing is MatPlotLibNet's. A pair with no coefficient — a column that
        /// never changes — is a cell with no colour, not one coloured as nought.
        /// </remarks>
        public string Heatmap() => HeatmapFigure(correlation).ToSvg();
    }

    extension(CorrelationMatrix matrix)
    {
        /// <summary>
        /// One kind of coefficient of a correlation as a heatmap: every coefficient on the whole of its scale, from minus one to
        /// one, blue to red, each cell labelled with its value.
        /// </summary>
        /// <param name="coefficient">Which coefficient to draw; Pearson's, which says nothing more than the picture, unless asked.</param>
        /// <returns>The chart, as the text of an SVG; Spearman's says so in a title.</returns>
        /// <exception cref="ArgumentException">Fewer than two rows were kept, which correlate nothing.</exception>
        /// <exception cref="ArgumentOutOfRangeException">The coefficient is none of those there are.</exception>
        /// <remarks>
        /// For a correlation worked out by hand or kept beside others, to draw it without the pipeline that made it. A pair with
        /// no coefficient is a cell with no colour, not one coloured as nought.
        /// </remarks>
        public string Heatmap(Coefficient coefficient = Coefficient.Pearson) => HeatmapFigure(matrix, coefficient).ToSvg();
    }

    /// <summary>The figure <see cref="Heatmap(CorrelationInput)"/> draws.</summary>
    internal static Figure HeatmapFigure(CorrelationInput correlation)
    {
        ArgumentNullException.ThrowIfNull(correlation);

        return HeatmapFigure(correlation.Correlate(), correlation.Coefficient);
    }

    /// <summary>The figure <see cref="Heatmap(CorrelationMatrix, Coefficient)"/> draws.</summary>
    internal static Figure HeatmapFigure(CorrelationMatrix matrix, Coefficient coefficient)
    {
        ArgumentNullException.ThrowIfNull(matrix);

        var coefficients = matrix.Of(coefficient);

        if (matrix.Kept < 2)
        {
            throw new ArgumentException("A correlation is drawn from two rows at least, and fewer were kept.", nameof(matrix));
        }

        var names = matrix.Columns.ToArray();
        var data = new double[names.Length, names.Length];
        var positions = Enumerable.Range(0, names.Length).Select(position => (double)position).ToArray();

        for (var row = 0; row < names.Length; row++)
        {
            for (var column = 0; column < names.Length; column++)
            {
                data[row, column] = coefficients[row, column];
            }
        }

        var size = 160 + (60 * names.Length);
        var figure = new FigureBuilder().WithSize(size + 120, size);

        // Pearson's picture is the one this has always drawn and stays as it was; the other says what it is.
        if (coefficient != Coefficient.Pearson)
        {
            figure = figure.WithTitle($"{coefficient} rank correlation");
        }

        return figure
            .AddSubPlot(1, 1, 1, axes => axes
                .Heatmap(data, series =>
                {
                    series.ColorMap = new BlankWhereUndefined(ColorMaps.Coolwarm);
                    series.Normalizer = new CenteredNormNormalizer(0, 1);
                    series.ShowLabels = true;
                    series.LabelFormat = "0.00";
                })
                .SetXTickLocator(new FixedLocator(positions))
                .SetXTickFormatter(new CategoryFormatter(names))
                .SetYTickLocator(new FixedLocator(positions))
                .SetYTickFormatter(new CategoryFormatter(names, reversed: true))
                .WithColorBar())
            .Build();
    }
}
