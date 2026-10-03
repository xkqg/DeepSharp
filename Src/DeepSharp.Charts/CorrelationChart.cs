// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models;
using MatPlotLibNet.Numerics;
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
        /// red, each cell labelled with its value.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">Fewer than two rows were kept, which correlate nothing.</exception>
        /// <remarks>The coefficients are MatPlotLibNet's, as is the drawing.</remarks>
        public string Heatmap() => HeatmapFigure(correlation).ToSvg();
    }

    /// <summary>The figure <see cref="Heatmap"/> draws.</summary>
    internal static Figure HeatmapFigure(CorrelationInput correlation)
    {
        ArgumentNullException.ThrowIfNull(correlation);

        if (correlation.Kept < 2)
        {
            throw new ArgumentException("A correlation is drawn from two rows at least, and fewer were kept.", nameof(correlation));
        }

        var names = correlation.Columns.ToArray();
        var matrix = NpStats.Corrcoef([.. Enumerable.Range(0, names.Length).Select(column => correlation.Rows.Select(row => row[column]).ToArray())]);
        var data = new double[names.Length, names.Length];
        var positions = Enumerable.Range(0, names.Length).Select(position => (double)position).ToArray();

        for (var row = 0; row < names.Length; row++)
        {
            for (var column = 0; column < names.Length; column++)
            {
                data[row, column] = matrix[row, column];
            }
        }

        var size = 160 + (60 * names.Length);

        return new FigureBuilder()
            .WithSize(size + 120, size)
            .AddSubPlot(1, 1, 1, axes => axes
                .Heatmap(data, series =>
                {
                    series.ColorMap = ColorMaps.Coolwarm;
                    series.Normalizer = FromMinusOneToOne.Instance;
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
