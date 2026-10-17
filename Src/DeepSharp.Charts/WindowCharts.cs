// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models;
using MatPlotLibNet.Models.Series;
using MatPlotLibNet.Styling;

namespace DeepSharp.Charts;

/// <summary>
/// The windows a served model met, drawn from what measuring them gave — <see cref="Watching.Watched(PreparedData, IRowSource)"/>
/// window by window — so nobody assembles an array to see a column drift away from the rows the model learned from.
/// </summary>
public static class WindowCharts
{
    extension(IReadOnlyList<WindowLikeness> windows)
    {
        /// <summary>
        /// Every watched column's distance from the training rows, window by window, as a line, under the bound the watch flags
        /// above: a column whose line climbs through the bound is one the served rows have moved away on.
        /// </summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">There is no window, or the windows were not all measured by one watch.</exception>
        /// <remarks>
        /// The windows stand at their first moment when every one of them has one, and are numbered from one in the order they
        /// were handed in when not. A window with no distance for a column — none of its rows held a value there — leaves that
        /// column's line open at it, as a distance nobody could work out is not nought.
        /// </remarks>
        public string Distances() => DistancesFigure(windows).ToSvg();
    }

    /// <summary>The figure <see cref="Distances"/> draws.</summary>
    internal static Figure DistancesFigure(IReadOnlyList<WindowLikeness> windows)
    {
        ArgumentNullException.ThrowIfNull(windows);

        if (windows.Count == 0)
        {
            throw new ArgumentException("A list of no windows has nothing to draw.", nameof(windows));
        }

        var first = windows[0];
        string[] columns = [.. first.Columns.Select(column => column.Column)];

        if (windows.Any(window => !window.WarnAbove.Equals(first.WarnAbove) || !window.Columns.Select(column => column.Column).SequenceEqual(columns)))
        {
            throw new ArgumentException("The windows are drawn against one watch, and these were measured over other columns or under another bound.", nameof(windows));
        }

        DateTime[]? moments = windows.All(window => window.From is not null) ? [.. windows.Select(window => window.From!.Value)] : null;
        double[] numbers = [.. Enumerable.Range(1, windows.Count).Select(window => (double)window)];

        return new FigureBuilder()
            .WithSize(640, 400)
            .AddSubPlot(1, 1, 1, axes =>
            {
                axes.WithTitle("served rows against the training rows").SetXLabel(moments is null ? "window" : "window from").SetYLabel("distance from the training rows").SetYLim(0, 1);

                for (var at = 0; at < columns.Length; at++)
                {
                    var column = at;
                    double[] apart = [.. windows.Select(window => window.Columns[column].Apart)];
                    void Drawn(LineSeries line)
                    {
                        line.Label = columns[column];
                        line.Marker = MarkerStyle.Circle;
                    }

                    if (moments is null)
                    {
                        axes.Plot(numbers, apart, Drawn);
                    }
                    else
                    {
                        axes.Plot(moments, apart, Drawn);
                    }
                }

                axes.AxHLine(first.WarnAbove, line => line.Label = "flagged above").WithLegend();
            })
            .Build();
    }
}
