// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// The windows a served model met, drawn one beside another as they were measured: a line to each watched column, its distance
/// from the training rows window by window, under the bound the watch flags above — so a column that drifts away is seen as a
/// line climbing through it. The windows stand at their first moment when every one of them has one, and are numbered from one in
/// the order they were handed in when not.
/// </summary>
public class WindowChartsTests
{
    private static WindowLikeness Window(DateTime? from, double weight, double farm, double warnAbove = 0.2) =>
        new(warnAbove, 10, [Column("weight", Distance.KolmogorovSmirnov, weight, warnAbove), Column("farm", Distance.TotalVariation, farm, warnAbove)])
        {
            From = from,
            To = from?.AddDays(1),
        };

    private static WindowColumn Column(string name, Distance distance, double apart, double warnAbove) => new(name, distance, apart, apart > warnAbove);

    [Fact]
    public void EveryColumn_IsALineOfItsDistanceWindowByWindow_UnderTheBound()
    {
        IReadOnlyList<WindowLikeness> windows = [Window(new DateTime(2025, 2, 1), 0.05, 0.1), Window(new DateTime(2025, 2, 2), 0.1, 0.15), Window(new DateTime(2025, 2, 3), 0.4, 0.1)];
        var axes = Assert.Single(WindowCharts.DistancesFigure(windows).SubPlots);
        var lines = axes.Series.OfType<LineSeries>().ToArray();

        Assert.Equal(["weight", "farm"], lines.Select(line => line.Label));
        Assert.Equal([0.05, 0.1, 0.4], lines[0].YData);
        Assert.Equal([0.1, 0.15, 0.1], lines[1].YData);
        Assert.Equal(windows.Select(window => window.From!.Value.ToOADate()), lines[0].XData);
        Assert.Equal(0.2, Assert.Single(axes.ReferenceLines).Value);
        Assert.Equal("distance from the training rows", axes.YAxis.Label);
        Assert.Equal((0.0, 1.0), (axes.YAxis.Min!.Value, axes.YAxis.Max!.Value));
        Assert.Equal(WindowCharts.DistancesFigure(windows).ToSvg(), windows.Distances());
        Assert.StartsWith("<svg", windows.Distances(), StringComparison.Ordinal);
    }

    [Fact]
    public void WindowsWithoutAMoment_AreNumberedFromOne_InTheOrderTheyWereHandedIn()
    {
        IReadOnlyList<WindowLikeness> windows = [Window(null, 0.05, 0.1), Window(new DateTime(2025, 2, 2), 0.3, 0.15)];
        var axes = Assert.Single(WindowCharts.DistancesFigure(windows).SubPlots);

        Assert.All(axes.Series.OfType<LineSeries>(), line => Assert.Equal([1.0, 2], line.XData));
        Assert.Equal("window", axes.XAxis.Label);
    }

    [Fact]
    public void AWindowWithNoDistance_LeavesItsColumnsLineOpenThere_AndIsStillDrawn()
    {
        IReadOnlyList<WindowLikeness> windows = [Window(null, 0.05, 0.1), Window(null, double.NaN, 0.15), Window(null, 0.1, 0.1)];
        var weight = WindowCharts.DistancesFigure(windows).SubPlots[0].Series.OfType<LineSeries>().First();

        Assert.True(double.IsNaN(weight.YData[1]));
        Assert.StartsWith("<svg", windows.Distances(), StringComparison.Ordinal);
    }

    [Fact]
    public void NoWindow_HasNothingToDraw_AndWindowsOfAnotherWatch_AreNotDrawnAsOne()
    {
        IReadOnlyList<WindowLikeness> none = [];
        IReadOnlyList<WindowLikeness> bounds = [Window(null, 0.1, 0.1), Window(null, 0.1, 0.1, warnAbove: 0.3)];
        IReadOnlyList<WindowLikeness> columns = [Window(null, 0.1, 0.1), new WindowLikeness(0.2, 10, [Column("weight", Distance.KolmogorovSmirnov, 0.1, 0.2)])];

        Assert.Throws<ArgumentException>(() => none.Distances());
        Assert.Contains("one watch", Assert.Throws<ArgumentException>(() => bounds.Distances()).Message, StringComparison.Ordinal);
        Assert.Contains("one watch", Assert.Throws<ArgumentException>(() => columns.Distances()).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => ((IReadOnlyList<WindowLikeness>)null!).Distances());
    }
}
