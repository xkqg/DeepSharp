// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// Why a model answers as it does, drawn from the results the questions give: the columns the answers lean on as bars with their spread,
/// what the answers do as a column moves as a line through a number's range or as bars over a category's values, and what one row's answer
/// is made of as the bars of what each column adds.
/// </summary>
public class ExplanationChartsTests
{
    private static readonly Reliance Leaned = new(["fare"], 7, 3, [new ColumnReliance("age", 4.5, 0.5), new ColumnReliance("sex", 2, 0.25), new ColumnReliance("ticket", 0, 0)]);

    [Fact]
    public void TheColumnsTheAnswersLeanOn_AreBarsWithTheirSpread_TheOneLeanedOnMostFirst()
    {
        var axes = Assert.Single(ExplanationCharts.LeaningFigure(Leaned).SubPlots);
        var bars = Assert.Single(axes.Series.OfType<BarSeries>());
        var spread = Assert.Single(axes.Series.OfType<ErrorBarSeries>());

        Assert.Equal(["age", "sex", "ticket"], bars.Categories);
        Assert.Equal([4.5, 2, 0], bars.Values);
        Assert.Equal([0.5, 0.25, 0], spread.YErrorLow);
        Assert.Contains("fare", axes.Title!, StringComparison.Ordinal);
        Assert.Contains("3 shuffles", axes.Title!, StringComparison.Ordinal);
        Assert.Equal(ExplanationCharts.LeaningFigure(Leaned).ToSvg(), Leaned.Leaning());
        Assert.StartsWith("<svg", Leaned.Leaning(), StringComparison.Ordinal);
    }

    [Fact]
    public void ANumbersCurve_IsALineThroughItsRange_OneToEachAnswer()
    {
        var curve = new Dependence("age", ["1", "2", "3"], [1, 2, 3], [[10, 1], [20, 2], [25, 1]], ["fare", "class"]);
        var axes = Assert.Single(ExplanationCharts.CurveFigure(curve).SubPlots);
        var lines = axes.Series.OfType<LineSeries>().ToArray();

        Assert.Equal(["fare", "class"], lines.Select(line => line.Label));
        Assert.Equal([1.0, 2, 3], lines[0].XData);
        Assert.Equal([10.0, 20, 25], lines[0].YData);
        Assert.Equal("age", axes.XAxis.Label);
        Assert.Equal(ExplanationCharts.CurveFigure(curve).ToSvg(), curve.Curve());
    }

    [Fact]
    public void ACategorysCurve_IsBarsOverItsValues_OneGroupToEachAnswer()
    {
        var curve = new Dependence("sex", ["female", "male"], null, [[30, 0.7], [12, 0.2]], ["fare", "survived"]);
        var axes = Assert.Single(ExplanationCharts.CurveFigure(curve).SubPlots);
        var bars = axes.Series.OfType<BarSeries>().ToArray();

        Assert.Equal(["fare", "survived"], bars.Select(bar => bar.Label));
        Assert.All(bars, bar => Assert.Equal(["female", "male"], bar.Categories));
        Assert.Equal([30.0, 12], bars[0].Values);
        Assert.StartsWith("<svg", curve.Curve(), StringComparison.Ordinal);
    }

    [Fact]
    public void ARowsMakeUp_IsTheBarsOfWhatEachColumnAdds_ToEachAnswer()
    {
        var makeUp = new RowMakeUp(
            3,
            50,
            ["fare", "class"],
            [32, 1],
            [20, 2],
            [new ColumnContribution("age", [9, -1]), new ColumnContribution("sex", [4, 0.5]), new ColumnContribution("ticket", [-1, 0])]);
        var figure = ExplanationCharts.MakeUpFigure(makeUp);

        Assert.Equal(2, figure.SubPlots.Count);

        var first = Assert.Single(figure.SubPlots[0].Series.OfType<BarSeries>());

        Assert.Equal(["age", "sex", "ticket"], first.Categories);
        Assert.Equal([9.0, 4, -1], first.Values);
        Assert.Contains("20", figure.SubPlots[0].Title!, StringComparison.Ordinal);
        Assert.Contains("32", figure.SubPlots[0].Title!, StringComparison.Ordinal);
        Assert.Equal([-1.0, 0.5, 0], Assert.Single(figure.SubPlots[1].Series.OfType<BarSeries>()).Values);
        Assert.Equal(figure.ToSvg(), makeUp.Parts());
    }

    [Fact]
    public void NothingToDraw_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => ExplanationCharts.LeaningFigure(null!));
        Assert.Throws<ArgumentNullException>(() => ExplanationCharts.CurveFigure(null!));
        Assert.Throws<ArgumentNullException>(() => ExplanationCharts.MakeUpFigure(null!));
        Assert.Throws<ArgumentException>(() => new Reliance(["fare"], 1, 1, []).Leaning());
    }
}
