// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// A correlation is drawn as a heatmap of its coefficients on the whole of their scale, minus one to one, so a table of weak
/// correlations is never drawn as if one of them were perfect: nought is the middle of the colours, always.
/// </summary>
public class CorrelationChartTests
{
    [Fact]
    public void ACorrelation_IsAHeatmapOfItsCoefficients_EachLabelled_OnTheWholeOfTheirScale()
    {
        var correlation = Correlated(Repository.Data("titanic.csv"));
        var heatmap = Assert.Single(Assert.Single(CorrelationChart.HeatmapFigure(correlation).SubPlots).Series.OfType<HeatmapSeries>());

        Assert.Equal(3, heatmap.Data.GetLength(0));
        Assert.All(Enumerable.Range(0, 3), at => Assert.Equal(1, heatmap.Data[at, at], 12));
        Assert.Equal(heatmap.Data[0, 1], heatmap.Data[1, 0], 12);
        Assert.True(heatmap.ShowLabels);
        Assert.Equal(0, heatmap.Normalizer!.Normalize(-1, -0.2, 0.3));
        Assert.Equal(0.5, heatmap.Normalizer.Normalize(0, -0.2, 0.3));
        Assert.Equal(1, heatmap.Normalizer.Normalize(2, -0.2, 0.3));
        Assert.Equal(CorrelationChart.HeatmapFigure(correlation).ToSvg(), correlation.Heatmap());
    }

    [Fact]
    public void ACorrelationOfFewerThanTwoRows_CorrelatesNothing()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", ""], ["", "2"], ["3", "4"]]), "three rows")
            .Declare(schema => schema.Optional("a", ColumnKind.Number).Optional("b", ColumnKind.Number))
            .Correlation(["a", "b"])
            .Build()
            .Run();

        Assert.Throws<ArgumentException>(() => prepared.Evidence.Values.OfType<CorrelationInput>().Single().Heatmap());
        Assert.Throws<ArgumentNullException>(() => ((CorrelationInput)null!).Heatmap());
    }

    private static CorrelationInput Correlated(string path) =>
        Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Integer("survived", "sibsp").Optional("age", ColumnKind.Number).Number("fare"))
            .Correlation(["age", "fare", "sibsp"])
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run()
            .Evidence.Values.OfType<CorrelationInput>().Single();
}
