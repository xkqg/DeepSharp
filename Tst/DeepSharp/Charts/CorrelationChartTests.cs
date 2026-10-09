// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;
using MatPlotLibNet.Numerics;
using MatPlotLibNet.Rendering.TickFormatters;
using MatPlotLibNet.Rendering.TickLocators;
using MatPlotLibNet.Styling.ColorMaps;

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

    [Fact]
    public void TheCoefficientsDrawn_AreTheOnesThePipelineWorkedOut_NoneOfTheDrawingLibrarysOwn()
    {
        // 0.8.0 drew MatPlotLibNet's own Corrcoef of the rows; the numbers are the pipeline's now, and for columns that all
        // change they are the same ones.
        var correlation = Correlated(Repository.Data("titanic.csv"));
        var worked = NpStats.Corrcoef([.. Enumerable.Range(0, 3).Select(column => correlation.Rows.Select(row => row[column]).ToArray())]);
        var drawn = Assert.Single(Assert.Single(CorrelationChart.HeatmapFigure(correlation).SubPlots).Series.OfType<HeatmapSeries>());

        for (var row = 0; row < 3; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                Assert.Equal(worked[row, column], drawn.Data[row, column], 12);
                Assert.Equal(correlation.Correlate().Pearson[row, column], drawn.Data[row, column]);
            }
        }
    }

    [Fact]
    public void TheCorrelationIsDrawnAsItWasBeforeItsScaleWasBorrowed()
    {
        // The picture 0.8.0 drew, kept here as it was written: its own normaliser and the library's coefficients. A fixed scale
        // taken from the library instead must leave every colour as it stood.
        var correlation = Correlated(Repository.Data("titanic.csv"));

        Assert.Equal(AsItWasBefore(correlation), correlation.Heatmap());
    }

    [Theory]
    [InlineData(-1.0, 0.0)]
    [InlineData(-0.3, 0.35)]
    [InlineData(0.0, 0.5)]
    [InlineData(0.5, 0.75)]
    [InlineData(1.0, 1.0)]
    [InlineData(2.0, 1.0)]
    public void NoughtIsTheMiddleOfTheColours_AndTheScaleDoesNotFollowTheCellsDrawn(double value, double place)
    {
        var correlation = Correlated(Repository.Data("titanic.csv"));
        var drawn = Assert.Single(Assert.Single(CorrelationChart.HeatmapFigure(correlation).SubPlots).Series.OfType<HeatmapSeries>());

        // Cells that only span -0.3 to 0.5 and cells that also hold both ends of the scale are coloured alike.
        Assert.Equal(place, drawn.Normalizer!.Normalize(value, -0.3, 0.5), 12);
        Assert.Equal(place, drawn.Normalizer.Normalize(value, -1, 1), 12);
        Assert.Equal(place, drawn.Normalizer.Normalize(value, 0.2, 0.4), 12);
    }

    [Fact]
    public void ACorrelationMatrix_IsDrawnAsItsInputIs()
    {
        var correlation = Correlated(Repository.Data("titanic.csv"));

        Assert.Equal(correlation.Heatmap(), correlation.Correlate().Heatmap());
        Assert.Equal(correlation.Heatmap(), correlation.Correlate().Heatmap(Coefficient.Pearson));
    }

    [Fact]
    public void TheRankedMatrix_IsDrawnFromSpearmansNumbers_AndSaysSo()
    {
        var matrix = Correlated(Repository.Data("titanic.csv")).Correlate();
        var figure = CorrelationChart.HeatmapFigure(matrix, Coefficient.Spearman);
        var drawn = Assert.Single(Assert.Single(figure.SubPlots).Series.OfType<HeatmapSeries>());

        for (var row = 0; row < 3; row++)
        {
            for (var column = 0; column < 3; column++)
            {
                Assert.Equal(matrix.Spearman[row, column], drawn.Data[row, column]);
            }
        }

        Assert.NotEqual(matrix.Pearson[0, 1], matrix.Spearman[0, 1]);
        Assert.Contains("Spearman", matrix.Heatmap(Coefficient.Spearman), StringComparison.Ordinal);
        Assert.DoesNotContain("Spearman", matrix.Heatmap(), StringComparison.Ordinal);
        Assert.NotEqual(matrix.Heatmap(), matrix.Heatmap(Coefficient.Spearman));
    }

    [Fact]
    public void AnInputDeclaredAsSpearman_IsDrawnAsSpearman()
    {
        var input = Correlated(Repository.Data("titanic.csv"), Coefficient.Spearman);

        Assert.Equal(Coefficient.Spearman, input.Coefficient);
        Assert.Equal(input.Correlate().Heatmap(Coefficient.Spearman), input.Heatmap());
        Assert.NotEqual(input.Correlate().Heatmap(), input.Heatmap());
    }

    [Fact]
    public void ACoefficientThatIsNotDefined_IsDrawnAsNothing_NotAsNought()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "k", "b"], [["1", "5", "2"], ["2", "5", "1"], ["3", "5", "4"], ["4", "5", "3"]]), "four rows")
            .Declare(schema => schema.Number("a", "k", "b"))
            .Correlation(["a", "k", "b"])
            .Build()
            .Run();
        var correlation = prepared.Evidence.Values.OfType<CorrelationInput>().Single();
        var drawn = Assert.Single(Assert.Single(CorrelationChart.HeatmapFigure(correlation).SubPlots).Series.OfType<HeatmapSeries>());
        var svg = correlation.Heatmap();

        Assert.All(Enumerable.Range(0, 3), at => Assert.True(double.IsNaN(drawn.Data[1, at])));
        Assert.All(Enumerable.Range(0, 3), at => Assert.True(double.IsNaN(drawn.Data[at, 1])));
        Assert.Equal(0.6, drawn.Data[0, 2], 12);

        // The cells of the constant column carry no colour at all: five of the nine are blank.
        Assert.Equal(5, svg.Split("fill-opacity=\"0\"").Length - 1);
    }

    [Fact]
    public void ACellWithNoNumber_HasNoColour_AndEveryOtherCellTheColourItsMapGivesIt()
    {
        // The library asks its map for the colour of any cell; on .NET 8 it fails for a number that is not one, so the map
        // answers for it, the same on every runtime.
        var colours = new BlankWhereUndefined(ColorMaps.Coolwarm);

        Assert.Equal(0, colours.GetColor(double.NaN).A);
        Assert.Equal(ColorMaps.Coolwarm.GetColor(0.25), colours.GetColor(0.25));
        Assert.Equal(ColorMaps.Coolwarm.Name, colours.Name);
        Assert.Equal(ColorMaps.Coolwarm.GetUnderColor(), colours.GetUnderColor());
        Assert.Equal(ColorMaps.Coolwarm.GetOverColor(), colours.GetOverColor());
        Assert.Equal(ColorMaps.Coolwarm.GetBadColor(), colours.GetBadColor());
    }

    [Fact]
    public void ACorrelationMatrixOfFewerThanTwoRows_CorrelatesNothing()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["a", "b"], [["1", ""], ["", "2"], ["3", "4"]]), "three rows")
            .Declare(schema => schema.Optional("a", ColumnKind.Number).Optional("b", ColumnKind.Number))
            .Correlation(["a", "b"])
            .Build()
            .Run();
        var matrix = prepared.Evidence.Values.OfType<CorrelationInput>().Single().Correlate();

        Assert.Throws<ArgumentException>(() => matrix.Heatmap());
        Assert.Throws<ArgumentNullException>(() => ((CorrelationMatrix)null!).Heatmap());
        Assert.Throws<ArgumentOutOfRangeException>(() => Correlated(Repository.Data("titanic.csv")).Correlate().Heatmap((Coefficient)5));
    }

    private static CorrelationInput Correlated(string path, Coefficient coefficient = Coefficient.Pearson) =>
        Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Integer("survived", "sibsp").Optional("age", ColumnKind.Number).Number("fare"))
            .Correlation(["age", "fare", "sibsp"], Shown.Drawn, coefficient)
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run()
            .Evidence.Values.OfType<CorrelationInput>().Single();

    // The heatmap 0.8.0 drew a correlation as, exactly as it was written: the library's coefficients and a normaliser of its own.
    private static string AsItWasBefore(CorrelationInput correlation)
    {
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
                    series.Normalizer = new WholeScale();
                    series.ShowLabels = true;
                    series.LabelFormat = "0.00";
                })
                .SetXTickLocator(new FixedLocator(positions))
                .SetXTickFormatter(new CategoryFormatter(names))
                .SetYTickLocator(new FixedLocator(positions))
                .SetYTickFormatter(new CategoryFormatter(names, reversed: true))
                .WithColorBar())
            .ToSvg();
    }

    // What 0.8.0 placed a coefficient on the colours with, from minus one to one whatever the cells span.
    private sealed class WholeScale : INormalizer
    {
        public double Normalize(double value, double min, double max) => Math.Clamp((value + 1) / 2, 0, 1);
    }
}
