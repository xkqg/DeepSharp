// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// A trained model's measures, drawn as the report declared them: a confusion matrix as a heatmap of counts, a row to each
/// class a row held; what was predicted against what was there, and what was left over, part by part; and every measure as
/// bars, each part's beside the training rows' average — all in the answer's own units, as the measures were taken.
/// </summary>
public class MeasureChartsTests
{
    private static readonly string[] Parts = ["train", "validation", "test"];

    [Fact]
    public void TheConfusionHeatmap_ShowsTheCounts_ARowToEachClassARowHeld()
    {
        var measures = Classes();
        var figure = MeasureCharts.ConfusionFigure(measures);

        Assert.Equal(3, figure.SubPlots.Count);

        for (var at = 0; at < 3; at++)
        {
            var axes = figure.SubPlots[at];
            var confusion = Assert.Single(measures.Parts[at].Confusions);
            var heatmap = Assert.Single(axes.Series.OfType<HeatmapSeries>());

            Assert.Equal($"{Parts[at]}: y", axes.Title);
            Assert.Equal("predicted", axes.XAxis.Label);
            Assert.Equal("actual", axes.YAxis.Label);
            Assert.True(heatmap.ShowLabels);

            for (var held = 0; held < 2; held++)
            {
                for (var predicted = 0; predicted < 2; predicted++)
                {
                    Assert.Equal(confusion.Counts[held][predicted], heatmap.Data[held, predicted]);
                }
            }
        }

        Assert.Equal(figure.ToSvg(), measures.ConfusionMatrices());
    }

    [Fact]
    public void WhatWasPredicted_IsDrawnAgainstWhatWasThere_PartByPart_BesideTheLineWhereTheyAgree()
    {
        var measures = Amounts();
        var figure = MeasureCharts.PredictedFigure(measures);

        Assert.Equal(3, figure.SubPlots.Count);

        for (var at = 0; at < 3; at++)
        {
            var axes = figure.SubPlots[at];
            var part = measures.Parts[at];
            var points = Assert.Single(axes.Series.OfType<ScatterSeries>());
            var agree = Assert.Single(axes.Series.OfType<LineSeries>());

            Assert.Equal(Parts[at], axes.Title);
            Assert.Equal(part.Actual.Select(row => row[0]), points.XData);
            Assert.Equal(part.Predicted.Select(row => row[0]), points.YData);
            Assert.Equal(agree.XData, agree.YData);
            Assert.Equal("actual y", axes.XAxis.Label);
            Assert.Equal("predicted y", axes.YAxis.Label);
        }

        Assert.Equal(figure.ToSvg(), measures.PredictedAgainstActual());
    }

    [Fact]
    public void WhatWasLeftOver_IsDrawnAgainstWhatWasPredicted_PartByPart()
    {
        var measures = Amounts();
        var figure = MeasureCharts.ResidualFigure(measures);

        Assert.Equal(3, figure.SubPlots.Count);

        for (var at = 0; at < 3; at++)
        {
            var axes = figure.SubPlots[at];
            var part = measures.Parts[at];
            var points = Assert.Single(axes.Series.OfType<ScatterSeries>());

            Assert.Equal(Parts[at], axes.Title);
            Assert.Equal(part.Predicted.Select(row => row[0]), points.XData);
            Assert.Equal(part.Actual.Zip(part.Predicted, (actual, predicted) => actual[0] - predicted[0]), points.YData);
            Assert.Equal("predicted y", axes.XAxis.Label);
            Assert.Equal("actual less predicted", axes.YAxis.Label);
        }

        Assert.Equal(figure.ToSvg(), measures.Residuals());
    }

    [Fact]
    public void EveryMeasure_IsDrawnAsBars_EachPartBesideTheTrainingRowsAverage()
    {
        var measures = Amounts();
        var figure = MeasureCharts.BarFigure(measures);

        Assert.Equal(["rmse", "mae", "r2"], figure.SubPlots.Select(axes => axes.Title));

        for (var metric = 0; metric < 3; metric++)
        {
            var bars = figure.SubPlots[metric].Series.OfType<BarSeries>().ToArray();

            Assert.Equal(["model", "average"], bars.Select(bar => bar.Label));
            Assert.All(bars, bar => Assert.Equal(Parts, bar.Categories));
            Assert.Equal(measures.Parts.Select(part => part.Values[metric].Value), bars[0].Values);
            Assert.Equal(measures.Parts.Select(part => part.Values[metric].Baseline), bars[1].Values);
        }

        Assert.Equal(figure.ToSvg(), measures.Bars());

        // The confusion matrix is no number, so it has no bars of its own.
        Assert.Equal(["accuracy", "rmse"], MeasureCharts.BarFigure(Classes()).SubPlots.Select(axes => axes.Title));
    }

    [Fact]
    public void AnOutputOfSeveralAnswers_HasAPanelForEachAnswerOfEachPart()
    {
        var measures = Shares();
        var figure = MeasureCharts.PredictedFigure(measures);

        Assert.Equal(
            ["train: c1", "validation: c1", "test: c1", "train: c2", "validation: c2", "test: c2"],
            figure.SubPlots.Select(axes => axes.Title));
        Assert.Equal(measures.Parts[2].Actual.Select(row => row[1]), figure.SubPlots[5].Series.OfType<ScatterSeries>().Single().XData);
        Assert.Equal(6, MeasureCharts.ResidualFigure(measures).SubPlots.Count);
    }

    [Fact]
    public void LabelsOfWhichARowHoldsOne_AreOneHeatmapAcrossThem_TitledByThePartAlone()
    {
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "a", "b"], [.. Enumerable.Range(1, 8).Select(t => (IReadOnlyList<string?>)[$"{t}", t % 2 == 0 ? "1" : "0", t % 2 == 0 ? "0" : "1"])]), "eight rows")
            .Declare(schema => schema.Integer("t", "a", "b"))
            .SplitByTime("t", 0.50)
            .Labels(["a", "b"], ones: 1)
            .Report(report => report.Measure(Metric.ConfusionMatrix).On(Part.Test).As(Shown.Drawn))
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Test);
        var measures = prepared.Measure([new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(_ => new[] { 0.3, 0.7 })])]);

        var axes = Assert.Single(MeasureCharts.ConfusionFigure(measures).SubPlots);

        Assert.Equal("test", axes.Title);
        Assert.Equal(2, Assert.Single(axes.Series.OfType<HeatmapSeries>()).Data[1, 1]);
    }

    [Fact]
    public void WhatTheMeasuresDoNotHold_IsNotDrawn()
    {
        var onlyMatrices = Measured(Twelve(), report => report.Measure(Metric.ConfusionMatrix).On(Part.Test).As(Shown.Drawn), (batch, row) => [0.3]);

        Assert.Throws<ArgumentException>(() => Amounts().ConfusionMatrices());
        Assert.Throws<ArgumentException>(() => onlyMatrices.Bars());
        Assert.Throws<ArgumentNullException>(() => ((Measures)null!).Bars());
        Assert.Throws<ArgumentNullException>(() => ((Measures)null!).ConfusionMatrices());
        Assert.Throws<ArgumentNullException>(() => ((Measures)null!).PredictedAgainstActual());
        Assert.Throws<ArgumentNullException>(() => ((Measures)null!).Residuals());
    }

    // Twelve rows t = 1…12 whose y is nought or one, divided in time: six to train on, three to choose, three to test.
    private static FittingBuilder Twelve() =>
        Pdd.Create()
            .Read(CsvRowSource.FromText("t,y\n" + string.Join('\n', "1,0,1,1,0,1,0,1,0,1,1,0".Split(',').Select((y, at) => $"{at + 1},{y}")) + "\n"), "twelve rows")
            .Declare(schema => schema.Integer("t", "y"))
            .SplitByTime("t", 0.50, 0.25)
            .Target("y");

    private static Measures Classes() =>
        Measured(
            Twelve(),
            report => report.Measure(Metric.Accuracy, Metric.ConfusionMatrix, Metric.Rmse).On(Part.Train, Part.Validation, Part.Test).As(Shown.Drawn),
            (batch, row) => [batch.Features[row][0] % 3 == 0 ? 0.8 : 0.3]);

    // Twelve rows whose y is twice x and a little, predicted as exactly twice x.
    private static Measures Amounts() =>
        Measured(
            Pdd.Create()
                .Read(CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 12).Select(t => string.Create(CultureInfo.InvariantCulture, $"{t},{t % 5},{(2 * (t % 5)) + (t % 2 == 0 ? 0.5 : -0.25)}"))) + "\n"), "twelve amounts")
                .Declare(schema => schema.Integer("t").Number("x", "y"))
                .SplitByTime("t", 0.50, 0.25)
                .Target("y"),
            report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Train, Part.Validation, Part.Test).As(Shown.Drawn),
            (batch, row) => [2 * batch.Features[row][batch.FeatureNames.ToList().IndexOf("x")]]);

    // Twelve rows of two counts, made shares of their sum and brought back as counts.
    private static Measures Shares() =>
        Measured(
            Pdd.Create()
                .Read(CsvRowSource.FromText("t,c1,c2\n" + string.Join('\n', Enumerable.Range(1, 12).Select(t => $"{t},{1 + (t % 3)},{2 + (t % 4)}")) + "\n"), "twelve pairs")
                .Declare(schema => schema.Integer("t").Number("c1", "c2"))
                .SplitByTime("t", 0.50, 0.25)
                .NormaliseRow(Norm.L1, "c1", "c2")
                .Distribution(["c1", "c2"]),
            report => report.Measure(Metric.Rmse).On(Part.Train, Part.Validation, Part.Test).As(Shown.Drawn),
            (batch, row) => [0.4, 0.6]);

    private static Measures Measured(FittingBuilder builder, Action<ReportBuilder> report, Func<Batch, int, double[]> model)
    {
        var prepared = builder.Report(report).Build().Run();

        return prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(row => model(batch, row))]);
        })]);
    }
}
