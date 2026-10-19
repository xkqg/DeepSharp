// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// What a run measured on each fold, drawn as it was laid beside itself: a panel to each measure, the parts along it, each part's
/// mean over the folds with its spread as a bar about it, beside the mean of the folds' baselines — so a number that moves by more
/// than the model beats the baseline by is seen as a bar that reaches across it.
/// </summary>
public class MeasuresAcrossChartsTests
{
    private const int Folds = 3;

    private static Measures Measured(int index)
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.80, new Fold(Folds, index), 42, 99)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Rmse).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        var chance = 0.2 + (0.2 * index);

        return prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { chance })]);
        })]);
    }

    private static MeasuresAcross Across() => MeasuresAcross.Of([.. Enumerable.Range(0, Folds).Select(Measured)]);

    [Fact]
    public void EveryMeasure_IsAPanel_WithEachPartsMeanAndSpreadBesideTheBaseline()
    {
        var across = Across();
        var figure = MeasureCharts.AcrossFoldsFigure(across);

        Assert.Equal(2, figure.SubPlots.Count);

        for (var metric = 0; metric < 2; metric++)
        {
            var axes = figure.SubPlots[metric];
            var bars = Assert.Single(axes.Series.OfType<ErrorBarSeries>());
            var baseline = Assert.Single(axes.Series.OfType<ScatterSeries>());
            var measured = across.Parts.Select(part => part.Values[metric]).ToArray();

            Assert.Equal(measured.Select(value => value.Mean), bars.YData);
            Assert.Equal(measured.Select(value => value.Spread), bars.YErrorLow);
            Assert.Equal(measured.Select(value => value.Spread), bars.YErrorHigh);
            Assert.Equal(measured.Select(value => value.Baseline), baseline.YData);
            Assert.Equal([0.0, 1.0], bars.XData);
            Assert.Equal([0.0, 1.0], baseline.XData);
            Assert.StartsWith(across.Parts[0].Values[metric].Metric.Word(), axes.Title!, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TheChart_SaysHowManyFoldsItIsTheMeanOf_AndIsTheSvgOfItsFigure()
    {
        var across = Across();

        Assert.Contains($"{Folds} folds", MeasureCharts.AcrossFoldsFigure(across).SubPlots[0].Title, StringComparison.Ordinal);
        Assert.Equal(MeasureCharts.AcrossFoldsFigure(across).ToSvg(), across.AcrossFolds());
        Assert.StartsWith("<svg", across.AcrossFolds(), StringComparison.Ordinal);
    }

    [Fact]
    public void AMeasureWithoutEnd_StandsAtNothing_InsteadOfBreakingTheDrawing()
    {
        // A prediction of nought in a band the answer holds is a divergence without end on every fold: its mean is infinite and its
        // spread no number, and neither has a height to be drawn at.
        var fold = Divergent();
        var across = MeasuresAcross.Of([fold, fold]);
        var axes = Assert.Single(MeasureCharts.AcrossFoldsFigure(across).SubPlots);
        var bars = Assert.Single(axes.Series.OfType<ErrorBarSeries>());

        Assert.Equal([0.0, 0.0, 0.0], bars.YData);
        Assert.Equal([0.0, 0.0, 0.0], bars.YErrorLow);
        Assert.DoesNotContain("Infinity", across.AcrossFolds(), StringComparison.Ordinal);
        Assert.DoesNotContain("NaN", across.AcrossFolds(), StringComparison.Ordinal);
    }

    [Fact]
    public void NothingToDraw_IsRefused_ANullAndAReportThatNamesTheConfusionMatrixAlone()
    {
        Measures Matrices(int index)
        {
            var prepared = Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "pclass"))
                .SplitAtRandom(0.80, new Fold(2, index), 42, 99)
                .Target("survived")
                .Report(report => report.Measure(Metric.ConfusionMatrix).On(Part.Test).As(Shown.Drawn))
                .Build()
                .Run();

            return prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
            {
                var batch = prepared.Batch(part);

                return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { 0.3 })]);
            })]);
        }

        Assert.Throws<ArgumentNullException>(() => MeasureCharts.AcrossFoldsFigure(null!));
        Assert.Throws<ArgumentException>(() => MeasuresAcross.Of([Matrices(0), Matrices(1)]).AcrossFolds());
    }

    private static Measures Divergent()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,c1,c2,c3\n" + string.Join('\n', Enumerable.Range(1, 12).Select(t => $"{t},{1 + (t % 3)},{2 + (t % 4)},{1 + (t % 2)}")) + "\n"), "twelve flocks")
            .Declare(schema => schema.Integer("t").Number("c1", "c2", "c3"))
            .SplitByTime("t", 0.50, 0.25)
            .NormaliseRow(Norm.L1, "c1", "c2", "c3")
            .Distribution(["c1", "c2", "c3"], scaleBy: null, ordered: true)
            .Report(report => report.Measure(Metric.Kl).On(Part.Train, Part.Validation, Part.Test).As(Shown.Drawn))
            .Build()
            .Run();

        return prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(_ => new[] { 0.5, 0.5, 0 })]);
        })]);
    }
}
