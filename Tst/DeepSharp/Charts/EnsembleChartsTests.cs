// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// An ensemble measured beside each of its members, drawn as the measures of one model are: a panel to each measure, the parts along
/// it, the ensemble's bar beside every member's and beside the training rows' average — so what combining bought is seen against
/// what each member did alone.
/// </summary>
public class EnsembleChartsTests
{
    private static readonly Part[] ThreeParts = [Part.Train, Part.Validation, Part.Test];

    private static EnsembleMeasures Measured(params double[] chances)
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Rmse).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();

        return prepared.MeasureEnsemble([.. chances.Select(chance => ThreeParts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { chance })]);
        }).ToArray())]);
    }

    [Fact]
    public void EveryMeasure_IsAPanel_TheEnsemblesBarBesideEachMembersAndTheAverages()
    {
        var measured = Measured(0.2, 0.6, 0.9);
        var figure = MeasureCharts.EnsembleBarFigure(measured);

        Assert.Equal(["accuracy", "rmse"], figure.SubPlots.Select(axes => axes.Title));

        for (var metric = 0; metric < 2; metric++)
        {
            var bars = figure.SubPlots[metric].Series.OfType<BarSeries>().ToArray();

            Assert.Equal(["ensemble", "member 1", "member 2", "member 3", "average"], bars.Select(bar => bar.Label));
            Assert.All(bars, bar => Assert.Equal(["train", "validation", "test"], bar.Categories));
            Assert.Equal(measured.Ensemble.Parts.Select(part => part.Values[metric].Value), bars[0].Values);

            for (var member = 0; member < 3; member++)
            {
                Assert.Equal(measured.Members[member].Parts.Select(part => part.Values[metric].Value), bars[member + 1].Values);
            }

            Assert.Equal(measured.Ensemble.Parts.Select(part => part.Values[metric].Baseline), bars[4].Values);
        }

        Assert.Equal(figure.ToSvg(), measured.BarsWithMembers());
        Assert.StartsWith("<svg", measured.BarsWithMembers(), StringComparison.Ordinal);
    }

    [Fact]
    public void AMeasureWithoutEnd_StandsAtNothing_ThePanelSayingWhoseAndWhere()
    {
        // A prediction of nought in a band the answer holds is a divergence without end, on the ensemble as on each member.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,c1,c2,c3\n" + string.Join('\n', Enumerable.Range(1, 12).Select(t => $"{t},{1 + (t % 3)},{2 + (t % 4)},{1 + (t % 2)}")) + "\n"), "twelve flocks")
            .Declare(schema => schema.Integer("t").Number("c1", "c2", "c3"))
            .SplitByTime("t", 0.50, 0.25)
            .NormaliseRow(Norm.L1, "c1", "c2", "c3")
            .Distribution(["c1", "c2", "c3"], scaleBy: null, ordered: true)
            .Report(report => report.Measure(Metric.Kl).On(ThreeParts).As(Shown.Drawn))
            .Build()
            .Run();
        IReadOnlyList<PartPredictions> Said() =>
            [.. ThreeParts.Select(part =>
            {
                var batch = prepared.Batch(part);

                return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(_ => new[] { 0.5, 0.5, 0 })]);
            })];

        var measured = prepared.MeasureEnsemble([Said(), Said()]);
        var figure = MeasureCharts.EnsembleBarFigure(measured);

        Assert.Equal("kl: the ensemble's is infinite on train, validation and test", Assert.Single(figure.SubPlots).Title);
        Assert.DoesNotContain("Infinity", measured.BarsWithMembers(), StringComparison.Ordinal);
        Assert.Equal([0.0, 0.0, 0.0], Assert.Single(figure.SubPlots).Series.OfType<BarSeries>().First().Values);
    }

    [Fact]
    public void NoEnsemble_HasNothingToDraw_NorOneMeasuredByTheConfusionMatrixAlone()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Target("survived")
            .Report(report => report.Measure(Metric.ConfusionMatrix).On(ThreeParts).As(Shown.Drawn))
            .Build()
            .Run();
        IReadOnlyList<PartPredictions> Said() =>
            [.. ThreeParts.Select(part =>
            {
                var batch = prepared.Batch(part);

                return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { 0.3 })]);
            })];

        Assert.Throws<ArgumentNullException>(() => MeasureCharts.EnsembleBarFigure(null!));
        Assert.Throws<ArgumentException>(() => prepared.MeasureEnsemble([Said(), Said()]).BarsWithMembers());
    }
}
