// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The measures 0.11.0 adds — F1, the area under the curve a threshold sweeps out, the error as a share of the answer, and the
/// Brier score — each scikit-learn's, worked out by it on the same twenty rows divided the same way (Fixtures/new-measures.py,
/// scikit-learn 1.9.1), beside the same measure of predicting the training rows' average answer; and refused where scikit-learn
/// has no number or one that says nothing.
/// </summary>
public class NewMeasureTests
{
    private const double Close = 1e-12;

    private static readonly Part[] ThreeParts = [Part.Train, Part.Validation, Part.Test];

    private static double[] Likely(int t) => [(t * 5 % 7) / 10.0, (t * 3 % 5) / 10.0, (t * 2 % 9) / 10.0];

    private static double[] Shares(int t)
    {
        double[] row = [.. Likely(t).Select(value => value + 0.1)];

        return [.. row.Select(value => value / row.Sum())];
    }

    [Fact]
    public void OneAnswer_IsMeasuredByF1AucAndBrier_AsScikitLearnMeasuresIt()
    {
        var measures = Measured(Rows(t => [t % 2], fitting => fitting.Target("a"), Metric.F1, Metric.Auc, Metric.Brier), t => [Likely(t)[0]]);

        AssertEach(measures, Metric.F1, [0.25, 0.8, 0.0], [0.6666666666666666, 0.75, 0.5714285714285714]);
        AssertEach(measures, Metric.Auc, [0.38, 0.8333333333333333, 0.33333333333333337], [0.5, 0.5, 0.5]);
        AssertEach(measures, Metric.Brier, [0.366, 0.24200000000000005, 0.332], [0.25, 0.25, 0.25]);
    }

    [Fact]
    public void LabelsOfWhichARowHoldsOne_AreMeasuredLabelByLabelAndAveraged_AndTheirBrierAcrossTheLabels()
    {
        var measures = Measured(
            Rows(t => [t * 7 % 3 == 0 ? 1 : 0, t * 7 % 3 == 1 ? 1 : 0, t * 7 % 3 == 2 ? 1 : 0], fitting => fitting.Labels(["a", "b", "c"], 1), Metric.F1, Metric.Auc, Metric.Brier),
            Shares);

        AssertEach(measures, Metric.F1, [0.25925925925925924, 0.16666666666666666, 0.48888888888888893], [0.19047619047619047, 0.1111111111111111, 0.19047619047619047]);
        AssertEach(measures, Metric.Auc, [0.3511904761904761, 0.5277777777777778, 0.888888888888889], [0.5, 0.5, 0.5]);
        AssertEach(measures, Metric.Brier, [0.8179696528077487, 0.6578508421882772, 0.4879891571533263], [0.6599999999999999, 0.7, 0.6599999999999999]);
    }

    [Fact]
    public void LabelsOfWhichARowHoldsAnyNumber_HaveAnF1ForEach_Averaged()
    {
        var measures = Measured(Rows(t => [t % 2, t % 3 == 0 ? 1 : 0, t % 5 == 0 ? 1 : 0], fitting => fitting.Labels(["a", "b", "c"], 0), Metric.F1), Likely);

        AssertEach(measures, Metric.F1, [0.08333333333333333, 0.26666666666666666, 0.0], [0.2222222222222222, 0.25, 0.19047619047619047]);
    }

    [Fact]
    public void AnAmount_IsMeasuredByItsErrorAsAShareOfIt()
    {
        var measures = Measured(Rows(t => [2 + (t * 3 % 7)], fitting => fitting.Target("a"), Metric.Mape), t => [2.5 + (t % 4)]);

        AssertEach(measures, Metric.Mape, [0.5109523809523809, 0.42952380952380953, 0.5172619047619048], [0.4103809523809524, 0.5527619047619048, 0.3547619047619047]);
    }

    [Fact]
    public void AnAnswerOfNought_IsRefusedAShareOfAnError_NamingTheRow()
    {
        var prepared = Rows(t => [t == 12 ? 0 : t], fitting => fitting.Target("a"), Metric.Mape);

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(prepared, t => [t]));

        Assert.StartsWith("Row 12: 'a' is nought, and mape measures each error as a share of its answer", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APartOfOneClass_HasNoAreaUnderTheCurve_AndSaysWhichPartAndWhichAnswer()
    {
        // The rows chosen on, t from 11 to 15, are all at least eleven.
        var prepared = Rows(t => [t >= 11 ? 1 : t % 2], fitting => fitting.Target("a"), Metric.Auc);

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(prepared, t => [Likely(t)[0]]));

        Assert.StartsWith("In 'validation', every row holds 'a' as one, and auc compares", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LabelsOfWhichARowHoldsAnyNumber_HaveNoBrierScore()
    {
        var prepared = Rows(t => [t % 2, t % 3 == 0 ? 1 : 0, t % 5 == 0 ? 1 : 0], fitting => fitting.Labels(["a", "b", "c"], 0), Metric.Brier);

        Assert.Contains("scikit-learn gives such chances no Brier score", Assert.Throws<InvalidOperationException>(() => Measured(prepared, Likely)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheNewMeasures_AreWordsOfTheFile_InTheirFamilies()
    {
        Assert.Equal(MetricFamily.Amounts, Metric.Mape.Family());
        Assert.All(new[] { Metric.F1, Metric.Auc, Metric.Brier }, metric => Assert.Equal(MetricFamily.Classes, metric.Family()));
        Assert.Equal(["f1", "auc", "mape", "brier"], new[] { Metric.F1, Metric.Auc, Metric.Mape, Metric.Brier }.Select(metric => metric.Word()));
    }

    [Fact]
    public void AMeasureOfClasses_IsRefusedWhereTheAnswersAreAmounts_AsEveryMeasureOfClassesIs()
    {
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .Read(new InMemoryRowSource(["t", "a"], [.. Enumerable.Range(1, 20).Select(t => (IReadOnlyList<string?>)[Written(t), Written(t)])]), "rows")
            .Declare(schema => schema.Integer("t").Number("a"))
            .OrderBy("t")
            .SplitByTime("t", 0.50, 0.25, 1)
            .Ahead("a", 1, AheadAs.Return)
            .Report(report => report.Measure(Metric.Auc).On(ThreeParts).As(Shown.Numbers))
            .Build());

        Assert.Contains("counts classes with auc", refused.Message, StringComparison.Ordinal);
    }

    private static PreparedData Rows(Func<int, double[]> answers, Func<FittingBuilder, FittingBuilder> output, params Metric[] metrics)
    {
        var width = answers(1).Length;
        string[] names = width == 1 ? ["a"] : ["a", "b", "c"];

        return output(Pdd.Create()
                .Read(new InMemoryRowSource(["t", .. names], [.. Enumerable.Range(1, 20).Select(t => (IReadOnlyList<string?>)[Written(t), .. answers(t).Select(Written)])]), "twenty rows")
                .Declare(schema => schema.Integer("t").Number(names))
                .SplitByTime("t", 0.50, 0.25))
            .Report(report => report.Measure(metrics).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();
    }

    private static Measures Measured(PreparedData prepared, Func<int, double[]> model) =>
        prepared.Measure([.. ThreeParts.Select(part =>
        {
            var batch = prepared.Batch(part);
            var t = batch.FeatureNames.ToList().IndexOf("t");

            return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(row => model((int)batch.Features[row][t]))]);
        })]);

    private static void AssertEach(Measures measures, Metric metric, double[] model, double[] average)
    {
        for (var at = 0; at < 3; at++)
        {
            var measured = measures.Parts[at].Values.Single(each => each.Metric == metric);

            Assert.Equal(model[at], measured.Value, Close);
            Assert.Equal(average[at], measured.Baseline, Close);
        }
    }

    private static string Written(double value) => value.ToString(CultureInfo.InvariantCulture);
}
