// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A distribution is measured as a distribution: each row of answers and of predictions, in the answer's own units, divided
/// by its own total, and the two compared as shares. The earth mover's distance and the ranked probability score follow the
/// order of the bands, so they take an output that says it has one; the Kullback–Leibler divergence takes any shares of a
/// whole. Every number is the one scipy and xskillscore give for the same rows (Fixtures/ordered-measures.py, its output
/// beside it).
/// </summary>
public class OrderedMeasureTests
{
    private const double Close = 1e-12;

    private static readonly double[][] Held = [[2, 5, 3, 0], [1, 1, 1, 1], [0, 0, 4, 4], [3, 1, 0, 0]];

    private static readonly double[][] Said = [[1.5, 4, 3.5, 1], [0.5, 2, 1, 0.5], [0, 1, 3, 4], [2.5, 1.2, 0.3, 0]];

    [Fact]
    public void TheThreeMeasuresOfShares_AreScipysAndXskillscoresOnTheSameRows()
    {
        var rows = new Answered(Held, Said);

        Assert.Equal(0.24999999999999997, rows.Of(Metric.Emd, 0), Close);
        Assert.Equal(0.13278789899053445, rows.Of(Metric.Kl, 0), Close);
        Assert.Equal(0.02968749999999999, rows.Of(Metric.Rps, 0), Close);
    }

    [Fact]
    public void APredictionOfNoughtWhereTheAnswerHoldsSome_IsAnInfiniteDivergence_AsItIs_WhileTheOrderedMeasuresStayFinite()
    {
        // No number is added to a nought to keep it finite: a number nobody chose would quietly become a measurement.
        var rows = new Answered([[1, 1, 0, 0], [2, 2, 0, 0]], [[2, 0, 1, 1], [1, 1, 1, 1]]);

        Assert.Equal(double.PositiveInfinity, rows.Of(Metric.Kl, 0));
        Assert.Equal(0.875, rows.Of(Metric.Emd, 0), Close);
        Assert.Equal(0.34375, rows.Of(Metric.Rps, 0), Close);
    }

    [Fact]
    public void WhatIsLeftOfTheWhole_StandsOutsideTheOrder_SoOnlyTheDivergenceComparesIt()
    {
        var rows = new Answered(
            [[2, 5, 3, 0, 2], [1, 1, 1, 1, 4], [0, 0, 4, 4, 0]],
            [[1.5, 4, 3.5, 1, 2], [0.5, 2, 1, 0.5, 4], [0, 1, 3, 4, 0.5]])
        {
            InTheOrder = 4,
        };

        Assert.Equal(0.26666666666666666, rows.Of(Metric.Emd, 0), Close);
        Assert.Equal(0.032499999999999994, rows.Of(Metric.Rps, 0), Close);
        Assert.Equal(0.13116495914835158, rows.Of(Metric.Kl, 0), Close);
    }

    [Fact]
    public void EveryMeasure_HasAFamily_AndEveryMeasureThatIsANumber_HasAnArmOfItsOwn()
    {
        // A measure added later cannot fall through to another's arithmetic: every one is said, and one that is none of them
        // is refused.
        var rows = new Answered([[1, 0], [0, 1]], [[0.75, 0.25], [0.5, 0.5]]);
        var families = new Dictionary<Metric, MetricFamily>
        {
            [Metric.Rmse] = MetricFamily.Amounts,
            [Metric.Mae] = MetricFamily.Amounts,
            [Metric.R2] = MetricFamily.Amounts,
            [Metric.Accuracy] = MetricFamily.Classes,
            [Metric.Precision] = MetricFamily.Classes,
            [Metric.Recall] = MetricFamily.Classes,
            [Metric.ConfusionMatrix] = MetricFamily.Classes,
            [Metric.Emd] = MetricFamily.OrderedShares,
            [Metric.Kl] = MetricFamily.Shares,
            [Metric.Rps] = MetricFamily.OrderedShares,
        };

        Assert.Equal(Enum.GetValues<Metric>(), families.Keys.Order());
        Assert.All(families, each => Assert.Equal(each.Value, each.Key.Family()));
        Assert.Throws<ArgumentOutOfRangeException>(() => ((Metric)99).Family());

        // Any value but one of the arms is refused, so a measure that is a number and is measured at all has an arm.
        Assert.All(
            Enum.GetValues<Metric>().Where(metric => metric != Metric.ConfusionMatrix),
            metric => Assert.True(double.IsFinite(rows.Of(metric, 1)), $"'{metric.Word()}' measured nothing."));
        Assert.Throws<ArgumentOutOfRangeException>(() => rows.Of((Metric)99, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => rows.Of(Metric.ConfusionMatrix, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => rows.Classes(0).Of(Metric.Rmse));
        Assert.Equal(["emd", "kl", "rps"], new[] { Metric.Emd, Metric.Kl, Metric.Rps }.Select(metric => metric.Word()));
    }

    // ---- A flock's weight in bands, planned and arrived ----

    private static string[] Bands { get; } = ["w500", "w550", "w600"];

    // Twelve flocks, numbered in time: how many were planned, and how many arrived in each band.
    private static InMemoryRowSource Flocks() =>
        new(
            ["t", "planned", .. Bands],
            [
                .. Enumerable.Range(1, 12).Select<int, IReadOnlyList<string?>>(t =>
                [
                    t.ToString(CultureInfo.InvariantCulture),
                    (20 + t).ToString(CultureInfo.InvariantCulture),
                    (3 + (t % 4)).ToString(CultureInfo.InvariantCulture),
                    (8 + (t % 3)).ToString(CultureInfo.InvariantCulture),
                    (2 + (t % 5)).ToString(CultureInfo.InvariantCulture),
                ]),
            ]);

    private static FittingBuilder Split() =>
        Pdd.Create()
            .Read(Flocks(), "twelve flocks")
            .Declare(schema => schema.Integer("t").Number("planned").Number(Bands))
            .SplitByTime("t", 0.50, 0.25);

    private static PreparedData Planned(params Metric[] metrics) =>
        Split()
            .Distribution(Bands, "planned", ordered: true, remainder: "other")
            .Report(report => report.Measure(metrics).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

    private static Measures Measured(PreparedData prepared, Func<int, double[]> model) =>
        prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(model)]);
        })]);

    [Fact]
    public void AFlockShortOfPlanned_IsMeasuredAsItsBirdsComeBack_TheBandsAlongTheirOrderAndEveryShareByTheDivergence()
    {
        var measures = Measured(Planned(Metric.Emd, Metric.Kl, Metric.Rps, Metric.Rmse), row => row % 2 == 0 ? [0.2, 0.4, 0.2, 0.2] : [0.1, 0.5, 0.3, 0.1]);

        foreach (var part in measures.Parts)
        {
            // In birds, as the way back brings them: the bands and what is left add up to the planned flock.
            var rows = new Answered(part.Actual, part.Predicted) { InTheOrder = 3 };

            Assert.All(part.Actual, birds => Assert.Equal(Math.Round(birds.Sum()), birds.Sum(), 9));
            Assert.Equal(rows.Of(Metric.Emd, 0), part.Values[0].Value, Close);
            Assert.Equal(rows.Of(Metric.Kl, 0), part.Values[1].Value, Close);
            Assert.Equal(rows.Of(Metric.Rps, 0), part.Values[2].Value, Close);
            Assert.All(part.Values, measured => Assert.True(double.IsFinite(measured.Baseline)));
        }

        Assert.Equal([Metric.Emd, Metric.Kl, Metric.Rps, Metric.Rmse], measures.Metrics);
    }

    [Fact]
    public void ADivergenceThatIsInfinite_IsReportedAsInfinite()
    {
        var measures = Measured(Planned(Metric.Kl, Metric.Emd), _ => [0.5, 0.5, 0, 0]);

        Assert.All(measures.Parts, part => Assert.Equal(double.PositiveInfinity, part.Values[0].Value));
        Assert.All(measures.Parts, part => Assert.True(double.IsFinite(part.Values[1].Value)));
    }

    [Fact]
    public void APredictionBelowNought_IsNoShare_AndIsRefusedNamingItsRow()
    {
        var prepared = Planned(Metric.Emd);

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(prepared, _ => [0.6, 0.6, -0.1, -0.1]));

        Assert.Contains("Row ", refused.Message, StringComparison.Ordinal);
        Assert.Contains("below nought", refused.Message, StringComparison.Ordinal);
        Assert.Contains("emd", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAnswerWhoseBandsHoldNothing_IsNoShareOfAnything_AndIsRefusedNamingItsRow()
    {
        // Every bird that arrived is in the bands; a flock of which none arrived has bands of nought, whose order says nothing.
        var prepared = Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["t", "planned", .. Bands],
                    [.. Enumerable.Range(1, 8).Select<int, IReadOnlyList<string?>>(t => [$"{t}", "10", t == 7 ? "0" : "2", t == 7 ? "0" : "3", t == 7 ? "0" : "4"])]),
                "eight flocks")
            .Declare(schema => schema.Integer("t").Number("planned").Number(Bands))
            .SplitByTime("t", 0.50, 0.25)
            .Distribution(Bands, "planned", ordered: true, remainder: "other")
            .Report(report => report.Measure(Metric.Rps).On(Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var refused = Assert.Throws<InvalidOperationException>(() => Measured(prepared, _ => [0.25, 0.25, 0.25, 0.25]));

        Assert.Contains("Row 7", refused.Message, StringComparison.Ordinal);
        Assert.Contains("nought", refused.Message, StringComparison.Ordinal);
        Assert.Contains("rps", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APredictionOfNothingAtAll_IsNoShareOfAnything_AndIsRefusedNamingItsRow()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Measured(Planned(Metric.Kl), _ => [0, 0, 0, 0]));

        Assert.Contains("prediction add up to nought,", refused.Message, StringComparison.Ordinal);
        Assert.Contains("kl", refused.Message, StringComparison.Ordinal);
    }

    // ---- Where the measures are written ----

    private static IReadOnlyList<DeclarationFault> Faults(INamesTheAnswer output, params Metric[] metrics) =>
        PipelineDeclaration.FaultsIn(
        [
            .. Split().Declaration.Steps,
            output,
            new ReportStep(metrics, [Part.Test], [Shown.Numbers]),
        ]);

    [Theory]
    [InlineData(Metric.Emd)]
    [InlineData(Metric.Rps)]
    public void AMeasureAlongAnOrder_IsRefusedWhereItIsWritten_AgainstAnOutputWithNoOrder(Metric metric)
    {
        var unordered = Assert.Single(Faults(new DistributionStep(Bands, "planned"), metric, Metric.Rmse));
        var target = Assert.Single(Faults(new TargetStep("planned"), metric));

        Assert.Equal("evidence.report", unordered.Verb);
        Assert.Contains(metric.Word(), unordered.Message, StringComparison.Ordinal);
        Assert.Contains("'ordered'", unordered.Message, StringComparison.Ordinal);
        Assert.Contains("'target'", target.Message, StringComparison.Ordinal);
        Assert.Empty(Faults(new DistributionStep(Bands, "planned", ordered: true), metric, Metric.Kl, Metric.Rmse));
    }

    [Fact]
    public void TheDivergence_IsRefusedWhereItIsWritten_AgainstAnOutputWhoseAnswersAreNoSharesOfAWhole()
    {
        var target = Assert.Single(Faults(new TargetStep("planned"), Metric.Kl));
        var numbers = Assert.Single(Faults(new NumbersStep(["planned", "w500"]), Metric.Kl));

        Assert.Contains("kl", target.Message, StringComparison.Ordinal);
        Assert.Contains("'target.distribution'", target.Message, StringComparison.Ordinal);
        Assert.Contains("'target.numbers'", numbers.Message, StringComparison.Ordinal);
        Assert.Empty(Faults(new DistributionStep(Bands, "planned"), Metric.Kl));
    }

    [Fact]
    public void ADistributionsClassesRefused_SaysWhichMeasuresItsSharesAreMeasuredBy()
    {
        var unordered = Assert.Single(Faults(new DistributionStep(Bands, "planned"), Metric.Accuracy));
        var ordered = Assert.Single(Faults(new DistributionStep(Bands, "planned", ordered: true), Metric.Accuracy));

        Assert.Contains("rmse, mae or r2, or as shares with kl.", unordered.Message, StringComparison.Ordinal);
        Assert.Contains("rmse, mae or r2, or as shares in their order with emd, kl or rps.", ordered.Message, StringComparison.Ordinal);
        Assert.Equal(string.Empty, new AheadStep("close", 5, AheadAs.Return).SharesMeasured());
        Assert.Throws<ArgumentOutOfRangeException>(() => new TargetStep("a").Takes((MetricFamily)99));
    }
}
