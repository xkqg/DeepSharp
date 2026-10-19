// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a run measured on each fold, laid beside itself: for every measure on every part, the value of each fold, their mean and
/// how far they spread. A small table measured on one deal is measured by that deal's luck; the spread across folds is how much of
/// it was. It gives no verdict — whether a spread is too wide is for the person who reads it — and it is the pipeline's own,
/// since the measures are, so a run of any learner can be laid beside the others.
/// </summary>
public class MeasuresAcrossTests
{
    private const int Folds = 4;

    // One fold of the passenger list, told that every passenger survives with a chance of its own: a different one on each fold,
    // so the folds measure differently.
    private static Measures Measured(int index)
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.80, new Fold(Folds, index), 42, 99)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Rmse, Metric.Mae).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        var chance = 0.2 + (0.2 * index);

        return prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { chance })]);
        })]);
    }

    private static double Mean(IEnumerable<double> values) => values.Average();

    private static double Spread(double[] values)
    {
        var mean = values.Average();

        return Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / (values.Length - 1));
    }

    [Fact]
    public void EveryMeasureOnEveryPart_ComesWithTheValueOfEachFold_TheirMeanAndTheirSpread()
    {
        var folds = Enumerable.Range(0, Folds).Select(Measured).ToArray();
        var across = MeasuresAcross.Of(folds);

        Assert.Equal(Folds, across.Folds);
        Assert.Equal([Part.Validation, Part.Test], across.Parts.Select(part => part.Part));

        foreach (var part in across.Parts)
        {
            Assert.Equal(folds.Select(fold => fold.Parts.Single(each => each.Part == part.Part).Rows), part.Rows);
            Assert.Equal([Metric.Accuracy, Metric.Rmse, Metric.Mae], part.Values.Select(value => value.Metric));

            foreach (var measured in part.Values)
            {
                double[] each = [.. folds.Select(fold => fold.Parts.Single(one => one.Part == part.Part).Values.Single(value => value.Metric == measured.Metric).Value)];
                double[] baselines = [.. folds.Select(fold => fold.Parts.Single(one => one.Part == part.Part).Values.Single(value => value.Metric == measured.Metric).Baseline)];

                Assert.Equal(each, measured.Values);
                Assert.Equal(Mean(each), measured.Mean, 12);
                Assert.Equal(Spread(each), measured.Spread, 12);
                Assert.Equal(Mean(baselines), measured.Baseline, 12);
            }
        }
    }

    [Fact]
    public void TheFoldsMeasuredDifferently_SoTheSpreadIsSomething()
    {
        var across = MeasuresAcross.Of([.. Enumerable.Range(0, Folds).Select(Measured)]);

        Assert.All(across.Parts.SelectMany(part => part.Values), measured => Assert.True(measured.Spread > 0, $"{measured.Metric} spread nothing"));
    }

    [Fact]
    public void TwoFolds_SpreadByTheirDifferenceOverTheRootOfTwo()
    {
        var one = Measured(0);
        var other = Measured(1);
        var across = MeasuresAcross.Of([one, other]);
        var accuracy = across.Parts[0].Values.Single(value => value.Metric == Metric.Accuracy);

        Assert.Equal(Math.Abs(accuracy.Values[0] - accuracy.Values[1]) / Math.Sqrt(2), accuracy.Spread, 12);
    }

    [Fact]
    public void FewerThanTwoFolds_AreRefused_ForOneValueSpreadsByNothingItCanSay()
    {
        Assert.Throws<ArgumentException>(() => MeasuresAcross.Of([Measured(0)]));
        Assert.Throws<ArgumentException>(() => MeasuresAcross.Of([]));
    }

    [Fact]
    public void FoldsThatMeasuredOtherThings_AreRefused_NamingWhatDiffers()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass"))
            .SplitAtRandom(0.80, new Fold(Folds, 0), 42, 99)
            .Target("survived")
            .Report(report => report.Measure(Metric.Rmse).On(Part.Validation).As(Shown.Numbers))
            .Build()
            .Run();
        var other = prepared.Measure([.. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. batch.Features.Select(_ => new[] { 0.5 })]);
        })]);
        var refused = Assert.Throws<ArgumentException>(() => MeasuresAcross.Of([Measured(0), other]));

        Assert.Contains("Accuracy", refused.Message, StringComparison.Ordinal);
    }
}
