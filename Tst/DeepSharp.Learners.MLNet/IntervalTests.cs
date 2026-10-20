// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A tree is given the width its answers are worth as a network is: the margin the errors on the rows the pipeline set aside to calibrate
/// on allow. The interval is the pipeline's, worked out from what the tree predicted for those rows, so the second learner needs nothing
/// of its own for it but a place in its file.
/// </summary>
public class IntervalTests
{
    private const double Alpha = 0.10;

    private static Pipeline Noisy()
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 1500).Select(t =>
        {
            var x = noise.NextDouble();
            var y = (3 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.1);

            return new string?[] { $"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture) };
        }).ToList();

        return Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "fifteen hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
            .Drop("t")
            .Target("y")
            .Report(report => report.Measure(Metric.Mae, Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithML(trainer => trainer.FastTree(trees: 30))
            .Build();
    }

    [Fact]
    public void ATreeGivenAnInterval_HasTheMarginOfItsOwnErrorsOnTheCalibrationRows_AndKeepsItsPromiseOnTheTestRows()
    {
        var tree = Noisy().TrainWithML();
        var with = tree.WithInterval(Alpha);
        var calibration = new PartPredictions(tree.Prepared.Batch(Part.Calibration, LearnMLStep.FeatureNeeds), [.. tree.Predict(Part.Calibration).Select(each => new[] { each })]);

        Assert.Null(tree.Interval);
        Assert.Equal(tree.Prepared.IntervalOf(calibration, Alpha).Margins, with.Interval!.Margins);
        Assert.Equal(300, with.Interval.Rows);

        var test = with.Measures!.Parts.Single(part => part.Part == Part.Test);
        var coverage = test.Values.Single(value => value.Metric == Metric.Coverage).Value;

        Assert.InRange(coverage, 1 - Alpha - 0.06, 1);
        Assert.Equal(2 * with.Interval.Margins[0], test.Values.Single(value => value.Metric == Metric.Width).Value, 12);
        Assert.DoesNotContain(Metric.Coverage, tree.Measures!.Metrics);
    }

    [Fact]
    public void ALevelThatIsNoShare_AClass_AndTooFewRows_AreRefused()
    {
        var tree = Noisy().TrainWithML();

        Assert.Throws<ArgumentOutOfRangeException>(() => tree.WithInterval(0));
        Assert.Throws<InvalidOperationException>(() => tree.WithInterval(0.001));

        var classes = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 3, testSeed: 7)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare")
            .Target("survived")
            .WithML(trainer => trainer.FastTree(trees: 10))
            .Build()
            .TrainWithML();

        Assert.Contains("class", Assert.Throws<InvalidOperationException>(() => classes.WithInterval(Alpha)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheFile_NamesTheIntervalOnlyWhenThereIsOne_AtTheNextVersion_AndReadsItBack()
    {
        var tree = Noisy().TrainWithML();
        var plain = JsonNode.Parse(tree.ToJson())!.AsObject();
        var with = tree.WithInterval(Alpha);
        var written = JsonNode.Parse(with.ToJson())!.AsObject();
        var read = MLModelFile.FromJson(with.ToJson());

        Assert.Equal(1, (int)plain["version"]!);
        Assert.False(plain.ContainsKey("interval"));
        Assert.Equal(2, (int)written["version"]!);
        Assert.True(written.ContainsKey("interval"));
        Assert.Equal(with.Interval!.Margins, read.Interval!.Margins);
        Assert.Equal(with.Interval.Alpha, read.Interval.Alpha);
        Assert.Null(MLModelFile.FromJson(tree.ToJson()).Interval);
        Assert.Equal(with.ToJson(), read.ToJson());
    }

    [Fact]
    public void AFileWhoseIntervalSaysNonsense_IsRefused()
    {
        var file = JsonNode.Parse(Noisy().TrainWithML().WithInterval(Alpha).ToJson())!.AsObject();

        file["interval"]!["alpha"] = 3;

        Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(file.ToJsonString()));
    }
}
