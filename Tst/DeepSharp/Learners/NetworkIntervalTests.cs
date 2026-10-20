// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A trained network, or an ensemble of them, can be given the width its answers are worth: the margin the errors on rows it was
/// neither fitted nor chosen on allow, at a level the caller names. The interval is kept with the model in its file, travels with every
/// answer it gives as a lower and an upper end, and is measured on the test rows by the pipeline's own report. Training the network further
/// gives a new network, which has none: the errors it was calibrated by were those of numbers that have since moved.
/// </summary>
public class NetworkIntervalTests
{
    private const double Alpha = 0.10;

    // y is three times x and a noise of a few hundredths, over 1 500 rows: 750 to learn from, 150 to choose with, 300 to calibrate on
    // and 300 to be measured on.
    private static PreparedData Prepared(bool report = true)
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 1500).Select(t =>
        {
            var x = noise.NextDouble();
            var y = (3 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.1);

            return new string?[] { $"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture) };
        }).ToList();
        var fitted = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "fifteen hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
            .Drop("t")
            .Target("y")
            .Normalise("x", Scale.MidRange);

        return (report ? fitted.Report(r => r.Measure(Metric.Mae, Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers)) : fitted)
            .WithTorch(network => network.Dense(8).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 40))
            .Build()
            .Run();
    }

    private static TrainedNetwork Trained(PreparedData prepared, int seed = 1) =>
        prepared.Train(((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed }, cancellation: TestContext.Current.CancellationToken);

    private static InMemoryRowSource Served(int count = 5, int from = 0) =>
        new(["t", "x"], [.. Enumerable.Range(from, count).Select(at => new string?[] { $"{9000 + at}", (at / 10.0).ToString("R", CultureInfo.InvariantCulture) })]);

    [Fact]
    public void ANetworkGivenAnInterval_WorksItOutFromTheCalibrationRows_AndLeavesTheNetworkWithoutOneAsItWas()
    {
        var prepared = Prepared();
        var trained = Trained(prepared);
        var with = trained.WithInterval(Alpha);

        Assert.Null(trained.Interval);
        Assert.Equal(Alpha, with.Interval!.Alpha);
        Assert.Equal(300, with.Interval.Rows);
        Assert.Equal(prepared.IntervalOf(trained.Predicted(prepared.Batch(Part.Calibration, TrainedNetwork.FeatureNeeds), new Chunking(new CpuBackend(), TrainedNetwork.ServingBatchSize)), Alpha).Margins, with.Interval.Margins);
        Assert.Same(trained.Network, with.Network);
        Assert.Same(prepared, with.Prepared);
    }

    [Fact]
    public void WhatTheNetworkPredicts_ComesWithALowerAndAnUpperEnd_AMarginOnEitherSide()
    {
        var trained = Trained(Prepared()).WithInterval(Alpha);
        var predicted = trained.Predict(Served());
        var margin = Assert.Single(trained.Interval!.Margins);

        Assert.Equal(5, predicted.Answers.Count);
        Assert.NotNull(predicted.Lower);
        Assert.NotNull(predicted.Upper);

        for (var row = 0; row < 5; row++)
        {
            Assert.Equal(predicted.Answers[row][0] - margin, predicted.Lower![row][0], 12);
            Assert.Equal(predicted.Answers[row][0] + margin, predicted.Upper![row][0], 12);
        }

        Assert.Null(Trained(Prepared()).Predict(Served()).Lower);
    }

    [Fact]
    public void TheReportMeasuresTheIntervalOnTheTestRows_AndItKeepsItsPromiseThere()
    {
        var trained = Trained(Prepared()).WithInterval(Alpha);
        var test = trained.Measures!.Parts.Single(part => part.Part == Part.Test);
        var coverage = test.Values.Single(value => value.Metric == Metric.Coverage);

        // Nine rows in ten land within the margin, to within the sampling of 300 test rows; the training rows' average, given the
        // same margin, covers a good deal fewer.
        Assert.InRange(coverage.Value, 1 - Alpha - 0.06, 1);
        Assert.True(coverage.Baseline < coverage.Value, $"{coverage.Baseline} against {coverage.Value}");
        Assert.Equal(2 * trained.Interval!.Margins[0], test.Values.Single(value => value.Metric == Metric.Width).Value, 12);
        Assert.DoesNotContain(Metric.Coverage, Trained(Prepared()).Measures!.Metrics);
    }

    [Fact]
    public void ALevelThatIsNoShare_AndRowsThatAreTooFew_AreRefused()
    {
        var trained = Trained(Prepared());

        Assert.Throws<ArgumentOutOfRangeException>(() => trained.WithInterval(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => trained.WithInterval(1));
        Assert.Throws<InvalidOperationException>(() => trained.WithInterval(0.001));
    }

    [Fact]
    public void ANetworkOfClasses_IsGivenNoInterval_ForAClassHasAChanceAndNoWidth()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .WithTorch(network => network.Dense(3).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 2))
            .Build()
            .Run();

        Assert.Contains("class", Assert.Throws<InvalidOperationException>(() => Trained(prepared).WithInterval(Alpha)).Message, StringComparison.Ordinal);
    }

    // ---- the file ----

    [Fact]
    public void AFileWithNoInterval_IsWrittenAsItWas_AndAFileWithOne_NamesItUnderOneKeyAtTheNextVersion()
    {
        var trained = Trained(Prepared());
        var plain = JsonNode.Parse(trained.ToJson())!.AsObject();
        var with = JsonNode.Parse(trained.WithInterval(Alpha).ToJson())!.AsObject();

        Assert.Equal(1, (int)plain["version"]!);
        Assert.False(plain.ContainsKey("interval"));
        Assert.Equal(2, (int)with["version"]!);
        Assert.Equal(["version", "network", "pipeline", "interval"], with.Select(each => each.Key));
        Assert.Equal(Alpha, (double)with["interval"]!["alpha"]!);
    }

    [Fact]
    public void AnIntervalSurvivesTheFile_AndPredictsAsItDid()
    {
        var trained = Trained(Prepared()).WithInterval(Alpha);
        var read = TrainedNetwork.FromJson(trained.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(trained.Interval!.Alpha, read.Interval!.Alpha);
        Assert.Equal(trained.Interval.Rows, read.Interval.Rows);
        Assert.Equal(trained.Interval.Margins, read.Interval.Margins);
        Assert.Equal(trained.Predict(Served()).Upper, read.Predict(Served()).Upper);
        Assert.Equal(trained.ToJson(), read.ToJson());
    }

    [Fact]
    public void AnIntervalThatSaysNonsense_IsRefusedWhereItStandsInTheFile()
    {
        var file = JsonNode.Parse(Trained(Prepared()).WithInterval(Alpha).ToJson())!.AsObject();

        file["interval"]!["alpha"] = 7;

        Assert.Contains("alpha", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(file.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIntervalWithAnotherNumberOfMarginsThanTheNetworkHasAnswers_IsRefused()
    {
        var file = JsonNode.Parse(Trained(Prepared()).WithInterval(Alpha).ToJson())!.AsObject();

        file["interval"]!["margins"]!.AsArray().Add(1.0);

        Assert.Contains("margin", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(file.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks())).Message, StringComparison.Ordinal);
    }

    // ---- an ensemble ----

    [Fact]
    public void AnEnsembleGivenAnInterval_HasTheMarginOfItsCombinedAnswer_NotOfAnyMember()
    {
        var prepared = Prepared();
        var ensemble = prepared.TrainEnsemble([.. Enumerable.Range(1, 3).Select(seed => ((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed })], new EnsembleOptions { Cancellation = TestContext.Current.CancellationToken });
        var with = ensemble.WithInterval(Alpha);
        var chunking = new Chunking(new CpuBackend(), TrainedNetwork.ServingBatchSize);
        var batch = prepared.Batch(Part.Calibration, TrainedNetwork.FeatureNeeds);
        var expected = prepared.IntervalOf([.. ensemble.Members.Select(member => member.Predicted(batch, chunking))], Blend.Average, Alpha);

        Assert.Null(ensemble.Interval);
        Assert.Equal(expected.Margins, with.Interval!.Margins);
        Assert.NotEqual(ensemble.Members[0].WithInterval(Alpha).Interval!.Margins, with.Interval.Margins);

        var predicted = with.Predict(Served());

        Assert.Equal(predicted.Answers[0][0] + with.Interval.Margins[0], predicted.Upper![0][0], 12);
        Assert.Contains(Metric.Coverage, with.Measures!.Ensemble.Metrics);
    }

    [Fact]
    public void AnEnsemblesIntervalSurvivesItsFile()
    {
        var prepared = Prepared();
        var ensemble = prepared.TrainEnsemble([.. Enumerable.Range(1, 2).Select(seed => ((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed })], new EnsembleOptions { Cancellation = TestContext.Current.CancellationToken }).WithInterval(Alpha);
        var file = JsonNode.Parse(ensemble.ToJson())!.AsObject();
        var read = TrainedEnsemble.FromJson(ensemble.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(["version", "blend", "networks", "pipeline", "interval"], file.Select(each => each.Key));
        Assert.Equal(ensemble.Interval!.Margins, read.Interval!.Margins);
        Assert.Equal(ensemble.Predict(Served()).Lower, read.Predict(Served()).Lower);
        Assert.Null(TrainedEnsemble.FromJson(prepared.TrainEnsemble([.. Enumerable.Range(1, 2).Select(seed => ((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed })], new EnsembleOptions { Cancellation = TestContext.Current.CancellationToken }).ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks()).Interval);
    }
}
