// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// What an ensemble refuses, each on its own: a blend that is none of the blends, a file whose networks say nothing of what they were
/// trained on or were trained for other features, an interval with the wrong number of margins, and rows a pipeline hands over as other
/// features than its members learned. An interval asked of a pipeline that reports nothing of one leaves the report as it was.
/// </summary>
public class TrainedEnsembleGuardTests
{
    private static PreparedData Prepared(bool report)
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 600).Select(t =>
        {
            var x = noise.NextDouble();
            var y = (3 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.1);

            return (IReadOnlyList<string?>)[$"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture)];
        }).ToList();
        var fitted = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "six hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
            .Drop("t")
            .Target("y")
            .Normalise("x", Scale.MidRange);

        return (report ? fitted.Report(r => r.Measure(Metric.Mae, Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers)) : fitted)
            .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 3))
            .Build()
            .Run();
    }

    private static TrainedNetwork Trained(PreparedData prepared, int seed) =>
        prepared.Train(((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed }, cancellation: TestContext.Current.CancellationToken);

    private static TrainedEnsemble Ensemble(PreparedData prepared) => TrainedEnsemble.Of([Trained(prepared, 1), Trained(prepared, 2)]);

    private static TrainedEnsemble Read(string json) => TrainedEnsemble.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks());

    [Fact]
    public void ABlendThatIsNoneOfTheBlends_IsNoWayToAnswerTogether()
    {
        var prepared = Prepared(report: true);

        Assert.Throws<ArgumentOutOfRangeException>(() => TrainedEnsemble.Of([Trained(prepared, 1), Trained(prepared, 2)], (Blend)99));
    }

    [Fact]
    public void ANetworkOfTheFileThatSaysNothingOfWhatItWasTrainedOn_IsRefused()
    {
        var file = JsonNode.Parse(Ensemble(Prepared(report: true)).ToJson())!.AsObject();

        file["networks"]![0]!.AsObject().Remove("trainedOn");

        var refused = Assert.Throws<NetworkFileException>(() => Read(file.ToJsonString()));

        Assert.Contains("Network 1 names nothing it was trained on", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkOfTheFileTrainedForOtherAnswers_IsRefusedNamingItsPlace()
    {
        var file = JsonNode.Parse(Ensemble(Prepared(report: true)).ToJson())!.AsObject();

        file["networks"]![1]!["trainedOn"]!["answers"]![0] = "elsewhere";

        var refused = Assert.Throws<NetworkFileException>(() => Read(file.ToJsonString()));

        Assert.Contains("Network 2:", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIntervalWithAnotherNumberOfMarginsThanTheAnswers_IsRefused()
    {
        var file = JsonNode.Parse(Ensemble(Prepared(report: true)).WithInterval(0.10).ToJson())!.AsObject();

        file["interval"]!["margins"] = new JsonArray(1.0, 2.0);

        var refused = Assert.Throws<NetworkFileException>(() => Read(file.ToJsonString()));

        Assert.Contains("it has one margin for each answer", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RowsThePipelineHandsOverAsOtherFeaturesThanTheMembersLearned_AreRefusedNamingBoth()
    {
        var prepared = Prepared(report: true);
        var members = new[] { Trained(prepared, 1), Trained(prepared, 2) }
            .Select(member => new TrainedNetwork(member.Network, member.Loss, member.Prepared, member.TrainedOn with { Features = ["elsewhere"] }))
            .ToArray();
        var served = new InMemoryRowSource(["t", "x"], [["9001", "0.5"]]);

        var refused = Assert.Throws<InvalidOperationException>(() => TrainedEnsemble.Of(members).Predict(served));

        Assert.Contains("'elsewhere'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("answers only what it learned from", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIntervalAskedOfAPipelineThatReportsNone_LeavesTheEnsemblesReportAsItWas()
    {
        var with = Ensemble(Prepared(report: false)).WithInterval(0.10);

        Assert.NotNull(with.Interval);
        Assert.Null(with.Measures);
    }
}
