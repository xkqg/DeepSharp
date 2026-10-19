// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// An ensemble is kept as one file of its own: its version, how its members' answers become one, every member's network, and the
/// one pipeline they were all trained behind. A single network's reader and the ensemble's reader each refuse the other's file by
/// name, and the file is held to its pipeline as a single network's is.
/// </summary>
public class TrainedEnsembleFileTests
{
    private static TrainedEnsemble Trained(Blend blend = Blend.Average)
    {
        var pipeline = Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.70, 0.15, 100, 7)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 3))
            .Build();
        var step = Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner);

        return pipeline.RunFor(step.Needs).TrainEnsemble([step, step with { Seed = 2 }, step with { Seed = 3 }], new EnsembleOptions { Blend = blend, Cancellation = TestContext.Current.CancellationToken });
    }

    private static CsvRowSource Served() =>
        CsvRowSource.FromText(string.Join('\n', File.ReadLines(Repository.Data("titanic.csv")).Take(13)) + "\n");

    private static TrainedEnsemble Read(string json) => TrainedEnsemble.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks());

    [Theory]
    [InlineData(Blend.Average)]
    [InlineData(Blend.Vote)]
    public void AnEnsembleWrittenAndReadAgain_AnswersToTheLastBit_UnderTheBlendItWasWrittenWith(Blend blend)
    {
        var ensemble = Trained(blend);
        var read = Read(ensemble.ToJson());

        Assert.Equal(blend, read.Blend);
        Assert.Equal(3, read.Members.Count);
        Assert.Equal(ensemble.Predict(Served()).Answers, read.Predict(Served()).Answers);
        Assert.Equal(ensemble.Members.Select(member => member.TrainedOn.Seed), read.Members.Select(member => member.TrainedOn.Seed));
        Assert.Equal(ensemble.ToJson(), read.ToJson());
    }

    [Fact]
    public void TheFile_HoldsItsVersion_TheBlend_EveryNetworkAndOnePipeline()
    {
        var file = JsonNode.Parse(Trained().ToJson())!.AsObject();

        Assert.Equal(["version", "blend", "networks", "pipeline"], file.Select(each => each.Key));
        Assert.Equal(1, (int)file["version"]!);
        Assert.Equal("average", (string)file["blend"]!);
        Assert.Equal(3, file["networks"]!.AsArray().Count);
        Assert.Equal(JsonValueKind.Object, file["pipeline"]!.GetValueKind());
    }

    [Fact]
    public void EachReaderRefuses_TheOthersFile_ByName()
    {
        var ensemble = Trained();
        var single = ensemble.Members[0].ToJson();

        var one = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(ensemble.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks()));
        var many = Assert.Throws<NetworkFileException>(() => Read(single));

        Assert.Contains("ensemble", one.Message, StringComparison.Ordinal);
        Assert.Contains("TrainedEnsemble", one.Message, StringComparison.Ordinal);
        Assert.Contains("single network", many.Message, StringComparison.Ordinal);
        Assert.Contains("TrainedNetwork", many.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkTrainedBehindAnotherFit_IsRefusedWhereItStands_NamingIt()
    {
        var file = JsonNode.Parse(Trained().ToJson())!.AsObject();
        var other = JsonNode.Parse(Trained().Members[0].ToJson())!["network"]!.DeepClone();

        other["trainedOn"]!["trainedBehind"] = "0000000000000000000000000000000000000000000000000000000000000000";
        file["networks"]!.AsArray()[1] = other;

        var refused = Assert.Throws<NetworkFileException>(() => Read(file.ToJsonString()));

        Assert.Contains("Network 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("another fit", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileWithFewerThanTwoNetworks_AnUnknownBlendAndNoBlend_AreRefused()
    {
        var file = JsonNode.Parse(Trained().ToJson())!.AsObject();

        var one = file.DeepClone().AsObject();
        one["networks"]!.AsArray().RemoveAt(2);
        one["networks"]!.AsArray().RemoveAt(1);

        var unknown = file.DeepClone().AsObject();
        unknown["blend"] = "median";

        var none = file.DeepClone().AsObject();
        none.Remove("blend");

        Assert.Contains("at least two", Assert.Throws<NetworkFileException>(() => Read(one.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("median", Assert.Throws<NetworkFileException>(() => Read(unknown.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("blend", Assert.Throws<NetworkFileException>(() => Read(none.ToJsonString())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileOfANewerVersion_AndOneWithAPartNoEnsembleHas_AreRefused()
    {
        var newer = JsonNode.Parse(Trained().ToJson())!.AsObject();
        newer["version"] = 2;

        var stray = JsonNode.Parse(Trained().ToJson())!.AsObject();
        stray["training"] = new JsonObject();

        Assert.Contains("newer DeepSharp", Assert.Throws<NetworkFileException>(() => Read(newer.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("'training'", Assert.Throws<NetworkFileException>(() => Read(stray.ToJsonString())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEnsembleOverTheBoundOfAText_IsRefusedBeforeItIsWritten_NamingTheBound()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => TrainedEnsemble.RequireWritable(TrainedEnsemble.MostParameters + 1));

        Assert.Contains("ensemble", refused.Message, StringComparison.Ordinal);
        Assert.Contains("40 million", refused.Message, StringComparison.Ordinal);
        TrainedEnsemble.RequireWritable(TrainedEnsemble.MostParameters);

        var tooLong = Assert.Throws<InvalidOperationException>(() => TrainedEnsemble.RequireTextFits(TrainedEnsemble.MostText + 1));

        Assert.Contains("characters", tooLong.Message, StringComparison.Ordinal);
        TrainedEnsemble.RequireTextFits(TrainedEnsemble.MostText);
    }
}
