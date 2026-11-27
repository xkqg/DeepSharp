// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// An ensemble is kept as one file of its own: its version, how its members' answers become one, and its members — every member's
/// network and the one pipeline they were all trained behind, the first version, which an ensemble behind one pipeline is still
/// written as; or each member whole, as a single network's file holds it, with its own pipeline, the second. A single network's reader
/// and the ensemble's reader each refuse the other's file by name, and the file is held to its pipelines as a single network's is.
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
    public void AnEnsembleBehindOnePipeline_IsWrittenAsTheFirstVersion_ItsVersionTheBlendEveryNetworkAndOnePipeline()
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
        var second = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(OfWholeModels().ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks()));

        Assert.Contains("ensemble", one.Message, StringComparison.Ordinal);
        Assert.Contains("TrainedEnsemble", one.Message, StringComparison.Ordinal);
        Assert.Contains("single network", many.Message, StringComparison.Ordinal);
        Assert.Contains("TrainedNetwork", many.Message, StringComparison.Ordinal);
        Assert.Contains("'members' is a part of an ensemble's file", second.Message, StringComparison.Ordinal);
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

        var counted = file.DeepClone().AsObject();
        counted["blend"] = "1";

        var none = file.DeepClone().AsObject();
        none.Remove("blend");

        Assert.Contains("at least two", Assert.Throws<NetworkFileException>(() => Read(one.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("median", Assert.Throws<NetworkFileException>(() => Read(unknown.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("'1' is none of", Assert.Throws<NetworkFileException>(() => Read(counted.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("blend", Assert.Throws<NetworkFileException>(() => Read(none.ToJsonString())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileOfANewerVersion_AndOneWithAPartNoEnsembleHas_AreRefused()
    {
        var newer = JsonNode.Parse(Trained().ToJson())!.AsObject();
        newer["version"] = TrainedEnsemble.Version + 1;

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

    // Two networks, each behind a pipeline of its own: the one trained on the answer as it is, the other on its log.
    private static TrainedEnsemble OfWholeModels() =>
        TrainedEnsemble.Of([TrainedEnsembleTests.Regressed(TrainedEnsembleTests.Regression()), TrainedEnsembleTests.Regressed(TrainedEnsembleTests.Regression(logged: true))]);

    private static string Indented(JsonNode file) => file.ToJsonString(new JsonSerializerOptions { WriteIndented = true }).ReplaceLineEndings("\n");

    // The line a text holds first at, from one.
    private static int LineOf(string text, string held) => text[..text.IndexOf(held, StringComparison.Ordinal)].Count(letter => letter == '\n') + 1;

    [Fact]
    public void AnEnsembleOfWholeModels_SurvivesItsFile_AsTheSecondVersion_EachMemberWithItsOwnPipeline()
    {
        var ensemble = OfWholeModels().WithInterval(0.10);
        var json = ensemble.ToJson();
        var file = JsonNode.Parse(json)!.AsObject();

        Assert.Equal(["version", "blend", "members", "interval"], file.Select(each => each.Key));
        Assert.Equal(2, (int)file["version"]!);
        Assert.All(file["members"]!.AsArray(), member => Assert.Equal(["version", "network", "pipeline"], member!.AsObject().Select(each => each.Key)));

        var read = Read(json);

        Assert.Equal(ensemble.Predict(TrainedEnsembleTests.Asked()).Answers, read.Predict(TrainedEnsembleTests.Asked()).Answers);
        Assert.Equal(ensemble.Interval!.Margins, read.Interval!.Margins);
        Assert.Equal(json, read.ToJson());
        Assert.NotEqual(read.Members[0].TrainedOn.TrainedBehind, read.Members[1].TrainedOn.TrainedBehind);
    }

    [Fact]
    public void AFileOfTheSecondVersion_HoldsItsMembersAsAListAndNothingOfTheFirsts_AndAFileOfTheFirstHoldsNoMembers()
    {
        var second = JsonNode.Parse(OfWholeModels().ToJson())!.AsObject();
        var first = JsonNode.Parse(Trained().ToJson())!.AsObject();

        var withPipeline = second.DeepClone().AsObject();
        withPipeline["pipeline"] = first["pipeline"]!.DeepClone();

        var notAList = second.DeepClone().AsObject();
        notAList["members"] = new JsonObject();

        var one = second.DeepClone().AsObject();
        one["members"]!.AsArray().RemoveAt(1);

        var withMembers = first.DeepClone().AsObject();
        withMembers["members"] = new JsonArray();

        Assert.Contains("'pipeline'", Assert.Throws<NetworkFileException>(() => Read(withPipeline.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("'members'", Assert.Throws<NetworkFileException>(() => Read(notAList.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("at least two", Assert.Throws<NetworkFileException>(() => Read(one.ToJsonString())).Message, StringComparison.Ordinal);
        Assert.Contains("'members'", Assert.Throws<NetworkFileException>(() => Read(withMembers.ToJsonString())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFaultInAMemberOfTheSecondVersion_IsPlacedAtItsLineAndColumnInTheWholeFile_NamingTheMember()
    {
        var file = JsonNode.Parse(OfWholeModels().ToJson())!.AsObject();

        var newer = file.DeepClone().AsObject();
        newer["members"]![1]!["version"] = 99;
        var newerText = Indented(newer);

        var unknown = file.DeepClone().AsObject();
        unknown["members"]![1]!["pipeline"]!["declaration"]![0]!["step"] = "nobody.knows";
        var unknownText = Indented(unknown);

        var bare = file.DeepClone().AsObject();
        bare["members"]![1] = 7;
        var bareText = Indented(bare);

        var version = Assert.Throws<NetworkFileException>(() => Read(newerText)).Faults[0];
        var verb = Assert.Throws<PipelineFileException>(() => Read(unknownText)).Faults[0];
        var number = Assert.Throws<NetworkFileException>(() => Read(bareText)).Faults[0];

        Assert.StartsWith("Network 2: ", version.Message, StringComparison.Ordinal);
        Assert.Equal(LineOf(newerText, "\"version\": 99"), version.Line);
        Assert.StartsWith("Network 2: ", verb.Message, StringComparison.Ordinal);
        // A step is placed where it starts, the line above its verb.
        Assert.Equal(LineOf(unknownText, "nobody.knows") - 1, verb.Line);
        Assert.StartsWith("Network 2: ", number.Message, StringComparison.Ordinal);
        Assert.Equal(LineOf(bareText, "\n    7\n") + 1, number.Line);
        Assert.Equal(5, number.Column);
    }

    [Fact]
    public void AFileOfTheSecondVersion_WhoseMembersAnswerOtherwise_IsRefusedAtItsMembers()
    {
        var file = JsonNode.Parse(OfWholeModels().ToJson())!.AsObject();

        file["members"]![1] = JsonNode.Parse(TrainedEnsembleTests.Regressed(TrainedEnsembleTests.Regression(answer: "x")).ToJson());

        var refused = Assert.Throws<NetworkFileException>(() => Read(file.ToJsonString()));

        Assert.Contains("Member 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'x'", refused.Message, StringComparison.Ordinal);
    }
}
