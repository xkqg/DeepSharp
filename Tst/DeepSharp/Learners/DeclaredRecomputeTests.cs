// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Whether a declared run throws the workings of its layers away and works them out again, so that a larger batch fits in the same
/// memory: a key a network's declaration gained after files were written without it. Left out, it means what a declaration meant before
/// it had it — every layer keeps its workings — and is not written; said, it is written, read back, and changes no number of the run,
/// to the bit. Which layers, rather than all of them, is code: a <see cref="DeepSharp.Networks.FitOptions"/> with its own <c>Recompute</c>.
/// </summary>
public class DeclaredRecomputeTests
{
    private static FittingBuilder Passengers() => Pdd.Create()
        .ReadCsv(Repository.Data("titanic.csv"))
        .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", 0.70, 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Normalise("sibsp", Scale.MidRange)
        .Normalise("parch", Scale.MidRange)
        .Target("survived");

    [Fact]
    public void ADeclarationThatSaysNothingOfIt_WritesNoKey()
    {
        var written = JsonNode.Parse(Passengers().WithTorch(network => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1)).Build().Declaration.ToJson())!;
        var learner = written["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();

        Assert.False(learner.ContainsKey("recompute"));
    }

    [Fact]
    public void ARecomputeThatIsSaid_IsWrittenUnderItsKey_ReadBack_AndWrittenAgainAsItWas()
    {
        var declaration = Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Recompute().Run(1)).Build().Declaration;
        var learner = JsonNode.Parse(declaration.ToJson())!["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.True((bool)learner["recompute"]!);
        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.True(Assert.IsType<LearnNetworkStep>(again.Learner).Recompute);
    }

    [Fact]
    public void ARecomputedDeclaredRun_TrainsToTheBit_WhatTheSameRunKeepingItsWorkingsTrains()
    {
        var kept = Passengers().WithTorch(network => network.Dense(8).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(7, epochs: 4)).Build().Train(new Engines());
        var again = Passengers().WithTorch(network => network.Dense(8).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Recompute().Run(7, epochs: 4)).Build().Train(new Engines());

        Assert.Equal(
            kept.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits)),
            again.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits)));
    }
}
