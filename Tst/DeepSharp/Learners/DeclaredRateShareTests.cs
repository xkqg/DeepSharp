// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The share of the run's rate a declared layer's numbers move at, said as a setting of the layer's word that leaving out means
/// the whole rate of — so a declaration that says nothing of it is written as it always was — and the door that says it of the
/// layer written last, in both vocabularies.
/// </summary>
public class DeclaredRateShareTests
{
    private static PartKind Word(string name) => NetworkWords.Layers.Single(kind => kind.Name == name);

    private static LearnNetworkStep Learner(Func<NetworkDeclaration, NetworkDeclaration> layers) =>
        Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTorch(network => layers(network).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 2))
            .Build().Declaration.Learner);

    [Theory]
    [InlineData("dense")]
    [InlineData("batchNorm")]
    [InlineData("layerNorm")]
    [InlineData("rmsNorm")]
    public void TheWordsOfTheLayersThatLearnNumbers_TakeAShareOfTheRate_AsASettingLeavingOutMeansTheWholeRateOf(string name)
    {
        var share = Word(name).Settings.Single(setting => setting.Key == "rateShare");

        Assert.Equal(1, Word(name).Declared().Number("rateShare"));
        Assert.Empty(share.RequiredKeys);
    }

    [Fact]
    public void ADenseWordWithAShare_BecomesALayerWhoseWeightsAndBiasMoveAtIt_ThatKeepsWhatTheOtherRulesPut()
    {
        var shared = new[]
        {
            Word("dense").Declared().With("units", PartValue.Of(3)).With("rateShare", PartValue.Of(0.1)).With("l2", PartValue.Of(0.02)),
        }.Described().Lower(new Shape(2), new RandomStream(1));
        var dense = Assert.IsType<Dense>(shared.Layers[0]);

        Assert.Equal(new SlotRules { L2 = 0.02, RateShare = 0.1 }, dense.Weight.Rules);
        Assert.Equal(new SlotRules { RateShare = 0.1 }, dense.Bias!.Rules);
    }

    [Fact]
    public void ANormalisationWordWithAShare_BecomesANormalisationWhoseScaleAndShiftMoveAtIt_AndOneThatIsHeldStaysHeld()
    {
        var made = new[]
        {
            Word("dense").Declared().With("units", PartValue.Of(3)),
            Word("batchNorm").Declared().With("rateShare", PartValue.Of(0.5)),
            Word("layerNorm").Declared().With("rateShare", PartValue.Of(0.25)).With("learns", PartValue.Of(false)),
        }.Described().Lower(new Shape(2), new RandomStream(1));

        Assert.Equal(0.5, Assert.IsType<BatchNorm>(made.Layers[1]).Weight!.Rules.RateShare);
        Assert.Equal(0.5, Assert.IsType<BatchNorm>(made.Layers[1]).Bias!.Rules.RateShare);

        var held = Assert.IsType<LayerNorm>(made.Layers[2]);

        Assert.Equal(new SlotRules { RateShare = 0.25, Held = true }, held.Weight!.Rules);
        Assert.Equal(default, Assert.IsType<Dense>(made.Layers[0]).Weight.Rules);
    }

    [Fact]
    public void TheDoor_SaysTheShareOfTheLayerWrittenLast_AndWritesItOnlyWhenSaid()
    {
        var step = Learner(network => network.Dense(4).RateShare(0.1).Relu().Dense(1));
        var plain = Learner(network => network.Dense(4).Relu().Dense(1));
        var text = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).MeanSquaredError().Run(1)).Build().Declaration.ToJson();

        Assert.Equal(0.1, step.Layers[0].Number("rateShare"));
        Assert.Equal(1, plain.Layers[0].Number("rateShare"));
        Assert.DoesNotContain("rateShare", text, StringComparison.Ordinal);
    }

    [Fact]
    public void AShareForALayerThatTakesNone_OrBeforeAnyLayer_IsRefusedByName()
    {
        Assert.Contains("relu", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).Relu().RateShare(0.1).Dense(1))).Message, StringComparison.Ordinal);
        Assert.Contains("rateShare", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).Dropout(0.2).RateShare(0.1).Dense(1))).Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => Learner(network => network.RateShare(0.1).Dense(1)));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AShareThatIsNotANumberAboveNothing_IsRefusedAtTheDoor(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).RateShare(wrong)));

    [Fact]
    public void ADeclarationWithAShare_GoesThroughTheFileAndComesBackTheSame()
    {
        var declaration = DeclaredNetworkTests.Passengers()
            .WithTensorflow(network => network.Dense(4).RateShare(0.1).Relu().Dense(1).BatchNorm().RateShare(2).Adam(0.01).MeanSquaredError().Run(1))
            .Build().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Contains("\"rateShare\": 0.1", declaration.ToJson(), StringComparison.Ordinal);
        Assert.Contains("\"rateShare\": 2", declaration.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void TheDenseWord_TakesALowRankPair_AsSettingsLeavingOutMeansNoneOf()
    {
        var dense = Word("dense");

        Assert.Equal(0, dense.Declared().Number("rank"));
        Assert.Equal(0, dense.Declared().Number("alpha"));
        Assert.Empty(dense.Settings.Single(setting => setting.Key == "rank").RequiredKeys);
        Assert.DoesNotContain(Word("batchNorm").Settings, setting => setting.Key == "rank");
    }

    [Fact]
    public void ADenseWordWithARank_BecomesALayerWithAPairBesideItsHeldKernel_TheAlphaTheRankWhereItIsLeftOut()
    {
        var pair = new[]
        {
            Word("dense").Declared().With("units", PartValue.Of(3)).With("rank", PartValue.Of(2)).With("alpha", PartValue.Of(8.0)),
            Word("dense").Declared().With("units", PartValue.Of(1)).With("rank", PartValue.Of(1)),
        }.Described().Lower(new Shape(2), new RandomStream(1));
        var first = Assert.IsType<Dense>(pair.Layers[0]);
        var second = Assert.IsType<Dense>(pair.Layers[1]);

        Assert.Equal(new LowRank(2, 8), first.LowRank);
        Assert.False(first.Weight.Learns);
        Assert.Equal(new LowRank(1), second.LowRank);
        Assert.Equal(1, second.LowRank!.Value.Scale);
    }

    [Fact]
    public void TheLowRankDoor_SaysThePairOfTheLayerWrittenLast_AndWritesItOnlyWhenSaid()
    {
        var step = Learner(network => network.Dense(4).LowRank(2, 6).Relu().Dense(1));
        var plain = Learner(network => network.Dense(4).Relu().Dense(1));
        var text = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).MeanSquaredError().Run(1)).Build().Declaration.ToJson();

        Assert.Equal(2, step.Layers[0].Number("rank"));
        Assert.Equal(6, step.Layers[0].Number("alpha"));
        Assert.Equal(0, plain.Layers[0].Number("rank"));
        Assert.DoesNotContain("\"rank\"", text, StringComparison.Ordinal);
        Assert.DoesNotContain("alpha", text, StringComparison.Ordinal);
    }

    [Fact]
    public void ALowRankPairForALayerThatTakesNone_OrOfNoRank_IsRefusedByName()
    {
        Assert.Contains("batchNorm", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).BatchNorm().LowRank(2).Dense(1))).Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => Learner(network => network.LowRank(2).Dense(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).LowRank(0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).LowRank(2, double.NaN)));
    }

    [Fact]
    public void ADeclarationWithAPair_GoesThroughTheFileAndComesBackTheSame_AndTrainsAHeldKernelStill()
    {
        var declaration = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).LowRank(2, 4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 2))
            .Build().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Contains("\"rank\": 2", declaration.ToJson(), StringComparison.Ordinal);
        Assert.Contains("\"alpha\": 4", declaration.ToJson(), StringComparison.Ordinal);
    }
}
