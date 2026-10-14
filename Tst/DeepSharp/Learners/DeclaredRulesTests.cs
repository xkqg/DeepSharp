// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The rules a declared layer's numbers are put under — a penalty, a bound, a hold — said as settings of the layer's word that
/// leaving out means none of, so a declaration that says nothing of them is written as it always was; the doors that say them
/// of the layer written last; and a declared network that trains as the same words written as code train.
/// </summary>
public class DeclaredRulesTests
{
    private static PartKind Word(string name) => NetworkWords.Layers.Single(kind => kind.Name == name);

    private static LearnNetworkStep Learner(Func<NetworkDeclaration, NetworkDeclaration> layers) =>
        Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTorch(network => layers(network).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 2))
            .Build().Declaration.Learner);

    [Fact]
    public void TheWordsOfTheLayers_TakeTheRulesAsSettingsThatLeavingOutMeansNoneOf()
    {
        Assert.Equal(["units", "bias", "weights", "biasStart", "l1", "l2", "maxNorm", "learns"], Word("dense").Settings.Select(setting => setting.Key));
        Assert.Equal(["momentum", "epsilon", "scale", "shift", "learns"], Word("batchNorm").Settings.Select(setting => setting.Key));
        Assert.Equal(["epsilon", "scale", "shift", "learns"], Word("layerNorm").Settings.Select(setting => setting.Key));
        Assert.Equal(["units"], Word("dense").Settings.Where(setting => setting.RequiredKeys.Count > 0).Select(setting => setting.Key));
        Assert.Equal(0, Word("dense").Declared().Number("l2"));
        Assert.True(Word("dense").Declared().YesOrNo("learns"));
    }

    [Fact]
    public void ADenseWordWithRules_BecomesALayerWhoseWeightsAreUnderThem_AndWhoseBiasIsNot()
    {
        var ruled = new[] { Word("dense").Declared().With("units", PartValue.Of(3)).With("l1", PartValue.Of(0.01)).With("l2", PartValue.Of(0.02)).With("maxNorm", PartValue.Of(2.0)) }
            .Described().Lower(new Shape(2), new RandomStream(1));
        var held = new[] { Word("dense").Declared().With("units", PartValue.Of(3)).With("learns", PartValue.Of(false)) }
            .Described().Lower(new Shape(2), new RandomStream(1));
        var plain = new[] { Word("dense").Declared().With("units", PartValue.Of(3)) }.Described().Lower(new Shape(2), new RandomStream(1));
        var dense = Assert.IsType<Dense>(ruled.Layers[0]);

        Assert.Equal(new SlotRules { L1 = 0.01, L2 = 0.02, MaxNorm = 2 }, dense.Weight.Rules);
        Assert.Equal(default, dense.Bias!.Rules);
        Assert.Equal(new SlotRules { Held = true }, Assert.IsType<Dense>(held.Layers[0]).Weight.Rules);
        Assert.Equal(new SlotRules { Held = true }, Assert.IsType<Dense>(held.Layers[0]).Bias!.Rules);
        Assert.Equal(default, Assert.IsType<Dense>(plain.Layers[0]).Weight.Rules);
    }

    [Fact]
    public void ANormalisationWordThatDoesNotLearn_BecomesANormalisationHeldStill()
    {
        var held = new[]
        {
            Word("dense").Declared().With("units", PartValue.Of(3)),
            Word("batchNorm").Declared().With("learns", PartValue.Of(false)),
            Word("layerNorm").Declared().With("learns", PartValue.Of(false)),
        }.Described().Lower(new Shape(2), new RandomStream(1));

        Assert.False(Assert.IsType<BatchNorm>(held.Layers[1]).Weight!.Learns);
        Assert.False(Assert.IsType<BatchNorm>(held.Layers[1]).Bias!.Learns);
        Assert.False(Assert.IsType<LayerNorm>(held.Layers[2]).Weight!.Learns);
    }

    [Fact]
    public void TheDoors_SayTheRulesOfTheLayerWrittenLast_AndWriteThemOnlyWhenSaid()
    {
        var step = Learner(network => network.Dense(4).L1(0.01).L2(0.02).MaxNorm(3).Relu().Dense(1).Held());
        var plain = Learner(network => network.Dense(4).Relu().Dense(1));
        var text = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).MeanSquaredError().Run(1)).Build().Declaration.ToJson();

        Assert.Equal([0.01, 0.02, 3], [step.Layers[0].Number("l1"), step.Layers[0].Number("l2"), step.Layers[0].Number("maxNorm")]);
        Assert.False(step.Layers[2].YesOrNo("learns"));
        Assert.True(step.Layers[0].YesOrNo("learns"));
        Assert.Equal(0, plain.Layers[0].Number("l2"));

        foreach (var key in new[] { "\"l1\"", "\"l2\"", "maxNorm", "learns" })
        {
            Assert.DoesNotContain(key, text, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ARuleForALayerThatTakesNone_OrBeforeAnyLayer_IsRefusedByName()
    {
        Assert.Contains("relu", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).Relu().L2(0.1).Dense(1))).Message, StringComparison.Ordinal);
        Assert.Contains("dropout", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).Dropout(0.2).Held().Dense(1))).Message, StringComparison.Ordinal);
        Assert.Contains("learns", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).Dropout(0.2).Held().Dense(1))).Message, StringComparison.Ordinal);
        Assert.Contains("maxNorm", Assert.Throws<NotSupportedException>(() => Learner(network => network.Dense(4).BatchNorm().MaxNorm(2).Dense(1))).Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => Learner(network => network.L2(0.1).Dense(1)));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void ANumberBelowNothing_IsRefusedAtTheDoor(double wrong)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).L1(wrong)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).L2(wrong)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).MaxNorm(wrong)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Dense(1).MaxNorm(0)));
    }

    [Fact]
    public void ADeclarationWithRules_GoesThroughTheFileAndComesBackTheSame()
    {
        var declaration = DeclaredNetworkTests.Passengers()
            .WithTensorflow(network => network.Dense(4).L2(0.01).MaxNorm(2).Relu().Dense(1).Held().BatchNorm().Adam(0.01).MeanSquaredError().Run(1))
            .Build().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Contains("\"learns\": false", declaration.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclaredNetworkWithRules_TrainsAsTheSameWordsWrittenAsCodeTrain_NumberForNumber()
    {
        var declared = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).L1(0.001).L2(0.01).MaxNorm(3).Relu().Dense(1).Held().Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 3))
            .Build()
            .Train();
        var written = new Sequential().Dense(4).Ruled(new SlotRules { L1 = 0.001, L2 = 0.01, MaxNorm = 3 }).Relu().Dense(1).Held()
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(DeclaredNetworkTests.Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 3 });
        var free = new Sequential().Dense(4).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(DeclaredNetworkTests.Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 3 });

        Assert.Equal(written.History!.Epochs, declared.History!.Epochs);
        Assert.Equal(
            written.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()),
            declared.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()));
        Assert.NotEqual(free.History!.Epochs, declared.History.Epochs);
    }
}
