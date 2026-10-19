// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// How a declared layer starts, and what it keeps: a dense layer that has no bias or says how its weights and bias begin,
/// a normalisation that keeps no scale or no shift. Each is a setting that leaving out means what it always meant, so a
/// declaration that says none of them is written as it always was.
/// </summary>
public class DeclaredLayerStartTests
{
    private static PartDeclaration Dense(params PartSetting[] settings) =>
        new("dense", [new PartSetting("units", PartValue.Of(3)), .. settings]);

    private static Layer First(string kind, params PartSetting[] settings) =>
        new[] { new PartDeclaration(kind, settings) }.Described().Lower(new Shape(3), new RandomStream(1)).HeldLayers().First().Layer;

    private static Dense LoweredDense(params PartSetting[] settings) =>
        Assert.IsType<Dense>(new[] { Dense(settings) }.Described().Lower(new Shape(2), new RandomStream(1)).HeldLayers().First().Layer);

    // The 'layers' of the declaration, wherever in the file the learning step stands.
    private static IEnumerable<JsonElement> Found(JsonProperty property) => property.Name == "layers"
        ? [property.Value]
        : property.Value.ValueKind switch
        {
            JsonValueKind.Object => property.Value.EnumerateObject().SelectMany(Found),
            JsonValueKind.Array => property.Value.EnumerateArray().Where(item => item.ValueKind == JsonValueKind.Object).SelectMany(item => item.EnumerateObject().SelectMany(Found)),
            _ => [],
        };

    [Fact]
    public void ADenseLayerThatSaysNothingOfHowItStarts_IsWrittenWithoutAnyOfTheNewKeys_AsItAlwaysWas()
    {
        var text = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).Relu().Dense(1).BatchNorm().LayerNorm().Adam(0.01).BinaryCrossEntropy().Run(1, epochs: 1))
            .Build().Declaration.ToJson();
        var layers = JsonDocument.Parse(text).RootElement.EnumerateObject().SelectMany(Found).Single();

        Assert.Equal(5, layers.GetArrayLength());

        foreach (var layer in layers.EnumerateArray())
        {
            Assert.DoesNotContain(layer.EnumerateObject().Select(property => property.Name), key => key is "bias" or "weights" or "biasStart" or "scale" or "shift");
        }
    }

    [Fact]
    public void ADenseLayerThatHasNoBias_IsDeclaredAsSuch_AndTheNetworkItTrainsHasNone()
    {
        var pipeline = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4, bias: false).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 2))
            .Build();
        var step = Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner);
        var trained = pipeline.Train();
        var layers = Assert.IsType<LayerStack>(trained.Network).Layers;

        Assert.False(step.Layers[0].YesOrNo("bias"));
        Assert.True(step.Layers[2].YesOrNo("bias"));
        Assert.Contains("\"bias\": false", pipeline.Declaration.ToJson(), StringComparison.Ordinal);
        Assert.Null(Assert.IsType<Dense>(layers[0]).Bias);
        Assert.NotNull(Assert.IsType<Dense>(layers[2]).Bias);
        Assert.Equal(2, trained.History!.Epochs.Count);
    }

    [Fact]
    public void ADeclaredStart_IsTheStartTheSameWordsWrittenAsCodeGive()
    {
        var declared = LoweredDense(
            new PartSetting("weights", PartValue.Of("ones")),
            new PartSetting("biasStart", PartValue.Of("zeros")));
        var written = Assert.IsType<Dense>(new Sequential().Dense(3, new Beginning { Weights = new Ones(), Bias = new Zeros() })
            .Lower(new Shape(2), new RandomStream(1)).HeldLayers().First().Layer);

        Assert.Equal(written.Weight.Value.Values.ToArray(), declared.Weight.Value.Values.ToArray());
        Assert.Equal(written.Bias!.Value.Values.ToArray(), declared.Bias!.Value.Values.ToArray());
        Assert.All(declared.Weight.Value.Values.ToArray(), value => Assert.Equal(1f, value));
        Assert.All(declared.Bias!.Value.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void ALayerThatSaysItStartsAsItAlwaysDid_StartsWhereItAlwaysDid_NumberForNumber()
    {
        var saying = LoweredDense(
            new PartSetting("weights", PartValue.Of("kaiminguniform")),
            new PartSetting("biasStart", PartValue.Of("fanInUniform")));
        var plain = LoweredDense();

        Assert.Equal(plain.Weight.Value.Values.ToArray(), saying.Weight.Value.Values.ToArray());
        Assert.Equal(plain.Bias!.Value.Values.ToArray(), saying.Bias!.Value.Values.ToArray());
    }

    [Theory]
    [InlineData("kaiminguniform", typeof(KaimingUniform))]
    [InlineData("glorotuniform", typeof(GlorotUniform))]
    [InlineData("fanInUniform", typeof(FanInUniform))]
    [InlineData("zeros", typeof(Zeros))]
    [InlineData("ones", typeof(Ones))]
    [InlineData("normal", typeof(Normal))]
    [InlineData("truncatedNormal", typeof(TruncatedNormal))]
    [InlineData("orthogonal", typeof(Orthogonal))]
    public void EveryWordForHowNumbersStart_IsTheInitialiserOfThatName(string word, Type expected)
    {
        var parameter = Assert.IsType<OneOfParameter<InitialiserWord>>(NetworkWords.Layers.Single(kind => kind.Name == "dense").Setting("weights"));

        Assert.Contains(word.ToLowerInvariant(), parameter.Choices);
        Assert.Equal(expected, Enum.Parse<InitialiserWord>(word, ignoreCase: true).Made().GetType());
    }

    [Fact]
    public void ADeclaredDenseWithAStartAndNoBias_GoesThroughTheFileAndComesBackTheSame()
    {
        var declaration = DeclaredNetworkTests.Passengers()
            .WithTensorflow(network => network
                .Dense(4, bias: false, weights: InitialiserWord.GlorotUniform)
                .Relu()
                .Dense(1, biasStart: InitialiserWord.Zeros)
                .BatchNorm(scale: false)
                .LayerNorm(scale: true, shift: false)
                .Adam(0.01)
                .BinaryCrossEntropy()
                .Run(1, epochs: 1))
            .Build().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());
        var layers = Assert.IsType<LearnNetworkStep>(again.Learner).Layers;

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Equal("glorotuniform", layers[0].Text("weights"));
        Assert.False(layers[0].YesOrNo("bias"));
        Assert.Equal("zeros", layers[2].Text("biasStart"));
        Assert.False(layers[3].YesOrNo("scale"));
        Assert.True(layers[3].YesOrNo("shift"));
        Assert.True(layers[4].YesOrNo("scale"));
        Assert.False(layers[4].YesOrNo("shift"));
    }

    [Fact]
    public void AFileThatLeavesOutHowALayerStarts_IsReadAsALayerThatStartsAsItAlwaysDid()
    {
        var step = Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(2).Adam(0.01).MeanSquaredError().Run(1, epochs: 1))
            .Build().Declaration.Learner);
        var layer = step.Layers[0];

        Assert.True(layer.YesOrNo("bias"));
        Assert.Equal("kaiminguniform", layer.Text("weights"));
        Assert.Equal("faninuniform", layer.Text("biasStart"));
    }

    [Fact]
    public void ABiasThatIsToStartAtSomething_ButIsNotThere_IsRefusedByName()
    {
        var refused = Assert.Throws<ArgumentException>(() => new LearnNetworkStep(
            [Dense(new PartSetting("bias", PartValue.Of(false)), new PartSetting("biasStart", PartValue.Of("zeros")))],
            NetworkWords.Optimizers[1].Declared(),
            NetworkWords.Losses[0].Declared()));

        Assert.Contains("bias", refused.Message, StringComparison.Ordinal);
        Assert.Contains("biasStart", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWordForHowNumbersStartThatNobodyKnows_IsRefusedNamingTheOnesThatAreKnown()
    {
        var text = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(2, weights: InitialiserWord.Ones).Adam(0.01).MeanSquaredError().Run(1, epochs: 1))
            .Build().Declaration.ToJson()
            .Replace("\"ones\"", "\"he\"", StringComparison.Ordinal);

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(text, StepCatalog.BuiltIn().WithNetworks()));

        Assert.Contains("'he'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("kaiminguniform", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANormalisationThatKeepsNoScaleOrNoShift_BecomesOneThatKeepsNone()
    {
        var batch = Assert.IsType<BatchNorm>(First("batchNorm", new PartSetting("momentum", PartValue.Of(0.9)), new PartSetting("epsilon", PartValue.Of(1e-5)), new PartSetting("scale", PartValue.Of(false))));
        var layer = Assert.IsType<LayerNorm>(First("layerNorm", new PartSetting("epsilon", PartValue.Of(1e-5)), new PartSetting("scale", PartValue.Of(false)), new PartSetting("shift", PartValue.Of(false))));

        Assert.Null(batch.Weight);
        Assert.NotNull(batch.Bias);
        Assert.Null(layer.Weight);
        Assert.Null(layer.Bias);
    }

    [Fact]
    public void EveryNewSetting_IsAWordOfTheKindItBelongsTo_WithTheMeaningOfLeavingItOut()
    {
        var dense = NetworkWords.Layers.Single(kind => kind.Name == "dense");
        var batch = NetworkWords.Layers.Single(kind => kind.Name == "batchNorm");
        var layer = NetworkWords.Layers.Single(kind => kind.Name == "layerNorm");

        Assert.Equal(["units", "bias", "weights", "biasStart", "l1", "l2", "maxNorm", "learns", "rateShare"], dense.Settings.Select(setting => setting.Key));
        Assert.Equal(["momentum", "epsilon", "scale", "shift", "learns", "rateShare"], batch.Settings.Select(setting => setting.Key));
        Assert.Equal(["epsilon", "scale", "shift", "learns", "rateShare"], layer.Settings.Select(setting => setting.Key));
        Assert.Equal(["units"], dense.Settings.Where(setting => setting.RequiredKeys.Count > 0).Select(setting => setting.Key));
        Assert.True(dense.Declared().YesOrNo("bias"));
        Assert.True(batch.Declared().YesOrNo("scale"));
        Assert.True(layer.Declared().YesOrNo("shift"));
    }

    [Fact]
    public void TheNewChainMembers_TakeTheWordsOfBothVocabularies()
    {
        var keras = Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTensorflow(network => network.Dense(2, bias: false).Adam().MeanSquaredError().Run(1)).Build().Declaration.Learner);
        var torch = Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(2, bias: false).Adam().MeanSquaredError().Run(1)).Build().Declaration.Learner);

        Assert.False(keras.Layers[0].YesOrNo("bias"));
        Assert.False(torch.Layers[0].YesOrNo("bias"));
    }

    [Fact]
    public void ASeedThatDrawsTheSameWeights_DrawsThemWhetherTheBiasIsThereOrNot()
    {
        var withBias = LoweredDense();
        var without = LoweredDense(new PartSetting("bias", PartValue.Of(false)));

        Assert.Equal(withBias.Weight.Value.Values.ToArray(), without.Weight.Value.Values.ToArray());
        Assert.Null(without.Bias);
    }
}
