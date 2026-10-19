// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The words 0.11.0 adds to a declared network, in both vocabularies: each door writes down the number its own library leaves
/// unsaid, every word is lowered onto the kind a network's file names, and the periodic embedding's spread has no default.
/// </summary>
public class DeclaredVocabularyTests
{
    [Fact]
    public void EachDoor_WritesTheNumbersItsOwnLibraryLeavesUnsaid()
    {
        var torch = Declared(Vocabularies.Torch);
        var keras = Declared(Vocabularies.Keras);

        Assert.Equal(0.01, torch.Layers[1].Number("negativeSlope"));
        Assert.Equal(0.3, keras.Layers[1].Number("negativeSlope"));
        Assert.False(torch.Layers[2].YesOrNo("eachFeature"));
        Assert.True(keras.Layers[2].YesOrNo("eachFeature"));
        Assert.Equal(1.1920928955078125e-07, torch.Layers[3].Number("epsilon"));
        Assert.Equal(1e-6, keras.Layers[3].Number("epsilon"));
        Assert.Equal(0.25, torch.Loss.Number("alpha"));
        Assert.Equal(-1, keras.Loss.Number("alpha"));
    }

    [Fact]
    public void ACountLoss_TakesTheLogarithmInPyTorchsWords_AndTheRateInKerass()
    {
        var torch = Step(Vocabularies.Torch, network => network.Dense(1).Poisson());
        var keras = Step(Vocabularies.Keras, network => network.Dense(1).Poisson());

        Assert.Equal((true, 1e-8), (torch.Loss.YesOrNo("logInput"), torch.Loss.Number("epsilon")));
        Assert.Equal((false, 1e-7), (keras.Loss.YesOrNo("logInput"), keras.Loss.Number("epsilon")));
        Assert.Equal(OutputActivation.Exp, torch.Loss.Judged().Activation);
        Assert.Equal(1e-7, Assert.IsType<Poisson>(keras.Loss.Judged()).Epsilon);
    }

    [Fact]
    public void EveryNewWord_IsLoweredOntoTheKindANetworksFileNames()
    {
        var step = Step(Vocabularies.Torch, network => network
            .Periodic(sigma: 0.05, frequencies: 4, width: 3).Relu().Dense(8).Gelu(approximate: true).Silu().Mish().LeakyRelu().PRelu(eachFeature: true).Elu(0.5).Selu().RmsNorm().Dense(2).Softmax()
            .CrossEntropy(0.1));
        var layers = step.Layers.Described().Lower(new Shape(5), new RandomStream(1)).Layers;

        Assert.Equal(
            [typeof(Periodic), typeof(Relu), typeof(Dense), typeof(Gelu), typeof(Silu), typeof(Mish), typeof(LeakyRelu), typeof(PRelu), typeof(Elu), typeof(Selu), typeof(RmsNorm), typeof(Dense), typeof(Softmax)],
            layers.Select(layer => layer.GetType()));
        Assert.Equal(15, Assert.IsType<Dense>(layers[2]).Inputs);
        Assert.Equal(0.05, Assert.IsType<Periodic>(layers[0]).Periodicity.Sigma);
        Assert.True(Assert.IsType<Gelu>(layers[3]).Approximate);
        Assert.Equal(8, Assert.IsType<PRelu>(layers[7]).Count);
        Assert.Equal(0.5, Assert.IsType<Elu>(layers[8]).Alpha);
        Assert.Equal(0.1, Assert.IsType<CrossEntropy>(step.Loss.Judged()).Smoothing);
    }

    [Theory]
    [InlineData("meanAbsoluteError", typeof(MeanAbsoluteError))]
    [InlineData("huber", typeof(Huber))]
    [InlineData("quantile", typeof(Quantile))]
    [InlineData("focal", typeof(Focal))]
    [InlineData("hinge", typeof(Hinge))]
    [InlineData("poisson", typeof(Poisson))]
    public void EveryNewLoss_IsJudgedAsTheKindANetworksFileNames(string word, Type kind)
    {
        Assert.IsType(kind, NetworkWords.Losses.Single(each => each.Name == word).Declared().Judged());
    }

    [Fact]
    public void AFocalLossOfNoWeight_IsWrittenAsTorchvisionWritesIt()
    {
        Assert.Null(Assert.IsType<Focal>(new PartDeclaration("focal", [new PartSetting("gamma", PartValue.Of(2.0)), new PartSetting("alpha", PartValue.Of(-1.0))]).Judged()).Alpha);
        Assert.Equal(0.25, Assert.IsType<Focal>(Step(Vocabularies.Torch, network => network.Dense(1).Focal()).Loss.Judged()).Alpha);
    }

    [Fact]
    public void ALossOfClassesThatSoftensNothing_IsDeclaredAsItAlwaysWas()
    {
        var plain = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(2).Adam().CrossEntropy(0).Run(seed: 1)).Build().Declaration;
        var softened = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(1).Adam().BinaryCrossEntropy(0.1).Run(seed: 1)).Build().Declaration;

        Assert.DoesNotContain("smoothing", plain.ToJson(), StringComparison.Ordinal);
        Assert.Contains("\"smoothing\": 0.1", softened.ToJson(), StringComparison.Ordinal);
        Assert.Equal(softened.ToJson(), PipelineDeclaration.FromJson(softened.ToJson(), StepCatalog.BuiltIn().WithNetworks()).ToJson());
        Assert.Equal(0, Assert.IsType<BinaryCrossEntropy>(new PartDeclaration("binaryCrossEntropy", []).Judged()).Smoothing);
    }

    [Fact]
    public void SmoothL1_IsRefusedByName_SayingItIsTheHuberLossOverItsBeta()
    {
        var refused = Assert.Throws<NotSupportedException>(() => new PartDeclaration("smoothL1", []).Judged());

        Assert.Contains("'huber', with its delta at that beta", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePeriodicEmbeddingsSpread_HasNoDefault()
    {
        var periodic = NetworkWords.Layers.Single(kind => kind.Name == "periodic");

        Assert.Contains("sigma", periodic.Settings.SelectMany(setting => setting.RequiredKeys));
        Assert.DoesNotContain("smoothing", NetworkWords.Losses.Single(kind => kind.Name == "crossEntropy").Settings.SelectMany(setting => setting.RequiredKeys));
    }

    [Fact]
    public void EveryLayerWord_IsAKindANetworksFileNames_TheVariationsOfAnImageAndTheFlatteningOfOneAmongThem()
    {
        var catalog = NetworkCatalog.BuiltIn().Names.ToHashSet(StringComparer.Ordinal);

        Assert.All(NetworkWords.Layers, kind => Assert.Contains(kind.Name, catalog));
        Assert.Equal(
            ["batchNorm", "dense", "dropout", "elu", "embedding", "flatten", "gaussianNoise", "gelu", "layerNorm", "leakyRelu", "mish", "periodic", "prelu", "randomCrop", "randomFlip", "relu", "rmsNorm", "selu", "sigmoid", "silu", "softmax", "tanh", "tokenEmbedding"],
            NetworkWords.Layers.Select(kind => kind.Name).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void ANetworkOfTheNewWords_TrainsBehindThePipeline()
    {
        var trained = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Periodic(sigma: 0.1, frequencies: 4, width: 2).Relu().Dense(4).Gelu().Dense(1).Adam().Focal().Run(seed: 1, epochs: 2))
            .Build()
            .Train();

        Assert.Equal(2, trained.History!.Epochs.Count);
        Assert.All(trained.History.Epochs, epoch => Assert.True(double.IsFinite(epoch.Loss)));
    }

    [Fact]
    public void AnEmbedding_ReadsTheBlockTheEncoderWrote_WhereThePipelineHandsItsRowsOver_AndTrains()
    {
        var pipeline = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Embedding("sex", 3).Dense(4).Relu().Dense(1).Adam().BinaryCrossEntropy().Run(seed: 1, epochs: 2))
            .Build();
        var prepared = pipeline.Run();
        var step = Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner);
        var handed = prepared.Batch(Part.Train, Needs.OneScale).FeatureNames;

        var layers = step.Layers.Described(prepared).Lower(new Shape(handed.Count), new RandomStream(1)).Layers;
        var embedding = Assert.IsType<Embedding>(layers[0]);
        var trained = pipeline.Train();

        Assert.Equal(prepared.CategoryBlock("sex").Select(name => handed.ToList().IndexOf(name)), embedding.Block);
        Assert.Equal(handed.Count - 4 + 3, Assert.IsType<Dense>(layers[1]).Inputs);
        Assert.All(trained.History!.Epochs, epoch => Assert.True(double.IsFinite(epoch.Loss)));
    }

    [Fact]
    public void AnEmbeddingAfterAnotherLayer_OrOfAColumnWithNoBlock_OrOutsideAPipeline_IsRefusedSayingWhy()
    {
        var late = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).Embedding("sex", 3).Dense(1).Adam().BinaryCrossEntropy().Run(seed: 1, epochs: 1))
            .Build();
        var numbers = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Embedding("fare", 3).Dense(1).Adam().BinaryCrossEntropy().Run(seed: 1, epochs: 1))
            .Build();

        Assert.Contains("stands before the first layer that is not one", Assert.Throws<InvalidOperationException>(() => late.Train()).Message, StringComparison.Ordinal);
        Assert.Contains("one column per category", Assert.Throws<InvalidOperationException>(() => numbers.Train()).Message, StringComparison.Ordinal);
        Assert.Contains(
            "train the network behind its pipeline",
            Assert.Throws<InvalidOperationException>(() => Assert.IsType<LearnNetworkStep>(late.Declaration.Learner).Layers.Skip(1).Prepend(new PartDeclaration("embedding", [new PartSetting("column", PartValue.Of("sex")), new PartSetting("width", PartValue.Of(3))])).ToList().Described()).Message,
            StringComparison.Ordinal);
    }

    private static LearnNetworkStep Declared(Vocabularies words) =>
        Step(words, network => network.Dense(4).LeakyRelu().PRelu().RmsNorm().Dense(1).Focal());

    private static LearnNetworkStep Step(Vocabularies words, Func<NetworkDeclaration, NetworkDeclaration> network)
    {
        Func<NetworkDeclaration, NetworkDeclaration> declared = each => network(each).Adam().Run(seed: 1, epochs: 1);
        var built = words == Vocabularies.Torch
            ? DeclaredNetworkTests.Passengers().WithTorch(declared)
            : DeclaredNetworkTests.Passengers().WithTensorflow(declared);

        return Assert.IsType<LearnNetworkStep>(built.Build().Declaration.Learner);
    }
}
