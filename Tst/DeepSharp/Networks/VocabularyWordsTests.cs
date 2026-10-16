// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The words 0.11.0 adds to a description in Keras's words: each is lowered onto the layer of its kind with the settings it was
/// told, or PyTorch's where it was told none, a word that cannot take the example reaching it says which word and why, and a last
/// softmax before a loss that applies the softmax itself is lifted into the loss as a last sigmoid is.
/// </summary>
public class VocabularyWordsTests
{
    [Fact]
    public void EveryActivationWord_IsLoweredOntoItsLayer_WithWhatItWasTold_OrPyTorchsDefault()
    {
        var layers = new Sequential()
            .Dense(4).Gelu().Gelu(approximate: true).Silu().Mish().LeakyRelu().LeakyRelu(0.3).PRelu().PRelu(eachChannel: true).Elu().Elu(0.5).Selu().Softmax()
            .Lower(new Shape(3), new RandomStream(1)).Layers;

        Assert.False(Assert.IsType<Gelu>(layers[1]).Approximate);
        Assert.True(Assert.IsType<Gelu>(layers[2]).Approximate);
        Assert.IsType<Silu>(layers[3]);
        Assert.IsType<Mish>(layers[4]);
        Assert.Equal(0.01, Assert.IsType<LeakyRelu>(layers[5]).NegativeSlope);
        Assert.Equal(0.3, Assert.IsType<LeakyRelu>(layers[6]).NegativeSlope);
        Assert.Equal(1, Assert.IsType<PRelu>(layers[7]).Count);
        Assert.Equal(4, Assert.IsType<PRelu>(layers[8]).Count);
        Assert.Equal(1, Assert.IsType<Elu>(layers[9]).Alpha);
        Assert.Equal(0.5, Assert.IsType<Elu>(layers[10]).Alpha);
        Assert.IsType<Selu>(layers[11]);
        Assert.IsType<Softmax>(layers[12]);
    }

    [Fact]
    public void EveryNormalisationWord_IsLoweredOntoItsLayer_OfAsManyChannelsAsTheLastAxisHolds()
    {
        var layers = new Sequential()
            .Dense(6).GroupNorm(3).GroupNorm(2, 1e-3, new Affine { WithoutShift = true }).InstanceNorm().InstanceNorm(1e-3, new Affine { WithoutScale = true, WithoutShift = true }).RmsNorm().RmsNorm(1e-6)
            .Lower(new Shape(3), new RandomStream(1)).Layers;

        var grouped = Assert.IsType<GroupNorm>(layers[1]);
        var keras = Assert.IsType<GroupNorm>(layers[2]);
        var plain = Assert.IsType<InstanceNorm>(layers[4]);
        var rms = Assert.IsType<RmsNorm>(layers[5]);

        Assert.Equal((6, 3, 1e-5), (grouped.Features, grouped.Groups, grouped.Epsilon));
        Assert.Equal((2, 1e-3), (keras.Groups, keras.Epsilon));
        Assert.Null(keras.Bias);
        Assert.Equal(6, Assert.IsType<InstanceNorm>(layers[3]).Features);
        Assert.Null(plain.Weight);
        Assert.Null(plain.Bias);
        Assert.Equal(1.1920928955078125e-07, rms.Epsilon);
        Assert.NotNull(rms.Weight);
        Assert.Null(rms.Bias);
        Assert.Equal(1e-6, Assert.IsType<RmsNorm>(layers[6]).Epsilon);
    }

    [Fact]
    public void AGroupNormalisation_WhoseGroupsDoNotDivideTheChannelsReachingIt_IsRefusedNamingTheWord()
    {
        var wrong = Assert.Throws<ArgumentException>(() => new Sequential().Dense(5).GroupNorm(2).Lower(new Shape(3), new RandomStream(1)));

        Assert.Contains("word 2, groupNorm, puts the channels in 2 groups, and the 5 channels", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APeriodicEmbedding_MakesEachNumberOfARowAsWideAsItIsTold_AndTakesARowAlone()
    {
        var periodicity = new Periodicity(4, 0.1, 5);
        var stack = new Sequential().Periodic(periodicity).Relu().Dense(1).Lower(new Shape(3), new RandomStream(1));
        var lite = new Sequential().Periodic(periodicity with { Lite = true }).Lower(new Shape(3), new RandomStream(1));

        Assert.Equal(15, Assert.IsType<Dense>(stack.Layers[2]).Inputs);
        Assert.Equal(new Shape(3, 4), Assert.IsType<Periodic>(stack.Layers[0]).Frequencies.Value.Shape);
        Assert.Equal(new Shape(8, 15), Assert.IsType<Periodic>(stack.Layers[0]).Weight.Value.Shape);
        Assert.Equal(new Shape(8, 5), Assert.IsType<Periodic>(lite.Layers[0]).Weight.Value.Shape);
        Assert.Contains("periodic, takes each example as a row of numbers", Assert.Throws<ArgumentException>(() => new Sequential().Periodic(periodicity).Lower(new Shape(2, 3), new RandomStream(1))).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential().Periodic(periodicity with { Sigma = 0 }));
    }

    [Fact]
    public void ALastSoftmaxBeforeALossThatAppliesIt_IsLiftedIntoTheLoss_AndAloneIsRefusedNamingTheSoftmax()
    {
        var rows = new TrainingData(Tensor.From(new Shape(2, 2), [0.1f, 0.2f, 0.3f, 0.4f]), Tensor.From(new Shape(2, 2), [1f, 0f, 0f, 1f]));
        var shares = new Sequential().Dense(2).Softmax().Compile(new Adam(), new CrossEntropy());
        var squared = new Sequential().Dense(2).Softmax().Compile(new Adam(), new MeanSquaredError());

        shares.Fit(rows, validation: null, new FitOptions(seed: 1) { Epochs = 1 });
        squared.Fit(rows, validation: null, new FitOptions(seed: 1) { Epochs = 1 });

        Assert.IsType<Dense>(Assert.IsType<LayerStack>(shares.Network).Layers[^1]);
        Assert.IsType<Softmax>(Assert.IsType<LayerStack>(squared.Network).Layers[^1]);
        Assert.Equal(
            "This network holds nothing but the softmax its loss applies itself: write in the layers before it, and the loss takes its outputs as the logits it applies the softmax to.",
            Assert.Throws<InvalidOperationException>(() => new Sequential().Softmax().Compile(new Adam(), new CrossEntropy())).Message);
    }

    [Fact]
    public void AStackEndingInTheSoftmaxItsLossAppliesItself_IsRefusedAtCompile_NamingTheLayer()
    {
        var stack = new LayerStack(new Dense(Tensor.From(new Shape(1, 2), [1f, 1f]), null), new Softmax());

        var wrong = Assert.Throws<ArgumentException>(() => stack.Compile(new Adam(), new CrossEntropy()));

        Assert.Equal(
            "CrossEntropy applies the softmax to the network's outputs itself, and this network's last layer, '1', is a Softmax: "
            + "every prediction would go through the softmax twice. Leave that layer out; the predictions still come out through the loss's softmax. (Parameter 'loss')",
            wrong.Message);
    }
}
