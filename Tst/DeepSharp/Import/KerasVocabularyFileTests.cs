// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Keras models made of the words 0.11.0 reads (Fixtures/keras-vocabulary-fixtures.py made them): the activations by name —
/// gelu, silu, mish, leaky_relu, elu, selu and softmax anywhere — and the layers LeakyReLU, PReLU, ELU, Softmax,
/// GroupNormalization (a group for every channel an instance normalisation), RMSNormalization and Embedding, each held to what
/// Keras answered through it. A layer normalisation that scales by <c>rms_scaling</c> is refused: Keras works it out from the
/// variance, not the root mean square, and says so as it deprecates it.
/// </summary>
public class KerasVocabularyFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-vocabulary.json"))).RootElement;

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    [Theory]
    [InlineData("keras-vocabulary-rows")]
    [InlineData("keras-vocabulary-images")]
    [InlineData("keras-vocabulary-embedding")]
    public void AModelOfTheNewWords_AnswersAsKerasDoes(string name)
    {
        var expected = Answers.GetProperty(name);
        var written = expected.GetProperty("examples");
        var shape = new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);
        var examples = Tensor.From(shape, Values(written.GetProperty("values")));

        var read = new KerasFile().Read(KerasFixtures.Open($"{name}.keras"));
        var here = read.Network.Predict(examples, read.Loss, Engine).Values.ToArray();
        var keras = Values(expected.GetProperty("outputs"));

        Assert.Equal(keras.Length, here.Length);
        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void EveryWordOfTheRowsModel_IsReadAsTheLayerOfItsKind_WithKerassSettings()
    {
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-vocabulary-rows.keras")).Network).Layers;

        Assert.Equal(
            [
                typeof(Dense), typeof(Gelu), typeof(Dense), typeof(Silu), typeof(Dense), typeof(Mish), typeof(Dense), typeof(LeakyRelu), typeof(LeakyRelu), typeof(PRelu),
                typeof(Elu), typeof(Dense), typeof(Elu), typeof(Selu), typeof(GroupNorm), typeof(RmsNorm), typeof(Dense), typeof(Softmax),
            ],
            layers.Select(layer => layer.GetType()));
        Assert.False(((Gelu)layers[1]).Approximate);
        Assert.Equal(0.2, ((LeakyRelu)layers[7]).NegativeSlope, 6);
        Assert.Equal(0.1, ((LeakyRelu)layers[8]).NegativeSlope, 6);
        Assert.Equal(6, ((PRelu)layers[9]).Count);
        Assert.Equal(0.7, ((Elu)layers[10]).Alpha, 6);
        Assert.Equal((3, 1e-3), (((GroupNorm)layers[14]).Groups, ((GroupNorm)layers[14]).Epsilon));
        Assert.Equal(1e-6, ((RmsNorm)layers[15]).Epsilon);
    }

    [Fact]
    public void AGroupNormalisationOfAGroupForEveryChannel_IsAnInstanceNormalisation_AndAPReLUSharedOverThePlaces_HasAShareForEachChannel()
    {
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-vocabulary-images.keras")).Network).Layers;

        Assert.Equal(2, Assert.IsType<GroupNorm>(layers[1]).Groups);
        Assert.Equal(4, Assert.IsType<PRelu>(layers[2]).Count);
        // Keras writes a group for every channel as -1, and as the channels it found once it is built: a group of one channel each.
        Assert.Equal((4, 4), (Assert.IsType<GroupNorm>(layers[3]).Groups, ((GroupNorm)layers[3]).Features));
        Assert.IsType<Softmax>(layers[4]);
    }

    [Fact]
    public void AGroupNormalisationSayingMinusOne_IsReadAsAnInstanceNormalisation_AndAnswersAsAGroupForEveryChannelDoes()
    {
        var expected = Answers.GetProperty("keras-vocabulary-images");
        var written = expected.GetProperty("examples");
        var examples = Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written.GetProperty("values")));

        var read = new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-images.keras", "config.layers.4.config.groups=-1"));
        var here = read.Network.Predict(examples, read.Loss, Engine).Values.ToArray();

        Assert.Equal(4, Assert.IsType<InstanceNorm>(Assert.IsType<LayerStack>(read.Network).Layers[3]).Features);
        Assert.InRange(Values(expected.GetProperty("outputs")).Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
        Assert.Contains("one group at least", Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-images.keras", "config.layers.4.config.groups=0"))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmbeddingFirst_TakesTheCategorysOnesAndNoughts_WhereKerasTakesItsPlace()
    {
        var read = new KerasFile().Read(KerasFixtures.Open("keras-vocabulary-embedding.keras"));
        var embedding = Assert.IsType<Embedding>(Assert.IsType<LayerStack>(read.Network).Layers[0]);

        Assert.Equal((5, 3), (embedding.Features, embedding.Width));
        Assert.Equal([0, 1, 2, 3, 4], embedding.Block);
    }

    [Fact]
    public void ALayerNormalisationThatScalesByRmsScaling_IsRefused_NamingTheVarianceKerasScalesByAndTheLayerThatIsRead()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Open("keras-vocabulary-rms-scaling.keras")));

        Assert.Contains("rms_scaling", refused.Message, StringComparison.Ordinal);
        Assert.Contains("variance", refused.Message, StringComparison.Ordinal);
        Assert.Contains("RMSNormalization", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("config.layers.1.config.activation=\"softsign\"", "its activation, softsign, is none of those read here")]
    [InlineData("config.layers.7.config.alpha=-1", "alpha")]
    [InlineData("config.layers.11.config.axis=1", "normalises over axis")]
    [InlineData("config.layers.10.config.axis=1", "normalises over axis")]
    [InlineData("config.layers.6.config.alpha_regularizer={\"class_name\": \"L2\", \"config\": {\"l2\": 0.01}}", "it penalises or bounds its shares")]
    [InlineData("config.layers.6.config.alpha_constraint={\"class_name\": \"NonNeg\", \"config\": {}}", "it penalises or bounds its shares")]
    public void WhatTheNewWordsCannotSay_IsRefusedWhereTheFileSaysIt(string edit, string said)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-rows.keras", edit)));

        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("config.layers.5.config.axis=-1")]
    [InlineData("config.layers.5.config.axis=[-1]")]
    [InlineData("config.layers.5.config.axis=-")]
    public void ASoftmaxOverTheLastAxis_IsRead_HoweverItSaysSo(string edit)
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-images.keras", edit));

        Assert.Contains(Assert.IsType<LayerStack>(read.Network).Layers, layer => layer is Softmax);
    }

    [Fact]
    public void ASoftmaxOverAnotherAxis_IsRefusedNamingIt()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-images.keras", "config.layers.5.config.axis=1")));

        Assert.Contains("it shares out axis 1, and a softmax here shares out the last axis", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("\"mae\"", typeof(MeanAbsoluteError))]
    [InlineData("\"mean_absolute_error\"", typeof(MeanAbsoluteError))]
    [InlineData("{\"class_name\": \"Huber\", \"config\": {\"delta\": 0.5}}", typeof(Huber))]
    [InlineData("\"poisson\"", typeof(Poisson))]
    [InlineData("\"hinge\"", typeof(Hinge))]
    [InlineData("{\"class_name\": \"BinaryFocalCrossentropy\", \"config\": {\"gamma\": 3.0, \"alpha\": 0.4, \"apply_class_balancing\": true, \"from_logits\": true}}", typeof(Focal))]
    public void TheLossesKerasNamesThatAreNowHere_AreReadWithTheirSettings(string loss, Type kind)
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", $"compile_config.loss={loss}", "config.layers.2.config.activation=\"linear\""));

        Assert.IsType(kind, read.Loss);

        switch (read.Loss)
        {
            case Huber huber:
                Assert.Equal(0.5, huber.Delta);
                break;
            case Poisson poisson:
                Assert.False(poisson.LogInput);
                Assert.Equal(1e-7, poisson.Epsilon);
                break;
            case Focal focal:
                Assert.Equal((3.0, (double?)0.4), (focal.Gamma, focal.Alpha));
                break;
        }
    }

    [Fact]
    public void ACrossEntropyKerasSmoothsTheLabelsOf_IsReadSmoothingThemAlike()
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"label_smoothing\": 0.1}}"));

        Assert.Equal(0.1, Assert.IsType<BinaryCrossEntropy>(read.Loss).Smoothing);
        Assert.Contains(
            "labels are softened by a share of nothing or more, below one",
            Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", "compile_config.loss={\"class_name\": \"BinaryCrossentropy\", \"config\": {\"label_smoothing\": 1.5}}"))).Message,
            StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("config.layers.1.config.mask_zero=true", "mask_zero")]
    [InlineData("config.layers.0.config.batch_shape=[null, 2]", "one category's place")]
    [InlineData("config.layers.1.config.lora_rank=2", "its vectors are low-rank factors, lora_rank 2")]
    public void AnEmbeddingTheNetworkCannotTake_IsRefused(string edit, string said)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-vocabulary-embedding.keras", edit)));

        Assert.Contains(said, refused.Message, StringComparison.Ordinal);
    }
}
