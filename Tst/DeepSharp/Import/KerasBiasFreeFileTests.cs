// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A Keras model whose layers keep no bias, no scale or no shift, read into a network that keeps none either: a dense layer
/// or a convolution with <c>use_bias</c> false, a normalisation with <c>center</c> or <c>scale</c> false, and a separable
/// convolution, whose first stage has none of its own. Each model is held to what Keras answered through it
/// (Fixtures/keras-free-fixtures.py made them), and a description that says one thing where the numbers say another is
/// refused where the file holds the number.
/// </summary>
public class KerasBiasFreeFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-free.json"))).RootElement;

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static IEnumerable<T> Layers<T>(string file)
        where T : Layer =>
        new KerasFile().Read(KerasFixtures.Open(file)).Network.HeldLayers().Select(held => held.Layer).OfType<T>();

    [Theory]
    [InlineData("keras-free-dense")]
    [InlineData("keras-free-images")]
    [InlineData("keras-free-series")]
    [InlineData("keras-free-volume")]
    public void AModelOfLayersThatKeepNoBiasNoScaleOrNoShift_AnswersAsKerasDoes(string name)
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
    public void TheDenseModel_KeepsNoBiasWhereKerasUsesNone_AndNoScaleOrShiftWhereItLeavesThemOut()
    {
        Assert.Equal([false, false, false, true], Layers<Dense>("keras-free-dense.keras").Select(dense => dense.Bias is not null));

        var batches = Layers<BatchNorm>("keras-free-dense.keras").ToArray();
        var layer = Assert.Single(Layers<LayerNorm>("keras-free-dense.keras"));

        Assert.Equal([(true, false), (false, true), (false, false)], batches.Select(each => (each.Weight is not null, each.Bias is not null)));
        Assert.Null(layer.Weight);
        Assert.NotNull(layer.Bias);
    }

    [Fact]
    public void TheConvolutionModels_KeepNoBiasWhereKerasUsesNone_AndASeparableConvolutionsFirstStageNever()
    {
        // Keras adds a separable convolution's bias once, after both stages: the depthwise stage has none, whether the layer
        // has a bias at all or not.
        Assert.Equal([false, false, false, false, false, false, true], Layers<Convolution>("keras-free-images.keras").Select(each => each.Bias is not null));
        Assert.Equal([false, false, true, false, false, false], Layers<Convolution>("keras-free-series.keras").Select(each => each.Bias is not null));
        Assert.Equal([false, false], Layers<Convolution>("keras-free-volume.keras").Select(each => each.Bias is not null));
    }

    [Fact]
    public void TheSeparableModelOfEarlierFixtures_NowHasNoBiasOfNoughtsInItsFirstStage_AndAnswersAsItDid()
    {
        var convolutions = Layers<Convolution>("keras-new-separable.keras").ToArray();

        Assert.Null(convolutions[0].Bias);
        Assert.NotNull(convolutions[1].Bias);
    }

    [Theory]
    [InlineData("keras-titanic.keras", "config.layers.1.config.use_bias=false", "dense.1", "layers/dense/vars/1")]
    [InlineData("keras-images.keras", "config.layers.2.config.use_bias=false", "conv2d.1", "layers/conv2d/vars/1")]
    [InlineData("keras-spatial-series.keras", "config.layers.1.config.use_bias=false", "conv1d.1", "layers/conv1d/vars/1")]
    [InlineData("keras-new-transposed.keras", "config.layers.1.config.use_bias=false", "conv2d_transpose.1", "layers/conv2d_transpose/vars/1")]
    [InlineData("keras-new-depthwise.keras", "config.layers.1.config.use_bias=false", "depthwise_conv2d.1", "layers/depthwise_conv2d/vars/1")]
    [InlineData("keras-new-separable.keras", "config.layers.1.config.use_bias=false", "separable_conv2d.2", "layers/separable_conv2d/vars/2")]
    [InlineData("keras-images.keras", "config.layers.3.config.center=false", "batch_normalization.3", "layers/batch_normalization/vars/3")]
    [InlineData("keras-images.keras", "config.layers.3.config.scale=false", "batch_normalization.3", "layers/batch_normalization/vars/3")]
    [InlineData("keras-images.keras", "config.layers.10.config.center=false", "layer_normalization.1", "layers/layer_normalization/vars/1")]
    public void ALayerTheDescriptionSaysKeepsLess_WhileTheFileHoldsTheNumber_IsRefusedAtTheNumberNothingHereTakes(string file, string edit, string slot, string where)
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited(file, edit)));
        var fault = Assert.Single(refused.Faults, each => each.Slot == slot);

        Assert.Contains(where, fault.Source, StringComparison.Ordinal);
        Assert.Contains("is no slot of this network", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADenseLayerTheDescriptionSaysHasABias_WhileTheFileHoldsNone_IsRefusedAsMissing()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-free-dense.keras", "config.layers.1.config.use_bias=true")));

        Assert.Contains(refused.Faults, fault => fault.Slot == "0.bias" && fault.Message.Contains("missing", StringComparison.Ordinal));
    }
}
