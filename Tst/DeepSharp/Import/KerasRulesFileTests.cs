// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A Keras model whose layers carry regularizers, MaxNorm constraints and a switch that keeps them from training, read into
/// a network whose parameters carry the same rules (Fixtures/keras-rules-fixtures.py made them): the penalty of each
/// regularizer on the number it is for, the bound of each MaxNorm on its unit's norm, and every layer that does not train held
/// still — and answering as Keras answers through them. What Keras can say that nothing here does is refused by name.
/// </summary>
public class KerasRulesFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-rules.json"))).RootElement;

    private static IEnumerable<T> Layers<T>(string file, params string[] edits)
        where T : Layer =>
        new KerasFile().Read(edits.Length == 0 ? KerasFixtures.Open(file) : KerasFixtures.Edited(file, edits)).Network.HeldLayers().Select(held => held.Layer).OfType<T>();

    // The rules a layer carries, to the single precision Keras writes its numbers in.
    private static void Rules(SlotRules expected, SlotRules actual)
    {
        Assert.Equal(expected.L1, actual.L1, 6);
        Assert.Equal(expected.L2, actual.L2, 6);
        Assert.Equal(expected.MaxNorm ?? 0, actual.MaxNorm ?? 0, 6);
        Assert.Equal(expected.Held, actual.Held);
    }

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    [Theory]
    [InlineData("keras-rules-dense")]
    [InlineData("keras-rules-images")]
    public void AModelWithRules_AnswersAsKerasDoes_SinceRulesChangeNothingOfWhatItAnswers(string name)
    {
        var expected = Answers.GetProperty(name);
        var written = expected.GetProperty("examples");
        var examples = Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written.GetProperty("values")));

        var read = new KerasFile().Read(KerasFixtures.Open($"{name}.keras"));
        var here = read.Network.Predict(examples, read.Loss, Engine).Values.ToArray();
        var keras = Values(expected.GetProperty("outputs"));

        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void TheDenseModel_CarriesTheRegularizersTheConstraintsAndTheSwitchesOfEachLayer()
    {
        var dense = Layers<Dense>("keras-rules-dense.keras").ToArray();
        var norm = Assert.Single(Layers<BatchNorm>("keras-rules-dense.keras"));

        Rules(new SlotRules { L2 = 0.01, MaxNorm = 2 }, dense[0].Weight.Rules);
        Rules(new SlotRules { L1 = 0.02 }, dense[0].Bias!.Rules);
        Rules(new SlotRules { L1 = 0.001, L2 = 0.002 }, dense[1].Weight.Rules);
        Rules(new SlotRules { MaxNorm = 1.5 }, dense[1].Bias!.Rules);
        Rules(new SlotRules { Held = true }, dense[2].Weight.Rules);
        Rules(new SlotRules { Held = true }, dense[2].Bias!.Rules);
        Rules(default, dense[3].Weight.Rules);
        Assert.False(norm.Weight!.Learns);
        Assert.False(norm.Bias!.Learns);
    }

    [Fact]
    public void TheConvolutionModel_CarriesTheRulesOfTheKernelsAndTheBiases_OfEachStageOfASeparableConvolution()
    {
        var convolutions = Layers<Convolution>("keras-rules-images.keras").ToArray();

        // The convolution, the depthwise convolution, and the separable one's two stages.
        Rules(new SlotRules { L2 = 0.01, MaxNorm = 2 }, convolutions[0].Weight.Rules);
        Rules(new SlotRules { MaxNorm = 1 }, convolutions[0].Bias!.Rules);
        Rules(new SlotRules { L1 = 0.01, MaxNorm = 1.5 }, convolutions[1].Weight.Rules);
        Rules(new SlotRules { L1 = 0.03 }, convolutions[2].Weight.Rules);
        Assert.Null(convolutions[2].Bias);
        Rules(new SlotRules { L2 = 0.02, MaxNorm = 2 }, convolutions[3].Weight.Rules);
        Rules(new SlotRules { L2 = 0.04 }, convolutions[3].Bias!.Rules);
    }

    [Theory]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=1", "kernel_constraint", "axis")]
    [InlineData("keras-rules-images.keras", "config.layers.1.config.kernel_constraint.config.axis=0", "kernel_constraint", "axis")]
    [InlineData("keras-rules-images.keras", "config.layers.2.config.depthwise_constraint.config.axis=[0, 1, 2]", "depthwise_constraint", "axis")]
    [InlineData("keras-rules-dense.keras", "config.layers.3.config.bias_constraint.config.axis=1", "bias_constraint", "axis")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.class_name=\"UnitNorm\"", "kernel_constraint", "UnitNorm")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer.class_name=\"OrthogonalRegularizer\"", "kernel_regularizer", "OrthogonalRegularizer")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.activity_regularizer={\"class_name\": \"L2\", \"config\": {\"l2\": 0.1}}", "activity_regularizer", "makes")]
    [InlineData("keras-rules-dense.keras", "config.layers.2.config.gamma_constraint={\"class_name\": \"NonNeg\", \"config\": {}}", "gamma_constraint", "NonNeg")]

    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer={\"class_name\": \"L2\", \"config\": {\"l2\": -0.5}}", "kernel_regularizer", "at least nothing")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer={\"class_name\": \"L1L2\", \"config\": {\"l1\": 0.1, \"l2\": -1}}", "kernel_regularizer", "at least nothing")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=\"x\"", "kernel_constraint", "axis")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=0.5", "kernel_constraint", "axis")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=[0, \"a\"]", "kernel_constraint", "axis")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.max_value=0", "kernel_constraint", "above nothing")]
    [InlineData("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.max_value=-2", "kernel_constraint", "above nothing")]
    public void WhatKerasCanSayThatNothingHereDoes_IsRefusedAtTheLayerThatSaysIt_ByName(string file, string edit, string setting, string says)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(file, edit)));

        Assert.Contains(setting, refused.Message, StringComparison.Ordinal);
        Assert.Contains(says, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARegularizerAndAConstraintOfAnotherAxisThatIsTheSameAxisWrittenOtherwise_AreRead()
    {
        // Axis 0 of a dense kernel written as a list or as a negative axis (-2 of a kernel of two axes) is the same axis.
        Assert.Equal(2, Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=[0]").First().Weight.Rules.MaxNorm);
        Assert.Equal(2, Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config.axis=-2").First().Weight.Rules.MaxNorm);
        Assert.Equal(2, Layers<Convolution>("keras-rules-images.keras", "config.layers.1.config.kernel_constraint.config.axis=[-4, -3, -2]").First().Weight.Rules.MaxNorm);
    }

    [Fact]
    public void AConstraintThatNamesNoAxis_IsKerasDefaultAxis_TheFirst()
    {
        // Keras's MaxNorm bounds over axis 0 when it is not told otherwise, which is a dense kernel's units.
        var dense = Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_constraint.config={\"max_value\": 3}").First();

        Assert.Equal(3, dense.Weight.Rules.MaxNorm);
    }

    [Fact]
    public void ARegularizerKerasLeavesItsNumberOfInOnlyItsClass_IsReadAsKerasDefaultsIt()
    {
        var l1 = Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer={\"class_name\": \"L1\", \"config\": {}}").First();
        var l2 = Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer={\"class_name\": \"L2\", \"config\": {}}").First();
        var both = Layers<Dense>("keras-rules-dense.keras", "config.layers.1.config.kernel_regularizer={\"class_name\": \"L1L2\", \"config\": {}}").First();

        Assert.Equal(0.01, l1.Weight.Rules.L1);
        Assert.Equal(0.01, l2.Weight.Rules.L2);
        Assert.Equal(0, both.Weight.Rules.L1);
        Assert.Equal(0, both.Weight.Rules.L2);
    }

    [Fact]
    public void AMaxNormOnATransposedConvolution_IsRefusedByName_SinceItsKernelIsNotLaidOutByUnit()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(
            "keras-new-transposed.keras", "config.layers.1.config.kernel_constraint={\"class_name\": \"MaxNorm\", \"config\": {\"max_value\": 2, \"axis\": [0, 1, 2]}}")));

        Assert.Contains("kernel_constraint", refused.Message, StringComparison.Ordinal);
        Assert.Contains("transposed", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerThatDoesNotTrain_IsHeldStill_AndAModelOfLayersThatTrainIsNot()
    {
        Assert.All(Layers<Dense>("keras-titanic.keras"), layer => Assert.True(layer.Weight.Learns));
        Assert.False(Layers<Dense>("keras-titanic.keras", "config.layers.1.config.trainable=false").First().Weight.Learns);
        Assert.False(Layers<Dense>("keras-titanic.keras", "config.layers.1.config.trainable=false").First().Bias!.Learns);
    }
}
