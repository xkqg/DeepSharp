// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The linear layer and the activations that bend it, run over the rows the published samples hand over, against the
/// numbers PyTorch gives for the same rows and weights.
/// </summary>
public class DenseTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void TitanicsFirstFourPassengers_ThroughDenseReluDense_ReachPyTorchsLogits()
    {
        var network = new LayerStack(
            new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()),
            new Relu(),
            new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

        var logits = network.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend));

        Assert.Equal(new Shape(4, 1), logits.Shape);
        AssertClose([0.20000000298023224, 0.20000000298023224, 0.22550064325332642, 0.20000000298023224], logits);
    }

    [Fact]
    public void ThePricesFirstFourDays_ThroughDenseTanhDense_ReachPyTorchsPredictions()
    {
        var network = new LayerStack(
            new Dense(WalkedRows.TanhWeights(), WalkedRows.TanhBias()),
            new Tanh(),
            new Dense(WalkedRows.ReturnWeights(), WalkedRows.ReturnBias()));

        var predictions = network.Forward(WalkedRows.Days(), Pass.Evaluation(_backend));

        AssertClose([-0.015954021364450455, -0.019673071801662445, 0.03601423650979996, 0.11015262454748154], predictions);
    }

    [Fact]
    public void Sigmoid_BendsEveryValueBetweenNothingAndOne()
    {
        var bent = new Sigmoid().Forward(Tensor.From(new Shape(1, 2), [0f, 0.2f]), Pass.Evaluation(_backend));

        AssertClose([0.5, 0.549834], bent);
    }

    [Fact]
    public void ADense_KeepsItsWeightAndBiasAsTheSlotsPyTorchNamesThem()
    {
        var dense = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());

        Assert.Equal(["weight", "bias"], dense.Slots().Select(slot => slot.Path));
        Assert.Equal(14, dense.Inputs);
        Assert.Equal(4, dense.Outputs);
    }

    [Fact]
    public void ADenseDrawnFromTheRunsStream_StartsAsPyTorchsLinearLayerStarts()
    {
        var dense = new Dense(14, 16, new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(new Shape(14, 16), dense.Weight.Value.Shape);
        Assert.Equal(new Shape(16), dense.Bias.Value.Shape);
        Assert.All(dense.Weight.Value.Values.ToArray(), value => Assert.InRange(value, -0.267261f, 0.267261f));
        Assert.All(dense.Bias.Value.Values.ToArray(), value => Assert.InRange(value, -0.267261f, 0.267261f));
        Assert.Contains(dense.Bias.Value.Values.ToArray(), value => value != 0);
    }

    [Fact]
    public void ADense_CanStartItsWeightsAsAnotherInitialiserHasIt()
    {
        var dense = new Dense(3, 2, new RandomStream(42).Draw("initialise:0", 0, 0), new Zeros());

        Assert.All(dense.Weight.Value.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void TheSameSeed_StartsTheSameDense()
    {
        var first = new Dense(14, 16, new RandomStream(42).Draw("initialise:0", 0, 0));
        var again = new Dense(14, 16, new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(first.Weight.Value.Values.ToArray(), again.Weight.Value.Values.ToArray());
        Assert.Equal(first.Bias.Value.Values.ToArray(), again.Bias.Value.Values.ToArray());
    }

    [Fact]
    public void ADenseOfNoInputsOrNoOutputs_OrWhoseBiasIsNotAsLongAsItHasOutputs_IsRefused()
    {
        var draws = new RandomStream(42).Draw("initialise:0", 0, 0);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(0, 2, draws));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dense(2, 0, draws));
        Assert.Throws<ArgumentException>(() => new Dense(Tensor.Zeros(new Shape(3, 2)), Tensor.Zeros(new Shape(3))));
        Assert.Throws<ArgumentException>(() => new Dense(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(3))));
    }

    [Fact]
    public void RowsOfAnotherWidthThanADenseReads_AreRefusedByTheBackend()
    {
        var dense = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());

        Assert.Throws<ArgumentException>(() => dense.Forward(Tensor.Zeros(new Shape(2, 3)), Pass.Evaluation(_backend)));
    }

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-6, Math.Abs(expected[at]) * 1e-6));
        }
    }
}
