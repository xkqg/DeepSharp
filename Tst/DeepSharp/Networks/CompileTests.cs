// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network is compiled with the loss it is trained to bring down, and a loss that applies an activation to the network's
/// outputs itself is not handed a network that applies it too: a stack whose last layer is that activation is refused where
/// the two are put together, naming the layer. Only a stack says what its last layer is; a network written as code writes
/// its own forward pass, which nothing here reads, so it is compiled as it is written.
/// </summary>
public class CompileTests
{
    [Fact]
    public void AStackEndingInTheSigmoidItsLossAppliesItself_IsRefusedAtCompile_NamingTheLayer()
    {
        // 0.4.0 compiled it, and every prediction went through the sigmoid twice: logits of -20, -1, 1 and 20 came out
        // 0.5000, 0.5668, 0.6750 and 0.7311, every one a half or more.
        var stack = new LayerStack(Identity(), new Sigmoid());

        var wrong = Assert.Throws<ArgumentException>(() => stack.Compile(new Adam(), new BinaryCrossEntropy()));

        Assert.Equal("loss", wrong.ParamName);
        Assert.Equal(
            "BinaryCrossEntropy applies the sigmoid to the network's outputs itself, and this network's last layer, '1', is a Sigmoid: "
            + "every prediction would go through the sigmoid twice. Leave that layer out; the predictions still come out through the loss's sigmoid. (Parameter 'loss')",
            wrong.Message);
    }

    [Fact]
    public void AStackWhoseLastLayerIsAStack_IsReadDownToItsLastLayer()
    {
        var stack = new LayerStack(Identity(), new LayerStack(new Relu(), new Sigmoid()));

        var wrong = Assert.Throws<ArgumentException>(() => stack.Compile(new Sgd(), new BinaryCrossEntropy()));

        Assert.Contains("last layer, '1.1', is a Sigmoid", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALastSigmoid_IsCompiledWithALossThatDoesNotApplyIt_AndASigmoidBeforeTheLastLayer_WithOneThatDoes()
    {
        Assert.IsType<MeanSquaredError>(new LayerStack(Identity(), new Sigmoid()).Compile(new Adam(), new MeanSquaredError()).Loss);
        Assert.IsType<CrossEntropy>(new LayerStack(Identity(), new Sigmoid()).Compile(new Adam(), new CrossEntropy()).Loss);
        Assert.IsType<BinaryCrossEntropy>(new LayerStack(new Sigmoid(), Identity()).Compile(new Adam(), new BinaryCrossEntropy()).Loss);
    }

    [Fact]
    public void ANetworkWrittenAsCode_IsCompiledAsItIsWritten()
    {
        // Its forward pass is its own code: nothing here runs it, or guesses at it, to find what it ends in.
        var network = new EndsInASigmoid();

        Assert.Same(network, network.Compile(new Adam(), new BinaryCrossEntropy()).Network);
    }

    private static Dense Identity() => new(Tensor.From(new Shape(1, 1), [1f]), Tensor.Zeros(new Shape(1)));

    private sealed class EndsInASigmoid : Network
    {
        private readonly Dense _dense;

        public EndsInASigmoid() => _dense = AddLayer("dense", Identity());

        protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Sigmoid(_dense.Forward(input, pass));
    }
}
