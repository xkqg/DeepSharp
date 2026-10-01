// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What turns a gradient into a change: every parameter moved one step against its gradient, by PyTorch's own formulas and
/// defaults. Three steps of each optimizer over the walked rows land where PyTorch's land, parameter by parameter, on any
/// engine.
/// </summary>
/// <param name="engine">The backend the steps run on.</param>
public abstract class OptimizerContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void Adam_OverThreeStepsOnTitanicsPassengers_MovesEveryParameterAsPyTorchsAdamDoes()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var adam = new Adam(0.01);

        var first = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6452890634536743, first, AgreementContract.Bound(WalkedRows.SurvivedLossTermsSize()));
        AssertClose([-0.30000001192092896, 0.20999999344348907, 0.009999997913837433, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21000000834465027], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.09000000357627869, 0.05000000074505806, 0.0], hidden.Bias.Value);
        Assert.Equal(-0.25f, hidden.Weight.Value.Values[0]);

        var second = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6404138207435608, second, 1e-6);
        AssertClose([-0.30000001192092896, 0.21991899609565735, 0.019997578114271164, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21999403834342957], output.Bias.Value);

        var third = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6339789628982544, third, 1e-6);
        AssertClose([-0.30000001192092896, 0.2296532243490219, 0.03001883253455162, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.2299758344888687], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.07021691650152206, 0.06573928892612457, 0.0], hidden.Bias.Value);
        AssertClose([0.10000000149011612, 0.22021692991256714, -0.1657392978668213, 0.0], Tensor.From(new Shape(4), hidden.Weight.Value.Values[4..8]));
        AssertClose([0.0, 0.1762383133172989, -0.23425878584384918, -0.10000000149011612], Tensor.From(new Shape(4), hidden.Weight.Value.Values[28..32]));
    }

    [Fact]
    public void SgdWithMomentum_OverThreeStepsOnThePricesDays_MovesEveryParameterAsPyTorchsSgdDoes()
    {
        var hidden = new Dense(WalkedRows.TanhWeights(), WalkedRows.TanhBias());
        var output = new Dense(WalkedRows.ReturnWeights(), WalkedRows.ReturnBias());
        var network = new LayerStack(hidden, new Tanh(), output);
        var sgd = new Sgd(0.1) { Momentum = 0.9 };

        Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

        AssertClose([-0.19669635593891144, -0.0012824939331039786, 0.19956445693969727], output.Weight.Value);
        AssertClose([-0.003436941420659423], output.Bias.Value);

        var second = Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

        Assert.Equal(0.004027019254863262, second, 1e-8);
        AssertClose([-0.19063809514045715, -0.003798568621277809, 0.1987387239933014], output.Weight.Value);
        AssertClose([-0.009116374887526035], output.Bias.Value);

        var third = Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

        Assert.Equal(0.003592291148379445, third, 1e-8);
        AssertClose([-0.18247266113758087, -0.007548476569354534, 0.19756300747394562], output.Weight.Value);
        AssertClose([-0.015407652594149113], output.Bias.Value);
    }

    [Fact]
    public void TheDefaults_ArePyTorchs()
    {
        var sgd = new Sgd();
        var adam = new Adam();

        Assert.Equal(0.001, sgd.Rate);
        Assert.Equal(0, sgd.Momentum);
        Assert.Equal(0.001, adam.Rate);
        Assert.Equal(new Betas(0.9, 0.999), adam.Betas);
        Assert.Equal(1e-8, adam.Epsilon);
    }

    [Fact]
    public void AMomentumBelowNothing_BetasOutsideNothingToOne_OrAnEpsilonOfNothing_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd { Momentum = -0.1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(1, 0.999) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(0.9, -0.1) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Epsilon = 0 });
    }

    // One training step: the batch forward through a recording backend, the loss, its gradients, and the optimizer's step.
    private double Stepped(Network network, Optimizer optimizer, Loss loss, Tensor rows, Tensor answers)
    {
        var recording = new RecordingBackend(_backend);
        var value = loss.Of(network.Forward(rows, Pass.Training(recording, new RandomStream(1), 0, 0)), answers, recording);
        var gradients = recording.GradientsOf(value, network.Parameters().Select(parameter => parameter.Value));

        optimizer.Step(network.Parameters(), gradients, optimizer.Rate, _backend);

        return value.Values[0];
    }

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-9, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
