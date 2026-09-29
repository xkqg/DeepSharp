// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What turns a gradient into a change: every parameter moved one step against its gradient, by PyTorch's own formulas and
/// defaults. Three steps of each optimizer over the walked rows land where PyTorch's land, parameter by parameter.
/// </summary>
public class OptimizerTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void Adam_OverThreeStepsOnTitanicsPassengers_MovesEveryParameterAsPyTorchsAdamDoes()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var adam = new Adam(0.01);

        var first = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6452890634536743, first, 1e-7);
        AssertClose([-0.30000001192092896, 0.20999999344348907, 0.009999997913837433, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21000000834465027], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.09000000357627869, 0.05000000074505806, 0.0], hidden.Bias.Value);
        Assert.Equal(-0.25f, hidden.Weight.Value.Values[0]);

        var second = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6404138207435608, second, 1e-6);
        AssertClose([-0.30000001192092896, 0.21991899609565735, 0.019997578114271164, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21999403834342957], output.Bias.Value);

        var third = Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());
        var moments = adam.MomentsOf(output.Weight);

        Assert.Equal(0.6339789628982544, third, 1e-6);
        AssertClose([-0.30000001192092896, 0.2296532243490219, 0.03001883253455162, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.2299758344888687], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.07021691650152206, 0.06573928892612457, 0.0], hidden.Bias.Value);
        AssertClose([0.10000000149011612, 0.22021692991256714, -0.1657392978668213, 0.0], Tensor.From(new Shape(4), hidden.Weight.Value.Values[4..8]));
        AssertClose([0.0, 0.1762383133172989, -0.23425878584384918, -0.10000000149011612], Tensor.From(new Shape(4), hidden.Weight.Value.Values[28..32]));
        Assert.Equal(3, moments.Steps);
        AssertClose([0.0, -0.0063126543536782265, -0.03658103942871094, 0.0], moments.Average);
        AssertClose([0.0, 1.7162035419460153e-06, 5.43771093362011e-05, 0.0], moments.SquaredAverage);
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
        AssertClose([-0.06058267503976822, 0.025160744786262512, 0.008257267996668816], sgd.VelocityOf(output.Weight));

        var third = Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

        Assert.Equal(0.003592291148379445, third, 1e-8);
        AssertClose([-0.18247266113758087, -0.007548476569354534, 0.19756300747394562], output.Weight.Value);
        AssertClose([-0.015407652594149113], output.Bias.Value);
        AssertClose([-0.08165434002876282, 0.0374990813434124, 0.011757144704461098], sgd.VelocityOf(output.Weight));
    }

    [Fact]
    public void SgdWithoutMomentum_MovesEachParameterByTheRateTimesItsGradient()
    {
        var parameter = new Dense(Tensor.From(new Shape(1, 2), [1f, -2f]), Tensor.Zeros(new Shape(2))).Weight;
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance)
        {
            [parameter.Value] = Tensor.From(new Shape(1, 2), [0.2f, -0.4f]),
        });

        new Sgd().Step([parameter], gradients, 0.5, _backend);

        AssertClose([0.8999999761581421, -1.7999999523162842], parameter.Value);
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
    public void ARecordingBackendHandedToAStep_RecordsNothing_ForAnOptimizersArithmeticIsNoPartOfAPass()
    {
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var recording = new RecordingBackend(_backend);
        var loss = new MeanSquaredError().Of(
            output.Forward(Tensor.From(new Shape(1, 4), [1f, 2f, 3f, 4f]), Pass.Training(recording, new RandomStream(1), 0, 0)),
            Tensor.Zeros(new Shape(1, 1)),
            recording);
        var gradients = recording.GradientsOf(loss, output.Parameters().Select(parameter => parameter.Value));
        var recorded = recording.Operations;

        new Adam().Step(output.Parameters(), gradients, 0.01, recording);

        Assert.Equal(recorded, recording.Operations);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ARateThatIsNotANumberAboveNothing_IsRefused(double rate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd(rate));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam(rate));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd().Step([], new Gradients(new Dictionary<Tensor, Tensor>()), rate, _backend));
    }

    [Fact]
    public void AMomentumBelowNothing_BetasOutsideNothingToOne_OrAnEpsilonOfNothing_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd { Momentum = -0.1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(1, 0.999) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(0.9, -0.1) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Epsilon = 0 });
    }

    [Fact]
    public void AStep_NeedsItsParametersItsGradientsAndABackend()
    {
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>());

        Assert.Throws<ArgumentNullException>(() => new Sgd().Step(null!, gradients, 0.1, _backend));
        Assert.Throws<ArgumentNullException>(() => new Sgd().Step([], null!, 0.1, _backend));
        Assert.Throws<ArgumentNullException>(() => new Sgd().Step([], gradients, 0.1, null!));
    }

    [Fact]
    public void AParameterWhoseGradientWasNotWorkedOut_IsRefused()
    {
        var parameter = new Dense(Tensor.Zeros(new Shape(1, 2)), Tensor.Zeros(new Shape(2))).Weight;

        Assert.Throws<ArgumentException>(() => new Adam().Step([parameter], new Gradients(new Dictionary<Tensor, Tensor>()), 0.1, _backend));
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

    [Fact]
    public void Adam_RemembersNothingOfAParameterItNeverMoved_AndRefusesAMemoryOfNoStep()
    {
        var weight = new Dense(1, 1, new RandomStream(1).Draw("initialise:test", 0, 0)).Weight;
        var adam = new Adam();

        Assert.Null(adam.MemoryOf(weight));

        var refused = Assert.Throws<ArgumentException>(() => adam.Recall(weight, new SlotMemory(0, new Dictionary<string, Tensor>())));

        Assert.Contains("counts no step", refused.Message, StringComparison.Ordinal);
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
