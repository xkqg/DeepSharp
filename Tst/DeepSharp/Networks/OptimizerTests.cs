// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What an optimizer remembers between steps, and a step handed gradients by hand: the door the library keeps to itself,
/// so these run on the light engine alone. Where three steps of each optimizer land, parameter by parameter, runs on every
/// engine from the contract.
/// </summary>
public class OptimizerTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void Adam_OverThreeStepsOnTitanicsPassengers_RemembersTheStepsAndBothRunningMeansAsPyTorchsAdamDoes()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var adam = new Adam(0.01);

        for (var step = 0; step < 3; step++)
        {
            Stepped(network, adam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());
        }

        var moments = adam.MomentsOf(output.Weight);

        Assert.Equal(3, moments.Steps);
        AssertClose([0.0, -0.0063126543536782265, -0.03658103942871094, 0.0], moments.Average);
        AssertClose([0.0, 1.7162035419460153e-06, 5.43771093362011e-05, 0.0], moments.SquaredAverage);
    }

    [Fact]
    public void SgdWithMomentum_OverThreeStepsOnThePricesDays_CarriesTheVelocityPyTorchsSgdCarries()
    {
        var hidden = new Dense(WalkedRows.TanhWeights(), WalkedRows.TanhBias());
        var output = new Dense(WalkedRows.ReturnWeights(), WalkedRows.ReturnBias());
        var network = new LayerStack(hidden, new Tanh(), output);
        var sgd = new Sgd(0.1) { Momentum = 0.9 };

        Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());
        Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

        AssertClose([-0.06058267503976822, 0.025160744786262512, 0.008257267996668816], sgd.VelocityOf(output.Weight));

        Stepped(network, sgd, new MeanSquaredError(), WalkedRows.Days(), WalkedRows.Returns());

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

        Assert.Null(adam.KeptOf(weight));

        var refused = Assert.Throws<ArgumentException>(() => adam.PutBack(weight, new SlotMemory(0, new Dictionary<string, Tensor>())));

        Assert.Contains("counts no step", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AdamWAndNadam_RememberTheStepsAndBothRunningMeans_UnderPyTorchsNames()
    {
        foreach (var optimizer in new Optimizer[] { new AdamW(0.01), new Nadam(0.01) })
        {
            var output = Walked(optimizer);
            var memory = optimizer.KeptOf(output.Weight)!.Value;

            Assert.Equal(3, memory.Steps);
            Assert.Equal(["exp_avg", "exp_avg_sq"], memory.Tensors.Keys.Order(StringComparer.Ordinal));
            Assert.All(memory.Tensors.Values, tensor => Assert.Equal(output.Weight.Value.Shape, tensor.Shape));
        }
    }

    [Fact]
    public void RmsProp_RemembersTheMeanOfTheSquares_AndTheBufferOnlyWithMomentum()
    {
        var plain = new RmsProp(0.001);
        var carried = new RmsProp(0.001) { Momentum = 0.9 };

        var without = plain.KeptOf(Walked(plain).Weight)!.Value;
        var with = carried.KeptOf(Walked(carried).Weight)!.Value;

        Assert.Equal(3, without.Steps);
        Assert.Equal(["square_avg"], without.Tensors.Keys);
        Assert.Equal(["momentum_buffer", "square_avg"], with.Tensors.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void RmsPropWithoutMomentum_PutsBackTheMeanOfTheSquaresItKept_AndNoBuffer()
    {
        var kept = new RmsProp(0.001);
        var weight = Walked(kept).Weight;
        var again = new RmsProp(0.001);

        again.PutBack(weight, kept.KeptOf(weight)!.Value);

        Assert.Equal(kept.KeptOf(weight)!.Value.Tensors["square_avg"], again.KeptOf(weight)!.Value.Tensors["square_avg"]);
        Assert.Equal(["square_avg"], again.KeptOf(weight)!.Value.Tensors.Keys);
        Assert.Equal(3, again.KeptOf(weight)!.Value.Steps);
    }

    [Fact]
    public void Nadam_KeepsTheProductOfItsMomentumsAsPyTorchsNAdamDoes_AndWorksItOutAgainFromTheStepsAMemoryCounts()
    {
        // PyTorch keeps mu_product as a float of its own; it is a function of the steps alone, so a checkpoint holds the
        // steps and the product is worked out again — to the same float — where the memory is put back.
        var nadam = new Nadam(0.01);
        var weight = Walked(nadam).Weight;
        var again = new Nadam(0.01);

        again.PutBack(weight, nadam.KeptOf(weight)!.Value);

        Assert.Equal(0.09121428430080414, nadam.MuProductOf(weight), 1e-9);
        Assert.Equal(BitConverter.SingleToInt32Bits(nadam.MuProductOf(weight)), BitConverter.SingleToInt32Bits(again.MuProductOf(weight)));
    }

    [Fact]
    public void AMemoryAnotherOptimizerKept_IsRefused_NamingWhatThisOneKeeps()
    {
        var adam = new Adam(0.01);
        var weight = Walked(adam).Weight;
        var adams = adam.KeptOf(weight)!.Value;
        var rmsProp = new RmsProp(0.001) { Momentum = 0.9 };
        var squares = new RmsProp(0.001);
        var squared = squares.KeptOf(Walked(squares).Weight)!.Value;

        Assert.Contains("square_avg", Assert.Throws<ArgumentException>(() => rmsProp.PutBack(weight, adams)).Message, StringComparison.Ordinal);
        Assert.Contains("momentum_buffer", Assert.Throws<ArgumentException>(() => rmsProp.PutBack(weight, squared)).Message, StringComparison.Ordinal);
        Assert.Contains("exp_avg", Assert.Throws<ArgumentException>(() => new AdamW().PutBack(weight, squared)).Message, StringComparison.Ordinal);
        Assert.Contains("exp_avg", Assert.Throws<ArgumentException>(() => new Nadam().PutBack(weight, squared)).Message, StringComparison.Ordinal);
        Assert.Contains("counts no step", Assert.Throws<ArgumentException>(() => new Nadam().PutBack(weight, adams with { Steps = 0 })).Message, StringComparison.Ordinal);
        Assert.Contains("counts no step", Assert.Throws<ArgumentException>(() => new AdamW().PutBack(weight, adams with { Steps = 0 })).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheNewOptimizers_RememberNothingOfAParameterTheyNeverMoved()
    {
        var weight = new Dense(1, 1, new RandomStream(1).Draw("initialise:test", 0, 0)).Weight;

        Assert.Null(new AdamW().KeptOf(weight));
        Assert.Null(new RmsProp().KeptOf(weight));
        Assert.Null(new Nadam().KeptOf(weight));
    }

    // Three steps of an optimizer over the Titanic walk; the output layer it moved.
    private Dense Walked(Optimizer optimizer)
    {
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), output);

        for (var step = 0; step < 3; step++)
        {
            Stepped(network, optimizer, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());
        }

        return output;
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
