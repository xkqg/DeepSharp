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
    public void AdamW_OverThreeStepsOnTitanicsPassengers_DecaysEveryParameterBeforeItsStep_AsPyTorchsAdamWDoes()
    {
        // Decoupled decay: every parameter is scaled by one less the rate times the decay before Adam's step — the ones whose
        // gradient is nothing too, which is where the decay alone shows (the first hidden weight, and the first output weight).
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var adamW = new AdamW(0.01) { WeightDecay = 0.1 };

        var first = Stepped(network, adamW, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6452890634536743, first, AgreementContract.Bound(WalkedRows.SurvivedLossTermsSize()));
        AssertClose([-0.2997000217437744, 0.20979999005794525, 0.009999997913837433, -0.19979999959468842], output.Weight.Value);
        AssertClose([0.20980000495910645], output.Bias.Value);
        AssertClose([0.09989999979734421, -0.08990000188350677, 0.049949999898672104, 0.0], hidden.Bias.Value);
        AssertClose([-0.24975000321865082, -0.10989999771118164, 0.049949999898672104, 0.19979999959468842], Slice(hidden.Weight.Value, 0));

        Assert.Equal(0.6404612064361572, Stepped(network, adamW, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.29940032958984375, 0.2195095270872116, 0.019987327978014946, -0.1996002048254013], output.Weight.Value);
        AssertClose([0.21958433091640472], output.Bias.Value);

        Assert.Equal(0.6340847015380859, Stepped(network, adamW, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.29910093545913696, 0.22902485728263855, 0.02998826466500759, -0.19940060377120972], output.Weight.Value);
        AssertClose([0.22934679687023163], output.Bias.Value);
        AssertClose([0.09970030188560486, -0.06994648277759552, 0.06558232754468918, 0.0], hidden.Bias.Value);
        AssertClose([-0.24925076961517334, -0.129564568400383, 0.03411778807640076, 0.19940060377120972], Slice(hidden.Weight.Value, 0));
        AssertClose([0.09970030188560486, 0.219496950507164, -0.16528265178203583, 0.0], Slice(hidden.Weight.Value, 4));
        AssertClose([0.0, 0.17576391994953156, -0.23351669311523438, -0.09970030188560486], Slice(hidden.Weight.Value, 28));
    }

    [Fact]
    public void RmsProp_OverThreeStepsOnTitanicsPassengers_MovesEveryParameterAsPyTorchsRmsPropDoes()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var rmsProp = new RmsProp(0.001);

        var first = Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6452890634536743, first, AgreementContract.Bound(WalkedRows.SurvivedLossTermsSize()));
        AssertClose([-0.30000001192092896, 0.2099999338388443, 0.00999999325722456, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.20999999344348907], output.Bias.Value);

        Assert.Equal(0.6404138207435608, Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.21829022467136383, 0.01705736666917801, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21701504290103912], output.Bias.Value);

        Assert.Equal(0.635928213596344, Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.22579747438430786, 0.023344002664089203, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.22270770370960236], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.07363978028297424, 0.0686170905828476, 0.0], hidden.Bias.Value);
        AssertClose([0.10000000149011612, 0.22363977134227753, -0.16861708462238312, 0.0], Slice(hidden.Weight.Value, 4));
        AssertClose([0.0, 0.17318235337734222, -0.231388658285141, -0.10000000149011612], Slice(hidden.Weight.Value, 28));
    }

    [Fact]
    public void RmsPropWithMomentum_OverThreeStepsOnTitanicsPassengers_CarriesTheBufferPyTorchsRmsPropCarries()
    {
        // PyTorch's buffer starts at nothing, so the first step is the one without momentum, and the second and third go further.
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var rmsProp = new RmsProp(0.001) { Momentum = 0.9 };

        Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        AssertClose([-0.30000001192092896, 0.2099999338388443, 0.00999999325722456, -0.20000000298023224], output.Weight.Value);

        Assert.Equal(0.6404138207435608, Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.22729015350341797, 0.02605736255645752, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.22601503133773804], output.Bias.Value);

        Assert.Equal(0.6302587389945984, Stepped(network, rmsProp, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.25123223662376404, 0.04674419015645981, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.24600830674171448], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.051209475845098495, 0.07831361889839172, 0.0], hidden.Bias.Value);
        AssertClose([-0.25, -0.14844974875450134, 0.021686876192688942, 0.20000000298023224], Slice(hidden.Weight.Value, 0));
        AssertClose([0.0, 0.18850119411945343, -0.2216913402080536, -0.10000000149011612], Slice(hidden.Weight.Value, 28));
    }

    [Fact]
    public void Nadam_OverThreeStepsOnTitanicsPassengers_MovesEveryParameterAsPyTorchsNAdamDoes()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var nadam = new Nadam(0.01);

        var first = Stepped(network, nadam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());

        Assert.Equal(0.6452890634536743, first, AgreementContract.Bound(WalkedRows.SurvivedLossTermsSize()));
        AssertClose([-0.30000001192092896, 0.2105645090341568, 0.010564515367150307, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.210564523935318], output.Bias.Value);

        Assert.Equal(0.64013671875, Stepped(network, nadam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.2196127325296402, 0.01836865209043026, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.21832452714443207], output.Bias.Value);

        Assert.Equal(0.6349390745162964, Stepped(network, nadam, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived()), 1e-6);
        AssertClose([-0.30000001192092896, 0.22891035676002502, 0.02621779777109623, -0.20000000298023224], output.Weight.Value);
        AssertClose([0.2255161851644516], output.Bias.Value);
        AssertClose([0.10000000149011612, -0.0707540288567543, 0.07057595998048782, 0.0], hidden.Bias.Value);
        AssertClose([-0.25, -0.12895378470420837, 0.029424497857689857, 0.20000000298023224], Slice(hidden.Weight.Value, 0));
        AssertClose([0.0, 0.17604176700115204, -0.22942867875099182, -0.10000000149011612], Slice(hidden.Weight.Value, 28));
    }

    [Fact]
    public void TheDefaults_ArePyTorchs()
    {
        var sgd = new Sgd();
        var adam = new Adam();
        var adamW = new AdamW();
        var rmsProp = new RmsProp();
        var nadam = new Nadam();

        Assert.Equal(0.001, sgd.Rate);
        Assert.Equal(0, sgd.Momentum);
        Assert.Equal(0.001, adam.Rate);
        Assert.Equal(new Betas(0.9, 0.999), adam.Betas);
        Assert.Equal(1e-8, adam.Epsilon);
        Assert.Equal(0.001, adamW.Rate);
        Assert.Equal(new Betas(0.9, 0.999), adamW.Betas);
        Assert.Equal(1e-8, adamW.Epsilon);
        Assert.Equal(0.01, adamW.WeightDecay);
        Assert.Equal(0.01, rmsProp.Rate);
        Assert.Equal(0.99, rmsProp.Alpha);
        Assert.Equal(1e-8, rmsProp.Epsilon);
        Assert.Equal(0, rmsProp.Momentum);
        Assert.Equal(0.002, nadam.Rate);
        Assert.Equal(new Betas(0.9, 0.999), nadam.Betas);
        Assert.Equal(1e-8, nadam.Epsilon);
        Assert.Equal(0.004, nadam.MomentumDecay);
    }

    [Fact]
    public void AMomentumBelowNothing_BetasOutsideNothingToOne_OrAnEpsilonOfNothing_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sgd { Momentum = -0.1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(1, 0.999) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Betas = new Betas(0.9, -0.1) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Adam { Epsilon = 0 });
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ADecayBelowNothingOrNoFiniteNumber_IsRefused_AsIsEverySettingTheNewOptimizersHoldOutsideItsRange(double wrong)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdamW { WeightDecay = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdamW { Epsilon = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdamW { Betas = new Betas(wrong, 0.999) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp { Alpha = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp { Epsilon = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp { Momentum = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Nadam { MomentumDecay = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Nadam { Epsilon = wrong });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Nadam { Betas = new Betas(0.9, wrong) });
        Assert.Throws<ArgumentOutOfRangeException>(() => new AdamW(wrong));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp(wrong));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Nadam(wrong));
    }

    [Fact]
    public void AnAlphaOfOne_AnEpsilonOfNothing_AndADecayOfNothing_AreTold_TheOneTwoRefusedAndTheThirdKept()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp { Alpha = 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsProp { Epsilon = 0 });
        Assert.Equal(0, new AdamW { WeightDecay = 0 }.WeightDecay);
        Assert.Equal(0, new Nadam { MomentumDecay = 0 }.MomentumDecay);
        Assert.Equal(0, new RmsProp { Alpha = 0 }.Alpha);
    }

    [Fact]
    public void AdamWWithNoDecay_StepsAsAdamDoes_BitForBit()
    {
        // One arithmetic for the running means, kept in one place: AdamW without its decay is Adam to the last bit.
        var adam = Walked(new Adam(0.01));
        var adamW = Walked(new AdamW(0.01) { WeightDecay = 0 });

        Assert.Equal(adam, adamW);
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

    // Every parameter of the Titanic walk after three steps of an optimizer, as the bits of its floats.
    private int[] Walked(Optimizer optimizer)
    {
        var network = new LayerStack(new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

        for (var step = 0; step < 3; step++)
        {
            Stepped(network, optimizer, new BinaryCrossEntropy(), WalkedRows.Passengers(), WalkedRows.Survived());
        }

        return [.. network.Parameters().SelectMany(parameter => parameter.Value.Values.ToArray()).Select(BitConverter.SingleToInt32Bits)];
    }

    // Four hidden weights from a place, in the inputs-by-outputs order the layer keeps them in.
    private static Tensor Slice(Tensor weights, int from) => Tensor.From(new Shape(4), weights.Values[from..(from + 4)]);

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-9, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
