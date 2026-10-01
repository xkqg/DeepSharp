// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// What a network is trained to bring down: one value for a batch, and the activation its outputs go through on their way
/// out as predictions. Every loss, gradient and prediction is PyTorch's for the same rows and weights; every refusal is of
/// an answer the loss could not have meant.
/// </summary>
public abstract class LossContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void BinaryCrossEntropy_OnTitanicsFirstFourPassengers_IsPyTorchs_AndSoAreItsGradients()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());
        var network = new LayerStack(hidden, new Relu(), output);
        var recording = new RecordingBackend(_backend);

        var loss = new BinaryCrossEntropy().Of(
            network.Forward(WalkedRows.Passengers(), Pass.Training(recording, new RandomStream(1), 0, 0)), WalkedRows.Survived(), recording);
        var gradients = recording.GradientsOf(loss, network.Parameters().Select(parameter => parameter.Value));

        Assert.Equal(new Shape(), loss.Shape);
        Assert.Equal(0.6452890634536743, loss.Values[0], AgreementContract.Bound(WalkedRows.SurvivedLossTermsSize()));
        AssertClose([0, -0.014148475602269173, -0.13113100826740265, 0], gradients[output.Weight.Value]);
        AssertClose([-0.19859017431735992], gradients[output.Bias.Value]);
        AssertClose([0, -0.022193128243088722, 0, 0], gradients[hidden.Bias.Value]);
    }

    [Fact]
    public void BinaryCrossEntropy_PredictsTheProbabilityOfAOne()
    {
        var predictions = new BinaryCrossEntropy().Predictions(Tensor.From(new Shape(2, 1), [0f, 0.2f]), _backend);

        AssertClose([0.5, 0.549834], predictions);
        Assert.Equal(OutputActivation.Sigmoid, new BinaryCrossEntropy().Activation);
    }

    [Fact]
    public void CrossEntropy_AgainstEachDaysShareOfItsHours_IsPyTorchs_AndItsPredictionsAreTheShares()
    {
        var days = Tensor.From(new Shape(2, 21),
        [
            0f, 0f, -0.2784217894077301f, -0.2529652416706085f, 0.657240092754364f, -0.430787593126297f, -0.7818315029144287f, 0.6234897971153259f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f,
            0f, 0f, -0.22953544557094574f, -0.27891865372657776f, 0.431541383266449f, -0.06757088750600815f, 0f, 1f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f,
        ]);
        var shares = Tensor.From(new Shape(2, 24),
        [
            0.01624365523457527f, 0.04060913622379303f, 0.03248731046915054f, 0.013197969645261765f, 0.0010152284521609545f, 0.0010152284521609545f, 0.002030456904321909f, 0.0030456853564828634f, 0.008121827617287636f, 0.014213197864592075f, 0.036548223346471786f, 0.0568527914583683f, 0.0852791890501976f, 0.09543146938085556f, 0.10761421173810959f, 0.11167512834072113f, 0.09441624581813812f, 0.0680203065276146f, 0.03553299605846405f, 0.03756345063447952f, 0.036548223346471786f, 0.034517765045166016f, 0.02842639572918415f, 0.039593908935785294f,
            0.02122347056865692f, 0.02122347056865692f, 0.01123595517128706f, 0.00749063678085804f, 0.00374531839042902f, 0f, 0.0024968788493424654f, 0.0012484394246712327f, 0.009987515397369862f, 0.024968789890408516f, 0.06616728752851486f, 0.08739075809717178f, 0.11610487103462219f, 0.09363295882940292f, 0.07365792989730835f, 0.09238451719284058f, 0.09488140046596527f, 0.08114856481552124f, 0.06616728752851486f, 0.03745318204164505f, 0.027465667575597763f, 0.0387016236782074f, 0.01123595517128706f, 0.009987515397369862f,
        ]);
        var layer = new Dense(
            Tensor.From(new Shape(21, 24), [.. Enumerable.Range(0, 21 * 24).Select(at => (((5 * (at / 24)) + (3 * (at % 24))) % 13 - 6) / 40f)]),
            Tensor.Zeros(new Shape(24)));
        var recording = new RecordingBackend(_backend);
        var loss = new CrossEntropy();

        var logits = layer.Forward(days, Pass.Training(recording, new RandomStream(1), 0, 0));
        var value = loss.Of(logits, shares, recording);
        var gradients = recording.GradientsOf(value, [layer.Bias.Value]);
        var predictions = loss.Predictions(logits, _backend);

        Assert.Equal(3.2375428676605225, value.Values[0], 1e-6);
        AssertClose(
            [0.029888564720749855, 0.021646419540047646, 0.00921641569584608, 0.0220717191696167, 0.04249461740255356, 0.044172242283821106],
            Tensor.From(new Shape(6), gradients[layer.Bias.Value].Values[..6]));
        AssertClose(
            [0.047344934195280075, 0.05210219323635101, 0.03003081865608692, 0.033837638795375824, 0.04443563520908356, 0.0446699894964695],
            Tensor.From(new Shape(6), predictions.Values[..6]));
        Assert.Equal(1f, predictions.Values[..24].ToArray().Sum(), 5);
        Assert.Equal(OutputActivation.Softmax, loss.Activation);
    }

    [Fact]
    public void MeanSquaredError_OnThePricesFirstFourDays_IsPyTorchs_AndSoAreItsGradients()
    {
        var hidden = new Dense(WalkedRows.TanhWeights(), WalkedRows.TanhBias());
        var output = new Dense(WalkedRows.ReturnWeights(), WalkedRows.ReturnBias());
        var network = new LayerStack(hidden, new Tanh(), output);
        var recording = new RecordingBackend(_backend);

        var loss = new MeanSquaredError().Of(
            network.Forward(WalkedRows.Days(), Pass.Training(recording, new RandomStream(1), 0, 0)), WalkedRows.Returns(), recording);
        var gradients = recording.GradientsOf(loss, network.Parameters().Select(parameter => parameter.Value));

        Assert.Equal(0.004323537927120924, loss.Values[0], 1e-8);
        AssertClose([-0.033036526292562485, 0.01282493956387043, 0.0043555255979299545], gradients[output.Weight.Value]);
        AssertClose([0.03436941280961037], gradients[output.Bias.Value]);
        AssertClose([-0.004057067446410656, 0, 0.006824049167335033], gradients[hidden.Bias.Value]);
    }

    [Fact]
    public void MeanSquaredError_PredictsTheOutputsAsTheyAre()
    {
        var outputs = Tensor.From(new Shape(2, 1), [1.5f, -2f]);

        Assert.Equal(outputs.Values.ToArray(), new MeanSquaredError().Predictions(outputs, _backend).Values.ToArray());
        Assert.Equal(OutputActivation.Identity, new MeanSquaredError().Activation);
    }

    [Fact]
    public void OutputsAndAnswersOfDifferentShapes_AreRefused()
    {
        Loss[] losses = [new MeanSquaredError(), new BinaryCrossEntropy(), new CrossEntropy()];

        Assert.All(losses, loss =>
            Assert.Throws<ArgumentException>(() => loss.Of(Tensor.Zeros(new Shape(2, 3)), Tensor.Zeros(new Shape(2, 2)), _backend)));
        Assert.Throws<ArgumentException>(() => new CrossEntropy().Of(Tensor.Zeros(new Shape(3)), Tensor.Zeros(new Shape(3)), _backend));
    }

    [Fact]
    public void BinaryCrossEntropy_RefusesAnAnswerOutsideNothingToOne()
    {
        var loss = new BinaryCrossEntropy();

        Assert.Null(loss.Refusal([0, 1, 0.25]));
        Assert.Contains("1.5", loss.Refusal([0, 1.5]), StringComparison.Ordinal);
        Assert.Contains("-1", loss.Refusal([-1]), StringComparison.Ordinal);
    }

    [Fact]
    public void CrossEntropy_RefusesARowThatIsNotShares()
    {
        var loss = new CrossEntropy();

        Assert.Null(loss.Refusal([0.25, 0.75]));
        Assert.Null(loss.Refusal([0, 1, 0]));
        Assert.Contains("0.9", loss.Refusal([0.4, 0.5]), StringComparison.Ordinal);
        Assert.Contains("-0.5", loss.Refusal([-0.5, 1.5]), StringComparison.Ordinal);
    }

    [Fact]
    public void MeanSquaredError_TakesAnyNumberAsAnAnswer()
    {
        Assert.Null(new MeanSquaredError().Refusal([-1e9, 0, 3.5]));
    }

    [Fact]
    public void ALoss_NeedsItsOutputsItsAnswersAndABackend()
    {
        var loss = new MeanSquaredError();

        Assert.Throws<ArgumentNullException>(() => loss.Of(null!, Tensor.Zeros(new Shape(1, 1)), _backend));
        Assert.Throws<ArgumentNullException>(() => loss.Of(Tensor.Zeros(new Shape(1, 1)), null!, _backend));
        Assert.Throws<ArgumentNullException>(() => loss.Of(Tensor.Zeros(new Shape(1, 1)), Tensor.Zeros(new Shape(1, 1)), null!));
        Assert.Throws<ArgumentNullException>(() => loss.Refusal(null!));
    }

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-6, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
