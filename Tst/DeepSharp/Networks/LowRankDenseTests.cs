// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A dense layer with a low-rank pair beside its held kernel, as Keras's <c>Dense</c> has one once <c>enable_lora</c> is called:
/// the answer is the kernel's plus the pair's, scaled by alpha over the rank; the kernel is held and the pair and the bias learn;
/// the pair starts so that the answer at the first step is the kernel's own; and a network that has one is written as the seventh
/// version of a network's part, while every dense layer without one is written as it always was.
/// </summary>
public class LowRankDenseTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void ARankBelowOne_IsRefused(int wrong) => Assert.Throws<ArgumentOutOfRangeException>(() => new LowRank(wrong));

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AnAlphaThatIsNotANumberAboveNothing_IsRefused(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new LowRank(2, wrong));

    [Fact]
    public void AnAlphaLeftOut_IsTheRank_SoTheScaleIsOne_AsKerasLeavesIt()
    {
        Assert.Equal(4, new LowRank(4).Alpha);
        Assert.Equal(1, new LowRank(4).Scale);
        Assert.Equal(2, new LowRank(2, 4).Scale);
    }

    [Fact]
    public void ADenseLayerWithoutAPair_HoldsTheSlotsItAlwaysHeld_InTheOrderItAlwaysHeldThem()
    {
        var dense = new Dense(3, 4, Draws());

        Assert.Equal(["weight", "bias"], dense.Slots().Select(named => named.Path));
        Assert.Null(dense.LowRank);
        Assert.Null(dense.A);
        Assert.Null(dense.B);
    }

    [Fact]
    public void ADenseLayerWithAPair_KeepsTheKernelTheBiasAndThePair_AndHoldsTheKernelStill()
    {
        var dense = LowRankDense(inputs: 3, outputs: 4, new LowRank(2, 4));

        Assert.Equal(["weight", "bias", "lora_a", "lora_b"], dense.Slots().Select(named => named.Path));
        Assert.Equal(new Shape(3, 2), dense.A!.Value.Shape);
        Assert.Equal(new Shape(2, 4), dense.B!.Value.Shape);
        Assert.False(dense.Weight.Learns);
        Assert.True(dense.Bias!.Learns);
        Assert.True(dense.A.Learns);
        Assert.True(dense.B.Learns);
        Assert.Equal(new LowRank(2, 4), dense.LowRank);
    }

    [Fact]
    public void ThePair_StartsWithBAtNothing_SoTheAnswerAtTheFirstStepIsTheKernelsOwn_AndAHeldAtAnEvenSpread()
    {
        var dense = LowRankDense(inputs: 3, outputs: 4, new LowRank(2));
        var plain = Dense(3, 4);

        Assert.All(dense.B!.Value.Values.ToArray(), value => Assert.Equal(0f, value));
        Assert.Contains(dense.A!.Value.Values.ToArray(), value => value is not 0f);
        Assert.All(dense.A.Value.Values.ToArray(), value => Assert.InRange(value, -Math.Sqrt(6.0 / 3), Math.Sqrt(6.0 / 3)));

        var rows = Tensor.From(new Shape(2, 3), [0.5f, -1f, 2f, 1f, 0.25f, -0.5f]);

        // The kernel and the bias are drawn first, so they are the ones a dense layer without a pair is begun with.
        Assert.Equal(plain.Weight.Value.Values.ToArray(), dense.Weight.Value.Values.ToArray());
        Assert.Equal(plain.Forward(rows, Pass.Evaluation(Backend)).Values.ToArray(), dense.Forward(rows, Pass.Evaluation(Backend)).Values.ToArray());
    }

    [Fact]
    public void TheAnswer_IsTheKernelsPlusThePairsTimesAlphaOverRank_PlusTheBias()
    {
        var dense = LowRankDense(inputs: 3, outputs: 2, new LowRank(2, 6));

        dense.Weight.Replace(Tensor.From(new Shape(3, 2), [1f, 0f, 0f, 1f, 1f, 1f]));
        dense.Bias!.Replace(Tensor.From(new Shape(2), [0.5f, -0.5f]));
        dense.A!.Replace(Tensor.From(new Shape(3, 2), [1f, 0f, 0f, 1f, 1f, 1f]));
        dense.B!.Replace(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]));

        var rows = Tensor.From(new Shape(2, 3), [1f, 2f, 3f, -1f, 0f, 2f]);
        var answered = dense.Forward(rows, Pass.Evaluation(Backend)).Values.ToArray();

        // x W: [4, 5] and [1, 2]; x A: [4, 5] and [1, 2]; x A B: [4+15, 8+20] = [19, 28] and [1+6, 2+8] = [7, 10]; scale 6 / 2 = 3.
        float[] expected = [4 + (3 * 19) + 0.5f, 5 + (3 * 28) - 0.5f, 1 + (3 * 7) + 0.5f, 2 + (3 * 10) - 0.5f];

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], answered[at], 4);
        }
    }

    [Fact]
    public void ARunOverARankedLayer_MovesNeitherTheKernel_ButMovesThePairAndTheBias_AndBringsTheLossDown()
    {
        var dense = LowRankDense(inputs: 3, outputs: 1, new LowRank(2));
        var kernel = dense.Weight.Value.Values.ToArray();
        var rows = new TrainingData(
            Tensor.From(new Shape(4, 3), [0.1f, 0.2f, 0.3f, 0.4f, 0.1f, -0.2f, -0.3f, 0.5f, 0.2f, 0.9f, -0.1f, 0.4f]),
            Tensor.From(new Shape(4, 1), [1f, 0f, 0.5f, -0.5f]));
        var history = new LayerStack(dense).Compile(new Adam(0.05), new MeanSquaredError()).Fit(rows, validation: null, new FitOptions(seed: 1) { Epochs = 60, BatchSize = 4 });

        Assert.Equal(kernel, dense.Weight.Value.Values.ToArray());
        Assert.Contains(dense.B!.Value.Values.ToArray(), value => value is not 0f);
        Assert.True(history.Epochs[^1].Loss < history.Epochs[0].Loss);
    }

    [Fact]
    public void ADenseLayerWithoutABias_CanHaveAPair_AndThenHoldsTheKernelAndThePairAlone()
    {
        var dense = new Dense(3, 2, new Beginning(Draws()) { WithoutBias = true, LowRank = new LowRank(1) });

        Assert.Equal(["weight", "lora_a", "lora_b"], dense.Slots().Select(named => named.Path));
    }

    [Fact]
    public void ANetworkWithoutAPair_IsWrittenAsItAlwaysWas_WithNoPairKeys_AndOneWithAPairAsTheSeventhVersion()
    {
        var plain = Written(new LayerStack(Dense(3, 2)));
        var ranked = Written(new LayerStack(LowRankDense(inputs: 3, outputs: 2, new LowRank(2, 6))));

        Assert.Contains("\"version\":2", plain, StringComparison.Ordinal);
        Assert.DoesNotContain("rank", plain, StringComparison.Ordinal);
        Assert.DoesNotContain("lora", plain, StringComparison.Ordinal);

        using var document = JsonDocument.Parse(ranked);
        var part = document.RootElement.GetProperty("network");
        var settings = part.GetProperty("layers").GetProperty("layers")[0];

        Assert.Equal(7, part.GetProperty("version").GetInt32());
        Assert.Equal(2, settings.GetProperty("rank").GetInt32());
        Assert.Equal(6, settings.GetProperty("alpha").GetDouble());
        Assert.False(part.GetProperty("parameters").GetProperty("0.weight").GetProperty("learns").GetBoolean());
        Assert.True(part.GetProperty("parameters").TryGetProperty("0.lora_a", out _));
        Assert.True(part.GetProperty("parameters").TryGetProperty("0.lora_b", out _));
    }

    [Fact]
    public void ANetworkWithAPair_IsReadBackAsItWas_TheKernelStillHeld_AndWrittenAgainToTheByte()
    {
        var dense = LowRankDense(inputs: 3, outputs: 2, new LowRank(2, 6));

        dense.A!.Replace(Tensor.From(new Shape(3, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f]));
        dense.B!.Replace(Tensor.From(new Shape(2, 2), [0.7f, 0.8f, 0.9f, 1f]));

        var text = Written(new LayerStack(dense));
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);
        var layer = Assert.IsType<Dense>(read.Layers[0]);

        Assert.Equal(new LowRank(2, 6), layer.LowRank);
        Assert.False(layer.Weight.Learns);
        Assert.Equal(dense.A.Value.Values.ToArray(), layer.A!.Value.Values.ToArray());
        Assert.Equal(dense.B.Value.Values.ToArray(), layer.B!.Value.Values.ToArray());
        Assert.Equal(text, Written(read));
    }

    [Fact]
    public void ARankWrittenWithoutItsAlpha_IsReadWithTheAlphaAtTheRank_AndAnAlphaWithoutARank_IsRefused()
    {
        var text = Written(new LayerStack(LowRankDense(inputs: 3, outputs: 2, new LowRank(2))));
        var withoutAlpha = text.Replace("\"alpha\":2,", string.Empty, StringComparison.Ordinal);
        var read = Assert.IsType<Dense>(Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(withoutAlpha, "network", NetworkCatalog.BuiltIn()).Network).Layers[0]);

        Assert.Equal(new LowRank(2), read.LowRank);

        var withoutRank = text.Replace("\"rank\":2,", string.Empty, StringComparison.Ordinal);

        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(withoutRank, "network", NetworkCatalog.BuiltIn()));
    }

    [Fact]
    public void ADescriptionInKerasWords_MakesTheLayerWithTheRankItIsToldAndNoneWhereItIsNot()
    {
        var stack = new Sequential()
            .Dense(4, new Beginning { LowRank = new LowRank(2, 4) })
            .Dense(1)
            .Lower(new Shape(3), new RandomStream(1));
        var dense = stack.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new LowRank(2, 4), dense[0].LowRank);
        Assert.Null(dense[1].LowRank);
        Assert.True(dense[1].Weight.Learns);
    }

    private static Dense Dense(int inputs, int outputs) => new(inputs, outputs, Draws());

    private static Dense LowRankDense(int inputs, int outputs, LowRank pair) =>
        new(inputs, outputs, new Beginning(Draws()) { LowRank = pair });

    private static Draws Draws() => new RandomStream(5).Draw("initialise:0", 0, 0);

    private static string Written(Network network)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
