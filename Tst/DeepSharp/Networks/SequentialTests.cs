// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network described in Keras's words is read once and lowered onto the same stack of layers a network written as code
/// is made of: every width worked out from the shape of an example, every layer drawing what it starts from out of a
/// stream by its place — so nothing downstream can tell which door a network came through, down to its file. This one puts
/// the walked weights into the lowered stack by the door the library keeps to itself, so it runs on the light engine alone;
/// the rest of the lowering runs on every engine from the contract.
/// </summary>
public class SequentialTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void TheSameNetwork_ThroughEitherDoor_GivesPyTorchsLogitsBitForBit_AndTheSameFile()
    {
        var lowered = new Sequential().Dense(4).Relu().Dense(1).Lower(new Shape(14), new RandomStream(1));
        var written = new LayerStack(
            new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

        foreach (var (slot, value) in lowered.Slots().Zip(written.Slots()))
        {
            slot.Slot.Replace(value.Slot.Value);
        }

        var fromWords = lowered.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray();
        var fromCode = written.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray();

        Assert.Equal(fromCode.Select(BitConverter.SingleToInt32Bits), fromWords.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal([0.2f, 0.2f, 0.22550064f, 0.2f], fromWords);
        Assert.Equal(Written(written), Written(lowered));
    }

    [Fact]
    public void ADenseLayerWithoutAnInitialiser_StartsAsPyTorchStartsIt_BitForBit_AsItAlwaysHas()
    {
        // The word as it was: PyTorch's start, drawn from the stream by the layer's place, exactly as a layer written as code
        // draws it — and naming PyTorch's own initialiser changes nothing.
        var stream = new RandomStream(7);
        var plain = new Sequential().Dense(3).Lower(new Shape(5), stream);
        var named = new Sequential().Dense(3, new KaimingUniform()).Lower(new Shape(5), new RandomStream(7));
        var code = new Dense(5, 3, new RandomStream(7).Draw("initialise:0", 0, 0));

        Assert.Equal(Bits(code), Bits(plain));
        Assert.Equal(Bits(code), Bits(named));
    }

    [Fact]
    public void ADenseLayerWithAnInitialiser_StartsItsWeightsAsThatInitialiserDraws_UnderTheSameSeed_AndItsBiasAsBefore()
    {
        // The initialiser decides the weights only: the bias is still drawn as PyTorch draws it, after the weights, from the
        // same draws, and the network's file holds the numbers, not how they started.
        var plainStack = new Sequential().Dense(3).Lower(new Shape(5), new RandomStream(7));
        var glorotStack = new Sequential().Dense(3, new GlorotUniform()).Lower(new Shape(5), new RandomStream(7));
        var plain = (Dense)plainStack.Layers[0];
        var glorot = (Dense)glorotStack.Layers[0];
        var zeros = (Dense)new Sequential().Dense(3, new Zeros()).Lower(new Shape(5), new RandomStream(7)).Layers[0];
        var bound = Math.Sqrt(6.0 / (5 + 3));

        Assert.NotEqual(plain.Weight.Value.Values.ToArray(), glorot.Weight.Value.Values.ToArray());
        Assert.All(glorot.Weight.Value.Values.ToArray(), value => Assert.InRange(value, -bound, bound));
        Assert.Contains(glorot.Weight.Value.Values.ToArray(), value => Math.Abs(value) > 1 / Math.Sqrt(5));
        Assert.All(zeros.Weight.Value.Values.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(plain.Bias.Value.Values.ToArray(), glorot.Bias.Value.Values.ToArray());
        Assert.Equal(LayersOf(plainStack), LayersOf(glorotStack));
        Assert.Throws<ArgumentNullException>(() => new Sequential().Dense(3, null!));
    }

    // What a network's file says of its layers: their kinds and settings, without the numbers they hold.
    private static string LayersOf(Network network) => JsonDocument.Parse(Written(network)).RootElement.GetProperty("layers").GetRawText();

    private static int[][] Bits(Layer network) =>
        [.. network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            NetworkDocument.WriteNetwork(writer, network, new BinaryCrossEntropy());
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }
}
