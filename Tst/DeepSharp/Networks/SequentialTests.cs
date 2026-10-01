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
