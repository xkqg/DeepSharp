// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// PyTorch's layouts are turned into DeepSharp's by two readers: the one that reads what PyTorch saved and the one that
/// reads the graph PyTorch exported to ONNX. The packages do not know each other, so each keeps its own copy of the turn;
/// a rule kept twice holds only while the two are compared, so the same trained network read through each is compared
/// here, slot for slot and bit for bit.
/// </summary>
public class LayoutParityTests
{
    [Theory]
    [InlineData("onnx-titanic.onnx")]
    [InlineData("onnx-titanic-torchscript.onnx")]
    public void TheSameTrainedNetwork_ReadFromWhatPyTorchSaved_AndFromWhatItExported_HoldsTheSameNumbers(string graph)
    {
        // Both scripts train the Titanic network from the same seed on the same rows, so the two files hold one network.
        var saved = new SafetensorsFile(
            new Sequential().Dense(16).Relu().Dense(1).Lower(new Shape(14), new RandomStream(7)),
            new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));
        using var exported = OnnxFixtures.Open(graph);
        var fromGraph = new OnnxFile(new BinaryCrossEntropy()).Read(exported);

        var weights = saved.Network.Slots().Select(named => named.Slot.Value.Values.ToArray()).ToArray();
        var turned = fromGraph.Network.Slots().Select(named => named.Slot.Value.Values.ToArray()).ToArray();

        Assert.Equal(weights.Length, turned.Length);
        Assert.All(weights.Zip(turned), pair => Assert.Equal(
            pair.First.Select(BitConverter.SingleToInt32Bits),
            pair.Second.Select(BitConverter.SingleToInt32Bits)));
    }
}
