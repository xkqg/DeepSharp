// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of networks that look across the steps of a series, as PyTorch saved it as safetensors and as torch.save writes it
/// (Fixtures/pytorch-attention.py made them): the projections of multi-head attention, which PyTorch keeps outputs by inputs, go in turned
/// round, and an encoder layer's attention, feed-forward layers and normalisations are found under the names its state gives them.
/// </summary>
public class PyTorchAttentionTests
{
    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(PyTorchFixture.Path("pytorch-attention.json"))).RootElement;

    // Each module of pytorch-attention.py written here, its layers called as its attributes are.
    private static LayerStack Network(string name) => name == "pooled"
        ? new Sequential().MultiHeadAttention(2).Named("attn").GlobalAvgPool1D().Dense(2).Named("head").Lower(new Shape(5, 8), new RandomStream(7))
        : new Sequential().SinePositions().EncoderBlock(new EncoderShape(2, 16)).Named("layer").GlobalAvgPool1D().Dense(2).Named("head").Lower(new Shape(5, 8), new RandomStream(7));

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "pooled", "encoder" })
            {
                files.Add(name, "safetensors");
                files.Add(name, "pt");
            }

            return files;
        }
    }

    private static PyTorchFile Reader(string kind, Network network, Shape example) =>
        kind == "pt"
            ? new TorchSaveFile(network, new MeanSquaredError()) { Example = example }
            : new SafetensorsFile(network, new MeanSquaredError()) { Example = example };

    private static Tensor Examples(string name)
    {
        var examples = Answers.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. examples.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), examples.GetProperty("values").Floats());
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AStateOfNetworksThatLookAcrossASeries_GoesIn_AndAnswersAsPyTorchDoes(string name, string kind)
    {
        var saved = Reader(kind, Network(name), new Shape(5, 8)).Read(PyTorchFixture.Open($"attention-{name}.{kind}"));
        var outputs = saved.Network.Predict(Examples(name), saved.Loss, Backend).Values.ToArray();
        var expected = Answers.GetProperty(name).GetProperty("outputs").Floats();

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(Farthest(outputs, expected), 0, 1e-5);
    }

    [Theory]
    [InlineData("pooled")]
    [InlineData("encoder")]
    public void TheStatesNames_AreThePathsOfTheLayersSlots(string name)
    {
        var paths = Network(name).Slots().Select(slot => slot.Path).Order(StringComparer.Ordinal);

        Assert.Equal(Answers.GetProperty(name).GetProperty("keys").EnumerateArray().Select(key => key.GetString()!).Order(StringComparer.Ordinal), paths);
    }
}
