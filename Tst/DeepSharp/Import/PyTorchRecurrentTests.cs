// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of networks that read a series a step at a time, as PyTorch saved it as safetensors and as torch.save writes it
/// (Fixtures/pytorch-recurrent.py made them): a recurrent layer's weights, which PyTorch keeps outputs by inputs, go into the layer
/// turned round, the gates in PyTorch's order, and the layer after it — a linear layer, or one a flatten of every step's state feeds —
/// takes its numbers in the order PyTorch's rows have them, the steps first, as a flatten of steps by channels leaves them.
/// </summary>
public class PyTorchRecurrentTests
{
    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(PyTorchFixture.Path("pytorch-recurrent.json"))).RootElement;

    // Each module of pytorch-recurrent.py written here, its layers called as its attributes are, and the shape of one example.
    private static (Func<LayerStack> Network, Shape Each) Case(string name) => name switch
    {
        "lstm" => (() => new Sequential().Lstm(4, bothWays: true, layers: 2).Named("rnn").Dense(2).Named("head").Lower(new Shape(5, 3), new RandomStream(7)), new Shape(5, 3)),
        "gru" => (() => new Sequential().Gru(4, sequences: true).Named("rnn").Flatten().Dense(2).Named("head").Lower(new Shape(5, 3), new RandomStream(7)), new Shape(5, 3)),
        "conv" => (() => new Sequential().Conv1D(3, new Window1D(2)).Named("conv").Relu().Lstm(4).Named("rnn").Dense(1).Named("head").Lower(new Shape(6, 2), new RandomStream(7)), new Shape(6, 2)),
        _ => (() => new Sequential().Conv1D(3, new Window1D(2)).Named("conv").Relu().Lstm(4, sequences: true).Named("rnn").Flatten().Dense(1).Named("head").Lower(new Shape(5, 2), new RandomStream(7)), new Shape(5, 2)),
    };

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "lstm", "gru", "conv", "conv-sequence" })
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
    public void AStateOfNetworksThatReadASeriesAStepAtATime_GoesIn_AndAnswersAsPyTorchDoes(string name, string kind)
    {
        var (network, each) = Case(name);
        var saved = Reader(kind, network(), each).Read(PyTorchFixture.Open($"recurrent-{name}.{kind}"));
        var outputs = saved.Network.Predict(Examples(name), saved.Loss, Backend).Values.ToArray();
        var expected = Answers.GetProperty(name).GetProperty("outputs").Floats();

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(Farthest(outputs, expected), 0, 1e-5);
    }

    [Theory]
    [InlineData("lstm")]
    [InlineData("gru")]
    [InlineData("conv")]
    [InlineData("conv-sequence")]
    public void TheStatesNames_AreThePathsOfTheLayersSlots(string name)
    {
        var paths = Case(name).Network().Slots().Select(slot => slot.Path).Order(StringComparer.Ordinal);

        Assert.Equal(Answers.GetProperty(name).GetProperty("keys").EnumerateArray().Select(key => key.GetString()!).Order(StringComparer.Ordinal), paths);
    }
}
