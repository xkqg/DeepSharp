// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of networks whose layers keep no bias and no shift, as PyTorch saved it as safetensors and as torch.save writes it
/// (Fixtures/pytorch-free.py made them): a state without a <c>bias</c> goes into the same network written here without one, and
/// answers as PyTorch does. A state that holds a bias for a layer written without one is refused at the tensor, and a layer
/// written with one that the state holds none for is refused as missing.
/// </summary>
public class PyTorchBiasFreeTests
{
    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(PyTorchFixture.Path("pytorch-free.json"))).RootElement;

    private static readonly Beginning Free = new() { WithoutBias = true };

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "rows", "images", "series", "volume" })
            {
                files.Add(name, "safetensors");
                files.Add(name, "pt");
            }

            return files;
        }
    }

    // Each network of pytorch-free.py written here, and the shape of one example it takes where it flattens.
    private static SpatialCase Case(string name, bool withBias = false)
    {
        var start = withBias ? default : Free;

        return name switch
        {
            "rows" => new(
                () => new Sequential().Dense(4, start).Relu().Dense(3, start).LayerNorm(new Affine { WithoutShift = !withBias }).Tanh().Dense(2).Lower(new Shape(5), new RandomStream(7)),
                new Shape(5)),
            "images" => new(
                () => new Sequential().Conv2D(4, new Window(3, 3), start).Relu().Conv2DTranspose(3, new Window(2, 2), start).Flatten().Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(7)),
                new Shape(6, 6, 2)),
            "series" => new(
                () => new Sequential().Conv1D(4, new Window1D(3), start).Relu().Conv1DTranspose(3, new Window1D(2), start).Flatten().Dense(2).Lower(new Shape(9, 2), new RandomStream(7)),
                new Shape(9, 2)),
            _ => new(
                () => new Sequential().Conv3D(3, new Window3D(2, 2, 2), start).Relu().Conv3DTranspose(2, new Window3D(2, 2, 2), start).Flatten().Dense(2).Lower(new Shape(4, 4, 4, 2), new RandomStream(7)),
                new Shape(4, 4, 4, 2)),
        };
    }

    private static PyTorchFile Reader(string kind, Network network, Shape? example) =>
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
    public void AStateOfLayersThatKeepNoBias_GoesIntoTheNetworkWrittenWithoutOne_AndAnswersAsPyTorchDoes(string name, string kind)
    {
        var made = Case(name);
        var saved = Reader(kind, made.Network(), made.Example).Read(PyTorchFixture.Open($"free-{name}.{kind}"));

        var outputs = saved.Network.Predict(Examples(name), saved.Loss, Backend).Values.ToArray();
        var expected = Answers.GetProperty(name).GetProperty("outputs").Floats();

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(Farthest(outputs, expected), 0, 1e-5);
    }

    [Theory]
    [InlineData("rows")]
    [InlineData("images")]
    [InlineData("series")]
    [InlineData("volume")]
    public void TheStateAsTorchSaveWritesIt_GoesInAsTheSafetensorsReaderPutsIt_BitForBit(string name)
    {
        var made = Case(name);
        var read = made.Network();
        var expected = made.Network();

        new TorchSaveFile(read, new MeanSquaredError()) { Example = made.Example }.Read(PyTorchFixture.Open($"free-{name}.pt"));
        new SafetensorsFile(expected, new MeanSquaredError()) { Example = made.Example }.Read(PyTorchFixture.Open($"free-{name}.safetensors"));

        Assert.Equal(Values(expected).Select(Bits), Values(read).Select(Bits));
    }

    [Theory]
    [InlineData("rows", "safetensors")]
    [InlineData("rows", "pt")]
    [InlineData("images", "safetensors")]
    public void ALayerWrittenWithABias_WhereTheStateHoldsNone_IsRefusedAsMissing_AndNoSlotChanges(string name, string kind)
    {
        var made = Case(name, withBias: true);
        var network = made.Network();
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(() => Reader(kind, network, made.Example).Read(PyTorchFixture.Open($"free-{name}.{kind}")));

        Assert.Contains(refused.Faults, fault => fault.Slot == "0.bias" && fault.Message.Contains("missing", StringComparison.Ordinal));
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Theory]
    [InlineData("safetensors")]
    [InlineData("pt")]
    public void ABiasTheStateHolds_ForALayerWrittenWithoutOne_IsRefusedAtTheTensorNothingHereTakes(string kind)
    {
        // The Titanic network's state holds a bias for each of its two linear layers; written without them, it has no slot for either.
        var network = new Sequential().Dense(16, Free).Relu().Dense(1, Free).Lower(new Shape(14), new RandomStream(7));
        var file = kind == "pt" ? "titanic.pt" : "titanic.safetensors";

        var refused = Assert.Throws<SlotLoadException>(() => Reader(kind, network, null).Read(PyTorchFixture.Open(file)));

        Assert.Equal(["0.bias", "2.bias"], refused.Faults.Select(fault => fault.Source).Order(StringComparer.Ordinal));
        Assert.All(refused.Faults, fault => Assert.Contains("is no slot of this network", fault.Message, StringComparison.Ordinal));
    }
}
