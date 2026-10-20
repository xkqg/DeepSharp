// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of three modules whose paths meet, as PyTorch saved it as safetensors and as torch.save writes it
/// (Fixtures/pytorch-branch.py made them): a state says what a module holds and not how its forward pass joins what its modules make,
/// so it goes into a network of steps written here, each step called what the module is called, and the network answers as PyTorch does.
/// </summary>
public class PyTorchBranchTests
{
    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(PyTorchFixture.Path("pytorch-branch.json"))).RootElement;

    // Each module of pytorch-branch.py written here: its steps called as its attributes are, and joined as its forward pass does.
    private static LayerStack Network(string name) => name switch
    {
        "residual" => new Sequential().Dense(6).Named("inner").Relu().Add("1", "input").Dense(2).Named("outer").Lower(new Shape(6), new RandomStream(7)),
        "dense" => new Sequential()
            .Dense(4).Named("a").Relu().Named("ra")
            .Concatenate("input", "ra").Dense(4).Named("b").Relu().Named("rb")
            .Concatenate("input", "ra", "rb").Dense(2).Named("out")
            .Lower(new Shape(5), new RandomStream(7)),
        _ => new Sequential()
            .Columns(0, 3).Takes("input").Dense(4).Named("left")
            .Columns(3, 3).Named("rightColumns").Takes("input").Dense(4).Named("right")
            .Add("left", "right")
            .Lower(new Shape(6), new RandomStream(7)),
    };

    private static Tensor Examples(string name)
    {
        var examples = Answers.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. examples.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), examples.GetProperty("values").Floats());
    }

    private static PyTorchFile Reader(string kind, Network network) =>
        kind == "pt" ? new TorchSaveFile(network, new MeanSquaredError()) : new SafetensorsFile(network, new MeanSquaredError());

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "residual", "dense", "halves" })
            {
                files.Add(name, "safetensors");
                files.Add(name, "pt");
            }

            return files;
        }
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AStateOfModulesWhosePathsMeet_GoesIntoTheNetworkOfStepsCalledAsTheyAre_AndAnswersAsPyTorchDoes(string name, string kind)
    {
        var saved = Reader(kind, Network(name)).Read(PyTorchFixture.Open($"branch-{name}.{kind}"));
        var outputs = saved.Network.Predict(Examples(name), saved.Loss, Backend).Values.ToArray();
        var expected = Answers.GetProperty(name).GetProperty("outputs").Floats();

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(outputs.Zip(expected, (here, torch) => Math.Abs(here - torch)).Max(), 0, 1e-5);
    }

    [Theory]
    [InlineData("residual")]
    [InlineData("dense")]
    [InlineData("halves")]
    public void TheStatesNames_AreThePathsOfTheStepsSlots(string name)
    {
        var paths = Network(name).Slots().Select(slot => slot.Path).Order(StringComparer.Ordinal);

        Assert.Equal(Answers.GetProperty(name).GetProperty("keys").EnumerateArray().Select(key => key.GetString()!).Order(StringComparer.Ordinal), paths);
    }

    [Fact]
    public void AStateGoesIntoAStackWhoseStepsAreNotCalledAsTheModulesAre_AsMissingAndAsAnExtra_AndNoSlotChanges()
    {
        var network = new Sequential().Dense(6).Relu().Add("1", "input").Dense(2).Lower(new Shape(6), new RandomStream(7));
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(() => Reader("pt", network).Read(PyTorchFixture.Open("branch-residual.pt")));

        Assert.Contains(refused.Faults, fault => fault.Slot == "inner.weight");
        Assert.Contains(refused.Faults, fault => fault.Slot == "0.weight");
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }
}
