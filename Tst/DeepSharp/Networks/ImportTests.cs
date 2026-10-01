// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Import.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A model trained somewhere else, read by an importer written outside the library — as a reader of another framework's
/// files is — through what the library publishes: its numbers go into a network by the path of each slot, all of them or
/// none, every fault named at once where the file holds it, by the same rule and in the same words as a network's own
/// file; and what comes back is a saved network like any other.
/// </summary>
public class ImportTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    // A number for every slot of a dense layer and a batch normalisation behind it, one slot to a line.
    private static readonly string[] Numbers =
    [
        "0.weight 2x2 1 0.5 -0.5 2",
        "0.bias 2 0.25 -0.25",
        "1.weight 2 1.5 0.5",
        "1.bias 2 0.125 -0.125",
        "1.running_mean 2 0.5 -1",
        "1.running_var 2 4 0.25",
    ];

    // The same numbers, as a network's own file holds them.
    private static readonly string OwnFile = """
        {"network": {"version": 2,
          "layers": {"kind": "stack", "layers": [{"kind": "dense", "inputs": 2, "outputs": 2}, {"kind": "batchNorm", "features": 2, "momentum": 0.1, "epsilon": 1e-05}]},
          "parameters": {
            "0.weight": {"shape": [2, 2], "values": [1, 0.5, -0.5, 2]},
            "0.bias": {"shape": [2], "values": [0.25, -0.25]},
            "1.weight": {"shape": [2], "values": [1.5, 0.5]},
            "1.bias": {"shape": [2], "values": [0.125, -0.125]}
          },
          "state": {
            "1.running_mean": {"shape": [2], "values": [0.5, -1]},
            "1.running_var": {"shape": [2], "values": [4, 0.25]}
          },
          "loss": {"kind": "binaryCrossEntropy"}}}
        """;

    private static readonly Tensor Rows = Tensor.From(new Shape(3, 2), [0.5f, -1f, 2f, 0.25f, -0.75f, 1f]);

    [Fact]
    public void AnImporterOutsideTheLibrary_LoadsABatchNormsWeightByItsPath()
    {
        var norm = new BatchNorm(2);
        var network = new LayerStack(Unset(2, 2), norm);

        var saved = new WeightsFile(network, new BinaryCrossEntropy()).Read(File(Numbers));

        Assert.Same(network, saved.Network);
        Assert.IsType<BinaryCrossEntropy>(saved.Loss);
        Assert.Null(saved.TrainedOn);
        Assert.Equal([1.5f, 0.5f], norm.Weight.Value.Values.ToArray());
        Assert.Equal([0.125f, -0.125f], norm.Bias.Value.Values.ToArray());
        Assert.Equal([0.5f, -1f], norm.RunningMean.Value.Values.ToArray());
        Assert.Equal([4f, 0.25f], norm.RunningVariance.Value.Values.ToArray());
    }

    [Fact]
    public void AnImportedNetwork_AnswersAsTheSameNumbersReadFromANetworksOwnFile_ToTheLastBit()
    {
        var own = NetworkDocument.ReadNetwork(OwnFile, "network", NetworkCatalog.BuiltIn());

        var imported = new WeightsFile(new LayerStack(Unset(2, 2), new BatchNorm(2)), new BinaryCrossEntropy()).Read(File(Numbers));

        Assert.Equal(Bits(own.Network.Predict(Rows, own.Loss, Backend)), Bits(imported.Network.Predict(Rows, imported.Loss, Backend)));
    }

    [Fact]
    public void EveryFault_Missing_Extra_OfAnotherShape_NotFinite_AndWrittenTwice_IsNamedAtOnce_AndNoSlotChanges()
    {
        var network = new LayerStack(Unset(2, 2), new BatchNorm(2));
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(() => new WeightsFile(network, new MeanSquaredError()).Read(File(
            "0.weight 2x2 1 0.5 -0.5 2",
            "0.bias 3 0 0 0",
            "1.weight 2 1.5 NaN",
            "1.bias 2 0.125 -0.125",
            "1.bias 2 0 0",
            "1.running_mean 2 0.5 -1",
            "2.weight 2 1 1")));

        SlotLoadFault[] faults =
        [
            new("line 2", "0.bias", "'0.bias' is a 2 slot here, and is written as 3."),
            new("line 3", "1.weight", "'1.weight' holds finite numbers, and its value at 1 is NaN."),
            new("line 5", "1.bias", "'1.bias' is written twice here, and only one of the two would be read."),
            new("line 7", "2.weight", "'2.weight' is no slot of this network."),
            new(null, "1.running_var", "'1.running_var' is missing: every slot of the network is written."),
        ];

        Assert.Equal(faults, refused.Faults);
        Assert.Equal(string.Join(Environment.NewLine, faults.Select(fault => fault.ToString())), refused.Message);
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void ANetworksOwnFile_AndAnImporter_RefuseTheSameSlotsInTheSameWords()
    {
        var file = """
            {"network": {"version": 2,
              "layers": {"kind": "stack", "layers": [{"kind": "dense", "inputs": 2, "outputs": 2}]},
              "parameters": {"0.weight": {"shape": [3, 2], "values": [0, 0, 0, 0, 0, 0]}, "0.biass": {"shape": [2], "values": [0, 0]}},
              "state": {},
              "loss": {"kind": "meanSquaredError"}}}
            """;

        var own = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(file, "network", NetworkCatalog.BuiltIn()));
        var imported = Assert.Throws<SlotLoadException>(
            () => new WeightsFile(new LayerStack(Unset(2, 2)), new MeanSquaredError()).Read(File("0.weight 3x2 0 0 0 0 0 0", "0.biass 2 0 0")));

        Assert.Equal(
            ["'0.bias' is missing: every slot of the network is written.", "'0.biass' is no slot of this network.", "'0.weight' is a 2x2 slot here, and is written as 3x2."],
            own.Faults.Select(fault => fault.Message).Order(StringComparer.Ordinal));
        Assert.Equal(
            own.Faults.Select(fault => fault.Message).Order(StringComparer.Ordinal),
            imported.Faults.Select(fault => fault.Message).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void AnImporterThatReadsTheNetworkToo_BuildsItFromTheFile_BehindTheSameSeam_AsOneThatIsHandedIt()
    {
        IImporter[] importers = [new WeightsFile(new LayerStack(Unset(2, 2), new BatchNorm(2)), new BinaryCrossEntropy()), new StackFile()];

        var read = importers.Select((importer, at) => importer.Read(File(at == 0 ? Numbers : ["dense 2 2", "batchNorm 2", "loss binaryCrossEntropy", .. Numbers]))).ToArray();
        var built = Assert.IsType<LayerStack>(read[1].Network);

        Assert.IsType<Dense>(built.Layers[0]);
        Assert.IsType<BatchNorm>(built.Layers[1]);
        Assert.IsType<BinaryCrossEntropy>(read[1].Loss);
        Assert.Equal(Bits(read[0].Network.Predict(Rows, read[0].Loss, Backend)), Bits(read[1].Network.Predict(Rows, read[1].Loss, Backend)));
        Assert.Throws<FormatException>(() => new StackFile().Read(File("relu")));
    }

    [Fact]
    public void AnImportedNetwork_IsWrittenToANetworksOwnFile_ReadBack_AndTrainedOn_AsAnyOtherIs()
    {
        var imported = new StackFile().Read(File(["dense 2 2", "batchNorm 2", "relu", "loss meanSquaredError", .. Numbers]));

        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, imported.Network, imported.Loss);
            writer.WriteEndObject();
        }

        var again = NetworkDocument.ReadNetwork(Encoding.UTF8.GetString(stream.ToArray()), "network", NetworkCatalog.BuiltIn());

        Assert.Equal(Bits(imported.Network.Predict(Rows, imported.Loss, Backend)), Bits(again.Network.Predict(Rows, again.Loss, Backend)));

        var answers = Tensor.From(new Shape(3, 2), [1f, 0f, 0f, 1f, 0.5f, 0.5f]);
        var history = imported.Network.Compile(new Sgd(0.1), imported.Loss).Fit(new TrainingData(Rows, answers), validation: null, new FitOptions(seed: 7));

        Assert.Single(history.Epochs);
        Assert.False(imported.Network.Parameters().First().Value.Values.SequenceEqual([1f, 0.5f, -0.5f, 2f]));
    }

    [Fact]
    public void ANetworkLoadsNothingItWasNotHanded_AndAnEntryLeavingSomethingOut_IsTheReadersFault()
    {
        var network = new LayerStack(Unset(1, 1));
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();
        var one = Tensor.From(new Shape(1), [1f]);

        new LayerStack(new Relu()).Load([]);

        Assert.Throws<ArgumentNullException>(() => network.Load(null!));
        Assert.Throws<ArgumentException>(() => network.Load([default]));
        Assert.Throws<ArgumentException>(() => network.Load([new SlotEntry("0.bias", null!, "here")]));
        Assert.Throws<ArgumentException>(() => network.Load([new SlotEntry("0.bias", one, null!)]));
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void ASingleValueHandedForASlotOfOneAxis_IsNamedAsTheShapeSaysIt()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new LayerStack(Unset(1, 1)).Load(
        [
            new SlotEntry("0.weight", Tensor.From(new Shape(1, 1), [1f]), "weights"),
            new SlotEntry("0.bias", Tensor.From(new Shape(), [1f]), "bias"),
        ]));

        Assert.Equal([new SlotLoadFault("bias", "0.bias", "'0.bias' is a 1 slot here, and is written as scalar.")], refused.Faults);
        Assert.Equal("bias: '0.bias' is a 1 slot here, and is written as scalar.", refused.Faults[0].ToString());
    }

    [Fact]
    public void ANameAFileHoldsForNoSlot_IsShownEscapedInTheRefusal_NeverBreakingItsLine()
    {
        // A name that would start a line of its own in a log, and one that would turn what follows it round.
        var network = new LayerStack(new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        var forged = "x\r\n[INFO] model verified";
        var turned = "y\u202Etxt.exe";

        var refused = Assert.Throws<SlotLoadException>(() => network.Load(
        [
            new SlotEntry(forged, Tensor.Zeros(new Shape(1)), forged),
            new SlotEntry(turned, Tensor.Zeros(new Shape(1)), turned),
        ]));

        Assert.Equal(forged, refused.Faults[0].Source);
        Assert.Equal(forged, refused.Faults[0].Slot);
        Assert.Equal(@"x\r\n[INFO] model verified: 'x\r\n[INFO] model verified' is no slot of this network.", refused.Faults[0].ToString());
        Assert.Equal(@"y\u202etxt.exe: 'y\u202etxt.exe' is no slot of this network.", refused.Faults[1].ToString());
        Assert.DoesNotContain(
            refused.Message.Replace(Environment.NewLine, string.Empty, StringComparison.Ordinal),
            character => char.IsControl(character) || char.GetUnicodeCategory(character) == System.Globalization.UnicodeCategory.Format);
    }

    [Fact]
    public void ARefusal_SaysEveryFault_AndIsARefusalOfTheFilesFormat()
    {
        SlotLoadFault[] faults = [new("tensor 'w'", "0.weight", "one"), new(null, "0.bias", "two")];
        var refused = new SlotLoadException(faults);

        Assert.IsAssignableFrom<FormatException>(refused);
        Assert.Same(faults, refused.Faults);
        Assert.Equal($"tensor 'w': one{Environment.NewLine}two", refused.Message);
        Assert.Throws<ArgumentNullException>(() => new SlotLoadException(null!));
    }

    // A dense layer of so many inputs and outputs whose numbers are all still to be read: nought, every one.
    private static Dense Unset(int inputs, int outputs) => new(Tensor.Zeros(new Shape(inputs, outputs)), Tensor.Zeros(new Shape(outputs)));

    private static MemoryStream File(params string[] lines) => new(Encoding.UTF8.GetBytes(string.Join('\n', lines)));

    private static int[] Bits(Tensor tensor) => [.. tensor.Values.ToArray().Select(BitConverter.SingleToInt32Bits)];
}
