// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using Onnxify.Safetensors;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A network PyTorch trained and saved as safetensors, read into the same network written here: each number turned into
/// the layout the slot it goes into keeps — a linear layer's weights turned round, a convolution's kernel channels last, the
/// rows a flatten makes of images read row by row rather than channel by channel — and every file safetensors' own reader
/// refuses refused, every fault among the tensors named at once by the tensor's name, and nothing put in until all of it is.
/// </summary>
public class SafetensorsFileTests
{
    [Fact]
    public void TheReadmesPassenger_AnswersAsPyTorchsTitanicNetworkDoes_WithinEightRoundings()
    {
        var served = PyTorchFixture.Json.GetProperty("titanic").GetProperty("served");
        var features = Rows(Titanic.Value.Served(Passengers, Needs.OneScale).Features);
        var saved = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        var chances = saved.Network.Predict(features, saved.Loss, Backend).Values.ToArray();

        // The very numbers PyTorch was handed, served here by the pipeline it was trained behind.
        Assert.Equal(served.GetProperty("features").Floats(), features.Values.ToArray());
        Assert.All(Roundings(chances, served.GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 8));
    }

    [Fact]
    public void EveryTestPassenger_AnswersAsPyTorchsTitanicNetworkDoes_WithinEightRoundings()
    {
        var test = PyTorchFixture.Json.GetProperty("titanic").GetProperty("test");
        var features = Rows(Titanic.Value.Batch(Part.Test, Needs.OneScale).Features);
        var saved = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        var chances = saved.Network.Predict(features, saved.Loss, Backend).Values.ToArray();

        Assert.Equal(test.GetProperty("features").Floats(), features.Values.ToArray());
        Assert.Equal(135, chances.Length);
        Assert.All(Roundings(chances, test.GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 8));
    }

    [Fact]
    public void ABatchNormalisationsNumbers_AnswerAsPyTorchDoes_ItsCountOfBatchesLeftOut_FromAFileThatSaysNothingOfItsLayout()
    {
        var batchnorm = PyTorchFixture.Json.GetProperty("batchnorm");
        var network = new Sequential().Dense(8).BatchNorm().Relu().Dense(1).Lower(new Shape(14), new RandomStream(7));
        var saved = new SafetensorsFile(network, new BinaryCrossEntropy()).Read(PyTorchFixture.Open("batchnorm.safetensors"));

        var served = saved.Network.Predict(Rows(Titanic.Value.Served(Passengers, Needs.OneScale).Features), saved.Loss, Backend).Values.ToArray();
        var test = saved.Network.Predict(Rows(Titanic.Value.Batch(Part.Test, Needs.OneScale).Features), saved.Loss, Backend).Values.ToArray();

        // safetensors.torch.save_model wrote it, which marks nothing, and it holds 1.num_batches_tracked, a whole number.
        Assert.False(Header("batchnorm.safetensors").TryGetProperty("__metadata__", out _));
        Assert.Equal("I64", Header("batchnorm.safetensors").GetProperty("1.num_batches_tracked").GetProperty("dtype").GetString());
        Assert.Equal(["0.weight", "0.bias", "1.weight", "1.bias", "1.running_mean", "1.running_var", "3.weight", "3.bias"], network.Slots().Select(named => named.Path));
        // PyTorch folds a batch normalisation at evaluation into one scale and one shift of each feature, and here the feature
        // is normalised and then scaled, so its roundings fall elsewhere: measured, at most 10 apart over these 137 chances.
        Assert.All(Roundings(served, batchnorm.GetProperty("served").GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 16));
        Assert.All(Roundings(test, batchnorm.GetProperty("test").GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 16));
    }

    [Fact]
    public void ConvolutionsFlattenedIntoALinearLayer_AnswerAsPyTorchDoes_WithinAHundredThousandth_ThroughAZeroExample()
    {
        var saved = new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Example = new Shape(8, 6, 2) }
            .Read(PyTorchFixture.Open("convolution.safetensors"));

        var outputs = saved.Network.Predict(Images(), saved.Loss, Backend).Values.ToArray();
        var expected = PyTorchFixture.Json.GetProperty("convolution").GetProperty("outputs").Floats();

        Assert.Equal(6, outputs.Length);
        Assert.All(outputs.Zip(expected, (ours, theirs) => Math.Abs(ours - theirs)), apart => Assert.InRange(apart, 0, 1e-5));
    }

    [Fact]
    public void WhatAFlattenIsHanded_StatedByTheCaller_OrReachedByAReshapeIntoRows_ReadsTheNumbersAnExampleDoes()
    {
        var byExample = new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Example = new Shape(8, 6, 2) }
            .Read(PyTorchFixture.Open("convolution.safetensors"));
        var stated = new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(4, 3, 3) } }
            .Read(PyTorchFixture.Open("convolution.safetensors"));
        var reshaped = new SafetensorsFile(ConvolutionInKerasWords(reshapedIntoRows: true), new MeanSquaredError()) { Example = new Shape(8, 6, 2) }
            .Read(PyTorchFixture.Open("convolution.safetensors"));

        Assert.Equal(Values(byExample.Network), Values(stated.Network));
        Assert.Equal(Values(byExample.Network), Values(reshaped.Network));
        Assert.IsType<Reshape>(((LayerStack)reshaped.Network).Layers[4]);
    }

    [Fact]
    public void AFlattenStatedAsHandedRows_PutsItsNumbersInUnturned_AndTheNetworkAnswersOtherwise()
    {
        // Said to be handed rows of 36, the flatten's linear layer takes PyTorch's weights row for row — channel by channel,
        // where the rows this network flattens run row by row — so what it answers is no longer what PyTorch answered.
        var saved = new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(36) } }
            .Read(PyTorchFixture.Open("convolution.safetensors"));

        var outputs = saved.Network.Predict(Images(), saved.Loss, Backend).Values.ToArray();
        var expected = PyTorchFixture.Json.GetProperty("convolution").GetProperty("outputs").Floats();

        Assert.True(outputs.Zip(expected, (ours, theirs) => Math.Abs(ours - theirs)).Max() > 0.01);
    }

    [Fact]
    public void AFlattenAnExampleHandsRows_TurnsNothing()
    {
        // The Titanic network behind a flatten of its rows: each tensor one place further on, and nothing to turn.
        var draws = new RandomStream(7).Draw("initialise", 0, 0);
        var network = new LayerStack(new Flatten(), new Dense(14, 16, draws), new Relu(), new Dense(16, 1, draws));
        var flattened = new SafetensorsFile(network, new BinaryCrossEntropy()) { Example = new Shape(14) }
            .Read(Renamed("titanic.safetensors", name => $"{name[0] - '0' + 1}{name[1..]}"));
        var plain = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        Assert.Equal(Values(plain.Network), Values(flattened.Network));
    }

    [Theory]
    [InlineData("f16")]
    [InlineData("bf16")]
    [InlineData("f64")]
    public void EachWidthOfFloat_IsReadAsPyTorchMakesItThirtyTwoBits(string width)
    {
        // Sixteen bits widen exactly; sixty-four are rounded to the nearest thirty-two, as PyTorch rounds them.
        var torch = PyTorchFixture.Json.GetProperty("widths").GetProperty(width);
        var network = TitanicInKerasWords();

        new SafetensorsFile(network, new BinaryCrossEntropy()).Read(PyTorchFixture.Open($"titanic-{width}.safetensors"));

        var slots = network.Slots().ToDictionary(named => named.Path, named => named.Slot.Value);
        Assert.Equal(Bits(Transposed(torch.GetProperty("0.weight").Floats(), 16, 14)), Bits(slots["0.weight"].Values.ToArray()));
        Assert.Equal(Bits(torch.GetProperty("0.bias").Floats()), Bits(slots["0.bias"].Values.ToArray()));
        Assert.Equal(Bits(torch.GetProperty("2.weight").Floats()), Bits(slots["2.weight"].Values.ToArray()));
        Assert.Equal(Bits(torch.GetProperty("2.bias").Floats()), Bits(slots["2.bias"].Values.ToArray()));
    }

    [Fact]
    public void EveryFileSafetensorsOwnReaderRefuses_IsRefusedHere_InItsWords_AndEveryOneItReads_IsRead()
    {
        var cases = PyTorchFixture.Json.GetProperty("reference").EnumerateArray().ToArray();

        Assert.Equal(20, cases.Length);

        foreach (var each in cases)
        {
            var name = each.GetProperty("case").GetString();
            var reference = each.GetProperty("reference").GetString()!;
            var network = new LayerStack(new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
            var before = network.Slots().Select(named => named.Slot.Value).ToArray();
            var read = new SafetensorsFile(network, new MeanSquaredError());
            using var file = new MemoryStream(Convert.FromBase64String(each.GetProperty("file").GetString()!));

            if (reference.StartsWith("reads", StringComparison.Ordinal))
            {
                read.Read(file);

                Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
                Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
                continue;
            }

            var refused = Assert.Throws<FormatException>(() => read.Read(file));

            // A refusal of the file, not of its numbers, naming the tensor wherever the reference names one.
            Assert.StartsWith("The file is no safetensors file: ", refused.Message, StringComparison.Ordinal);
            Assert.All(new[] { "0.weight", "0.bias" }.Where(tensor => reference.Contains($"`{tensor}`", StringComparison.Ordinal)), tensor => Assert.Contains($"`{tensor}`", refused.Message, StringComparison.Ordinal));
            Assert.True(before.SequenceEqual(network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance), name);
        }
    }

    [Fact]
    public void AFileMarkedAsAnotherFrameworksLayout_IsRefused_AndOneMarkedWithAnythingElse_IsReadAsPyTorchs()
    {
        var network = new LayerStack(new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        Held[] tensors = [new("0.weight", DataType.F32, [1, 2], Floats(1.5f, -2f)), new("0.bias", DataType.F32, [1], Floats(0.5f))];

        var refused = Assert.Throws<FormatException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(tensors, new Dictionary<string, string> { ["format"] = "tf" })));
        new SafetensorsFile(network, new MeanSquaredError()).Read(Written(tensors, new Dictionary<string, string> { ["note"] = "trained on Tuesdays" }));

        Assert.Equal(
            "The file says its tensors are laid out as 'tf' lays them out, and this reader reads PyTorch's layout, which a file marks as \"format\": \"pt\".",
            refused.Message);
        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void EveryFaultAmongTheTensors_IsNamedAtOnce_WhereTheFileHoldsIt_AndNoSlotChanges()
    {
        var network = new LayerStack(
            new Dense(Tensor.Zeros(new Shape(3, 2)), Tensor.Zeros(new Shape(2))), new Relu(), new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("0.weight", DataType.F32, [3, 2], Floats(1, 2, 3, 4, 5, 6)),
            new("0.bias", DataType.I64, [2], new byte[16]),
            new("1.num_batches_tracked", DataType.I64, [], new byte[8]),
            new("2.weight", DataType.F64, [1, 2], [.. BitConverter.GetBytes(1e39), .. BitConverter.GetBytes(0d)]),
            new("3.weight", DataType.F32, [1], Floats(1)),
        ])));

        SlotLoadFault[] faults =
        [
            new("0.bias", "0.bias", "'0.bias' holds I64 numbers, and a slot holds 32-bit floats: F16 and BF16 are widened to them and F64 is narrowed, and no other kind of number is read."),
            new("1.num_batches_tracked", "1.num_batches_tracked", "'1.num_batches_tracked' is no slot of this network."),
            new("2.weight", "2.weight", "'2.weight' holds finite numbers, and its value at 0 is Infinity."),
            new("0.weight", "0.weight", "'0.weight' is a 2x3 tensor in PyTorch's layout here, and is written as 3x2."),
            new("3.weight", "3.weight", "'3.weight' is no slot of this network."),
            new(null, "2.bias", "'2.bias' is missing: every slot of the network is written."),
        ];

        Assert.Equal(faults, refused.Faults);
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void ASingleNumberWrittenForAVector_IsNamedAScalar()
    {
        var network = new LayerStack(new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
            [new("0.weight", DataType.F32, [1, 2], Floats(1, 2)), new("0.bias", DataType.F32, [], Floats(0.5f))])));

        Assert.Equal([new SlotLoadFault("0.bias", "0.bias", "'0.bias' is a 1 tensor in PyTorch's layout here, and is written as scalar.")], refused.Faults);
    }

    [Fact]
    public void WithNothingSaidOfWhatAFlattenIsHanded_EveryTensorThatTurnsOnIt_IsRefused_AndNoSlotChanges()
    {
        var network = ConvolutionInKerasWords();
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(PyTorchFixture.Open("convolution.safetensors")));

        const string Why = "reads the rows the layer at 4 flattens, which PyTorch lays out channel by channel: hand the reader an example the network takes, or state what that layer is handed.";
        Assert.Equal(
            ["5.bias", "5.running_mean", "5.running_var", "5.weight", "6.weight"],
            refused.Faults.Select(fault => fault.Source));
        Assert.All(refused.Faults, fault => Assert.Equal($"'{fault.Slot}' {Why}", fault.Message));
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void AFlattenAnExampleHandsNeitherAnImageNorARow_IsRefusedAtTheLinearLayerReadingIt()
    {
        var network = new LayerStack(new Reshape(new Shape(2, 7)), new Flatten(), new Dense(Tensor.Zeros(new Shape(14, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()) { Example = new Shape(14) }.Read(Written(
            [new("2.weight", DataType.F32, [1, 14], Floats(new float[14])), new("2.bias", DataType.F32, [1], Floats(0))])));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal(
            "'2.weight' reads the rows the layer at 1 makes of 2x7, which is neither an image nor a row: an image's channels are turned to the end, and a row is read as it is.",
            fault.Message);
    }

    [Fact]
    public void ALinearLayerMakingTheRowsAReshapeLaysOutAsImages_IsRefused_AsIsWhatNormalisesThem()
    {
        var network = new LayerStack(
            new Dense(Tensor.Zeros(new Shape(2, 12)), Tensor.Zeros(new Shape(12))),
            new LayerNorm(12),
            new Reshape(new Shape(2, 2, 3)),
            new Conv2D(Tensor.Zeros(new Shape(3, 1)), Tensor.Zeros(new Shape(1)), new Window(1, 1)),
            new Flatten(),
            new Dense(Tensor.Zeros(new Shape(4, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()) { Example = new Shape(2) }.Read(Written(
        [
            new("0.weight", DataType.F32, [12, 2], Floats(new float[24])),
            new("0.bias", DataType.F32, [12], Floats(new float[12])),
            new("1.weight", DataType.F32, [12], Floats(new float[12])),
            new("1.bias", DataType.F32, [12], Floats(new float[12])),
            new("3.weight", DataType.F32, [1, 3, 1, 1], Floats(new float[3])),
            new("3.bias", DataType.F32, [1], Floats(0)),
            new("5.weight", DataType.F32, [1, 4], Floats(new float[4])),
            new("5.bias", DataType.F32, [1], Floats(0)),
        ])));

        Assert.Equal(["0.bias", "0.weight", "1.bias", "1.weight"], refused.Faults.Select(fault => fault.Slot));
        Assert.Equal(
            "'0.weight' makes the rows the layer at 2 lays out as 2x2x3, which PyTorch lays out channel by channel: rows a flatten makes of images are turned, and images a reshape makes of rows are not.",
            refused.Faults[1].Message);
    }

    [Fact]
    public void TheTitanicNetworkWrittenAsCode_AnswersTheReadmesPassengerAsPyTorchsDoes_WithinEightRoundings()
    {
        // As a PyTorch module written with a hidden and an output layer saves itself: each tensor under its layer's name.
        var served = PyTorchFixture.Json.GetProperty("titanic").GetProperty("served");
        var network = new TitanicAsCode();
        var saved = new SafetensorsFile(network, new BinaryCrossEntropy()).Read(Renamed("titanic.safetensors", ByTitanicCodeName));

        var chances = saved.Network.Predict(Rows(Titanic.Value.Served(Passengers, Needs.OneScale).Features), saved.Loss, Backend).Values.ToArray();
        var stack = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        Assert.Equal(["hidden.weight", "hidden.bias", "output.weight", "output.bias"], network.Slots().Select(named => named.Path));
        Assert.Equal(Values(stack.Network), Values(network));
        Assert.All(Roundings(chances, served.GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 8));
    }

    [Fact]
    public void ConvolutionsWrittenAsCode_AnswerAsPyTorchDoes_WithinAHundredThousandth_OnceWhatEachLinearLayerAndNormalisationReadsIsStated()
    {
        var saved = new SafetensorsFile(new ConvolutionAsCode(), new MeanSquaredError()) { Flattened = StatedForTheCode }
            .Read(Renamed("convolution.safetensors", ByConvolutionCodeName));

        var outputs = saved.Network.Predict(Images(), saved.Loss, Backend).Values.ToArray();
        var expected = PyTorchFixture.Json.GetProperty("convolution").GetProperty("outputs").Floats();

        Assert.Equal(6, outputs.Length);
        Assert.All(outputs.Zip(expected, (ours, theirs) => Math.Abs(ours - theirs)), apart => Assert.InRange(apart, 0, 1e-5));
    }

    [Fact]
    public void ConvolutionsWrittenAsCode_WithNothingStated_AreRefusedAtEveryLinearLayerAndNormalisation_ForTheirOrderIsTheCodes()
    {
        var network = new ConvolutionAsCode();
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(
            () => new SafetensorsFile(network, new MeanSquaredError()).Read(Renamed("convolution.safetensors", ByConvolutionCodeName)));

        Assert.Equal(
            ["linear1.weight", "linear2.weight", "norm1.bias", "norm1.running_mean", "norm1.running_var", "norm1.weight", "norm2.bias", "norm2.running_mean", "norm2.running_var", "norm2.weight"],
            refused.Faults.Select(fault => fault.Slot));
        Assert.Equal(
            "'linear1.weight' is held by a network written as code, whose forward pass keeps to itself whether the layer at linear1 reads rows made of "
            + "images, which PyTorch lays out channel by channel: state what that layer reads in Flattened — the rows, columns and channels of the images, or "
            + "the length of a row.",
            refused.Faults[0].Message);
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void ANetworkWrittenAsCode_HandedRowsAStacksFlattenMade_IsReadAsTheFlattenWasHandedThem_OrAsStated()
    {
        // The flatten stands in the stack and the linear layers reading its rows inside the code, whose order is its own.
        static MemoryStream File() => Renamed("titanic.safetensors", name => $"1.{ByTitanicCodeName(name)}");

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(new LayerStack(new Flatten(), new TitanicAsCode()), new BinaryCrossEntropy()).Read(File()));
        var handedRows = new LayerStack(new Flatten(), new TitanicAsCode());
        new SafetensorsFile(handedRows, new BinaryCrossEntropy()) { Example = new Shape(14) }.Read(File());
        var stated = new LayerStack(new Flatten(), new TitanicAsCode());
        new SafetensorsFile(stated, new BinaryCrossEntropy())
        {
            Flattened = new Dictionary<string, Shape> { ["1.hidden"] = new Shape(14), ["1.output"] = new Shape(16) },
        }.Read(File());

        var stack = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));
        Assert.Equal(["1.hidden.weight", "1.output.weight"], refused.Faults.Select(fault => fault.Slot));
        Assert.Equal(Values(stack.Network), Values(handedRows));
        Assert.Equal(Values(stack.Network), Values(stated));
    }

    [Fact]
    public void ANetworkWrittenAsCode_ThatLaysRowsOutAsImages_IsRefusedAtEveryLinearLayerAndNormalisation_WhateverIsStated()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(new ReshapedAsCode(), new MeanSquaredError())
        {
            Flattened = new Dictionary<string, Shape> { ["linear"] = new Shape(2) },
        }.Read(Written(
        [
            new("linear.weight", DataType.F32, [12, 2], Floats(new float[24])),
            new("linear.bias", DataType.F32, [12], Floats(new float[12])),
            new("norm.weight", DataType.F32, [12], Floats(new float[12])),
            new("norm.bias", DataType.F32, [12], Floats(new float[12])),
        ])));

        Assert.Equal(["linear.bias", "linear.weight", "norm.bias", "norm.weight"], refused.Faults.Select(fault => fault.Slot));
        Assert.Equal(
            "'linear.bias' is held by a network written as code that lays rows out as images with a reshape, which PyTorch reads channel by channel, "
            + "and its forward pass keeps to itself which rows those are: rows a flatten makes of images are turned, and images a reshape makes of rows are not.",
            refused.Faults[0].Message);
    }

    [Fact]
    public void ANetworkWrittenAsCode_RefusesWhatItKeepsOfItsOwn_AndALayerOfAKindTheReaderDoesNotKnow()
    {
        var network = new KeepsItsOwn();

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("scale", DataType.F32, [2], Floats(1, 1)),
            new("scaled.scale", DataType.F32, [2], Floats(1, 1)),
            new("linear.weight", DataType.F32, [1, 2], Floats(1, 2)),
            new("linear.bias", DataType.F32, [1], Floats(0)),
        ])));

        Assert.Equal(
            [
                "'scale' is held by a KeepsItsOwn, and how PyTorch lays out a layer of that kind is not known here.",
                "'scaled.scale' is held by a Scaled, and how PyTorch lays out a layer of that kind is not known here.",
            ],
            refused.Faults.Select(fault => fault.Message));
        Assert.Equal([0f, 0f], network.Linear.Weight.Value.Values.ToArray());
    }

    [Fact]
    public void ALayerOfAKindTheReaderDoesNotKnow_IsRefusedAtItsTensors_AndOneHoldingNothing_IsPassedAsItStands()
    {
        var network = new LayerStack(new PassedOn(), new Scaled(), new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("1.scale", DataType.F32, [2], Floats(1, 1)),
            new("2.weight", DataType.F32, [1, 2], Floats(1, 2)),
            new("2.bias", DataType.F32, [1], Floats(0)),
        ])));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal("'1.scale' is held by a Scaled, and how PyTorch lays out a layer of that kind is not known here.", fault.Message);
    }

    [Fact]
    public void WhatTheCallerSays_IsHeldToTheNetworkItIsSaidOf()
    {
        static ArgumentException Said(SafetensorsFile file) => Assert.ThrowsAny<ArgumentException>(() => file.Read(PyTorchFixture.Open("convolution.safetensors")));

        var noImage = Said(new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Example = new Shape(5) });
        var noFlatten = Said(new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["3"] = new Shape(4, 3, 3) } });
        var both = Said(new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError())
        {
            Example = new Shape(8, 6, 2),
            Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(4, 3, 3) },
        });
        var neither = Said(new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(12, 3) } });
        var tooFew = Said(new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(2, 3, 3) } });
        var tooFewForALinearLayer = Assert.Throws<ArgumentException>(() => new SafetensorsFile(
                new LayerStack(new Flatten(), new Dense(Tensor.Zeros(new Shape(14, 1)), Tensor.Zeros(new Shape(1)))), new MeanSquaredError())
            { Flattened = new Dictionary<string, Shape> { ["0"] = new Shape(2, 2, 2) } }
            .Read(PyTorchFixture.Open("titanic.safetensors")));

        Assert.Equal("Example", noImage.ParamName);
        Assert.StartsWith("An example of 5 does not go through the network: ", noImage.Message, StringComparison.Ordinal);
        Assert.Equal("Flattened", noFlatten.ParamName);
        Assert.StartsWith(
            "'3' is neither a layer of a stack that makes rows of what it is handed nor a linear layer or normalisation of a network written as code, so nothing can be stated of it.",
            noFlatten.Message,
            StringComparison.Ordinal);
        Assert.StartsWith("An example and what each flatten is handed say one thing twice: hand one of them.", both.Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 4 is handed is stated as 12x3, and what is stated is an image — rows, columns and channels — or a row.", neither.Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 4 is handed is stated as 2x3x3, which holds 18 values, and the layer at 5 reads 36.", tooFew.Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 0 is handed is stated as 2x2x2, which holds 8 values, and the layer at 1 reads 14.", tooFewForALinearLayer.Message, StringComparison.Ordinal);
        Assert.All([noFlatten, neither, tooFew, tooFewForALinearLayer, both], said => Assert.Equal("Flattened", said.ParamName));
    }

    [Fact]
    public void NothingIsReadWithoutANetwork_ALoss_AndAFile()
    {
        Assert.Throws<ArgumentNullException>(() => new SafetensorsFile(null!, new MeanSquaredError()));
        Assert.Throws<ArgumentNullException>(() => new SafetensorsFile(TitanicInKerasWords(), null!));
        Assert.Throws<ArgumentNullException>(() => new SafetensorsFile(TitanicInKerasWords(), new MeanSquaredError()).Read(null!));
        Assert.Throws<ArgumentNullException>(() => new SafetensorsFile(TitanicInKerasWords(), new MeanSquaredError()) { Flattened = null! });
    }

    [Fact]
    public void AnImport_IsWrittenToANetworksOwnFile_AndReadBack_AnsweringToTheLastBit()
    {
        var imported = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));
        var features = Rows(Titanic.Value.Served(Passengers, Needs.OneScale).Features);

        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, imported.Network, imported.Loss);
            writer.WriteEndObject();
        }

        var read = NetworkDocument.ReadNetwork(System.Text.Encoding.UTF8.GetString(stream.ToArray()), "network", NetworkCatalog.BuiltIn());

        Assert.Equal(Bits(imported.Network.Predict(features, imported.Loss, Backend).Values.ToArray()), Bits(read.Network.Predict(features, read.Loss, Backend).Values.ToArray()));
    }

    private static JsonElement Header(string file)
    {
        var bytes = File.ReadAllBytes(PyTorchFixture.Path(file));

        return JsonDocument.Parse(bytes.AsMemory(8, (int)BinaryPrimitives.ReadUInt64LittleEndian(bytes))).RootElement;
    }

    private static float[] Transposed(float[] values, int rows, int columns) =>
        [.. Enumerable.Range(0, values.Length).Select(at => values[((at % rows) * columns) + (at / rows)])];

    private static byte[] Floats(params float[] values) => [.. values.SelectMany(BitConverter.GetBytes)];

    // The Titanic file with each tensor under another name.
    private static MemoryStream Renamed(string file, Func<string, string> name)
    {
        var archive = SafeTensors.Deserialize(File.ReadAllBytes(PyTorchFixture.Path(file)));

        return new MemoryStream(SafeTensors.Serialize(archive.Tensors().Select(pair => new KeyValuePair<string, TensorView>(name(pair.Key), pair.Value)), archive.Metadata.MetadataEntries));
    }

    // A safetensors file of the given tensors, written by the borrowed writer.
    private static MemoryStream Written(Held[] tensors, IReadOnlyDictionary<string, string>? notes = null) =>
        new(SafeTensors.Serialize(tensors.Select(held => new KeyValuePair<string, TensorView>(held.Name, new TensorView(held.Type, held.Shape, held.Bytes))), notes));

    private readonly record struct Held(string Name, DataType Type, ulong[] Shape, byte[] Bytes);

    // What each linear layer and normalisation of the convolutions written as code reads.
    private static readonly Dictionary<string, Shape> StatedForTheCode = new()
    {
        ["norm1"] = new Shape(4),
        ["norm2"] = new Shape(4, 3, 3),
        ["linear1"] = new Shape(4, 3, 3),
        ["linear2"] = new Shape(5),
    };

    // The Titanic tensors under the names the network written as code gives its layers.
    private static string ByTitanicCodeName(string name) => (name[0] == '0' ? "hidden" : "output") + name[1..];

    // The convolutions' tensors under the names the network written as code gives its layers.
    private static string ByConvolutionCodeName(string name) =>
        name[0] switch
        {
            '0' => "conv1",
            '1' => "norm1",
            '3' => "conv2",
            '5' => "norm2",
            '6' => "linear1",
            _ => "linear2",
        } + name[1..];

    // Titanic's network written as code, as a PyTorch module with a hidden and an output layer is.
    private sealed class TitanicAsCode : Network
    {
        private readonly Dense _hidden;
        private readonly Dense _output;

        public TitanicAsCode()
        {
            var draws = new RandomStream(7).Draw("initialise", 0, 0);

            _hidden = AddLayer("hidden", new Dense(14, 16, draws));
            _output = AddLayer("output", new Dense(16, 1, draws));
        }

        protected override Tensor Compute(Tensor input, Pass pass) => _output.Forward(pass.Backend.Relu(_hidden.Forward(input, pass)), pass);
    }

    // pytorch.py's convolutions written as code, flattening the images in its own forward pass.
    private sealed class ConvolutionAsCode : Network
    {
        private readonly Conv2D _conv1;
        private readonly BatchNorm _norm1;
        private readonly Conv2D _conv2;
        private readonly BatchNorm _norm2;
        private readonly Dense _linear1;
        private readonly Dense _linear2;

        public ConvolutionAsCode()
        {
            var draws = new RandomStream(7).Draw("initialise", 0, 0);

            _conv1 = AddLayer("conv1", new Conv2D(2, 4, new Window(3, 2) { Padding = 1 }, draws));
            _norm1 = AddLayer("norm1", new BatchNorm(4));
            _conv2 = AddLayer("conv2", new Conv2D(4, 3, new Window(2, 2) { Stride = 2 }, draws));
            _norm2 = AddLayer("norm2", new BatchNorm(36));
            _linear1 = AddLayer("linear1", new Dense(36, 5, draws));
            _linear2 = AddLayer("linear2", new Dense(5, 1, draws));
        }

        protected override Tensor Compute(Tensor input, Pass pass)
        {
            var backend = pass.Backend;
            var images = _conv2.Forward(backend.Relu(_norm1.Forward(_conv1.Forward(input, pass), pass)), pass);
            var rows = backend.Reshape(images, new Shape(images.Shape[0], 36));

            return _linear2.Forward(backend.Tanh(_linear1.Forward(_norm2.Forward(rows, pass), pass)), pass);
        }
    }

    // A network written as code that lays the rows a linear layer makes out as images, having normalised them.
    private sealed class ReshapedAsCode : Network
    {
        public ReshapedAsCode()
        {
            AddLayer("linear", new Dense(Tensor.Zeros(new Shape(2, 12)), Tensor.Zeros(new Shape(12))));
            AddLayer("norm", new LayerNorm(12));
            AddLayer("images", new Reshape(new Shape(2, 2, 3)));
        }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    // A network written as code keeping a number of its own, beside a layer of somebody's own and a linear layer.
    private sealed class KeepsItsOwn : Network
    {
        public KeepsItsOwn()
        {
            AddParameter("scale", Tensor.From(new Shape(2), [1f, 1f]));
            AddLayer("scaled", new Scaled());
            Linear = AddLayer("linear", new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        }

        public Dense Linear { get; }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    // A layer of somebody's own that scales each feature by a number it learns.
    private sealed class Scaled : Layer
    {
        public Scaled() => AddParameter("scale", Tensor.From(new Shape(2), [1f, 1f]));

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }

    // A layer of somebody's own that holds nothing and hands on what it is handed.
    private sealed class PassedOn : Layer
    {
        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
