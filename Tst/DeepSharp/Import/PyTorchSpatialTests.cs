// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnxify.Safetensors;
using static DeepSharp.Tests.Import.HandWrittenPickle;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of networks that walk series, images and volumes — convolutions along one, two and three axes, poolings, global
/// poolings, dropouts of whole channels — as PyTorch saved it as safetensors and as torch.save writes it, read into the same
/// networks written here and held to what PyTorch answered: each kernel laid out channels last, and the rows a flatten makes of a
/// series, an image or a volume read place by place where PyTorch reads them channel by channel.
/// </summary>
public class PyTorchSpatialTests
{
    private static readonly string[] Names = ["series", "series-flat", "pooling", "pooling-global", "volume", "volume-global"];

    public static TheoryData<string> Networks => [.. Names];

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in Names)
            {
                files.Add(name, "safetensors");
                files.Add(name, "pt");
            }

            return files;
        }
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void ANetworkWalkingSeriesImagesOrVolumes_ReadFromWhatPyTorchSaved_AnswersAsPyTorchDoes_WithinAHundredThousandth(string name, string kind)
    {
        var spatial = Spatial(name);
        var saved = Reader(kind, spatial.Network(), spatial.Example).Read(PyTorchFixture.Open($"spatial-{name}.{kind}"));

        var outputs = saved.Network.Predict(SpatialExamples(name), saved.Loss, Backend).Values.ToArray();
        var expected = SpatialOutputs(name);

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(Farthest(outputs, expected), 0, 1e-5);
    }

    [Theory]
    [MemberData(nameof(Networks))]
    public void TheStateAsTorchSaveWritesIt_GoesInAsTheSafetensorsReaderPutsIt_BitForBit(string name)
    {
        var spatial = Spatial(name);
        var read = spatial.Network();
        var expected = spatial.Network();

        new TorchSaveFile(read, new MeanSquaredError()) { Example = spatial.Example }.Read(PyTorchFixture.Open($"spatial-{name}.pt"));
        new SafetensorsFile(expected, new MeanSquaredError()) { Example = spatial.Example }.Read(PyTorchFixture.Open($"spatial-{name}.safetensors"));

        Assert.Equal(Values(expected).Select(Bits), Values(read).Select(Bits));
    }

    [Theory]
    [InlineData("series-flat", "4", new[] { 5, 3 })]
    [InlineData("pooling", "7", new[] { 4, 4, 3 })]
    [InlineData("volume", "7", new[] { 4, 4, 3, 2 })]
    public void WhatAFlattenIsHanded_StatedByTheCaller_ReadsTheNumbersAnExampleDoes_AndStatedAsARow_AnswersOtherwise(string name, string flatten, int[] handed)
    {
        var spatial = Spatial(name);
        var file = $"spatial-{name}.safetensors";
        var byExample = new SafetensorsFile(spatial.Network(), new MeanSquaredError()) { Example = spatial.Example }.Read(PyTorchFixture.Open(file));
        var stated = new SafetensorsFile(spatial.Network(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { [flatten] = new Shape(handed) } }.Read(PyTorchFixture.Open(file));
        var asRows = new SafetensorsFile(spatial.Network(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { [flatten] = new Shape(new Shape(handed).Count) } }.Read(PyTorchFixture.Open(file));

        Assert.Equal(Values(byExample.Network).Select(Bits), Values(stated.Network).Select(Bits));
        // Said to be handed rows, the linear layer takes PyTorch's weights row for row — channel by channel, where the rows this
        // network flattens run place by place — so what it answers is no longer what PyTorch answered.
        Assert.True(Farthest(asRows.Network.Predict(SpatialExamples(name), asRows.Loss, Backend).Values.ToArray(), SpatialOutputs(name)) > 1e-3f);
    }

    [Theory]
    [InlineData("series-flat", "safetensors", new[] { "5.bias", "5.running_mean", "5.running_var", "5.weight", "6.weight" }, 4)]
    [InlineData("pooling", "pt", new[] { "8.weight" }, 7)]
    [InlineData("volume", "safetensors", new[] { "8.weight" }, 7)]
    public void WithNothingSaidOfWhatAFlattenIsHanded_EveryTensorThatTurnsOnIt_IsRefused_AndNoSlotChanges(string name, string kind, string[] turned, int flatten)
    {
        var network = Spatial(name).Network();
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();
        var reader = Reader(kind, network, null);

        var refused = Assert.Throws<SlotLoadException>(() => reader.Read(PyTorchFixture.Open($"spatial-{name}.{kind}")));

        Assert.Equal(turned, refused.Faults.Select(fault => fault.Source).Order(StringComparer.Ordinal));
        Assert.All(
            refused.Faults,
            fault => Assert.Equal(
                $"'{fault.Slot}' reads the rows the layer at {flatten} flattens, which PyTorch lays out channel by channel: hand the reader an example the network takes, or state what that layer is handed.",
                fault.Message));
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void TwoAxesAreASeries_OnlyWhereTheLayerThatWalksAxesNearestBeforeTheFlattenWalksOne()
    {
        // After a convolution along a series: steps by channels, turned. After a reshape, or a convolution over images or
        // volumes with a statement that says two axes: no series, and the reader says so rather than turn rows that are not one.
        static ArgumentException Said(SafetensorsFile file) => Assert.ThrowsAny<ArgumentException>(() => file.Read(PyTorchFixture.Open("convolution.safetensors")));

        var afterAnImage = new LayerStack(new Conv2D(Tensor.Zeros(new Shape(4, 3)), Tensor.Zeros(new Shape(3)), new Window(2, 2)), new Flatten(), new Dense(Tensor.Zeros(new Shape(12, 1)), Tensor.Zeros(new Shape(1))));
        var afterAReshape = new LayerStack(new Reshape(new Shape(3, 4)), new Conv1D(Tensor.Zeros(new Shape(4, 2)), Tensor.Zeros(new Shape(2)), new Window1D(1)), new Reshape(new Shape(3, 2)), new Flatten(), new Dense(Tensor.Zeros(new Shape(6, 1)), Tensor.Zeros(new Shape(1))));

        var image = Said(new SafetensorsFile(afterAnImage, new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["1"] = new Shape(6, 2) } });
        var reshaped = Said(new SafetensorsFile(afterAReshape, new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["3"] = new Shape(3, 2) } });

        Assert.StartsWith("What the layer at 1 is handed is stated as 6x2, and two axes are a series", image.Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 3 is handed is stated as 3x2, and two axes are a series", reshaped.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoAxesAreNoSeries_AfterALinearLayerACodeNetworkOrNothingAtAll()
    {
        static ArgumentException Said(Network network, string path, Shape stated) => Assert.ThrowsAny<ArgumentException>(
            () => new SafetensorsFile(network, new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { [path] = stated } }.Read(PyTorchFixture.Open("convolution.safetensors")));

        var afterALinearLayer = new LayerStack(
            new Dense(Tensor.Zeros(new Shape(2, 6)), Tensor.Zeros(new Shape(6))), new Flatten(), new Dense(Tensor.Zeros(new Shape(6, 1)), Tensor.Zeros(new Shape(1))));
        var afterCode = new LayerStack(new HoldsSpatial("maxpool1d"), new Flatten(), new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        var first = new LayerStack(new Flatten(), new Dense(Tensor.Zeros(new Shape(6, 1)), Tensor.Zeros(new Shape(1))));

        Assert.StartsWith("What the layer at 1 is handed is stated as 3x2, and two axes are a series", Said(afterALinearLayer, "1", new Shape(3, 2)).Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 1 is handed is stated as 2x1, and two axes are a series", Said(afterCode, "1", new Shape(2, 1)).Message, StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 0 is handed is stated as 3x2, and two axes are a series", Said(first, "0", new Shape(3, 2)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoAxesStatedForALayerOfCode_AreASeriesOnlyWhereTheCodeHoldsALayerAlongOneAxis()
    {
        static Exception Read(Network network) => Assert.ThrowsAny<Exception>(
            () => new SafetensorsFile(network, new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["linear"] = new Shape(1, 2) } }.Read(PyTorchFixture.Open("convolution.safetensors")));

        // Along one axis the statement stands, and the read goes on to the file, whose tensors are for another network.
        Assert.IsType<SlotLoadException>(Read(new HoldsSpatial("maxpool1d")));
        Assert.IsType<SlotLoadException>(Read(new HoldsSpatial("globalavgpool1d")));

        var refused = Assert.IsType<ArgumentException>(Read(new HoldsSpatial("avgpool2d")));

        Assert.StartsWith("What the layer at linear is handed is stated as 1x2, and two axes are a series", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFlattenHandedASeriesOfOneStep_TurnsNothing_ForTheChannelsAreAllThereIs()
    {
        // A global pooling that keeps its axes leaves one place: the same rows whichever of the two orders reads them.
        var spatial = Spatial("series");
        var byExample = new SafetensorsFile(spatial.Network(), new MeanSquaredError()) { Example = spatial.Example }.Read(PyTorchFixture.Open("spatial-series.safetensors"));
        var asRows = new SafetensorsFile(spatial.Network(), new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["7"] = new Shape(5) } }.Read(PyTorchFixture.Open("spatial-series.safetensors"));

        Assert.Equal(Values(byExample.Network).Select(Bits), Values(asRows.Network).Select(Bits));
    }

    [Fact]
    public void AnAdaptivePoolingToOnePlace_WrittenAsAGlobalPoolingThatDropsItsAxis_AnswersAsPyTorchDoes_ForTheFlattenAfterItIsHandedARow()
    {
        var kept = new SafetensorsFile(SeriesInKerasWords(keepsAxes: true), new MeanSquaredError()) { Example = new Shape(17, 2) }.Read(PyTorchFixture.Open("spatial-series.safetensors"));
        var dropped = new SafetensorsFile(SeriesInKerasWords(keepsAxes: false), new MeanSquaredError()) { Example = new Shape(17, 2) }.Read(PyTorchFixture.Open("spatial-series.safetensors"));

        var outputs = dropped.Network.Predict(SpatialExamples("series"), dropped.Loss, Backend).Values.ToArray();

        Assert.Equal(Values(kept.Network).Select(Bits), Values(dropped.Network).Select(Bits));
        Assert.InRange(Farthest(outputs, SpatialOutputs("series")), 0, 1e-5);
    }

    [Fact]
    public void AnAveragePoolingThatLeavesOutItsPadding_AnswersOtherwiseThanPyTorchsDefault_ThatCountsIt()
    {
        var leavesOut = new SafetensorsFile(SeriesFlatInKerasWords(countsPadding: false), new MeanSquaredError()) { Example = new Shape(10, 2) }
            .Read(PyTorchFixture.Open("spatial-series-flat.safetensors"));

        var outputs = leavesOut.Network.Predict(SpatialExamples("series-flat"), leavesOut.Loss, Backend).Values.ToArray();

        Assert.True(Farthest(outputs, SpatialOutputs("series-flat")) > 1e-3f);
    }

    [Fact]
    public void AFlattenAnExampleHandsMoreAxesThanAVolumeHas_IsRefusedAtTheLinearLayerReadingIt()
    {
        var network = new LayerStack(new Reshape(new Shape(1, 1, 1, 2, 7)), new Flatten(), new Dense(Tensor.Zeros(new Shape(14, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()) { Example = new Shape(14) }.Read(Written(
            [new("2.weight", DataType.F32, [1, 14], Floats(new float[14])), new("2.bias", DataType.F32, [1], Floats(0))])));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal(
            "'2.weight' reads the rows the layer at 1 makes of 1x1x1x2x7, which is neither a row, a series, an image nor a volume: the channels of a series, an image or a volume are turned to the end, and a row is read as it is.",
            fault.Message);
    }

    [Fact]
    public void RowsAReshapeLaysOutAsASeries_AreNotTurned_ButTheLinearLayerReadingThemBeforeIt_IsRefused()
    {
        var network = new LayerStack(
            new Dense(Tensor.Zeros(new Shape(2, 6)), Tensor.Zeros(new Shape(6))),
            new Reshape(new Shape(3, 2)),
            new Conv1D(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1)), new Window1D(1)),
            new Flatten(),
            new Dense(Tensor.Zeros(new Shape(3, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()) { Example = new Shape(2) }.Read(Written(
        [
            new("0.weight", DataType.F32, [6, 2], Floats(new float[12])),
            new("0.bias", DataType.F32, [6], Floats(new float[6])),
            new("2.weight", DataType.F32, [1, 2, 1], Floats(0, 0)),
            new("2.bias", DataType.F32, [1], Floats(0)),
            new("4.weight", DataType.F32, [1, 3], Floats(0, 0, 0)),
            new("4.bias", DataType.F32, [1], Floats(0)),
        ])));

        Assert.Equal(["0.bias", "0.weight"], refused.Faults.Select(fault => fault.Slot).Order(StringComparer.Ordinal));
        Assert.Equal(
            "'0.weight' makes the rows the layer at 1 lays out as 3x2, which PyTorch lays out channel by channel: rows a flatten makes of images are turned, and images a reshape makes of rows are not.",
            refused.Faults.Single(fault => fault.Slot == "0.weight").Message);
    }

    [Fact]
    public void AConvolutionAlongASeries_TakesPyTorchsKernelWithItsWindowFirst_ThenItsChannelsIn_ThenItsChannelsOut()
    {
        var network = new LayerStack(new Conv1D(Tensor.Zeros(new Shape(4, 2)), Tensor.Zeros(new Shape(2)), new Window1D(2)));

        new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("0.weight", DataType.F32, [2, 2, 2], Floats(1, 2, 3, 4, 5, 6, 7, 8)),
            new("0.bias", DataType.F32, [2], Floats(10, 20)),
        ]));

        var convolution = (Conv1D)network.Layers[0];
        Assert.Equal([1f, 5f, 3f, 7f, 2f, 6f, 4f, 8f], convolution.Weight.Value.Values.ToArray());
        Assert.Equal([10f, 20f], convolution.Bias.Value.Values.ToArray());
    }

    [Fact]
    public void AConvolutionThroughAVolume_TakesPyTorchsKernelWithItsWindowFirst_ThenItsChannelsIn_ThenItsChannelsOut()
    {
        var network = new LayerStack(new Conv3D(Tensor.Zeros(new Shape(8, 1)), Tensor.Zeros(new Shape(1)), new Window3D(1, 2, 2)));

        new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("0.weight", DataType.F32, [1, 2, 1, 2, 2], Floats(0, 1, 2, 3, 4, 5, 6, 7)),
            new("0.bias", DataType.F32, [1], Floats(9)),
        ]));

        var convolution = (Conv3D)network.Layers[0];
        Assert.Equal([0f, 4f, 1f, 5f, 2f, 6f, 3f, 7f], convolution.Weight.Value.Values.ToArray());
        Assert.Equal([9f], convolution.Bias.Value.Values.ToArray());
    }

    [Fact]
    public void AKernelOfAnotherShapeThanItsLayerKeeps_IsNamedByTheShapesOfBoth_AsOneInGroupsIs()
    {
        // PyTorch keeps a grouped convolution's kernel with fewer channels in: it has no layer here, and shows as a shape that does not fit.
        var series = new LayerStack(new Conv1D(Tensor.Zeros(new Shape(6, 4)), Tensor.Zeros(new Shape(4)), new Window1D(3)));
        var volume = new LayerStack(new Conv3D(Tensor.Zeros(new Shape(8, 3)), Tensor.Zeros(new Shape(3)), new Window3D(2, 2, 2)));

        var grouped = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(series, new MeanSquaredError()).Read(Written(
            [new("0.weight", DataType.F32, [4, 1, 3], Floats(new float[12])), new("0.bias", DataType.F32, [4], Floats(new float[4]))])));
        var planar = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(volume, new MeanSquaredError()).Read(Written(
            [new("0.weight", DataType.F32, [3, 1, 2, 2], Floats(new float[12])), new("0.bias", DataType.F32, [3], Floats(new float[3]))])));

        Assert.Equal([new SlotLoadFault("0.weight", "0.weight", "'0.weight' is a 4x2x3 tensor in PyTorch's layout here, and is written as 4x1x3.")], grouped.Faults);
        Assert.Equal([new SlotLoadFault("0.weight", "0.weight", "'0.weight' is a 3x1x2x2x2 tensor in PyTorch's layout here, and is written as 3x1x2x2.")], planar.Faults);
    }

    [Fact]
    public void PoolingsAndDropoutsHoldNoNumbers_ATensorForOneOfThem_IsForNoSlot_AndTheLinearLayerAfterThemIsReadAsItStands()
    {
        var network = new LayerStack(
            new MaxPool1D(new Window1D(2)),
            new GlobalAvgPool1D(),
            new SpatialDropout1D(0.5),
            new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("0.weight", DataType.F32, [1], Floats(1)),
            new("1.weight", DataType.F32, [1], Floats(1)),
            new("2.weight", DataType.F32, [1], Floats(1)),
            new("3.weight", DataType.F32, [1, 2], Floats(1, 2)),
            new("3.bias", DataType.F32, [1], Floats(3)),
        ])));

        Assert.Equal(
            [
                new SlotLoadFault("0.weight", "0.weight", "'0.weight' is no slot of this network."),
                new SlotLoadFault("1.weight", "1.weight", "'1.weight' is no slot of this network."),
                new SlotLoadFault("2.weight", "2.weight", "'2.weight' is no slot of this network."),
            ],
            refused.Faults);

        var read = new LayerStack(new MaxPool1D(new Window1D(2)), new GlobalAvgPool1D(), new SpatialDropout1D(0.5), new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        new SafetensorsFile(read, new MeanSquaredError()).Read(Written([new("3.weight", DataType.F32, [1, 2], Floats(1, 2)), new("3.bias", DataType.F32, [1], Floats(3))]));
        Assert.Equal([1f, 2f], ((Dense)read.Layers[3]).Weight.Value.Values.ToArray());
    }

    [Fact]
    public void ASeriesWrittenAsCode_AnswersAsPyTorchDoes_WithinAHundredThousandth_OnceWhatEachLinearLayerAndNormalisationReadsIsStated()
    {
        var saved = new SafetensorsFile(new SeriesAsCode(), new MeanSquaredError()) { Flattened = StatedForTheSeries }
            .Read(Renamed("spatial-series-flat.safetensors", BySeriesCodeName));

        var outputs = saved.Network.Predict(SpatialExamples("series-flat"), saved.Loss, Backend).Values.ToArray();

        Assert.InRange(Farthest(outputs, SpatialOutputs("series-flat")), 0, 1e-5);
    }

    [Fact]
    public void ASeriesWrittenAsCode_WithNothingStated_IsRefusedAtEveryLinearLayerAndNormalisation_ForTheirOrderIsTheCodes()
    {
        var network = new SeriesAsCode();
        var before = network.Slots().Select(named => named.Slot.Value).ToArray();

        var refused = Assert.Throws<SlotLoadException>(
            () => new SafetensorsFile(network, new MeanSquaredError()).Read(Renamed("spatial-series-flat.safetensors", BySeriesCodeName)));

        Assert.Equal(
            ["linear1.weight", "linear2.weight", "norm.bias", "norm.running_mean", "norm.running_var", "norm.weight"],
            refused.Faults.Select(fault => fault.Slot));
        Assert.Equal(
            "'linear1.weight' is held by a network written as code, whose forward pass keeps to itself whether the layer at linear1 reads rows made of "
            + "images, which PyTorch lays out channel by channel: state what that layer reads in Flattened — the steps and channels of a series, the rows, columns and channels "
            + "of an image, the planes, rows, columns and channels of a volume, or the length of a row.",
            refused.Faults[0].Message);
        Assert.Equal(before, network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Theory]
    [InlineData("maxpool1d")]
    [InlineData("avgpool2d")]
    [InlineData("globalmaxpool3d")]
    [InlineData("globalavgpool1d")]
    [InlineData("spatialdropout2d")]
    public void ANetworkWrittenAsCode_HoldingALayerThatWalksSeriesImagesOrVolumes_MayHandItsLinearLayerRowsMadeOfThem_SoThatLayerIsReadOnlyAsStated(string kind)
    {
        var tensors = new Held[] { new("linear.weight", DataType.F32, [1, 2], Floats(1, 2)), new("linear.bias", DataType.F32, [1], Floats(3)) };

        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(new HoldsSpatial(kind), new MeanSquaredError()).Read(Written(tensors)));
        var network = new HoldsSpatial(kind);
        new SafetensorsFile(network, new MeanSquaredError()) { Flattened = new Dictionary<string, Shape> { ["linear"] = new Shape(2) } }.Read(Written(tensors));

        Assert.Equal(["linear.weight"], refused.Faults.Select(fault => fault.Slot));
        Assert.Equal([1f, 2f], network.Slots().Single(named => named.Path == "linear.weight").Slot.Value.Values.ToArray());
    }

    [Fact]
    public void WhatTheCallerStatesOfARowASeriesAnImageOrAVolume_IsHeldToTheSameNumberOfValuesTheLayerReads()
    {
        var spatial = Spatial("series-flat");

        var tooMany = Assert.Throws<ArgumentException>(() => new SafetensorsFile(spatial.Network(), new MeanSquaredError())
        {
            Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(5, 4) },
        }.Read(PyTorchFixture.Open("spatial-series-flat.safetensors")));
        var neither = Assert.Throws<ArgumentException>(() => new SafetensorsFile(spatial.Network(), new MeanSquaredError())
        {
            Flattened = new Dictionary<string, Shape> { ["4"] = new Shape(1, 5, 1, 3, 1) },
        }.Read(PyTorchFixture.Open("spatial-series-flat.safetensors")));
        var nothing = Assert.Throws<ArgumentException>(() => new SafetensorsFile(spatial.Network(), new MeanSquaredError())
        {
            Flattened = new Dictionary<string, Shape> { ["4"] = new Shape() },
        }.Read(PyTorchFixture.Open("spatial-series-flat.safetensors")));

        Assert.StartsWith("What the layer at 4 is handed is stated as 5x4, which holds 20 values, and the layer at 5 reads 15.", tooMany.Message, StringComparison.Ordinal);
        Assert.StartsWith(
            "What the layer at 4 is handed is stated as 1x5x1x3x1, and what is stated is a row, a series — steps and channels —, an image — rows, columns and channels — or a volume — planes, rows, columns and channels.",
            neither.Message,
            StringComparison.Ordinal);
        Assert.StartsWith("What the layer at 4 is handed is stated as scalar, and what is stated is a row, ", nothing.Message, StringComparison.Ordinal);
        Assert.All([tooMany, neither, nothing], said => Assert.Equal("Flattened", said.ParamName));
    }

    private static PyTorchFile Reader(string kind, Network network, Shape? example) =>
        kind == "pt"
            ? new TorchSaveFile(network, new MeanSquaredError()) { Example = example }
            : new SafetensorsFile(network, new MeanSquaredError()) { Example = example };

    // What each normalisation and linear layer of the series written as code reads.
    private static readonly Dictionary<string, Shape> StatedForTheSeries = new()
    {
        ["norm"] = new Shape(5, 3),
        ["linear1"] = new Shape(5, 3),
        ["linear2"] = new Shape(4),
    };

    // The series network's tensors under the names the network written as code gives its layers.
    private static string BySeriesCodeName(string name) =>
        name[0] switch
        {
            '0' => "conv1",
            '3' => "conv2",
            '5' => "norm",
            '6' => "linear1",
            _ => "linear2",
        } + name[1..];

    // pytorch-spatial.py's second series network written as code, flattening the series in its own forward pass.
    private sealed class SeriesAsCode : Network
    {
        private readonly Conv1D _conv1;
        private readonly AvgPool1D _pool;
        private readonly Conv1D _conv2;
        private readonly BatchNorm _norm;
        private readonly Dense _linear1;
        private readonly Dense _linear2;

        public SeriesAsCode()
        {
            var draws = new RandomStream(7).Draw("initialise", 0, 0);

            _conv1 = AddLayer("conv1", new Conv1D(2, 4, new Window1D(4) { PaddingMode = PaddingMode.Same }, draws));
            _pool = AddLayer("pool", new AvgPool1D(new Window1D(2) { Stride = 2, Padding = 1 }) { CountsPadding = true });
            _conv2 = AddLayer("conv2", new Conv1D(4, 3, new Window1D(2), draws));
            _norm = AddLayer("norm", new BatchNorm(15));
            _linear1 = AddLayer("linear1", new Dense(15, 4, draws));
            _linear2 = AddLayer("linear2", new Dense(4, 1, draws));
        }

        protected override Tensor Compute(Tensor input, Pass pass)
        {
            var backend = pass.Backend;
            var series = _conv2.Forward(_pool.Forward(backend.Relu(_conv1.Forward(input, pass)), pass), pass);
            var rows = backend.Reshape(series, new Shape(series.Shape[0], 15));

            return _linear2.Forward(backend.Tanh(_linear1.Forward(_norm.Forward(rows, pass), pass)), pass);
        }
    }

    // A network written as code that holds a layer walking series, images or volumes beside a linear layer; it is never run.
    private sealed class HoldsSpatial : Network
    {
        public HoldsSpatial(string kind)
        {
            Layer spatial = kind switch
            {
                "maxpool1d" => new MaxPool1D(new Window1D(2)),
                "avgpool2d" => new AvgPool2D(new Window(2, 2)),
                "globalmaxpool3d" => new GlobalMaxPool3D(),
                "globalavgpool1d" => new GlobalAvgPool1D(),
                _ => new SpatialDropout2D(0.5),
            };

            AddLayer("spatial", spatial);
            AddLayer("linear", new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));
        }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
