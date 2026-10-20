// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// A network that walks a window along a series, over an image or through a volume, written as an ONNX graph that takes its examples
/// channels first, as ONNX's operators do: the kernels written channels out by channels in by the window's places, the border worked out
/// for the examples that reach each layer, and the rows a flatten makes of them laid out channel by channel, with the numbers after the
/// flatten in the order the graph has the row. What is written is read back by DeepSharp's reader and answered through by ONNX Runtime.
/// </summary>
public class OnnxSpatialExportTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    // ReadBack: whether the reader reads the graph back; a dense layer over channels first is turned round in the graph by transposes it does not read.
    private sealed record Walk(Func<LayerStack> Network, Shape Each, bool ReadBack = true);

    private static readonly Dictionary<string, Walk> Walks = new()
    {
        ["series"] = new(() => new Sequential().Conv1D(4, new Window1D(3) { Padding = 1 }).Relu().MaxPool1D(new Window1D(2) { Stride = 2 }).Conv1D(3, new Window1D(2)).Flatten().Dense(2).Lower(new Shape(8, 2), new RandomStream(1)), new Shape(8, 2)),
        ["image"] = new(() => new Sequential().Conv2D(4, new Window(3, 3) { PaddingMode = PaddingMode.Same }).BatchNorm().Relu().MaxPool2D(new Window(2, 2) { Stride = 2 }).AvgPool2D(new Window(2, 2)).Flatten().Dense(3)
            .Lower(new Shape(6, 6, 2), new RandomStream(2)), new Shape(6, 6, 2)),
        ["volume"] = new(() => new Sequential().Conv3D(3, new Window3D(2, 2, 2)).Relu().Flatten().Dense(2).Lower(new Shape(4, 4, 4, 2), new RandomStream(3)), new Shape(4, 4, 4, 2)),
        ["transposed"] = new(() => new Sequential().Conv2D(3, new Window(2, 2) { Stride = 2 }).Tanh().Conv2DTranspose(2, new Window(2, 2) { Stride = 2 }).Flatten().Dense(1).Lower(new Shape(6, 6, 2), new RandomStream(4)), new Shape(6, 6, 2)),
        ["transposed series and volume"] = new(() => new Sequential().Conv1DTranspose(3, new Window1D(2) { Stride = 2 }).GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 2), new RandomStream(5)), new Shape(5, 2)),
        ["global pooling dropped"] = new(() => new Sequential().Conv2D(4, new Window(3, 3)).Relu().GlobalAvgPool2D().Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(6)), new Shape(6, 6, 2)),
        ["global pooling kept"] = new(() => new Sequential().Conv2D(4, new Window(3, 3)).GlobalMaxPool2D(keepsAxes: true).Flatten().Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(7)), new Shape(6, 6, 2)),
        ["pads"] = new(() => new Sequential().Pad2D(new Margins(new Sides(1, 2), new Sides(0, 1)), 0.5f).Conv2D(3, new Window(2, 2)).Pad2D(new Margins(new Sides(1, 1), new Sides(2, 0)), Filling.Mirror)
            .Pad2D(new Margins(new Sides(1, 0), new Sides(0, 1)), Filling.Repeat).Flatten().Dense(1).Lower(new Shape(5, 5, 2), new RandomStream(8)), new Shape(5, 5, 2)),
        ["groups, dilation and stride"] = new(() => new Sequential().Conv2D(4, new Window(2, 2) { Dilation = 2, Stride = 2 }, 2).Relu().Flatten().Dense(2).Lower(new Shape(8, 8, 2), new RandomStream(9)), new Shape(8, 8, 2)),
        ["normalisations of channels"] = new(() => new Sequential().Conv2D(4, new Window(3, 3)).GroupNorm(2).InstanceNorm().PRelu(true).Flatten().Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(10)), new Shape(6, 6, 2)),
        ["batch normalisation of flattened"] = new(() => new Sequential().Conv2D(3, new Window(3, 3)).Flatten().BatchNorm().LayerNorm().Dense(2).Lower(new Shape(5, 5, 2), new RandomStream(11)), new Shape(5, 5, 2)),
        ["dropouts"] = new(() => new Sequential().Conv2D(4, new Window(3, 3)).SpatialDropout2D(0.3).Reshape(new Shape(64)).Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(12)), new Shape(6, 6, 2)),
        ["residual image"] = new(() => new Sequential().Conv2D(2, new Window(3, 3) { PaddingMode = PaddingMode.Same }).Named("c").Add("input", "c").Conv2D(3, new Window(2, 2)).Flatten().Dense(1)
            .Lower(new Shape(5, 5, 2), new RandomStream(13)), new Shape(5, 5, 2)),
        ["columns of channels"] = new(() => new Sequential().Conv2D(4, new Window(2, 2)).Named("c").Columns(0, 2).Takes("c").Named("lo").Columns(2, 2).Takes("c").Named("hi").Concatenate("hi", "lo").Flatten().Dense(1)
            .Lower(new Shape(5, 5, 2), new RandomStream(14)), new Shape(5, 5, 2)),
        ["a dense layer on the channels of every place"] = new(() => new Sequential().Conv2D(3, new Window(2, 2)).Dense(4).Relu().GlobalAvgPool2D().Dense(1).Lower(new Shape(5, 5, 2), new RandomStream(16)), new Shape(5, 5, 2), ReadBack: false),
        ["a dense layer with a pair on a series"] = new(() => new Sequential().Conv1D(3, new Window1D(2)).Dense(4, new Beginning { LowRank = new LowRank(2, 4) }).GlobalMaxPool1D().Dense(1).Lower(new Shape(6, 2), new RandomStream(17)), new Shape(6, 2), ReadBack: false),
        ["a softmax over channels"] = new(() => new Sequential().Conv2D(3, new Window(2, 2)).Softmax().Flatten().Dense(1).Lower(new Shape(4, 4, 2), new RandomStream(15)), new Shape(4, 4, 2)),
    };

    public static TheoryData<string> EveryWalk => [.. Walks.Keys];

    public static TheoryData<string> EveryWalkTheReaderReads => [.. Walks.Where(walk => walk.Value.ReadBack).Select(walk => walk.Key)];

    private static LayerStack Filled(string name) => Runtime.Filled(Walks[name].Network());

    [Theory]
    [MemberData(nameof(EveryWalk))]
    public void AWrittenGraph_IsAnsweredThroughByOnnxRuntime_AsTheNetworkAnswers(string name)
    {
        var network = Filled(name);
        var loss = new MeanSquaredError();
        var each = Walks[name].Each;
        var rows = Runtime.Batch(3, each);
        var graph = new OnnxExport(each).ToBytes(network, loss);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, Runtime.ChannelsFirst(rows));

        Assert.Equal(here.Length, there.Length);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Theory]
    [MemberData(nameof(EveryWalkTheReaderReads))]
    public void AWrittenGraph_IsReadBackByTheReader_AnsweringAsTheNetworkDoes(string name)
    {
        var network = Filled(name);
        var loss = new MeanSquaredError();
        var each = Walks[name].Each;
        var rows = Runtime.Batch(3, each);
        var read = new OnnxFile(loss).Read(new MemoryStream(new OnnxExport(each).ToBytes(network, loss)));

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var back = read.Network.Predict(rows, read.Loss, Engine).Values.ToArray();

        Assert.InRange(here.Zip(back, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-6);
    }

    [Fact]
    public void TheGraphTakesItsExamplesChannelsFirst_AndEachKernelIsWrittenChannelsOutByChannelsInByTheWindow()
    {
        var model = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(6, 6, 2)).ToBytes(Filled("image"), new MeanSquaredError()));
        var input = Assert.Single(model.Graph.Input);

        Assert.Equal([2L, 6L, 6L], input.Type.TensorType.Shape.Dim.Skip(1).Select(dim => dim.DimValue));
        Assert.Equal([4L, 2L, 3L, 3L], model.Graph.Initializer.Single(each => each.Name == "0.weight").Dims);
    }

    [Fact]
    public void TheFlattenOfAnImage_IsOnnxsFlatten_AndTheDenseLayerAfterItIsWrittenInTheOrderOfItsRow()
    {
        var network = Filled("image");
        var model = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(6, 6, 2)).ToBytes(network, new MeanSquaredError()));
        var dense = (Dense)network.Layers[^1];
        var written = model.Graph.Initializer.Single(each => each.Name == $"{network.Steps.Count - 1}.weight");

        Assert.Contains("Flatten", model.Graph.Node.Select(node => node.OpType));
        Assert.NotEqual(dense.Weight.Value.Values.ToArray(), [.. Enumerable.Range(0, written.RawData.Length / sizeof(float)).Select(at => BitConverter.ToSingle(written.RawData.Span.Slice(at * sizeof(float), sizeof(float))))]);
    }

    [Fact]
    public void ANetworkThatEndsInAFlattenedImage_IsRefusedByName_ForTheGraphWouldGiveItsValuesInAnotherOrder()
    {
        var network = Runtime.Filled(new Sequential().Conv2D(2, new Window(2, 2)).Flatten().Lower(new Shape(4, 4, 2), new RandomStream(1)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("channel by channel", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("conv", "A layer normalisation")]
    [InlineData("rms", "A root-mean-square normalisation")]
    public void ANormalisationOfTheChannelsOfEveryPlace_IsRefusedByName_ForOnnxsOperatorsNormaliseToTheLastAxis(string kind, string message)
    {
        var words = new Sequential().Conv2D(2, new Window(2, 2));
        var network = Runtime.Filled((kind == "conv" ? words.LayerNorm() : words.RmsNorm(1e-6)).Flatten().Dense(1).Lower(new Shape(4, 4, 2), new RandomStream(1)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains(message, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFlattenedImageLaidBesideOtherValues_OrAddedToARowOfAnotherOrder_IsRefusedByName()
    {
        var beside = Runtime.Filled(new Sequential().Conv2D(2, new Window(2, 2)).Flatten().Named("flat").Dense(4).Named("dense").Concatenate("flat", "dense").Dense(1).Lower(new Shape(4, 4, 2), new RandomStream(1)));
        var added = Runtime.Filled(new Sequential().Conv2D(2, new Window(2, 2)).Flatten().Named("flat").Dense(18).Named("dense").Add("flat", "dense").Dense(1).Lower(new Shape(4, 4, 2), new RandomStream(1)));

        Assert.Contains("two orders", Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(beside, new MeanSquaredError())).Message, StringComparison.Ordinal);
        Assert.Contains("one order", Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(added, new MeanSquaredError())).Message, StringComparison.Ordinal);
    }

}
