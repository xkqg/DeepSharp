// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Networks PyTorch exported to ONNX that walk a series, an image or a volume — convolutions, max and average poolings, a
/// pooling of every place into one — read into layers here: each node lowered onto the layer it is, its numbers laid out
/// for the channels-last layout, and answering as PyTorch answered. Borders PyTorch cannot write — TensorFlow's 'same' —
/// are held to the answers PyTorch gave when the same borders were padded by hand.
/// </summary>
public class OnnxSpatialFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    // The nodes, by name, the TorchScript exporter gave the first of the trunk's average poolings, and a convolution.
    private const string FirstAverage = "/0/0.5/AveragePool";

    private const string FirstConvolution = "/0/0.0/Conv";

    private const string LastConvolution = "/0/0.7/Conv";

    public static TheoryData<string> Nets => new("series", "image", "volume");

    public static TheoryData<string, string, string> Exports
    {
        get
        {
            var data = new TheoryData<string, string, string>();

            foreach (var net in new[] { "series", "image", "volume" })
            {
                data.Add(net, "flat", $"onnx-{net}.onnx");
                data.Add(net, "flat", $"onnx-{net}-torchscript.onnx");
                data.Add(net, "average", $"onnx-{net}-global-average.onnx");
                data.Add(net, "max", $"onnx-{net}-global-max.onnx");
                data.Add(net, "average", $"onnx-{net}-global-average-default.onnx");
                data.Add(net, "max", $"onnx-{net}-global-max-default.onnx");
                data.Add(net, "max", $"onnx-{net}-global-max-opset17.onnx");
            }

            return data;
        }
    }

    // The poolings of every place into one, by the net, whether it is an average, and the file that has it: written by the
    // TorchScript exporter as a global pooling, by the default exporter as a ReduceMean or a ReduceMax.
    public static TheoryData<string, bool, string> Poolings
    {
        get
        {
            var data = new TheoryData<string, bool, string>();

            foreach (var net in new[] { "series", "image", "volume" })
            {
                data.Add(net, true, $"onnx-{net}-global-average.onnx");
                data.Add(net, false, $"onnx-{net}-global-max.onnx");
                data.Add(net, true, $"onnx-{net}-global-average-default.onnx");
                data.Add(net, false, $"onnx-{net}-global-max-default.onnx");
                data.Add(net, false, $"onnx-{net}-global-max-opset17.onnx");
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Exports))]
    public void ANetworkPyTorchExported_GivesEachExampleTheOutputPyTorchGaveIt_WithinAHundredThousandth(string net, string head, string file)
    {
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        using var graph = OnnxFixtures.Open(file);
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);

        // The examples go in with their channels last, as every example here does; PyTorch was handed the same ones turned.
        var examples = OnnxFixtures.Images(torch.GetProperty("input"));
        var outputs = saved.Network.Predict(examples, saved.Loss, Engine);

        Assert.Equal(new Shape(examples.Shape[0], 3), outputs.Shape);
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty(head)), outputs), 0, 1e-5);
    }

    [Theory]
    [InlineData("onnx-series.onnx")]
    [InlineData("onnx-series-torchscript.onnx")]
    public void ASeriesNetwork_IsBuiltAsItsNodesSay_ItsDropoutOfWholeChannelsNoLayerAtAll(string file)
    {
        using var graph = OnnxFixtures.Open(file);
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;

        Assert.Collection(
            layers,
            layer => AssertConvolution(layer, 3, 4, Assert.IsType<Conv1D>(layer).Window, new Window1D(3) { Stride = 2, Padding = 1 }),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(4, Assert.IsType<BatchNorm>(layer).Features),
            layer => Assert.Equal(new Window1D(3) { Stride = 2, Padding = 1 }, Assert.IsType<MaxPool1D>(layer).Window),
            layer => AssertAverage(new Window1D(3) { Padding = 1 }, true, Assert.IsType<AvgPool1D>(layer).Window, layer),
            layer => AssertAverage(new Window1D(3) { Stride = 2, Padding = 1 }, false, Assert.IsType<AvgPool1D>(layer).Window, layer),
            layer => AssertConvolution(layer, 4, 5, Assert.IsType<Conv1D>(layer).Window, new Window1D(2)),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(15, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Theory]
    [InlineData("onnx-image.onnx")]
    [InlineData("onnx-image-torchscript.onnx")]
    public void AnImageNetwork_IsBuiltAsItsNodesSay_ItsDropoutOfWholeChannelsNoLayerAtAll(string file)
    {
        using var graph = OnnxFixtures.Open(file);
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;

        Assert.Collection(
            layers,
            layer => AssertConvolution(layer, 3, 4, Assert.IsType<Conv2D>(layer).Window, new Window(3, 3) { Padding = 1 }),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(4, Assert.IsType<BatchNorm>(layer).Features),
            layer => Assert.Equal(new Window(3, 3) { Stride = 2, Padding = 1 }, Assert.IsType<MaxPool2D>(layer).Window),
            layer => AssertAverage(new Window(3, 3) { Padding = 1 }, true, Assert.IsType<AvgPool2D>(layer).Window, layer),
            layer => AssertAverage(new Window(3, 3) { Stride = 2, Padding = 1 }, false, Assert.IsType<AvgPool2D>(layer).Window, layer),
            layer => AssertConvolution(layer, 4, 5, Assert.IsType<Conv2D>(layer).Window, new Window(2, 1)),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(30, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Theory]
    [InlineData("onnx-volume.onnx")]
    [InlineData("onnx-volume-torchscript.onnx")]
    public void AVolumeNetwork_IsBuiltAsItsNodesSay_ItsDropoutOfWholeChannelsNoLayerAtAll(string file)
    {
        using var graph = OnnxFixtures.Open(file);
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;

        Assert.Collection(
            layers,
            layer => AssertConvolution(layer, 2, 3, Assert.IsType<Conv3D>(layer).Window, new Window3D(3, 3, 3) { Padding = 1 }),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(3, Assert.IsType<BatchNorm>(layer).Features),
            layer => Assert.Equal(new Window3D(2, 2, 2) { Stride = 2 }, Assert.IsType<MaxPool3D>(layer).Window),
            layer => AssertAverage(new Window3D(2, 2, 2) { Padding = 1 }, true, Assert.IsType<AvgPool3D>(layer).Window, layer),
            layer => AssertAverage(new Window3D(3, 3, 3) { Stride = 2, Padding = 1 }, false, Assert.IsType<AvgPool3D>(layer).Window, layer),
            layer => AssertConvolution(layer, 3, 4, Assert.IsType<Conv3D>(layer).Window, new Window3D(1, 2, 2)),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(24, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Theory]
    [MemberData(nameof(Poolings))]
    public void APoolingOfEveryPlaceIntoOne_KeepsTheAxesItPools_AsOnnxsGlobalPoolingsDo_AndAFlattenTakesThemAway(string net, bool average, string file)
    {
        var channels = net switch { "series" => 5, "image" => 5, _ => 4 };
        using var graph = OnnxFixtures.Open(file);
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network).Layers;

        Assert.IsType<Flatten>(layers[^2]);
        Assert.Equal(new Shape(channels, 3), Assert.IsType<Dense>(layers[^1]).Weight.Value.Shape);
        Assert.True(KeepsAxes(layers[^3], net, average));
    }

    [Theory]
    [MemberData(nameof(Nets))]
    public void TheDefaultExporter_AndTheTorchScriptOne_WriteNoNodeForADropoutOfWholeChannels(string net)
    {
        foreach (var file in new[] { $"onnx-{net}.onnx", $"onnx-{net}-torchscript.onnx" })
        {
            Assert.DoesNotContain(OnnxFixtures.Model(file).Graph.Node, node => node.OpType.Contains("Dropout", StringComparison.Ordinal));
        }
    }

    [Theory]
    [InlineData("series", FirstAverage, 4, "SAME_UPPER")]
    [InlineData("image", FirstAverage, 4, "SAME_UPPER")]
    [InlineData("volume", FirstConvolution, 0, "SAME_UPPER")]
    [InlineData("series", FirstAverage, 4, "SAME_LOWER")]
    [InlineData("image", FirstAverage, 4, "SAME_LOWER")]
    [InlineData("volume", FirstConvolution, 0, "SAME_LOWER")]
    public void APaddingWrittenAsSameUpperOrLower_WhereTheStatedOneWasTheSame_GivesPyTorchsAnswersStill(string net, string node, int layer, string autoPad)
    {
        // A window of three at a stride of one pads one place on each side, which the odd place of either borders is not before or after.
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited($"onnx-{net}-torchscript.onnx", model =>
        {
            model.Node(node).SetText("auto_pad", autoPad);
            model.Node(node).Unset("pads");
        }));

        Assert.Equal(PaddingMode.Same, PaddingOf(Assert.IsType<LayerStack>(read.Network).Layers[layer]));
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty("flat")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Theory]
    [MemberData(nameof(Nets))]
    public void APaddingWrittenAsValid_IsNoPaddingAtAll_AsTheConvolutionHadItsOwn(string net)
    {
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited($"onnx-{net}-torchscript.onnx", model =>
        {
            model.Node(LastConvolution).SetText("auto_pad", "VALID");
            model.Node(LastConvolution).Unset("pads");
        }));

        Assert.Equal(PaddingMode.Stated, PaddingOf(Assert.IsType<LayerStack>(read.Network).Layers[6]));
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty("flat")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Theory]
    [MemberData(nameof(Nets))]
    public void TheBordersPyTorchCannotWrite_ASameUpperOrPadsWrittenOutAsTensorFlowsSame_GiveTheOutputsPyTorchGaveWhenItPaddedThemByHand(string net)
    {
        var torch = OnnxFixtures.Spatial.GetProperty($"borders-{net}");
        using var graph = OnnxFixtures.Open($"onnx-borders-{net}.onnx");
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);
        var layers = Assert.IsType<LayerStack>(saved.Network).Layers;

        // Every layer pads as TensorFlow's 'same', the two average poolings leaving their border out of the average or counting it
        // as the node says; the pads that depend on the length — at a stride of two — are held to the length once it is known.
        Assert.All(layers, layer => Assert.Equal(PaddingMode.Same, PaddingOf(layer)));
        Assert.Equal([false, true], layers.Where(layer => layer is AvgPool1D or AvgPool2D or AvgPool3D).Select(CountsPaddingOf));

        var examples = OnnxFixtures.Images(torch.GetProperty("input"));
        var outputs = saved.Network.Predict(examples, saved.Loss, Engine);
        var expected = OnnxFixtures.Images(torch.GetProperty("output"));

        Assert.Equal(expected.Shape, outputs.Shape);
        Assert.InRange(Farthest(expected.Values.ToArray(), outputs), 0, 1e-5);
    }

    [Fact]
    public void APaddingWrittenAsSameLower_WhereTheLengthLeavesNoOddPlace_IsTheSamePaddingSameUpperIs()
    {
        // The first pooling of the borders written at a stride of two stands on a series of 23 steps: three steps at a stride
        // of two pad one place on each side, so the odd place SAME_LOWER puts before is none.
        var torch = OnnxFixtures.Spatial.GetProperty("borders-series");
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-borders-series.onnx", model => model.Node("averagepool_1").SetText("auto_pad", "SAME_LOWER")));
        var outputs = read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine);

        Assert.Equal(PaddingMode.Same, PaddingOf(Assert.IsType<LayerStack>(read.Network).Layers[1]));
        Assert.InRange(Farthest(OnnxFixtures.Images(torch.GetProperty("output")).Values.ToArray(), outputs), 0, 1e-5);
    }

    [Theory]
    [InlineData("image", true, "node_mean")]
    [InlineData("image", false, "n0")]
    [InlineData("volume", true, "node_mean")]
    [InlineData("volume", false, "n0")]
    public void AReduceThatDropsTheAxesItReduces_IsAGlobalPoolingThatKeepsNone_AndLeavesARowOfChannels(string net, bool average, string node)
    {
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, $"onnx-{net}-global-{(average ? "average" : "max")}-default.onnx", model => model.Node(node).SetWhole("keepdims", 0));
        var read = new OnnxFile(new MeanSquaredError()).Read(graph);

        Assert.False(KeepsAxes(Assert.IsType<LayerStack>(read.Network).Layers[^3], net, average));
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty(average ? "average" : "max")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Theory]
    [InlineData("image", "node_mean", true)]
    [InlineData("image", "n0", false)]
    [InlineData("volume", "node_mean", true)]
    [InlineData("volume", "n0", false)]
    public void AReducesAxesWrittenAsAnAttribute_AsBeforeOpsetEighteen_AreReadAsOnesWrittenAsAnInput(string net, string node, bool average)
    {
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        var file = $"onnx-{net}-global-{(average ? "average" : "max")}-default.onnx";
        using var original = OnnxFixtures.Open(file);
        var expected = Bits(new OnnxFile(new MeanSquaredError()).Read(original).Network);

        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, file, model =>
        {
            var reduce = model.Node(node);
            var axes = model.Initializer(reduce.Input[1]).RawData.ToByteArray();
            reduce.SetWholes("axes", [.. Enumerable.Range(0, axes.Length / 8).Select(at => BitConverter.ToInt64(axes, at * 8))]);
            reduce.Input.RemoveAt(1);
        });
        var read = new OnnxFile(new MeanSquaredError()).Read(graph);

        Assert.Equal(expected, Bits(read.Network));
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty(average ? "average" : "max")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Theory]
    [InlineData("image", "node_mean", new[] { 2L, 3L })]
    [InlineData("image", "node_mean", new[] { 3L, 2L })]
    [InlineData("image", "n0", new[] { -2L, 3L })]
    [InlineData("image", "n0", new[] { 2L, -1L })]
    [InlineData("volume", "node_mean", new[] { 4L, 3L, 2L })]
    [InlineData("volume", "n0", new[] { -3L, 3L, -1L })]
    public void TheAxesOfAReduce_AreTheSameWhateverOrderTheyAreWrittenIn_AndWhicheverEndTheyAreCountedFrom(string net, string node, long[] axes)
    {
        var torch = OnnxFixtures.Spatial.GetProperty(net);
        var head = node == "node_mean" ? "average" : "max";
        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, $"onnx-{net}-global-{head}-default.onnx", model => model.SetAxes(node, axes));
        var read = new OnnxFile(new MeanSquaredError()).Read(graph);

        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty(head)), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Fact]
    public void TheUnsqueezeAndSqueezeRoundAPoolingOfASeries_WithTheirAxesAndTheReducesWrittenFromTheStart_AreTheSamePooling()
    {
        var torch = OnnxFixtures.Spatial.GetProperty("series");
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-series-global-average-default.onnx", model =>
        {
            // The axis the Unsqueeze makes and the Squeeze takes away, the third of four, and the two the ReduceMean reduces.
            model.SetAxes("node_unsqueeze", 2);
            model.SetAxes("node_mean", 3, 2);
        }));

        Assert.IsType<GlobalAvgPool1D>(Assert.IsType<LayerStack>(read.Network).Layers[^3]);
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty("average")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Fact]
    public void AReduceOfASeriesWithNoUnsqueezeRoundIt_IsAGlobalPoolingOfTheSeries_AsTheOneTheTorchScriptExporterWrites()
    {
        var torch = OnnxFixtures.Spatial.GetProperty("series");
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-series-global-average.onnx", model =>
        {
            var node = model.Node("/1/GlobalAveragePool");
            node.OpType = "ReduceMean";
            node.SetWholes("axes", -1);
        }));

        Assert.True(Assert.IsType<GlobalAvgPool1D>(Assert.IsType<LayerStack>(read.Network).Layers[^3]).KeepsAxes);
        Assert.InRange(Farthest(OnnxFixtures.Values(torch.GetProperty("average")), read.Network.Predict(OnnxFixtures.Images(torch.GetProperty("input")), read.Loss, Engine)), 0, 1e-5);
    }

    [Fact]
    public void APoolingsSettingsLeftOut_AreWhatOnnxGivesThem()
    {
        using var original = OnnxFixtures.Open("onnx-series.onnx");
        var expected = new OnnxFile(new MeanSquaredError()).Read(original).Network;

        // A dilation of one, rounding down, indices in the usual order, no count of the padding, and a padding left open, are
        // what ONNX means when a node says none.
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-series.onnx", model =>
        {
            var max = model.Node("node_max_pool1d");
            max.Unset("dilations");
            max.Unset("ceil_mode");
            max.Unset("storage_order");
            max.Unset("auto_pad");

            var average = model.Node("node_avg_pool1d_1");
            average.Unset("count_include_pad");
            average.Unset("ceil_mode");
        })).Network;

        Assert.Equal(Bits(expected), Bits(read));
        Assert.Equal(
            Assert.IsType<LayerStack>(expected).Layers.Select(layer => layer.GetType()),
            Assert.IsType<LayerStack>(read).Layers.Select(layer => layer.GetType()));
        Assert.Equal(new Window1D(3) { Stride = 2, Padding = 1 }, Assert.IsType<MaxPool1D>(Assert.IsType<LayerStack>(read).Layers[3]).Window);
        Assert.False(Assert.IsType<AvgPool1D>(Assert.IsType<LayerStack>(read).Layers[5]).CountsPadding);
    }

    [Fact]
    public void AMaxPoolingThatNamesAValueForItsIndices_NoNodeTakesAndNoGraphGives_IsNoFaultOfItsOwn()
    {
        using var original = OnnxFixtures.Open("onnx-series-torchscript.onnx");
        var expected = Bits(new OnnxFile(new MeanSquaredError()).Read(original).Network);

        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-series-torchscript.onnx", model => model.Node("/0/0.3/MaxPool").Output.Add("indices")));

        Assert.Equal(expected, Bits(read.Network));
    }

    [Fact]
    public void AMaxPoolingsIndicesLeftEmpty_AreNotNamed_SoNothingTakesThem()
    {
        using var original = OnnxFixtures.Open("onnx-series-torchscript.onnx");
        var expected = Bits(new OnnxFile(new MeanSquaredError()).Read(original).Network);

        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-series-torchscript.onnx", model => model.Node("/0/0.3/MaxPool").Output.Add(string.Empty)));

        Assert.Equal(expected, Bits(read.Network));
    }

    // Whether the global pooling of the net, an average or a largest, keeps the axes it pools.
    private static bool KeepsAxes(Layer layer, string net, bool average) => (net, average) switch
    {
        ("series", true) => Assert.IsType<GlobalAvgPool1D>(layer).KeepsAxes,
        ("series", false) => Assert.IsType<GlobalMaxPool1D>(layer).KeepsAxes,
        ("image", true) => Assert.IsType<GlobalAvgPool2D>(layer).KeepsAxes,
        ("image", false) => Assert.IsType<GlobalMaxPool2D>(layer).KeepsAxes,
        (_, true) => Assert.IsType<GlobalAvgPool3D>(layer).KeepsAxes,
        _ => Assert.IsType<GlobalMaxPool3D>(layer).KeepsAxes,
    };

    private static void AssertConvolution<TWindow>(Layer layer, int inChannels, int outChannels, TWindow actual, TWindow expected)
    {
        var convolution = Assert.IsAssignableFrom<Convolution>(layer);

        Assert.Equal(expected, actual);
        Assert.Equal(inChannels, convolution.InChannels);
        Assert.Equal(outChannels, convolution.OutChannels);
    }

    private static void AssertAverage<TWindow>(TWindow expected, bool counts, TWindow actual, Layer layer)
    {
        Assert.Equal(expected, actual);
        Assert.Equal(counts, CountsPaddingOf(layer));
    }

    private static bool CountsPaddingOf(Layer layer) => Assert.IsAssignableFrom<AveragePooling>(layer).CountsPadding;

    // How a layer that walks pads, whichever number of axes it walks.
    private static PaddingMode PaddingOf(Layer layer) => layer switch
    {
        Conv1D conv => conv.Window.PaddingMode,
        Conv2D conv => conv.Window.PaddingMode,
        Conv3D conv => conv.Window.PaddingMode,
        MaxPool1D pool => pool.Window.PaddingMode,
        MaxPool2D pool => pool.Window.PaddingMode,
        MaxPool3D pool => pool.Window.PaddingMode,
        AvgPool1D pool => pool.Window.PaddingMode,
        AvgPool2D pool => pool.Window.PaddingMode,
        AvgPool3D pool => pool.Window.PaddingMode,
        _ => throw new InvalidOperationException($"{layer.GetType().Name} walks nothing."),
    };

    // The farthest any value lies from PyTorch's.
    private static double Farthest(float[] torch, Tensor here)
    {
        Assert.Equal(torch.Length, here.Values.Length);

        return torch.Zip(here.Values.ToArray(), (expected, actual) => Math.Abs((double)expected - actual)).Max();
    }

    private static int[][] Bits(Network network) =>
        [.. network.Slots().Select(named => named.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
