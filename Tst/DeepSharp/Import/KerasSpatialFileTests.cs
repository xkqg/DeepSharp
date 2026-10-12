// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using PureHDF;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A model Keras 3 trained and saved whose layers walk a series, an image or a volume — convolutions, poolings, global
/// poolings and the dropouts of whole channels — read into a network here: each layer built as the window and the settings
/// it says, its kernel's numbers in the order Keras wrote them, and every example answered as Keras answered it. Where
/// Keras on PyTorch pools a padded average otherwise than TensorFlow does, the answer held to is the one TensorFlow's rule
/// gives, worked out by hand (Fixtures/keras-spatial-fixtures.py says which networks, and measures it).
/// </summary>
public class KerasSpatialFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    [Theory]
    [InlineData("series", "keras-spatial-series.keras")]
    [InlineData("series", "keras-spatial-series.h5")]
    [InlineData("image", "keras-spatial-image.keras")]
    [InlineData("image", "keras-spatial-image.h5")]
    [InlineData("volume", "keras-spatial-volume.keras")]
    [InlineData("volume", "keras-spatial-volume.h5")]
    [InlineData("seriesSame", "keras-spatial-series-same.keras")]
    [InlineData("imageSame", "keras-spatial-image-same.keras")]
    [InlineData("volumeSame", "keras-spatial-volume-same.keras")]
    public void ANetworkWalkingASeriesAnImageOrAVolume_AnswersEveryExampleAsKerasDid_ToAHundredThousandth(string network, string file)
    {
        var answered = KerasFixtures.SpatialAnswers.GetProperty(network);
        var rows = answered.GetProperty("rows").EnumerateArray().Select(Values).ToArray();
        var shape = new Shape([rows.Length, .. answered.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);
        var saved = new KerasFile().Read(KerasFixtures.Open(file));

        var here = saved.Network.Predict(Tensor.From(shape, [.. rows.SelectMany(row => row)]), saved.Loss, Engine).Values.ToArray();
        var keras = Values(answered.GetProperty("answers"));

        Assert.Equal(keras.Length, here.Length);
        Assert.InRange(keras.Zip(here, (expected, actual) => Math.Abs(expected - actual)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ThePaddedAverageKerasOnPyTorchPoolsOtherwise_IsHeldToTensorFlowsRule_NotToThePyTorchBackendsAnswer()
    {
        var answered = KerasFixtures.SpatialAnswers.GetProperty("imageSame");

        Assert.StartsWith("numpy, TensorFlow's rule", answered.GetProperty("answeredBy").GetString(), StringComparison.Ordinal);
        Assert.Equal(
            ["keras", "keras", "keras", "keras", "keras"],
            new[] { "series", "image", "volume", "seriesSame", "volumeSame" }.Select(network => KerasFixtures.SpatialAnswers.GetProperty(network).GetProperty("answeredBy").GetString()));
    }

    [Fact]
    public void TheSeriesNetwork_IsBuiltAsItsDescriptionSays_ACausalWindow_PoolingsAndAGlobalAverage()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-series.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer =>
            {
                var convolution = Assert.IsType<Conv1D>(layer);
                Assert.Equal(new Window1D(3) { PaddingMode = PaddingMode.Causal }, convolution.Window);
                Assert.Equal(3, convolution.InChannels);
                Assert.Equal(5, convolution.OutChannels);
                Assert.Equal(new Shape(9, 5), convolution.Weight.Value.Shape);
            },
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Window1D(2) { Stride = 2 }, Assert.IsType<MaxPool1D>(layer).Window),
            layer => Assert.Equal(0.3, Assert.IsType<SpatialDropout1D>(layer).Rate),
            layer =>
            {
                var pooling = Assert.IsType<AvgPool1D>(layer);
                Assert.Equal(new Window1D(3) { Stride = 2 }, pooling.Window);
                Assert.False(pooling.CountsPadding);
            },
            layer => Assert.False(Assert.IsType<GlobalAvgPool1D>(layer).KeepsAxes),
            layer => Assert.Equal(new Shape(5, 2), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.IsType<MeanSquaredError>(saved.Loss);
    }

    [Fact]
    public void TheImageNetwork_IsBuiltAsItsDescriptionSays_APoolingPaddedAsSame_ASquareOfTwoAndAGlobalMaximum()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-image.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Window(3, 2), Assert.IsType<Conv2D>(layer).Window),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Window(3, 2) { Stride = 2, PaddingMode = PaddingMode.Same }, Assert.IsType<MaxPool2D>(layer).Window),
            layer => Assert.Equal(new Window(2, 2) { Stride = 2 }, Assert.IsType<AvgPool2D>(layer).Window),
            layer => Assert.Equal(0.25, Assert.IsType<SpatialDropout2D>(layer).Rate),
            layer => Assert.False(Assert.IsType<GlobalMaxPool2D>(layer).KeepsAxes),
            layer => Assert.Equal(new Shape(4, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.IsType<CrossEntropy>(saved.Loss);
    }

    [Fact]
    public void TheVolumeNetwork_IsBuiltAsItsDescriptionSays_AWindowOfUnequalSides_AndAGlobalAverageThatKeepsItsAxes()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-volume.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer =>
            {
                var convolution = Assert.IsType<Conv3D>(layer);
                Assert.Equal(new Window3D(2, 3, 2), convolution.Window);
                Assert.Equal(2, convolution.InChannels);
                Assert.Equal(3, convolution.OutChannels);
                Assert.Equal(new Shape(24, 3), convolution.Weight.Value.Shape);
            },
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Window3D(2, 2, 2) { Stride = 2 }, Assert.IsType<MaxPool3D>(layer).Window),
            layer => Assert.Equal(new Window3D(1, 1, 2), Assert.IsType<AvgPool3D>(layer).Window),
            layer => Assert.True(Assert.IsType<GlobalAvgPool3D>(layer).KeepsAxes),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(3, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.IsType<BinaryCrossEntropy>(saved.Loss);
    }

    [Fact]
    public void TheSeriesNetworkOfPaddedWindows_IsBuiltAsItsDescriptionSays_AValidWindowASameWindowAtAStrideOfTwoAndAGlobalMaximumThatKeepsItsAxis()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-series-same.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Window1D(3), Assert.IsType<Conv1D>(layer).Window),
            layer => Assert.IsType<Tanh>(layer),
            layer => Assert.Equal(new Window1D(4) { Stride = 2, PaddingMode = PaddingMode.Same }, Assert.IsType<Conv1D>(layer).Window),
            layer => Assert.Equal(new Window1D(2) { PaddingMode = PaddingMode.Same }, Assert.IsType<AvgPool1D>(layer).Window),
            layer => Assert.True(Assert.IsType<GlobalMaxPool1D>(layer).KeepsAxes),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(4, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Fact]
    public void TheImageNetworkOfPaddedWindows_IsBuiltAsItsDescriptionSays_ASigmoidAfterTheConvolution_AndAnAveragePaddedAsSame()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-image-same.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same }, Assert.IsType<Conv2D>(layer).Window),
            layer => Assert.IsType<Sigmoid>(layer),
            layer =>
            {
                var pooling = Assert.IsType<AvgPool2D>(layer);
                Assert.Equal(new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same }, pooling.Window);
                Assert.False(pooling.CountsPadding);
            },
            layer => Assert.False(Assert.IsType<GlobalAvgPool2D>(layer).KeepsAxes),
            layer => Assert.Equal(new Shape(3, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Fact]
    public void TheVolumeNetworkOfPaddedWindows_IsBuiltAsItsDescriptionSays_AConvolutionAndAMaximumPaddedAsSame_AndADropoutOfWholeChannels()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-spatial-volume-same.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Window3D(2, 2, 2) { PaddingMode = PaddingMode.Same }, Assert.IsType<Conv3D>(layer).Window),
            layer => Assert.IsType<Tanh>(layer),
            layer => Assert.Equal(new Window3D(2, 2, 2) { Stride = 2, PaddingMode = PaddingMode.Same }, Assert.IsType<MaxPool3D>(layer).Window),
            layer => Assert.Equal(0.2, Assert.IsType<SpatialDropout3D>(layer).Rate),
            layer => Assert.False(Assert.IsType<GlobalMaxPool3D>(layer).KeepsAxes),
            layer => Assert.Equal(new Shape(3, 2), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Theory]
    [InlineData("keras-spatial-series", "conv1d", 0)]
    [InlineData("keras-spatial-series-same", "conv1d", 0)]
    [InlineData("keras-spatial-series-same", "conv1d_1", 1)]
    [InlineData("keras-spatial-image", "conv2d", 0)]
    [InlineData("keras-spatial-image-same", "conv2d", 0)]
    [InlineData("keras-spatial-volume", "conv3d", 0)]
    [InlineData("keras-spatial-volume-same", "conv3d", 0)]
    public void AKernelsNumbers_ReachItsSlotAsKerasWroteThem_RowByRowInTheOrderOfItsWindowThenItsChannelsIn(string model, string group, int which)
    {
        var stack = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open($"{model}.keras")).Network);
        var convolution = stack.Layers.OfType<Convolution>().ElementAt(which);
        using var archive = new MemoryStream(KerasFixtures.Entries($"{model}.keras")["model.weights.h5"]);
        using var weights = H5File.Open(archive);

        var kernel = weights.Dataset($"layers/{group}/vars/0");

        // Keras writes (window..., in, out); the slot is the same numbers as (the window's places times in) by out.
        Assert.Equal(kernel.Read<float[]>(), convolution.Weight.Value.Values.ToArray());
        Assert.Equal(weights.Dataset($"layers/{group}/vars/1").Read<float[]>(), convolution.Bias.Value.Values.ToArray());
        Assert.Equal((int)kernel.Space.Dimensions[^1], convolution.OutChannels);
        Assert.Equal((int)kernel.Space.Dimensions[^2], convolution.InChannels);
        Assert.Equal(kernel.Space.Dimensions.SkipLast(1).Aggregate(1, (rows, length) => rows * (int)length), convolution.Weight.Value.Shape[0]);
    }

    [Theory]
    [InlineData("keras-spatial-series")]
    [InlineData("keras-spatial-image")]
    [InlineData("keras-spatial-volume")]
    public void AModelsArchive_AndTheHdf5FileKerasSavedItToBefore_PutTheSameNumbersIntoTheSameSlots(string model)
    {
        var archive = new KerasFile().Read(KerasFixtures.Open($"{model}.keras")).Network;
        var legacy = new KerasFile().Read(KerasFixtures.Open($"{model}.h5")).Network;

        Assert.Equal(archive.Slots().Select(named => named.Path), legacy.Slots().Select(named => named.Path));
        Assert.Equal(
            archive.Slots().Select(named => named.Slot.Value.Values.ToArray()),
            legacy.Slots().Select(named => named.Slot.Value.Values.ToArray()));
    }

    [Theory]
    // A pool size or a stride Keras writes as a whole number stands for every axis alike; a stride it writes as nothing is the pool size.
    [InlineData(new[] { "config.layers.2.config.pool_size=3" }, 3, 3, 2)]
    [InlineData(new[] { "config.layers.2.config.strides=3" }, 3, 2, 3)]
    [InlineData(new[] { "config.layers.2.config.pool_size=[2, 2]", "config.layers.2.config.strides=null" }, 2, 2, 2)]
    [InlineData(new[] { "config.layers.2.config.pool_size=[2, 2]", "config.layers.2.config.strides=-" }, 2, 2, 2)]
    // A pool size whose sides differ is a window whose sides differ.
    [InlineData(new[] { "config.layers.2.config.pool_size=[1, 4]" }, 1, 4, 2)]
    public void APoolSizeAndAStrideKerasWritesInAnyOfItsWays_AreReadAsTheWindowTheyStand(string[] edits, int height, int width, int stride)
    {
        var saved = new KerasFile().Read(KerasFixtures.Edited("keras-spatial-image.keras", edits));
        var pooling = Assert.IsType<MaxPool2D>(Assert.IsType<LayerStack>(saved.Network).Layers[2]);

        Assert.Equal(new Window(height, width) { Stride = stride, PaddingMode = PaddingMode.Same }, pooling.Window);
    }

    [Fact]
    public void ASpatialDropoutThatSaysANoiseShape_IsReadAsKerasReadsIt_SinceItWorksOutTheShapeItselfAndTakesNone()
    {
        var saved = new KerasFile().Read(KerasFixtures.Edited("keras-spatial-series.keras", "config.layers.3.config.noise_shape=[null, 1, 5]"));

        Assert.Equal(0.3, Assert.IsType<SpatialDropout1D>(Assert.IsType<LayerStack>(saved.Network).Layers[3]).Rate);
    }

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    /// <summary>
    /// The Keras layers this release reads for the first time: a convolution with its channels in groups, a depthwise and a
    /// separable convolution, one that gives the places back, a layer that only pads, and a dilated window padded as
    /// 'causal' — each along an image, and then along a series and through a volume. Each model is read and held to what
    /// Keras answered through it.
    /// </summary>
    [Theory]
    [InlineData("keras-new-grouped")]
    [InlineData("keras-new-depthwise")]
    [InlineData("keras-new-separable")]
    [InlineData("keras-new-transposed")]
    [InlineData("keras-new-padded")]
    [InlineData("keras-new-dilated")]
    [InlineData("keras-new-series")]
    [InlineData("keras-new-volume")]
    public void AModelOfTheLayersReadForTheFirstTime_AnswersAsKerasDoes(string name)
    {
        var expected = KerasFixtures.New.GetProperty(name);
        var written = expected.GetProperty("examples");
        var shape = new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);
        var examples = Tensor.From(shape, Values(written.GetProperty("values")));

        using var file = KerasFixtures.Open($"{name}.keras");
        var read = new KerasFile().Read(file);

        var here = read.Network.Predict(examples, read.Loss, Engine).Values.ToArray();
        var keras = Values(expected.GetProperty("outputs"));

        Assert.Equal(keras.Length, here.Length);
        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void TheGroupedModel_IsBuiltAsItsDescriptionSays_ChannelsInGroups_AndAStrideOfItsOwnForEachAxis()
    {
        using var file = KerasFixtures.Open("keras-new-grouped.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        var grouped = Assert.IsType<Conv2D>(layers[0]);
        var strided = Assert.IsType<Conv2D>(layers[1]);

        Assert.Equal(new Window(3, 3) { PaddingMode = PaddingMode.Same }, grouped.Window);
        Assert.Equal(new Channels { In = 4, Out = 6, Groups = 2 }, grouped.Channels);
        Assert.Equal(new Window(3, 3) { Stride = new Steps(2, 1) }, strided.Window);
        Assert.Equal(new Channels { In = 6, Out = 4 }, strided.Channels);
    }

    [Fact]
    public void TheDepthwiseModel_IsBuiltAsItsDescriptionSays_AGroupForEveryChannel_MakingTwoOfEach()
    {
        using var file = KerasFixtures.Open("keras-new-depthwise.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        var depthwise = Assert.IsType<Conv2D>(layers[0]);

        Assert.Equal(new Window(3, 3) { PaddingMode = PaddingMode.Same }, depthwise.Window);
        Assert.Equal(new Channels { In = 3, Out = 6, Groups = 3 }, depthwise.Channels);
        Assert.Equal(new Channels { In = 6, Out = 3 }, Assert.IsType<Conv2D>(layers[1]).Channels);
    }

    [Fact]
    public void TheTransposedModel_IsBuiltAsItsDescriptionSays_ABorderWorkedOutFromTheSizesKerasMakes()
    {
        using var file = KerasFixtures.Open("keras-new-transposed.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        var grown = Assert.IsType<Conv2DTranspose>(layers[0]);
        var after = Assert.IsType<Conv2DTranspose>(layers[1]);

        // Padded as 'valid' Keras drops nothing, so the border worked out from the sizes it makes is nothing at either end.
        Assert.Equal(new Window(3, 3) { Stride = new Steps(2, 2), Border = default(Margins) }, grown.Window);
        Assert.Equal(new Channels { In = 2, Out = 4 }, grown.Channels);
        Assert.Equal([0, 0], grown.OutputPadding);
        Assert.Equal(new Window(2, 2) { Border = default(Margins) }, after.Window);
        Assert.Equal(new Channels { In = 4, Out = 3 }, after.Channels);
    }

    [Fact]
    public void APaddingKerasWritesAsOneNumberForEachAxis_IsThatManyPlacesAtBothEndsOfIt()
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-new-padded.keras", "config.layers.1.config.padding=[1, 2]"));

        Assert.Equal(
            new Margins(new Sides(1, 1), new Sides(2, 2)),
            Assert.IsType<Pad2D>(Assert.IsType<LayerStack>(read.Network).Layers[0]).Border);
    }

    [Fact]
    public void APaddingKerasWritesAsOneNumberForEveryEnd_IsThatManyPlacesAtEachOfThem()
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-new-series.keras", "config.layers.1.config.padding=2"));

        Assert.Equal(new Sides(2, 2), Assert.IsType<Pad1D>(Assert.IsType<LayerStack>(read.Network).Layers[0]).Border);
    }

    [Fact]
    public void ThePaddedModel_IsBuiltAsItsDescriptionSays_ThePlacesAtBothEndsOfEachAxis_AndNothingElse()
    {
        using var file = KerasFixtures.Open("keras-new-padded.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        var padding = Assert.IsType<Pad2D>(layers[0]);

        Assert.Equal(new Margins(new Sides(1, 1), new Sides(2, 2)), padding.Border);
        Assert.Equal(0f, padding.Value);
        Assert.Equal(new Window(3, 3), Assert.IsType<Conv2D>(layers[1]).Window);
    }

    [Fact]
    public void TheDilatedModel_IsBuiltAsItsDescriptionSays_ACausalWindowWhoseCellsStandTwoPlacesApart()
    {
        using var file = KerasFixtures.Open("keras-new-dilated.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        Assert.Equal(
            new Window1D(3) { Dilation = 2, PaddingMode = PaddingMode.Causal },
            Assert.IsType<Conv1D>(layers[0]).Window);
    }

    [Fact]
    public void TheSeriesModel_IsBuiltAsItsDescriptionSays_TheSameWindowsAlongASeries_ThePlacesGivenBackAndTheOneKeptAtTheEnd()
    {
        using var file = KerasFixtures.Open("keras-new-series.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        Assert.Equal(new Sides(2, 2), Assert.IsType<Pad1D>(layers[0]).Border);

        // The separable convolution's two stages: a group for every channel, and then a window of one place.
        Assert.Equal(new Channels { In = 2, Out = 4, Groups = 2 }, Assert.IsType<Conv1D>(layers[1]).Channels);
        Assert.Equal(new Window1D(1), Assert.IsType<Conv1D>(layers[2]).Window);
        Assert.Equal(new Channels { In = 4, Out = 4 }, Assert.IsType<Conv1D>(layers[2]).Channels);
        Assert.Equal(new Channels { In = 4, Out = 8, Groups = 4 }, Assert.IsType<Conv1D>(layers[3]).Channels);

        // Padded as 'same' the window gives back one place beyond the stride, and that one is dropped at the end; padded as
        // 'valid' with a place kept at the end, nothing is dropped and one is added.
        var same = Assert.IsType<Conv1DTranspose>(layers[4]);
        var kept = Assert.IsType<Conv1DTranspose>(layers[5]);

        Assert.Equal(new Window1D(3) { Stride = 2, Border = new Sides(0, 1) }, same.Window);
        Assert.Equal(new Window1D(3) { Stride = 2, Border = default(Sides) }, kept.Window);
        Assert.Equal([1], kept.OutputPadding);
    }

    [Fact]
    public void TheVolumeModel_IsBuiltAsItsDescriptionSays_APairOfPlacesForEachAxisOfAVolume_AndAStrideOfItsOwnForEachAxis()
    {
        using var file = KerasFixtures.Open("keras-new-volume.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        Assert.Equal(
            new Margins3D(new Sides(1, 0), new Sides(0, 1), new Sides(1, 1)),
            Assert.IsType<Pad3D>(layers[0]).Border);

        // The window gives back two places where it moves one, and the place beyond the stride is dropped on that axis alone.
        Assert.Equal(
            new Window3D(2, 2, 2)
            {
                Stride = new Steps3D(2, 1, 2),
                Border = new Margins3D(new Sides(0, 0), new Sides(0, 1), new Sides(0, 0)),
            },
            Assert.IsType<Conv3DTranspose>(layers[1]).Window);
    }

    [Fact]
    public void ASeparableConvolution_IsTwoLayers_TheFirstWithABiasOfNoughts()
    {
        using var file = KerasFixtures.Open("keras-new-separable.keras");
        var layers = Assert.IsType<LayerStack>(new KerasFile().Read(file).Network).Layers;

        var depthwise = Assert.IsType<Conv2D>(layers[0]);
        var pointwise = Assert.IsType<Conv2D>(layers[1]);

        Assert.Equal(new Channels { In = 3, Out = 6, Groups = 3 }, depthwise.Channels);
        Assert.Equal(new Window(1, 1), pointwise.Window);
        Assert.Equal(new Channels { In = 6, Out = 5 }, pointwise.Channels);
        Assert.All(depthwise.Bias.Value.Values.ToArray(), value => Assert.Equal(0f, value));
    }
}
