// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network described in Keras's words can walk a series, an image or a volume: convolutions, poolings, global poolings and
/// the dropouts of whole channels, each lowered to the layer it names with the shapes an example gives it, as Keras works
/// them out.
/// </summary>
public class SequentialSpatialTests
{
    private static readonly CpuBackend Engine = new();

    [Fact]
    public void EveryWordThatWalksASeries_LowersToTheLayerItNames_WithTheShapesAnExampleGives()
    {
        var stack = new Sequential()
            .Input(new Shape(20, 2))
            .Conv1D(4, new Window1D(3))
            .Relu()
            .MaxPool1D(2)
            .SpatialDropout1D(0.25)
            .Conv1D(6, new Window1D(3) { PaddingMode = PaddingMode.Causal })
            .AvgPool1D(new Window1D(2) { Stride = 2 })
            .GlobalMaxPool1D()
            .Dense(3)
            .Lower(new Shape(20, 2), new RandomStream(3));

        Assert.Equal(
            [typeof(Conv1D), typeof(Relu), typeof(MaxPool1D), typeof(SpatialDropout1D), typeof(Conv1D), typeof(AvgPool1D), typeof(GlobalMaxPool1D), typeof(Dense)],
            stack.Layers.Select(layer => layer.GetType()));
        Assert.Equal(2, ((Conv1D)stack.Layers[0]).InChannels);
        Assert.Equal(6, ((Conv1D)stack.Layers[4]).OutChannels);
        Assert.Equal(new Window1D(2) { Stride = 2 }, ((MaxPool1D)stack.Layers[2]).Window);
        Assert.Equal(6, ((Dense)stack.Layers[7]).Inputs);
        Assert.Equal(new Shape(5, 3), stack.Forward(Tensor.Zeros(new Shape(5, 20, 2)), Pass.Evaluation(Engine)).Shape);
    }

    [Fact]
    public void EveryWordThatWalksAnImage_LowersToTheLayerItNames_WithTheShapesAnExampleGives()
    {
        var stack = new Sequential()
            .Input(new Shape(8, 8, 1))
            .Conv2D(3, new Window(3, 3) { Padding = 1 })
            .MaxPool2D(2)
            .SpatialDropout2D(0.5)
            .AvgPool2D(new Window(2, 2) { Stride = 2 })
            .GlobalAvgPool2D()
            .Dense(2)
            .Lower(new Shape(8, 8, 1), new RandomStream(3));

        Assert.Equal(
            [typeof(Conv2D), typeof(MaxPool2D), typeof(SpatialDropout2D), typeof(AvgPool2D), typeof(GlobalAvgPool2D), typeof(Dense)],
            stack.Layers.Select(layer => layer.GetType()));
        Assert.Equal(3, ((Dense)stack.Layers[5]).Inputs);
        Assert.Equal(new Shape(4, 2), stack.Forward(Tensor.Zeros(new Shape(4, 8, 8, 1)), Pass.Evaluation(Engine)).Shape);
    }

    [Fact]
    public void EveryWordThatWalksAVolume_LowersToTheLayerItNames_WithTheShapesAnExampleGives()
    {
        var stack = new Sequential()
            .Input(new Shape(6, 6, 6, 1))
            .Conv3D(2, new Window3D(3, 3, 3))
            .MaxPool3D(2)
            .SpatialDropout3D(0.2)
            .AvgPool3D(new Window3D(2, 2, 2))
            .GlobalMaxPool3D(keepsAxes: true)
            .Flatten()
            .Dense(1)
            .Lower(new Shape(6, 6, 6, 1), new RandomStream(3));

        Assert.Equal(
            [typeof(Conv3D), typeof(MaxPool3D), typeof(SpatialDropout3D), typeof(AvgPool3D), typeof(GlobalMaxPool3D), typeof(Flatten), typeof(Dense)],
            stack.Layers.Select(layer => layer.GetType()));
        Assert.True(((GlobalMaxPool3D)stack.Layers[4]).KeepsAxes);
        Assert.Equal(2, ((Dense)stack.Layers[6]).Inputs);
        Assert.Equal(new Shape(3, 1), stack.Forward(Tensor.Zeros(new Shape(3, 6, 6, 6, 1)), Pass.Evaluation(Engine)).Shape);
    }

    [Fact]
    public void APoolingWrittenWithASize_WalksNonOverlappingRuns_AsKerasPoolsByDefault()
    {
        var line = new Sequential().MaxPool1D(3).AvgPool1D(2).Lower(new Shape(40, 1), new RandomStream(1));
        var plane = new Sequential().MaxPool2D(2).AvgPool2D(3).Lower(new Shape(12, 12, 1), new RandomStream(1));
        var volume = new Sequential().MaxPool3D(2).AvgPool3D(3).Lower(new Shape(12, 12, 12, 1), new RandomStream(1));

        Assert.Equal(new Window1D(3) { Stride = 3 }, ((MaxPool1D)line.Layers[0]).Window);
        Assert.Equal(new Window1D(2) { Stride = 2 }, ((AvgPool1D)line.Layers[1]).Window);
        Assert.Equal(new Window(2, 2) { Stride = 2 }, ((MaxPool2D)plane.Layers[0]).Window);
        Assert.Equal(new Window(3, 3) { Stride = 3 }, ((AvgPool2D)plane.Layers[1]).Window);
        Assert.Equal(new Window3D(2, 2, 2) { Stride = 2 }, ((MaxPool3D)volume.Layers[0]).Window);
        Assert.Equal(new Window3D(3, 3, 3) { Stride = 3 }, ((AvgPool3D)volume.Layers[1]).Window);
    }

    [Fact]
    public void AnAveragePoolingThatCountsItsBorder_SaysSo_AndOneThatDoesNotLeavesItOut()
    {
        var stack = new Sequential()
            .AvgPool1D(new Window1D(3) { Padding = 1 }, countsPadding: true)
            .AvgPool1D(new Window1D(3) { Padding = 1 })
            .Lower(new Shape(8, 1), new RandomStream(1));

        Assert.True(((AvgPool1D)stack.Layers[0]).CountsPadding);
        Assert.False(((AvgPool1D)stack.Layers[1]).CountsPadding);
    }

    [Fact]
    public void AWordThatWalksAnExampleOfOtherAxes_IsRefused_NamingTheWordAndWhatReachesIt()
    {
        string Refusal(Sequential description, Shape example) =>
            Assert.Throws<ArgumentException>(() => description.Lower(example, new RandomStream(1))).Message;

        Assert.Contains("word 1, conv1d, takes each example as a series of steps and channels, and each reaching it is 5", Refusal(new Sequential().Conv1D(2, new Window1D(3)), new Shape(5)), StringComparison.Ordinal);
        Assert.Contains("conv3d, takes each example as a volume of planes, rows, columns and channels, and each reaching it is 4x4x1", Refusal(new Sequential().Conv3D(2, new Window3D(2, 2, 2)), new Shape(4, 4, 1)), StringComparison.Ordinal);
        Assert.Contains("maxpool2d, takes each example as an image of rows, columns and channels, and each reaching it is 16", Refusal(new Sequential().MaxPool2D(2), new Shape(16)), StringComparison.Ordinal);
        Assert.Contains("avgpool3d, takes each example as a volume", Refusal(new Sequential().AvgPool3D(2), new Shape(4, 4, 1)), StringComparison.Ordinal);
        Assert.Contains("globalavgpool1d, takes each example as a series", Refusal(new Sequential().GlobalAvgPool1D(), new Shape(4, 4, 1)), StringComparison.Ordinal);
        Assert.Contains("globalmaxpool2d, takes each example as an image", Refusal(new Sequential().GlobalMaxPool2D(), new Shape(4, 1)), StringComparison.Ordinal);
        Assert.Contains("spatialdropout2d, takes each example as an image", Refusal(new Sequential().SpatialDropout2D(0.5), new Shape(4, 1)), StringComparison.Ordinal);
    }

    [Fact]
    public void AWindowLargerThanTheExample_IsRefused_NamingTheWordAndTheWindow()
    {
        string Refusal(Sequential description, Shape example) =>
            Assert.Throws<ArgumentException>(() => description.Lower(example, new RandomStream(1))).Message;

        Assert.Contains("conv1d, slides a window 3 (stride 1, padding 0) over each example, and an example 2x1 is smaller than it", Refusal(new Sequential().Conv1D(2, new Window1D(3)), new Shape(2, 1)), StringComparison.Ordinal);
        Assert.Contains("maxpool3d, slides a window 2x2x2 (stride 2, padding 0) over each example", Refusal(new Sequential().MaxPool3D(2), new Shape(4, 4, 1, 1)), StringComparison.Ordinal);
    }

    [Fact]
    public void ADenseLayerAfterAGlobalPoolingThatKeepsItsAxes_IsRefused_AsAfterAnyImage()
    {
        var wrong = Assert.Throws<ArgumentException>(
            () => new Sequential().GlobalAvgPool2D(keepsAxes: true).Dense(2).Lower(new Shape(4, 4, 3), new RandomStream(1)));

        Assert.Contains("dense, takes each example as a row of numbers, and each reaching it is 1x1x3: flatten it first", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWordWrittenWrongly_IsRefusedWhereItIsWritten()
    {
        var description = new Sequential();

        Assert.Throws<ArgumentOutOfRangeException>(() => description.Conv1D(0, new Window1D(3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.Conv3D(0, new Window3D(3, 3, 3)));
        Assert.Throws<ArgumentException>(() => description.Conv1D(2, new Window1D(0)));
        Assert.Throws<ArgumentException>(() => description.Conv3D(2, new Window3D(3, 0, 3)));
        Assert.Throws<ArgumentException>(() => description.MaxPool1D(0));
        Assert.Throws<ArgumentException>(() => description.AvgPool2D(0));
        Assert.Throws<ArgumentException>(() => description.MaxPool3D(new Window3D(2, 2, 2) { Stride = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.SpatialDropout1D(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.SpatialDropout2D(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.SpatialDropout3D(double.NaN));
    }

    [Fact]
    public void ASeriesNetwork_LearnsWhereABumpIs_ThroughAConvolutionAGlobalMaxAndADense()
    {
        var train = Bumps(96, 0);
        var validation = Bumps(32, 96);
        var compiled = new Sequential()
            .Conv1D(6, new Window1D(5))
            .Relu()
            .MaxPool1D(2)
            .SpatialDropout1D(0.1)
            .Conv1D(6, new Window1D(3) { PaddingMode = PaddingMode.Same })
            .Relu()
            .GlobalMaxPool1D()
            .Dense(1)
            .Compile(new Adam(0.03), new BinaryCrossEntropy());

        var history = compiled.Fit(train, validation, new FitOptions(seed: 5) { Backend = Engine, Epochs = 40, BatchSize = 16 });

        Assert.True(history.Epochs[^1].Loss < history.Epochs[0].Loss * 0.5, $"{history.Epochs[0].Loss} -> {history.Epochs[^1].Loss}");
        Assert.True(Accuracy(compiled, validation) >= 0.9, $"{Accuracy(compiled, validation)}");
    }

    [Fact]
    public void AnImageNetwork_LearnsWhichSideABlobIsOn_ThroughAConvolutionAPoolingAndAGlobalAverage()
    {
        var train = Blobs(96, 0);
        var validation = Blobs(32, 96);
        var compiled = new Sequential()
            .Conv2D(6, new Window(3, 3) { Padding = 1 })
            .Relu()
            .MaxPool2D(2)
            .Conv2D(2, new Window(3, 3) { PaddingMode = PaddingMode.Same })
            .GlobalAvgPool2D()
            .Dense(1)
            .Compile(new Adam(0.03), new BinaryCrossEntropy());

        var history = compiled.Fit(train, validation, new FitOptions(seed: 5) { Backend = Engine, Epochs = 40, BatchSize = 16 });

        Assert.True(history.Epochs[^1].Loss < history.Epochs[0].Loss * 0.5, $"{history.Epochs[0].Loss} -> {history.Epochs[^1].Loss}");
        Assert.True(Accuracy(compiled, validation) >= 0.9, $"{Accuracy(compiled, validation)}");
    }

    [Fact]
    public void AVolumeNetwork_LearnsOnThroughAConvolutionAnAveragePoolingAndAGlobalMax()
    {
        var train = Cubes(48, 0);
        var validation = Cubes(16, 48);
        var compiled = new Sequential()
            .Conv3D(4, new Window3D(2, 2, 2))
            .Relu()
            .AvgPool3D(new Window3D(2, 2, 2) { Stride = 1 })
            .GlobalMaxPool3D()
            .Dense(1)
            .Compile(new Adam(0.03), new BinaryCrossEntropy());

        var history = compiled.Fit(train, validation, new FitOptions(seed: 5) { Backend = Engine, Epochs = 40, BatchSize = 16 });

        Assert.True(history.Epochs[^1].Loss < history.Epochs[0].Loss * 0.6, $"{history.Epochs[0].Loss} -> {history.Epochs[^1].Loss}");
    }

    // A noisy series of sixteen steps, with a bump of three steps at some place in half of them, whose answer is one.
    private static TrainingData Bumps(int count, int from)
    {
        var features = new float[count * 16];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            var bumped = (from + row) % 2 == 0;

            for (var step = 0; step < 16; step++)
            {
                features[(row * 16) + step] = Noise(from + row, step) * 0.3f;
            }

            if (bumped)
            {
                var at = 1 + ((from + row) * 7 % 12);

                for (var step = at; step < at + 3; step++)
                {
                    features[(row * 16) + step] += 2f;
                }
            }

            answers[row] = bumped ? 1f : 0f;
        }

        return new TrainingData(Tensor.From(new Shape(count, 16, 1), features), Tensor.From(new Shape(count, 1), answers));
    }

    // An image of eight by eight with a bright blob of two by two on its left half, or on its right, whose answer is one on the left.
    private static TrainingData Blobs(int count, int from)
    {
        var features = new float[count * 64];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            var left = (from + row) % 2 == 0;

            for (var cell = 0; cell < 64; cell++)
            {
                features[(row * 64) + cell] = Noise(from + row, cell) * 0.2f;
            }

            var top = (from + row) * 5 % 6;
            var column = (left ? 0 : 4) + ((from + row) * 3 % 3);

            for (var down = 0; down < 2; down++)
            {
                for (var across = 0; across < 2; across++)
                {
                    features[(row * 64) + ((top + down) * 8) + column + across] += 2f;
                }
            }

            answers[row] = left ? 1f : 0f;
        }

        return new TrainingData(Tensor.From(new Shape(count, 8, 8, 1), features), Tensor.From(new Shape(count, 1), answers));
    }

    // A volume of four cubed with one bright corner or none, whose answer is one where there is a corner.
    private static TrainingData Cubes(int count, int from)
    {
        var features = new float[count * 64];
        var answers = new float[count];

        for (var row = 0; row < count; row++)
        {
            var bright = (from + row) % 2 == 0;

            for (var cell = 0; cell < 64; cell++)
            {
                features[(row * 64) + cell] = Noise(from + row, cell) * 0.2f;
            }

            if (bright)
            {
                // A block of two by two by two somewhere in the cube.
                var plane = (from + row) % 3;
                var line = (from + row) * 5 % 3;
                var column = (from + row) * 7 % 3;

                for (var cell = 0; cell < 8; cell++)
                {
                    features[(row * 64) + ((plane + (cell / 4)) * 16) + ((line + (cell / 2 % 2)) * 4) + column + (cell % 2)] += 2f;
                }
            }

            answers[row] = bright ? 1f : 0f;
        }

        return new TrainingData(Tensor.From(new Shape(count, 4, 4, 4, 1), features), Tensor.From(new Shape(count, 1), answers));
    }

    // Noise between minus one and one, the same for the same row and cell.
    private static float Noise(int row, int cell) => MathF.Sin((row * 12.9898f) + (cell * 78.233f));

    private static double Accuracy(CompiledNetwork compiled, TrainingData data)
    {
        var predicted = compiled.Predict(data.Features, Engine).Values.ToArray();
        var right = predicted.Select((value, at) => (value > 0.5f) == (data.Answers.Values[at] > 0.5f)).Count(correct => correct);

        return right / (double)predicted.Length;
    }
}
