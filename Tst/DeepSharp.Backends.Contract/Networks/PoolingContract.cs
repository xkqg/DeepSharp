// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A pooling walks a window over a series, an image or a volume and makes one value of what it covers, for each channel on
/// its own: the largest, or the average. A global pooling makes one value of a whole channel. Border places never count as
/// values: the largest is not a nought a border supplies, and the average of a border is left out unless a window is asked to
/// count it. Every value, and every gradient of a loss through them, is PyTorch's; the border PyTorch cannot state is
/// worked out as TensorFlow works it out.
/// </summary>
public abstract class PoolingContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Windowed { get; } = [.. SpatialReference.Named("max").Concat(SpatialReference.Named("average"))];

    public static TheoryData<string> Global { get; } = [.. SpatialReference.Named("globalAverage").Concat(SpatialReference.Named("globalMax"))];

    [Theory]
    [MemberData(nameof(Windowed), MemberType = typeof(PoolingContract))]
    [MemberData(nameof(Global), MemberType = typeof(PoolingContract))]
    public void APooling_IsPyTorchsForTheSameInput(string name)
    {
        var reference = SpatialReference.Of(name);

        var output = reference.Pooling().Forward(reference.Images(), Pass.Evaluation(_backend));

        Assert.Equal(reference.OutShape, output.Shape);
        AssertClose(reference.Output, output.Values);
    }

    [Theory]
    [MemberData(nameof(Windowed), MemberType = typeof(PoolingContract))]
    [MemberData(nameof(Global), MemberType = typeof(PoolingContract))]
    public void TheGradientOfALossThroughAPooling_IsPyTorchs(string name)
    {
        var reference = SpatialReference.Of(name);
        var recording = new RecordingBackend(_backend);
        var images = reference.Images();

        var loss = reference.LossOver(recording, reference.Pooling().Forward(images, Pass.Evaluation(recording)));
        var gradients = recording.GradientsOf(loss, [images]);

        AssertClose(reference.GradInput, gradients[images].Values);
    }

    [Fact]
    public void ALargestOfNegativeValues_IsNeverTheNoughtABorderSupplies()
    {
        // Padded as 'same', a window over the border and the values sees only the values: the largest of -1 and -2 is -1.
        var series = Tensor.From(new Shape(1, 4, 1), [-1f, -2f, -3f, -4f]);
        var image = Tensor.From(new Shape(1, 3, 3, 1), [.. Enumerable.Range(1, 9).Select(at => -(float)at)]);

        var line = new MaxPool1D(new Window1D(3) { PaddingMode = PaddingMode.Same }).Forward(series, Pass.Evaluation(_backend));
        var plane = new MaxPool2D(new Window(2, 2) { PaddingMode = PaddingMode.Same }).Forward(image, Pass.Evaluation(_backend));

        Assert.Equal<float[]>([-1f, -1f, -2f, -3f], line.Values.ToArray());
        Assert.Equal<float[]>([-1f, -2f, -3f, -4f, -5f, -6f, -7f, -8f, -9f], plane.Values.ToArray());
    }

    [Fact]
    public void AnAverageOverTheBorder_LeavesTheBorderOutOfTheCount_UnlessAskedToCountIt()
    {
        var ones = Tensor.From(new Shape(1, 3, 3, 1), [.. Enumerable.Repeat(1f, 9)]);
        var window = new Window(2, 2) { PaddingMode = PaddingMode.Same };

        var leaving = new AvgPool2D(window).Forward(ones, Pass.Evaluation(_backend));
        var counting = new AvgPool2D(window) { CountsPadding = true }.Forward(ones, Pass.Evaluation(_backend));

        Assert.Equal<float[]>([.. Enumerable.Repeat(1f, 9)], leaving.Values.ToArray());
        Assert.Equal<float[]>([1f, 1f, 0.5f, 1f, 1f, 0.5f, 0.5f, 0.5f, 0.25f], counting.Values.ToArray());
    }

    [Fact]
    public void ValuesThatTie_GiveTheWholeGradientToTheFirstOfThem_AsPyTorchsMaxPoolDoes()
    {
        var recording = new RecordingBackend(_backend);
        var images = Tensor.From(new Shape(1, 4, 4, 1), [.. Enumerable.Repeat(1f, 16)]);
        var layer = new MaxPool2D(new Window(2, 2) { Stride = 2 });

        var loss = recording.Mean(layer.Forward(images, Pass.Evaluation(recording)));
        var gradient = recording.GradientsOf(loss, [images])[images];

        // Four windows, a quarter of the mean each, all to the top left value of the window.
        Assert.Equal<float[]>([.25f, 0f, .25f, 0f, 0f, 0f, 0f, 0f, .25f, 0f, .25f, 0f, 0f, 0f, 0f, 0f], gradient.Values.ToArray());
    }

    [Fact]
    public void AGlobalPooling_KeepsTheAxesItPoolsAsOnesWhenAskedTo()
    {
        var volume = Tensor.From(new Shape(2, 3, 4, 5, 6), [.. Enumerable.Range(0, 720).Select(at => MathF.Sin(at * 0.1f))]);

        var dropped = new GlobalAvgPool3D().Forward(volume, Pass.Evaluation(_backend));
        var kept = new GlobalAvgPool3D { KeepsAxes = true }.Forward(volume, Pass.Evaluation(_backend));
        var largest = new GlobalMaxPool3D { KeepsAxes = true }.Forward(volume, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(2, 6), dropped.Shape);
        Assert.Equal(new Shape(2, 1, 1, 1, 6), kept.Shape);
        Assert.Equal(new Shape(2, 1, 1, 1, 6), largest.Shape);
        Assert.Equal(dropped.Values.ToArray(), kept.Values.ToArray());
    }

    [Fact]
    public void APooling_LearnsNothing_AndKeepsEachChannelAndEachExampleToItself()
    {
        var both = Tensor.From(new Shape(2, 4, 4, 3), [.. Enumerable.Range(0, 96).Select(at => MathF.Sin(at * 0.37f))]);
        var first = Tensor.From(new Shape(1, 4, 4, 3), both.Values[..48]);
        Layer[] layers = [new MaxPool2D(new Window(2, 2) { Stride = 2 }), new AvgPool2D(new Window(3, 3) { Padding = 1, Stride = 2 })];

        foreach (var layer in layers)
        {
            var together = layer.Forward(both, Pass.Evaluation(_backend));
            var alone = layer.Forward(first, Pass.Evaluation(_backend));

            Assert.Empty(layer.Parameters());
            Assert.Equal(alone.Values.ToArray(), together.Values[..alone.Values.Length].ToArray());
        }
    }

    [Fact]
    public void Input_ThatIsNotWhatThePoolingWalks_IsRefused()
    {
        var line = new MaxPool1D(new Window1D(2));
        var plane = new AvgPool2D(new Window(2, 2));
        var volume = new MaxPool3D(new Window3D(2, 2, 2));

        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(1, 4, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(4, 4)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => plane.Forward(Tensor.Zeros(new Shape(1, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => volume.Forward(Tensor.Zeros(new Shape(1, 4, 4, 4)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new GlobalAvgPool2D().Forward(Tensor.Zeros(new Shape(1, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new GlobalMaxPool1D().Forward(Tensor.Zeros(new Shape(4, 4, 4, 4)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void AGlobalPoolingOfAnExampleThatHoldsNoPlace_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new GlobalAvgPool1D().Forward(Tensor.Zeros(new Shape(2, 0, 3)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new GlobalMaxPool2D().Forward(Tensor.Zeros(new Shape(2, 4, 0, 3)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void ASeriesOrAVolumeSmallerThanTheWindowWithItsBorder_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new MaxPool1D(new Window1D(5)).Forward(Tensor.Zeros(new Shape(1, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new AvgPool3D(new Window3D(2, 2, 5)).Forward(Tensor.Zeros(new Shape(1, 3, 3, 4, 1)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void AWindowThatCannotStand_IsRefusedWhereThePoolingIsMade()
    {
        Assert.Throws<ArgumentException>(() => new MaxPool1D(new Window1D(0)));
        Assert.Throws<ArgumentException>(() => new MaxPool1D(new Window1D(3) { Stride = 0 }));
        Assert.Throws<ArgumentException>(() => new AvgPool1D(new Window1D(3) { Padding = 1, PaddingMode = PaddingMode.Same }));
        Assert.Throws<ArgumentException>(() => new MaxPool2D(new Window(2, 0)));
        Assert.Throws<ArgumentException>(() => new AvgPool2D(new Window(2, 2) { Padding = -1 }));
        Assert.Throws<ArgumentException>(() => new MaxPool3D(new Window3D(2, 2, 0)));
        Assert.Throws<ArgumentException>(() => new AvgPool3D(new Window3D(2, 2, 2) { Stride = 0 }));
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
