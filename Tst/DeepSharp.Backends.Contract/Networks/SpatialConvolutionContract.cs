// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A convolution along one axis of a series, or three of a volume, is the two-dimensional convolution's own arithmetic over
/// a window of its own: the patch the window covers, times a kernel as many rows as the window holds values, channels in
/// included, by as many columns as channels out. Every value, and every gradient of a loss through them, is PyTorch's.
/// </summary>
public abstract class SpatialConvolutionContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Cases => SpatialReference.Named("convolution");

    [Theory]
    [MemberData(nameof(Cases), MemberType = typeof(SpatialConvolutionContract))]
    public void AConvolution_IsPyTorchsForTheSameInputKernelAndBias(string name)
    {
        var reference = SpatialReference.Of(name);

        var output = reference.Convolution().Forward(reference.Images(), Pass.Evaluation(_backend));

        Assert.Equal(reference.OutShape, output.Shape);
        AssertClose(reference.Output, output.Values);
    }

    [Theory]
    [MemberData(nameof(Cases), MemberType = typeof(SpatialConvolutionContract))]
    public void TheGradientsOfALossThroughAConvolution_ArePyTorchs_ForTheInputTheKernelAndTheBias(string name)
    {
        var reference = SpatialReference.Of(name);
        var recording = new RecordingBackend(_backend);
        var convolution = reference.Convolution();
        var images = reference.Images();

        var loss = reference.LossOver(recording, convolution.Forward(images, Pass.Evaluation(recording)));
        var gradients = recording.GradientsOf(loss, [images, convolution.Weight.Value, convolution.Bias.Value]);

        AssertClose(reference.GradInput, gradients[images].Values);
        AssertClose(reference.GradKernel, gradients[convolution.Weight.Value].Values);
        AssertClose(reference.GradBias, gradients[convolution.Bias.Value].Values);
    }

    [Fact]
    public void AConvolutionAlongOneAxis_PaddedAsCausal_LetsNoPlaceSeeWhatComesAfterIt()
    {
        var convolution = new Conv1D(Tensor.From(new Shape(3, 1), [1f, 10f, 100f]), Tensor.From(new Shape(1), [0f]), new Window1D(3) { PaddingMode = PaddingMode.Causal });
        var series = Tensor.From(new Shape(1, 5, 1), [1f, 2f, 3f, 4f, 5f]);
        var later = Tensor.From(new Shape(1, 5, 1), [1f, 2f, 3f, 40f, 50f]);

        var output = convolution.Forward(series, Pass.Evaluation(_backend));
        var changed = convolution.Forward(later, Pass.Evaluation(_backend));

        Assert.Equal<float[]>([100f, 210f, 321f, 432f, 543f], output.Values.ToArray());
        Assert.Equal(output.Values[..3].ToArray(), changed.Values[..3].ToArray());
        Assert.NotEqual(output.Values[3], changed.Values[3]);
    }

    [Fact]
    public void AConvolutionDrawnFromTheRunsStream_StartsAsPyTorchsDoes_AlongOneAxisAndThree()
    {
        // One over the root of the values a window holds, channels in included: three times three, nine; two by three by three
        // times two, thirty-six.
        var line = new Conv1D(3, 8, new Window1D(3), new RandomStream(42).Draw("initialise:0", 0, 0));
        var volume = new Conv3D(2, 4, new Window3D(2, 3, 3), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(new Shape(9, 8), line.Weight.Value.Shape);
        Assert.Equal(new Shape(8), line.Bias.Value.Shape);
        Assert.All(line.Weight.Value.Values.ToArray(), value => Assert.InRange(value, -1f / 3f, 1f / 3f));
        Assert.Equal(new Shape(36, 4), volume.Weight.Value.Shape);
        Assert.Equal(new Shape(4), volume.Bias.Value.Shape);
        Assert.All(volume.Weight.Value.Values.ToArray(), value => Assert.InRange(value, -1f / 6f, 1f / 6f));
        Assert.All(volume.Bias.Value.Values.ToArray(), value => Assert.InRange(value, -1f / 6f, 1f / 6f));
        Assert.Equal(["weight", "bias"], volume.Slots().Select(slot => slot.Path));
        Assert.Equal(3, line.InChannels);
        Assert.Equal(8, line.OutChannels);
        Assert.Equal(2, volume.InChannels);
        Assert.Equal(4, volume.OutChannels);
        Assert.Equal(new Window1D(3), line.Window);
        Assert.Equal(new Window3D(2, 3, 3), volume.Window);
    }

    [Fact]
    public void ABatchOfTwo_IsConvolvedAsEachOfThemAlone_AlongOneAxisAndThree()
    {
        var line = new Conv1D(2, 3, new Window1D(3) { Padding = 1, Stride = 2 }, new RandomStream(42).Draw("initialise:0", 0, 0));
        var volume = new Conv3D(2, 3, new Window3D(2, 2, 2) { Padding = 1, Stride = 2 }, new RandomStream(42).Draw("initialise:0", 0, 0));

        AssertApart(line, new Shape(2, 5, 2));
        AssertApart(volume, new Shape(2, 3, 3, 3, 2));
    }

    [Fact]
    public void Input_ThatIsNotWhatTheConvolutionReads_IsRefused()
    {
        var line = new Conv1D(3, 2, new Window1D(2), new RandomStream(42).Draw("initialise:0", 0, 0));
        var volume = new Conv3D(3, 2, new Window3D(2, 2, 2), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(1, 4, 2)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(1, 4, 4, 3)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(4, 3)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => volume.Forward(Tensor.Zeros(new Shape(1, 3, 3, 3, 2)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => volume.Forward(Tensor.Zeros(new Shape(1, 3, 3, 3)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void ASeriesOrAVolumeSmallerThanTheWindowWithItsBorder_IsRefused()
    {
        var line = new Conv1D(1, 1, new Window1D(5), new RandomStream(42).Draw("initialise:0", 0, 0));
        var volume = new Conv3D(1, 1, new Window3D(2, 2, 5), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Throws<ArgumentException>(() => line.Forward(Tensor.Zeros(new Shape(1, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => volume.Forward(Tensor.Zeros(new Shape(1, 3, 3, 4, 1)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void AKernelThatIsNotAWholeWindowOfChannels_OrABiasOfAnotherLength_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new Conv1D(Tensor.Zeros(new Shape(10, 2)), Tensor.Zeros(new Shape(2)), new Window1D(3)));
        Assert.Throws<ArgumentException>(() => new Conv1D(Tensor.Zeros(new Shape(9, 2)), Tensor.Zeros(new Shape(3)), new Window1D(3)));
        Assert.Throws<ArgumentException>(() => new Conv1D(Tensor.Zeros(new Shape(9)), Tensor.Zeros(new Shape(1)), new Window1D(3)));
        Assert.Throws<ArgumentException>(() => new Conv3D(Tensor.Zeros(new Shape(9, 2)), Tensor.Zeros(new Shape(2)), new Window3D(2, 2, 2)));
        Assert.Throws<ArgumentException>(() => new Conv3D(Tensor.Zeros(new Shape(8, 2)), Tensor.Zeros(new Shape(3)), new Window3D(2, 2, 2)));
        Assert.Throws<ArgumentException>(() => new Conv3D(Tensor.Zeros(new Shape(8)), Tensor.Zeros(new Shape(1)), new Window3D(2, 2, 2)));
    }

    [Fact]
    public void AConvolutionOfNoChannels_OrThroughAWindowThatCannotStand_IsRefused()
    {
        var draws = new RandomStream(42).Draw("initialise:0", 0, 0);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv1D(0, 2, new Window1D(3), draws));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv1D(2, 0, new Window1D(3), draws));
        Assert.Throws<ArgumentException>(() => new Conv1D(2, 2, new Window1D(0), draws));
        Assert.Throws<ArgumentException>(() => new Conv1D(2, 2, new Window1D(3) { Stride = 0 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv1D(2, 2, new Window1D(3) { Padding = -1 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv1D(2, 2, new Window1D(3) { Padding = 1, PaddingMode = PaddingMode.Same }, draws));
        Assert.Throws<ArgumentException>(() => new Conv1D(Tensor.Zeros(new Shape(3, 1)), Tensor.Zeros(new Shape(1)), new Window1D(3) { Padding = 1, PaddingMode = PaddingMode.Causal }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv3D(0, 2, new Window3D(2, 2, 2), draws));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv3D(2, 0, new Window3D(2, 2, 2), draws));
        Assert.Throws<ArgumentException>(() => new Conv3D(2, 2, new Window3D(2, 0, 2), draws));
        Assert.Throws<ArgumentException>(() => new Conv3D(2, 2, new Window3D(2, 2, 2) { Stride = 0 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv3D(2, 2, new Window3D(2, 2, 2) { Padding = -1 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv3D(Tensor.Zeros(new Shape(8, 1)), Tensor.Zeros(new Shape(1)), new Window3D(2, 2, 2) { Padding = 1, PaddingMode = PaddingMode.Same }));
    }

    // The batch convolved at once is each example convolved alone, one after another.
    private void AssertApart(Convolution convolution, Shape batch)
    {
        var both = Tensor.From(batch, [.. Enumerable.Range(0, batch.Count).Select(at => MathF.Sin(at * 0.37f))]);
        var each = batch.Count / batch[0];
        var rest = new Shape([1, .. batch.Axes[1..]]);
        var first = Tensor.From(rest, both.Values[..each]);
        var second = Tensor.From(rest, both.Values[each..]);

        var together = convolution.Forward(both, Pass.Evaluation(_backend));
        var apart = convolution.Forward(first, Pass.Evaluation(_backend)).Values.ToArray()
            .Concat(convolution.Forward(second, Pass.Evaluation(_backend)).Values.ToArray());

        Assert.Equal(2, together.Shape[0]);
        Assert.Equal(apart, together.Values.ToArray());
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], Math.Max(1e-4, Math.Abs(expected[at]) * 1e-4));
        }
    }
}
