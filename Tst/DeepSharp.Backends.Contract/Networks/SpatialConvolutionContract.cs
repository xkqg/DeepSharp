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

    /// <summary>The cases whose output padding is nothing: what a layer handed a kernel reads, since the padding is not in the kernel.</summary>
    public static TheoryData<string> TransposedCases =>
        [.. SpatialReference.All.Where(reference => reference.Kind == "transposed" && reference.OutputPadding.All(places => places == 0))
            .Select(reference => reference.Name)];

    /// <summary>The cases that add places at the end of an axis, which a layer is told and a kernel cannot say.</summary>
    public static TheoryData<string> GrownCases =>
        [.. SpatialReference.All.Where(reference => reference.Kind == "transposed" && reference.OutputPadding.Any(places => places != 0))
            .Select(reference => reference.Name)];

    [Theory]
    [MemberData(nameof(GrownCases), MemberType = typeof(SpatialConvolutionContract))]
    public void ATransposedConvolutionThatAddsPlacesAtTheEnd_IsPyTorchs_ValueAndGradient(string name)
    {
        var reference = SpatialReference.Of(name);
        var recording = new RecordingBackend(_backend);
        var layer = reference.TransposedGrown();
        var images = reference.Images();

        var output = layer.Forward(images, Pass.Evaluation(recording));
        var gradients = recording.GradientsOf(reference.LossOver(recording, output), [images, layer.Weight.Value, layer.Bias.Value]);

        Assert.Equal(reference.OutShape, output.Shape);
        AssertClose(reference.Output, output.Values);
        AssertClose(reference.GradInput, gradients[images].Values);
        AssertClose(reference.GradKernel, gradients[layer.Weight.Value].Values);
        AssertClose(reference.GradBias, gradients[layer.Bias.Value].Values);
    }

    [Fact]
    public void ATransposedConvolution_RefusesABorderItCannotRead_AndPlacesAtTheEndItsWindowWouldGiveAnyway()
    {
        var channels = new Channels { In = 2, Out = 2 };
        var draws = new RandomStream(3).Draw("initialise:test", 0, 0);

        // A window padded as 'same' pads by what the example reaching it is, and a transposed convolution MAKES that example.
        Assert.Throws<ArgumentException>(() => new Conv2DTranspose(channels, new Window(3, 3) { PaddingMode = PaddingMode.Same }, draws));
        Assert.Throws<ArgumentException>(() => new Conv1DTranspose(channels, new Window1D(3) { PaddingMode = PaddingMode.Causal }, draws));

        // Beyond the larger of the stride and the dilation, the window gives the place itself; PyTorch refuses the same.
        Assert.Throws<ArgumentException>(() => new Conv1DTranspose(channels, new Window1D(3) { Stride = 2 }, draws, 2));
        Assert.Throws<ArgumentException>(() => new Conv1DTranspose(channels, new Window1D(3) { Stride = 1 }, draws, 1));

        // One below it is a place the window cannot give, and is taken.
        _ = new Conv1DTranspose(channels, new Window1D(3) { Stride = 2 }, draws, 1);
        _ = new Conv1DTranspose(channels, new Window1D(3) { Dilation = 3 }, draws, 2);
    }

    [Fact]
    public void ATransposedConvolutionStartedAtAKernelAndABias_IsTheOneStartedFromDrawsAndLoaded_AndRefusesAnotherShape()
    {
        var channels = new Channels { In = 2, Out = 3 };
        var window = new Window1D(3) { Stride = 2 };
        var drawn = new Conv1DTranspose(channels, window, new RandomStream(5).Draw("initialise:test", 0, 0));
        var series = Tensor.From(new Shape(1, 4, 2), [.. Enumerable.Range(0, 8).Select(at => MathF.Sin(at * 0.4f))]);

        // The other constructor is for numbers a caller already holds: the same layer, started at them.
        var held = new Conv1DTranspose(drawn.Weight.Value, drawn.Bias.Value, window, channels);

        Assert.Equal(new Shape(2, 9), drawn.Weight.Value.Shape);
        Assert.Equal([0], held.OutputPadding);
        Assert.Equal(
            drawn.Forward(series, Pass.Evaluation(_backend)).Values.ToArray(),
            held.Forward(series, Pass.Evaluation(_backend)).Values.ToArray());

        // The channels say the shape the kernel must be, and the refusal says both; a window that works its border out is
        // refused here too, since this constructor makes the example as the other does.
        var wrong = Assert.Throws<ArgumentException>(() => new Conv1DTranspose(Tensor.Zeros(new Shape(2, 6)), Tensor.Zeros(new Shape(3)), window, channels));
        var grouped = Assert.Throws<ArgumentException>(
            () => new Conv3DTranspose(Tensor.Zeros(new Shape(4, 8)), Tensor.Zeros(new Shape(4)), new Window3D(2, 2, 2), new Channels { In = 4, Out = 4, Groups = 2 }));

        Assert.Contains("2 channels in, 3 out", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("4 channels in, 4 out, in 2 groups", grouped.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(
            () => new Conv2DTranspose(Tensor.Zeros(new Shape(2, 18)), Tensor.Zeros(new Shape(2)), new Window(3, 3) { PaddingMode = PaddingMode.Same }, new Channels { In = 2, Out = 2 }));
    }

    [Theory]
    [MemberData(nameof(TransposedCases), MemberType = typeof(SpatialConvolutionContract))]
    public void ATransposedConvolution_IsPyTorchsForTheSameInputKernelAndBias(string name)
    {
        var reference = SpatialReference.Of(name);

        var output = reference.Transposed().Forward(reference.Images(), Pass.Evaluation(_backend));

        Assert.Equal(reference.OutShape, output.Shape);
        AssertClose(reference.Output, output.Values);
    }

    [Theory]
    [MemberData(nameof(TransposedCases), MemberType = typeof(SpatialConvolutionContract))]
    public void TheGradientsOfALossThroughATransposedConvolution_ArePyTorchs(string name)
    {
        var reference = SpatialReference.Of(name);
        var recording = new RecordingBackend(_backend);
        var layer = reference.Transposed();
        var images = reference.Images();

        var loss = reference.LossOver(recording, layer.Forward(images, Pass.Evaluation(recording)));
        var gradients = recording.GradientsOf(loss, [images, layer.Weight.Value, layer.Bias.Value]);

        AssertClose(reference.GradInput, gradients[images].Values);
        AssertClose(reference.GradKernel, gradients[layer.Weight.Value].Values);
        AssertClose(reference.GradBias, gradients[layer.Bias.Value].Values);
    }

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

    [Fact]
    public void TheTwoWaysAGroupedConvolutionIsWorkedOut_AgreeWithEachOther()
    {
        // A depthwise convolution is a group for every channel, so it can be worked out either way: by spreading each
        // channel's own kernel over every place, or by spreading the kernel into the places the groups give it. The first
        // costs a fraction of the second, which is why it is taken; that they agree is what makes taking it safe.
        var images = Tensor.From(new Shape(2, 5, 5, 4), [.. Enumerable.Range(0, 200).Select(at => MathF.Sin(at * 0.21f))]);
        var kernel = Tensor.From(new Shape(9, 4), [.. Enumerable.Range(0, 36).Select(at => MathF.Cos(at * 0.37f))]);
        var bias = Tensor.From(new Shape(4), [0.1f, -0.2f, 0.3f, -0.4f]);
        var window = new Window(3, 3) { Padding = 1 };

        var apart = new Conv2D(kernel, bias, window, new Channels { In = 4, Out = 4, Groups = 4 });

        // The same convolution said as two groups of two would spread the kernel instead — so the one kernel is laid out
        // for that shape and the values held against each other channel by channel.
        var depthwise = apart.Forward(images, Pass.Evaluation(_backend));
        var grouped = new Conv2D(Spread(kernel), bias, window, new Channels { In = 4, Out = 4, Groups = 2 })
            .Forward(images, Pass.Evaluation(_backend));

        AssertClose(depthwise.Values.ToArray(), grouped.Values);

        // Where every channel is in a group of its own, the kernel a group holds is one column.
        static Tensor Spread(Tensor depthwise)
        {
            var spread = new float[9 * 2 * 4];

            for (var place = 0; place < 9; place++)
            {
                for (var channel = 0; channel < 4; channel++)
                {
                    spread[(((place * 2) + (channel % 2)) * 4) + channel] = depthwise.Values[(place * 4) + channel];
                }
            }

            return Tensor.From(new Shape(18, 4), spread);
        }
    }

    [Fact]
    public void AKernelOfMostlyNoughts_WouldLearnBetweenTheGroups_WhichIsWhyTheKernelAConvolutionHoldsIsTheCompactOne()
    {
        // The reason the kernel a grouped convolution holds is as small as its groups, and not a kernel of the whole shape
        // with noughts between them: a product sends its gradient to every entry of its right operand, mask or no mask, so
        // those noughts would be moved by the first step and the convolution would stop being grouped. Measured here rather
        // than remembered, on the shape a grouped convolution would have used.
        var recording = new RecordingBackend(_backend);
        var patches = Tensor.From(new Shape(4, 8), [.. Enumerable.Range(0, 32).Select(at => MathF.Sin(at * 0.5f) + 1f)]);
        var blocked = Tensor.From(
            new Shape(8, 4),
            [.. Enumerable.Range(0, 32).Select(at => (at % 4 / 2) == (at / 4 % 2) ? MathF.Cos(at) : 0f)]);

        var loss = recording.Mean(recording.MatMul(patches, blocked));
        var gradient = recording.GradientsOf(loss, [blocked])[blocked];

        var offDiagonal = Enumerable.Range(0, 32).Where(at => (at % 4 / 2) != (at / 4 % 2)).Select(at => Math.Abs(gradient.Values[at])).Max();

        Assert.True(offDiagonal > 0f, "A nought between the groups receives gradient, so it cannot be a number the layer holds.");
    }

    [Fact]
    public void AGroupedConvolution_KeepsTheChannelsItWasToldOf_AndWritesThemToItsFile()
    {
        // The channels of a grouped convolution cannot be read off its kernel: the kernel holds one group's worth. So they
        // are held, and the file says them, which is the only thing that fixes the kernel's shape when it is read back.
        var layer = new Conv2D(new Channels { In = 6, Out = 9, Groups = 3 }, new Window(3, 3), new RandomStream(4).Draw("initialise:test", 0, 0));

        Assert.Equal(6, layer.InChannels);
        Assert.Equal(9, layer.OutChannels);
        Assert.Equal(3, layer.Groups);
        Assert.Equal(new Shape(9 * 2, 9), layer.Weight.Value.Shape);

        var read = NetworkDocument.ReadNetwork(
            Written(writer =>
            {
                writer.WritePropertyName("network");
                NetworkDocument.WriteNetwork(writer, new LayerStack(layer, new Flatten()), new MeanSquaredError());
            }),
            "network",
            NetworkCatalog.BuiltIn());
        var again = Assert.IsType<Conv2D>(Assert.IsType<LayerStack>(read.Network).Layers[0]);

        Assert.Equal(layer.Channels, again.Channels);
        Assert.Equal(layer.Weight.Value.Shape, again.Weight.Value.Shape);
    }

    [Fact]
    public void AValueThatIsNotANumber_ReachesEveryChannelAProductReads_WhichIsEveryGroupWhereTheKernelIsSpreadWithNoughts()
    {
        // Measured rather than assumed, because the two ways a grouped convolution is worked out differ here and the
        // difference is worth saying out loud. The channels of a group are what a product READS, and a value that is not a
        // number spreads over everything the product reads: nought times it is not a number either. So where the groups are
        // made by spreading the kernel into their places — which is how a convolution of some groups is worked out — such a
        // value reaches every channel out, and where every channel has a group of its own it does not, because a depthwise
        // convolution multiplies a channel by its own kernel and reads no other. Every number a network holds is finite, so
        // this is the limit of the lowering and not of the layer; it is stated beside the layer for that reason.
        var images = Tensor.From(new Shape(1, 3, 3, 4), [.. Enumerable.Range(0, 36).Select(at => at == 0 ? float.NaN : 1f)]);
        var window = new Window(3, 3) { Padding = 1 };
        var bias = Tensor.Zeros(new Shape(4));

        var whole = new Conv2D(Tensor.From(new Shape(36, 4), [.. Enumerable.Repeat(0.5f, 144)]), bias, window)
            .Forward(images, Pass.Evaluation(_backend)).Values.ToArray();
        var grouped = new Conv2D(
                Tensor.From(new Shape(18, 4), [.. Enumerable.Repeat(0.5f, 72)]),
                bias,
                window,
                new Channels { In = 4, Out = 4, Groups = 2 })
            .Forward(images, Pass.Evaluation(_backend)).Values.ToArray();
        var apart = new Conv2D(
                Tensor.From(new Shape(9, 4), [.. Enumerable.Repeat(0.5f, 36)]),
                bias,
                window,
                new Channels { In = 4, Out = 4, Groups = 4 })
            .Forward(images, Pass.Evaluation(_backend)).Values.ToArray();

        Assert.All(Enumerable.Range(0, 4), at => Assert.True(float.IsNaN(whole[at]), "Every channel in reaches every channel out."));
        Assert.All(Enumerable.Range(0, 4), at => Assert.True(float.IsNaN(grouped[at]), "The kernel spread into the groups' places is read whole."));
        Assert.True(float.IsNaN(apart[0]), "The channel the value is in holds no number.");
        Assert.All(Enumerable.Range(1, 3), at => Assert.False(float.IsNaN(apart[at]), "A channel with a kernel of its own reads no other."));
    }

    private static string Written(Action<System.Text.Json.Utf8JsonWriter> write)
    {
        using var stream = new MemoryStream();
        using (var writer = new System.Text.Json.Utf8JsonWriter(stream, new System.Text.Json.JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            write(writer);
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
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
