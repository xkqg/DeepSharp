// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A layer that only pads: so many places of nothing before and after each axis of a series, an image or a volume, and
/// nothing else. Every value, and every gradient of a loss through it, is PyTorch's own <c>ConstantPad</c>.
/// </summary>
/// <remarks>
/// It holds no numbers and learns nothing, so a gradient reaching it goes back to the places it padded around and nowhere
/// else — the places of nothing it added have nobody to send anything to.
/// </remarks>
public abstract class PaddingContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Padded { get; } = SpatialReference.Named("pad");

    [Theory]
    [MemberData(nameof(Padded), MemberType = typeof(PaddingContract))]
    public void APadding_IsPyTorchsForTheSameInput(string name)
    {
        var reference = SpatialReference.Of(name);

        var output = reference.Pads().Forward(reference.Images(), Pass.Evaluation(_backend));

        Assert.Equal(reference.OutShape, output.Shape);
        AssertClose(reference.Output, output.Values);
    }

    [Theory]
    [MemberData(nameof(Padded), MemberType = typeof(PaddingContract))]
    public void TheGradientOfALossThroughAPadding_IsPyTorchs(string name)
    {
        var reference = SpatialReference.Of(name);
        var recording = new RecordingBackend(_backend);
        var images = reference.Images();

        var loss = reference.LossOver(recording, reference.Pads().Forward(images, Pass.Evaluation(recording)));
        var gradients = recording.GradientsOf(loss, [images]);

        AssertClose(reference.GradInput, gradients[images].Values);
    }

    [Fact]
    public void APadding_AddsThePlacesItSays_AtEachEndOfEachAxis()
    {
        var series = Tensor.From(new Shape(1, 3, 1), [1f, 2f, 3f]);
        var image = Tensor.From(new Shape(1, 2, 2, 1), [1f, 2f, 3f, 4f]);

        var along = new Pad1D(new Sides(2, 1)).Forward(series, Pass.Evaluation(_backend));
        var over = new Pad2D(new Margins(new Sides(1, 0), new Sides(0, 2))).Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 6, 1), along.Shape);
        Assert.Equal<float[]>([0f, 0f, 1f, 2f, 3f, 0f], along.Values.ToArray());
        Assert.Equal(new Shape(1, 3, 4, 1), over.Shape);
        Assert.Equal<float[]>([0f, 0f, 0f, 0f, 1f, 2f, 0f, 0f, 3f, 4f, 0f, 0f], over.Values.ToArray());
    }

    [Fact]
    public void APaddingOfAValue_PutsThatValueInEveryPlaceItAdds_AndLeavesTheExampleAsItIs()
    {
        var image = Tensor.From(new Shape(1, 2, 2, 1), [1f, 2f, 3f, 4f]);

        var padded = new Pad2D(new Margins(new Sides(1, 0), new Sides(0, 1)), -2.5f).Forward(image, Pass.Evaluation(_backend));

        Assert.Equal<float[]>([-2.5f, -2.5f, -2.5f, 1f, 2f, -2.5f, 3f, 4f, -2.5f], padded.Values.ToArray());
    }

    [Fact]
    public void APadding_LearnsNothing_AndKeepsEachChannelAndEachExampleToItself()
    {
        var both = Tensor.From(new Shape(2, 3, 3, 2), [.. Enumerable.Range(0, 36).Select(at => MathF.Sin(at * 0.37f))]);
        var first = Tensor.From(new Shape(1, 3, 3, 2), both.Values[..18]);
        var layer = new Pad2D(new Margins(new Sides(1, 1), new Sides(2, 0)));

        var together = layer.Forward(both, Pass.Evaluation(_backend));
        var alone = layer.Forward(first, Pass.Evaluation(_backend));

        Assert.Empty(layer.Parameters());
        Assert.Equal(alone.Values.ToArray(), together.Values[..alone.Values.Length].ToArray());
    }

    [Fact]
    public void APadding_RefusesAnInputThatIsNotWhatItPads_AndPlacesBelowNothing()
    {
        Assert.Throws<ArgumentException>(() => new Pad1D(new Sides(1, 1)).Forward(Tensor.Zeros(new Shape(1, 4, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new Pad2D(new Margins(new Sides(1, 1), new Sides(1, 1))).Forward(Tensor.Zeros(new Shape(1, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new Pad3D(new Margins3D(new Sides(1, 1), new Sides(1, 1), new Sides(1, 1))).Forward(Tensor.Zeros(new Shape(1, 4, 4, 1)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new Pad1D(new Sides(-1, 0)));
        Assert.Throws<ArgumentException>(() => new Pad2D(new Margins(new Sides(0, 0), new Sides(0, -2))));
        Assert.Throws<ArgumentException>(() => new Pad3D(new Margins3D(new Sides(0, -1), new Sides(0, 0), new Sides(0, 0))));
    }

    [Fact]
    public void APaddingOfNothing_AddsNoPlace_AndIsTheExampleItself()
    {
        var image = Tensor.From(new Shape(1, 2, 2, 1), [1f, 2f, 3f, 4f]);

        var padded = new Pad2D(new Margins(new Sides(0, 0), new Sides(0, 0))).Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(image.Shape, padded.Shape);
        Assert.Equal(image.Values.ToArray(), padded.Values.ToArray());
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
