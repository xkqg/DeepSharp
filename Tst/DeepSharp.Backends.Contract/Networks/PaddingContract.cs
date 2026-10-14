// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
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

    // What PyTorch's reflect, replicate and circular padding make of the same examples, and the gradient of a loss through each:
    // the values Fixtures/border-pytorch.py printed.
    private static readonly JsonElement Fills = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "border-pytorch.json")))
        .RootElement.GetProperty("cases");

    public static TheoryData<string> Filled { get; } = [.. Fills.EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

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

    [Theory]
    [MemberData(nameof(Filled), MemberType = typeof(PaddingContract))]
    public void APaddingFilledFromTheExample_IsPyTorchsForTheSameInput(string name)
    {
        var (layer, images, expected) = FilledCase(name);

        var output = layer.Forward(images, Pass.Evaluation(_backend));

        Assert.Equal(new Shape([.. expected.GetProperty("outShape").EnumerateArray().Select(length => length.GetInt32())]), output.Shape);
        AssertClose(Floats(expected, "output"), output.Values);
    }

    [Theory]
    [MemberData(nameof(Filled), MemberType = typeof(PaddingContract))]
    public void TheGradientOfALossThroughAPaddingFilledFromTheExample_IsPyTorchs(string name)
    {
        var (layer, images, expected) = FilledCase(name);
        var recording = new RecordingBackend(_backend);
        var weights = Tensor.From(new Shape([.. expected.GetProperty("outShape").EnumerateArray().Select(length => length.GetInt32())]), Floats(expected, "gradOutput"));

        // The loss is the sum of the padded example times numbers drawn for it, so the gradient is those numbers added back.
        var loss = recording.Mean(recording.Multiply(layer.Forward(images, Pass.Evaluation(recording)), weights));
        var gradients = recording.GradientsOf(loss, [images]);
        var scale = (float)weights.Values.Length;

        AssertClose([.. Floats(expected, "gradInput").Select(value => value / scale)], gradients[images].Values);
    }

    [Fact]
    public void ThePlacesAddedFromTheExample_AreThoseOfPyTorchsOwnDocumentation_OnASeriesOfFour()
    {
        var series = Tensor.From(new Shape(1, 4, 1), [0f, 1f, 2f, 3f]);

        Assert.Equal<float[]>([2f, 1f, 0f, 1f, 2f, 3f, 2f, 1f], new Pad1D(new Sides(2, 2), Filling.Mirror).Forward(series, Pass.Evaluation(_backend)).Values.ToArray());
        Assert.Equal<float[]>([0f, 0f, 0f, 1f, 2f, 3f, 3f, 3f], new Pad1D(new Sides(2, 2), Filling.Repeat).Forward(series, Pass.Evaluation(_backend)).Values.ToArray());
        Assert.Equal<float[]>([2f, 3f, 0f, 1f, 2f, 3f, 0f, 1f], new Pad1D(new Sides(2, 2), Filling.Wrap).Forward(series, Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Fact]
    public void AFillingThatIsConstant_IsTheNothingAPaddingAddsUnlessItIsToldAValue()
    {
        var series = Tensor.From(new Shape(1, 3, 1), [1f, 2f, 3f]);
        var layer = new Pad1D(new Sides(2, 1), Filling.Constant);

        Assert.Equal(Filling.Constant, layer.Filling);
        Assert.Equal(0f, layer.Value);
        Assert.Equal<float[]>([0f, 0f, 1f, 2f, 3f, 0f], layer.Forward(series, Pass.Evaluation(_backend)).Values.ToArray());
        Assert.Equal(Filling.Constant, new Pad1D(new Sides(2, 1), 0.5f).Filling);
        Assert.Equal(Filling.Mirror, new Pad1D(new Sides(1, 1), Filling.Mirror).Filling);
        Assert.Equal(Filling.Repeat, new Pad2D(new Margins(new Sides(1, 1), new Sides(1, 1)), Filling.Repeat).Filling);
        Assert.Equal(Filling.Wrap, new Pad3D(new Margins3D(new Sides(1, 1), new Sides(1, 1), new Sides(1, 1)), Filling.Wrap).Filling);
    }

    [Fact]
    public void AMirrorAddsNoMoreThanTheAxisHasBesidesItsEdge_AndAWrapNoMoreThanTheAxisHolds()
    {
        var series = Tensor.From(new Shape(1, 4, 1), [0f, 1f, 2f, 3f]);

        // Three places beside an edge is as many as four places have; four would reflect the edge itself.
        Assert.Equal(10, new Pad1D(new Sides(3, 3), Filling.Mirror).Forward(series, Pass.Evaluation(_backend)).Values.Length);
        Assert.Contains("mirror", Assert.Throws<ArgumentException>(() => new Pad1D(new Sides(4, 0), Filling.Mirror).Forward(series, Pass.Evaluation(_backend))).Message, StringComparison.Ordinal);
        Assert.Contains("mirror", Assert.Throws<ArgumentException>(() => new Pad1D(new Sides(0, 4), Filling.Mirror).Forward(series, Pass.Evaluation(_backend))).Message, StringComparison.Ordinal);

        // A wrap goes round the axis once at each end, and no further.
        Assert.Equal(12, new Pad1D(new Sides(4, 4), Filling.Wrap).Forward(series, Pass.Evaluation(_backend)).Values.Length);
        Assert.Contains("wrap", Assert.Throws<ArgumentException>(() => new Pad1D(new Sides(5, 0), Filling.Wrap).Forward(series, Pass.Evaluation(_backend))).Message, StringComparison.Ordinal);

        // Repeating the edge has no limit.
        Assert.Equal(104, new Pad1D(new Sides(50, 50), Filling.Repeat).Forward(series, Pass.Evaluation(_backend)).Values.Length);
    }

    [Fact]
    public void APaddingFilledFromTheExample_OfNoPlaces_IsTheExampleItself_AndKeepsEveryChannelAndExampleToItself()
    {
        var both = Tensor.From(new Shape(2, 3, 3, 2), [.. Enumerable.Range(0, 36).Select(at => MathF.Sin(at * 0.37f))]);
        var first = Tensor.From(new Shape(1, 3, 3, 2), both.Values[..18]);
        var layer = new Pad2D(new Margins(new Sides(1, 2), new Sides(2, 0)), Filling.Mirror);

        var together = layer.Forward(both, Pass.Evaluation(_backend));
        var alone = layer.Forward(first, Pass.Evaluation(_backend));
        var nothing = new Pad2D(new Margins(new Sides(0, 0), new Sides(0, 0)), Filling.Wrap).Forward(both, Pass.Evaluation(_backend));

        Assert.Empty(layer.Parameters());
        Assert.Equal(alone.Values.ToArray(), together.Values[..alone.Values.Length].ToArray());
        Assert.Equal(both.Values.ToArray(), nothing.Values.ToArray());
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

    // One case of border-pytorch.json: the layer it pads with, the examples it was handed, and what PyTorch answered.
    private static (Padding Layer, Tensor Images, JsonElement Expected) FilledCase(string name)
    {
        var expected = Fills.EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var mode = expected.GetProperty("mode").GetString() switch
        {
            "mirror" => Filling.Mirror,
            "repeat" => Filling.Repeat,
            _ => Filling.Wrap,
        };
        var pads = expected.GetProperty("pads").EnumerateArray().Select(places => places.GetInt32()).ToArray();
        var images = Tensor.From(
            new Shape([expected.GetProperty("n").GetInt32(), .. expected.GetProperty("extents").EnumerateArray().Select(length => length.GetInt32()), expected.GetProperty("channels").GetInt32()]),
            Floats(expected, "input"));

        Padding layer = expected.GetProperty("rank").GetInt32() switch
        {
            1 => new Pad1D(new Sides(pads[0], pads[1]), mode),
            2 => new Pad2D(new Margins(new Sides(pads[0], pads[1]), new Sides(pads[2], pads[3])), mode),
            _ => new Pad3D(new Margins3D(new Sides(pads[0], pads[1]), new Sides(pads[2], pads[3]), new Sides(pads[4], pads[5])), mode),
        };

        return (layer, images, expected);
    }

    private static float[] Floats(JsonElement written, string key) => [.. written.GetProperty(key).EnumerateArray().Select(value => (float)value.GetDouble())];

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
