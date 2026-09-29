// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The operations that move values without changing them: another shape for the same values, and the patches a window
/// covers as it walks over an image — which is what a convolution multiplies by its kernel. Images are laid out with
/// their channels last: image, row, column, channel.
/// </summary>
public partial class CpuBackendTests
{
    // One image of three rows of three, a single channel, holding one to nine.
    private static Tensor NineInASquare() => Tensor.From(new Shape(1, 3, 3, 1), [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f]);

    [Fact]
    public void Reshaping_KeepsTheValuesInTheirOrder_UnderAnotherShape()
    {
        var reshaped = _backend.Reshape(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]), new Shape(3, 2));

        Assert.Equal(new Shape(3, 2), reshaped.Shape);
        Assert.Equal<float[]>([1f, 2f, 3f, 4f, 5f, 6f], reshaped.Values.ToArray());
        Assert.Equal(new Shape(6), _backend.Reshape(reshaped, new Shape(6)).Shape);
    }

    [Fact]
    public void AShapeHoldingAnotherNumberOfValues_IsRefusedAndTheMessageNamesBoth()
    {
        var wrong = Assert.Throws<ArgumentException>(() => _backend.Reshape(Tensor.Zeros(new Shape(2, 3)), new Shape(4)));

        Assert.Contains("2x3", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("4", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Unfolding_GivesEveryPlaceTheWindowStandsARowOfWhatItCovers()
    {
        var patches = _backend.Unfold(NineInASquare(), new Window(2, 2));

        Assert.Equal(new Shape(4, 4), patches.Shape);
        Assert.Equal<float[]>([1f, 2f, 4f, 5f, 2f, 3f, 5f, 6f, 4f, 5f, 7f, 8f, 5f, 6f, 8f, 9f], patches.Values.ToArray());
    }

    [Fact]
    public void Unfolding_WithPadding_CoversTheBorderWithNothing()
    {
        var patches = _backend.Unfold(NineInASquare(), new Window(3, 3) { Padding = 1 });

        Assert.Equal(new Shape(9, 9), patches.Shape);
        Assert.Equal<float[]>([0f, 0f, 0f, 0f, 1f, 2f, 0f, 4f, 5f], patches.Values[..9].ToArray());
        Assert.Equal<float[]>([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], patches.Values[36..45].ToArray());
    }

    [Fact]
    public void Unfolding_WithAStride_SkipsThePlacesBetween()
    {
        var patches = _backend.Unfold(NineInASquare(), new Window(1, 1) { Stride = 2 });

        Assert.Equal<float[]>([1f, 3f, 7f, 9f], patches.Values.ToArray());
    }

    [Fact]
    public void Unfolding_TakesEveryChannelOfAPlace_BeforeTheNextPlace()
    {
        // One image of two by two with two channels: the one window covers all of it, row by row, each place's channels
        // side by side.
        var image = Tensor.From(new Shape(1, 2, 2, 2), [1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f]);

        var patches = _backend.Unfold(image, new Window(2, 2));

        Assert.Equal(new Shape(1, 8), patches.Shape);
        Assert.Equal<float[]>([1f, 10f, 2f, 20f, 3f, 30f, 4f, 40f], patches.Values.ToArray());
    }

    [Fact]
    public void Folding_AddsEveryPatchValueBackToThePlaceItCameFrom()
    {
        // Folded back, each value arrives as often as a window covered it: the corners once, the middle four times.
        var folded = _backend.Fold(_backend.Unfold(NineInASquare(), new Window(2, 2)), new Shape(1, 3, 3, 1), new Window(2, 2));

        Assert.Equal(new Shape(1, 3, 3, 1), folded.Shape);
        Assert.Equal<float[]>([1f, 4f, 3f, 8f, 20f, 12f, 7f, 16f, 9f], folded.Values.ToArray());
    }

    [Fact]
    public void Folding_IsUnfoldingTurnedAround()
    {
        // The defining property of an adjoint: what unfolding an image and weighing its patches adds up to is what weighing
        // the image by the patches folded back adds up to. Two images, three channels, a window of three with a stride of
        // two and a border of one.
        var window = new Window(3, 3) { Stride = 2, Padding = 1 };
        var images = Sequence(new Shape(2, 5, 4, 3), 0.37f);
        var patches = _backend.Unfold(images, window);
        var weights = Sequence(patches.Shape, 0.61f);

        var unfolded = Dot(patches, weights);
        var folded = Dot(images, _backend.Fold(weights, images.Shape, window));

        Assert.Equal(unfolded, folded, 1e-3);
    }

    [Fact]
    public void OnlyAnImageBatch_IsUnfolded()
    {
        Assert.Throws<ArgumentException>(() => _backend.Unfold(Tensor.Zeros(new Shape(3, 3)), new Window(2, 2)));
    }

    [Theory]
    [InlineData(0, 2, 1, 0)]
    [InlineData(2, 0, 1, 0)]
    [InlineData(2, 2, 0, 0)]
    [InlineData(2, 2, 1, -1)]
    [InlineData(4, 2, 1, 0)]
    [InlineData(2, 4, 1, 0)]
    public void AWindowThatCannotStandOnTheImage_IsRefused(int height, int width, int stride, int padding)
    {
        var window = new Window(height, width) { Stride = stride, Padding = padding };

        Assert.Throws<ArgumentException>(() => _backend.Unfold(NineInASquare(), window));
    }

    [Fact]
    public void PatchesThatDoNotFitTheImageTheyAreFoldedInto_AreRefused()
    {
        var patches = _backend.Unfold(NineInASquare(), new Window(2, 2));

        Assert.Throws<ArgumentException>(() => _backend.Fold(patches, new Shape(1, 4, 4, 1), new Window(2, 2)));
        Assert.Throws<ArgumentException>(() => _backend.Fold(patches, new Shape(1, 3, 3), new Window(2, 2)));
        Assert.Throws<ArgumentException>(() => _backend.Fold(Tensor.Zeros(new Shape(16)), new Shape(1, 3, 3, 1), new Window(2, 2)));
    }

    // Values that differ from place to place, so a value in the wrong place cannot go unnoticed.
    private static Tensor Sequence(Shape shape, float step) =>
        Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(at => MathF.Sin(at * step))]);

    private static double Dot(Tensor left, Tensor right)
    {
        var total = 0d;

        for (var at = 0; at < left.Values.Length; at++)
        {
            total += left.Values[at] * (double)right.Values[at];
        }

        return total;
    }
}
