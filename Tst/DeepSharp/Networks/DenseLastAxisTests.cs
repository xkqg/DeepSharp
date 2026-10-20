// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A dense layer works on the last axis of what it is handed, as Keras's and PyTorch's do: over a batch of series, images or volumes it is
/// the same layer applied to the channels of every place, and what it makes keeps the places.
/// </summary>
public class DenseLastAxisTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static Tensor Batch(params int[] axes) =>
        Tensor.From(new Shape(axes), [.. Enumerable.Range(0, new Shape(axes).Count).Select(at => (float)Math.Sin((at * 0.37) + 0.3))]);

    [Fact]
    public void ADenseLayerOfARankThreeBatch_IsTheSameLayerAppliedToEveryRowAlone()
    {
        var dense = new Dense(3, 2, new RandomStream(1).Draw("dense", 0, 0));
        var series = Batch(4, 5, 3);
        var each = Engine.Reshape(series, new Shape(20, 3));

        var together = dense.Forward(series, Pass.Evaluation(Engine));
        var alone = dense.Forward(each, Pass.Evaluation(Engine));

        Assert.Equal(new Shape(4, 5, 2), together.Shape);
        Assert.Equal(alone.Values.ToArray(), together.Values.ToArray());
    }

    [Fact]
    public void ADenseLayerWithAPair_OverARankFourBatch_KeepsThePlacesAndAppliesThePairToEveryRow()
    {
        var dense = new Dense(2, 3, new Beginning { LowRank = new LowRank(1, 2), Draws = new RandomStream(1).Draw("dense", 0, 0) });
        var images = Batch(2, 3, 3, 2);

        var together = dense.Forward(images, Pass.Evaluation(Engine));
        var alone = dense.Forward(Engine.Reshape(images, new Shape(18, 2)), Pass.Evaluation(Engine));

        Assert.Equal(new Shape(2, 3, 3, 3), together.Shape);
        Assert.Equal(alone.Values.ToArray(), together.Values.ToArray());
    }

    [Fact]
    public void ADenseLayerOfAnotherWidthThanTheLastAxis_IsRefused()
    {
        var dense = new Dense(3, 2, new RandomStream(1).Draw("dense", 0, 0));

        Assert.Throws<ArgumentException>(() => dense.Forward(Batch(2, 4, 5), Pass.Evaluation(Engine)));
    }

    [Fact]
    public void ADenseWordAfterAnImageOrASeries_MakesTheSeriesOrImageOfItsUnits_WithoutAFlatten()
    {
        var series = new Sequential().Dense(4).Relu().Dense(2).Lower(new Shape(7, 3), new RandomStream(1));
        var image = new Sequential().Conv2D(3, new Window(2, 2)).Dense(5).Lower(new Shape(6, 6, 2), new RandomStream(1));

        Assert.Equal(new Shape(2, 7, 2), series.Forward(Batch(2, 7, 3), Pass.Evaluation(Engine)).Shape);
        Assert.Equal(new Shape(2, 5, 5, 5), image.Forward(Batch(2, 6, 6, 2), Pass.Evaluation(Engine)).Shape);
    }
}
