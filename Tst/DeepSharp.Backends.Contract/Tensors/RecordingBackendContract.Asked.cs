// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The way back is worked out only towards the tensors asked for. Nobody asks how the images of a batch or its rows moved
/// the loss, so the engine is asked for nothing that would only reach them — no fold back onto the images, no product for
/// the rows — and what was asked for comes to the same bits whether or not everything else the pass read was asked for too.
/// </summary>
public abstract partial class RecordingBackendContract
{
    [Fact]
    public void WorkingBackFromAConvolution_FoldsNothingOntoImagesNobodyAskedAbout()
    {
        var (images, convolution) = Convolution();

        var alone = WayBack(pass => SquaredOf(convolution, images, pass), convolution.Weight.Value, convolution.Bias.Value);
        var withImages = WayBack(pass => SquaredOf(convolution, images, pass), images, convolution.Weight.Value, convolution.Bias.Value);

        Assert.Equal(0, alone.Count(nameof(ITensorBackend.Fold)));
        Assert.Equal(1, alone.Count(nameof(ITensorBackend.MatMul)));
        Assert.Equal(1, withImages.Count(nameof(ITensorBackend.Fold)));
        Assert.Equal(2, withImages.Count(nameof(ITensorBackend.MatMul)));
    }

    [Fact]
    public void WorkingBackFromADenseLayer_TakesNoProductForTheRowsNobodyAskedAbout()
    {
        var (rows, dense, answers) = Passengers();

        var alone = WayBack(pass => CrossEntropyOf(dense, rows, answers, pass), dense.Weight.Value, dense.Bias.Value);
        var withRows = WayBack(pass => CrossEntropyOf(dense, rows, answers, pass), rows, dense.Weight.Value, dense.Bias.Value);

        Assert.Equal(1, alone.Count(nameof(ITensorBackend.MatMul)));
        Assert.Equal(2, withRows.Count(nameof(ITensorBackend.MatMul)));
    }

    [Fact]
    public void WhatIsAskedFor_ComesToTheSameBits_WhetherOrNotEveryTensorThePassReadIsAskedForToo()
    {
        var (images, convolution) = Convolution();
        var (rows, dense, answers) = Passengers();

        var convolved = WayBack(pass => SquaredOf(convolution, images, pass), convolution.Weight.Value, convolution.Bias.Value);
        var everyConvolved = WayBack(pass => SquaredOf(convolution, images, pass), images, convolution.Weight.Value, convolution.Bias.Value);
        var weighed = WayBack(pass => CrossEntropyOf(dense, rows, answers, pass), dense.Weight.Value, dense.Bias.Value);
        var everyWeighed = WayBack(pass => CrossEntropyOf(dense, rows, answers, pass), rows, dense.Weight.Value, dense.Bias.Value, answers);

        Assert.Equal(everyConvolved.Gradients[1..], convolved.Gradients);
        Assert.Equal(everyWeighed.Gradients[1..3], weighed.Gradients);
    }

    // Two images of five by five, one channel, through two kernels of three by three with a border of one.
    private static ConvolvedImages Convolution() =>
        new(Random(2, 5, 5, 1), new Conv2D(Random(9, 2), Random(2), new Window(3, 3) { Padding = 1 }));

    // Four rows of three features, a dense layer to one logit, and whether each row's answer is one.
    private static WeighedRows Passengers() =>
        new(Random(4, 3), new Dense(Random(3, 1), Random(1)), Tensor.From(new Shape(4, 1), [1f, 0f, 0f, 1f]));

    private static Tensor SquaredOf(Conv2D convolution, Tensor images, RecordingBackend pass)
    {
        var output = convolution.Forward(images, Pass.Training(pass, new RandomStream(1), 0, 0));

        return pass.Mean(pass.Multiply(output, output));
    }

    private static Tensor CrossEntropyOf(Dense dense, Tensor rows, Tensor answers, RecordingBackend pass) =>
        new BinaryCrossEntropy().Of(dense.Forward(rows, Pass.Training(pass, new RandomStream(1), 0, 0)), answers, pass);

    // The loss worked out through a recording over an engine that notes what it is asked, then the way back to the tensors
    // asked for: what the way back asked the engine for, and the gradient of each tensor asked for, in the order asked.
    private WayBackTaken WayBack(Func<RecordingBackend, Tensor> loss, params Tensor[] asked)
    {
        var noted = new NotingBackend(_backend);
        var pass = new RecordingBackend(noted);
        var value = loss(pass);
        var from = noted.Asked.Count;
        var gradients = pass.GradientsOf(value, asked);

        return new WayBackTaken(noted, from, [.. asked.Select(tensor => gradients[tensor].Values.ToArray())]);
    }

    private readonly record struct ConvolvedImages(Tensor Images, Conv2D Convolution);

    private readonly record struct WeighedRows(Tensor Rows, Dense Dense, Tensor Answers);

    // What one way back asked of the engine — everything it was asked from the given place on — and what it came to.
    private sealed record WayBackTaken(NotingBackend Noted, int From, float[][] Gradients)
    {
        public int Count(string operation) => Noted.Count(operation, From);
    }
}
