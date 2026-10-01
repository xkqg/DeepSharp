// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A convolution walks a window over a batch of images and gives every place a value per channel it makes: the patches the
/// window covers, times its kernel. Images are laid out with their channels last — image, row, column, channel — and a
/// kernel as many rows as the window holds values, by as many columns as channels it makes. The layers around it turn a row
/// into an image and an image back into a row.
/// </summary>
public abstract class ConvolutionContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void AConvolutionWithABorder_IsPyTorchsForTheSameImageKernelAndBias()
    {
        var convolution = new Conv2D(
            Tensor.From(new Shape(9, 1), [0.1f, 0f, -0.1f, 0.2f, 0.5f, -0.2f, 0.1f, 0f, -0.1f]),
            Tensor.From(new Shape(1), [0.05f]),
            new Window(3, 3) { Padding = 1 });
        var image = Tensor.From(new Shape(1, 4, 4, 1), [.. Enumerable.Range(1, 16).Select(at => at / 16f)]);

        var output = convolution.Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 4, 4, 1), output.Shape);
        AssertClose(
            [0.01874999701976776, 0.07500000298023224, 0.10625000298023224, 0.2562499940395355, 0.056249991059303284, 0.1875000298023224, 0.2187500149011612, 0.4749999940395355,
             0.08125000447034836, 0.3125000596046448, 0.3437500298023224, 0.699999988079071, 0.21875, 0.44999998807907104, 0.4812500476837158, 0.8062500357627869],
            output);
    }

    [Fact]
    public void AConvolutionOfTwoChannelsIntoTwo_WithAStrideOfTwo_IsPyTorchs()
    {
        var convolution = new Conv2D(
            Tensor.From(new Shape(18, 2), [.. Enumerable.Range(0, 36).Select(at => (((at % 2) * 18) + ((at / 2 % 2) * 9) + (at / 4)) / 36f - 0.25f)]),
            Tensor.From(new Shape(2), [0f, 0.1f]),
            new Window(3, 3) { Stride = 2 });
        var image = Tensor.From(new Shape(1, 5, 5, 2), [.. Enumerable.Range(0, 50).Select(at => (((at % 2) * 25) + (at / 2)) / 50f - 0.5f)]);

        var output = convolution.Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 2, 2, 2), output.Shape);
        AssertClose([0.7016666531562805, -0.3683333396911621, 0.6916666030883789, -0.018333330750465393, 0.6516666412353516, 1.3816665410995483, 0.6416666507720947, 1.7316666841506958], output);
        Assert.Equal(2, convolution.InChannels);
        Assert.Equal(2, convolution.OutChannels);
    }

    [Fact]
    public void AConvolutionPaddedAsSame_IsTensorFlowsSame_WithTheOddRowAndColumnOfItsBorderAfter()
    {
        // The probe that found it: a window of three at a stride of two over 28 by 28, as Keras's padding='same' walks it.
        // TensorFlow pads none before and one row and one column after; a border of one stated for every side gives the same
        // fourteen by fourteen places and values up to 4.56 away.
        const int Side = 28;
        var image = Tensor.From(new Shape(1, Side, Side, 1), [.. Enumerable.Range(0, Side * Side).Select(at => (float)Math.Sin((at * 0.7) + 0.3))]);
        var kernel = Tensor.From(new Shape(9, 1), [.. Enumerable.Range(1000, 9).Select(at => (float)Math.Sin((at * 0.7) + 0.3))]);
        var window = new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same };

        var output = new Conv2D(kernel, Tensor.From(new Shape(1), [0f]), window).Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 14, 14, 1), output.Shape);
        Assert.Equal(new Borders(0, 1, 0, 1), window.BordersOver(Side, Side));
        AssertClose(TensorFlowsSame(image, kernel, 3, 2), output);
    }

    [Fact]
    public void AnEvenWindowPaddedAsSame_KeepsEveryRowAndColumnOfTheImage()
    {
        // No border stated for every side keeps 28 rows through a window of two: none gives 27, one gives 29.
        var convolution = new Conv2D(Tensor.From(new Shape(4, 1), [0.25f, 0.25f, 0.25f, 0.25f]), Tensor.From(new Shape(1), [0f]), new Window(2, 2) { PaddingMode = PaddingMode.Same });
        var image = Tensor.From(new Shape(1, 28, 28, 1), [.. Enumerable.Range(0, 784).Select(at => (float)Math.Cos(at))]);

        var output = convolution.Forward(image, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 28, 28, 1), output.Shape);
        Assert.Equal(27, new Window(2, 2).RowsOver(28));
        Assert.Equal(29, new Window(2, 2) { Padding = 1 }.RowsOver(28));
        AssertClose(TensorFlowsSame(image, convolution.Weight.Value, 2, 1), output);
    }

    [Fact]
    public void ABatchOfTwoImages_IsConvolvedAsEachImageAlone()
    {
        var convolution = new Conv2D(3, 5, new Window(3, 3) { Padding = 1, Stride = 2 }, new RandomStream(42).Draw("initialise:0", 0, 0));
        var both = Tensor.From(new Shape(2, 4, 4, 3), [.. Enumerable.Range(0, 96).Select(at => MathF.Sin(at * 0.37f))]);
        var first = Tensor.From(new Shape(1, 4, 4, 3), both.Values[..48]);
        var second = Tensor.From(new Shape(1, 4, 4, 3), both.Values[48..]);

        var together = convolution.Forward(both, Pass.Evaluation(_backend));
        var apart = convolution.Forward(first, Pass.Evaluation(_backend)).Values.ToArray()
            .Concat(convolution.Forward(second, Pass.Evaluation(_backend)).Values.ToArray());

        Assert.Equal(new Shape(2, 2, 2, 5), together.Shape);
        Assert.Equal(apart, together.Values.ToArray());
    }

    [Fact]
    public void AConvolutionDrawnFromTheRunsStream_StartsAsPyTorchsDoes()
    {
        // One over the root of the window's values times the channels in: three by three by three, twenty-seven.
        var convolution = new Conv2D(3, 8, new Window(3, 3), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(new Shape(27, 8), convolution.Weight.Value.Shape);
        Assert.Equal(new Shape(8), convolution.Bias.Value.Shape);
        Assert.All(convolution.Weight.Value.Values.ToArray(), value => Assert.InRange(value, -0.19245f, 0.19245f));
        Assert.All(convolution.Bias.Value.Values.ToArray(), value => Assert.InRange(value, -0.19245f, 0.19245f));
        Assert.Equal(["weight", "bias"], convolution.Slots().Select(slot => slot.Path));
        Assert.Equal(new Window(3, 3), convolution.Window);
    }

    [Fact]
    public void ImagesWithOtherChannelsThanTheConvolutionReads_AreRefused()
    {
        var convolution = new Conv2D(3, 2, new Window(2, 2), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Throws<ArgumentException>(() => convolution.Forward(Tensor.Zeros(new Shape(1, 4, 4, 2)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => convolution.Forward(Tensor.Zeros(new Shape(4, 4)), Pass.Evaluation(_backend)));
    }

    [Fact]
    public void AKernelThatIsNotAWholeWindowOfChannels_OrABiasOfAnotherLength_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new Conv2D(Tensor.Zeros(new Shape(10, 2)), Tensor.Zeros(new Shape(2)), new Window(3, 3)));
        Assert.Throws<ArgumentException>(() => new Conv2D(Tensor.Zeros(new Shape(9, 2)), Tensor.Zeros(new Shape(3)), new Window(3, 3)));
        Assert.Throws<ArgumentException>(() => new Conv2D(Tensor.Zeros(new Shape(9)), Tensor.Zeros(new Shape(1)), new Window(3, 3)));
    }

    [Fact]
    public void AConvolutionOfNoChannels_OrThroughAWindowThatCannotStand_IsRefused()
    {
        var draws = new RandomStream(42).Draw("initialise:0", 0, 0);

        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv2D(0, 2, new Window(3, 3), draws));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Conv2D(2, 0, new Window(3, 3), draws));
        Assert.Throws<ArgumentException>(() => new Conv2D(2, 2, new Window(0, 3), draws));
        Assert.Throws<ArgumentException>(() => new Conv2D(2, 2, new Window(3, 3) { Stride = 0 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv2D(2, 2, new Window(3, 3) { Padding = -1 }, draws));
        Assert.Throws<ArgumentException>(() => new Conv2D(2, 2, new Window(3, 3) { Padding = 1, PaddingMode = PaddingMode.Same }, draws));
        Assert.Throws<ArgumentException>(() => new Conv2D(Tensor.Zeros(new Shape(9, 1)), Tensor.Zeros(new Shape(1)), new Window(3, 3) { Padding = 1, PaddingMode = PaddingMode.Same }));
    }

    [Fact]
    public void Flattening_KeepsEachImageAsOneRow()
    {
        var flat = new Flatten().Forward(Tensor.Zeros(new Shape(2, 2, 2, 3)), Pass.Evaluation(_backend));

        Assert.Equal(new Shape(2, 12), flat.Shape);
    }

    [Fact]
    public void Reshaping_TurnsEachRowIntoTheShapeSaid()
    {
        var rows = Tensor.From(new Shape(2, 16), [.. Enumerable.Range(0, 32).Select(at => (float)at)]);

        var images = new Reshape(new Shape(4, 4, 1)).Forward(rows, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(2, 4, 4, 1), images.Shape);
        Assert.Equal(rows.Values.ToArray(), images.Values.ToArray());
        Assert.Equal(new Shape(4, 4, 1), new Reshape(new Shape(4, 4, 1)).Each);
    }

    [Fact]
    public void RowsThatDoNotHoldTheShapeSaid_OrABatchWithNoRows_AreRefused()
    {
        Assert.Throws<ArgumentException>(() => new Reshape(new Shape(4, 4, 1)).Forward(Tensor.Zeros(new Shape(2, 15)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new Flatten().Forward(Tensor.Zeros(new Shape(3)), Pass.Evaluation(_backend)));
        Assert.Throws<ArgumentException>(() => new Reshape(new Shape()));
        Assert.Throws<ArgumentException>(() => new Reshape(new Shape(3, 0)));
    }

    // One channel convolved as TensorFlow's kernel_shape_util.cc works 'same' out, in double: as many places as the stride
    // fits into the side, the border needed for them split with the odd row or column after the image.
    private static double[] TensorFlowsSame(Tensor image, Tensor kernel, int size, int stride)
    {
        var side = image.Shape[1];
        var places = (side + stride - 1) / stride;
        var before = Math.Max(0, ((places - 1) * stride) + size - side) / 2;
        var output = new double[places * places];

        for (var row = 0; row < places; row++)
        {
            for (var column = 0; column < places; column++)
            {
                var total = 0d;

                for (var down = 0; down < size; down++)
                {
                    for (var across = 0; across < size; across++)
                    {
                        var y = (row * stride) + down - before;
                        var x = (column * stride) + across - before;

                        if (y >= 0 && y < side && x >= 0 && x < side)
                        {
                            total += image.Values[(y * side) + x] * (double)kernel.Values[(down * size) + across];
                        }
                    }
                }

                output[(row * places) + column] = total;
            }
        }

        return output;
    }

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
