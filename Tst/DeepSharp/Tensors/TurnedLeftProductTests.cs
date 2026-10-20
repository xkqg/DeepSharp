// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// How much a weight moved a loss is the activations turned, times the gradient that reached them. The light engine works that
/// product out from the activations as they lie, without making the turned copy first; an engine that cannot is asked for the
/// turned copy and the product, as every engine was. Either way the gradient is the same bits.
/// </summary>
public class TurnedLeftProductTests
{
    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 3, 7)]
    [InlineData(32, 14, 16)]
    [InlineData(17, 9, 33)]
    [InlineData(64, 70, 128)]
    public void TheProductOfTheLeftTurned_IsTheBitsOfTurningItAndMultiplying(int inner, int rows, int columns)
    {
        var left = Tensor.From(new Shape(inner, rows), Values(inner * rows, 1));
        var right = Tensor.From(new Shape(inner, columns), Values(inner * columns, 2));
        var backend = new CpuBackend();

        var turned = ((IMultipliesTurnedLeft)backend).MatMulTurnedLeft(left, right);
        var released = ReleasedCpuArithmetic.MatMul(backend.Transpose(left).Values, right.Values, new Shape(rows, inner), right.Shape);

        Assert.Equal(new Shape(rows, columns), turned.Shape);
        Assert.True(released.AsSpan().SequenceEqual(turned.Values), "A weight's gradient moved when the turned copy was left out.");
    }

    [Fact]
    public void AGradientThroughAProduct_IsTheSameBits_OnTheEngineThatTurnsInPlaceAndOnOneThatCannot()
    {
        var turning = Gradients(new CpuBackend());
        var copying = Gradients(new CopyCountingBackend());

        Assert.True(turning.Left.Values.SequenceEqual(copying.Left.Values));
        Assert.True(turning.Right.Values.SequenceEqual(copying.Right.Values));
    }

    [Fact]
    public void AWeightsGradient_MakesNoTurnedCopyOfTheActivations()
    {
        var activations = Tensor.From(new Shape(256, 256), Values(256 * 256, 3));
        var weight = Tensor.From(new Shape(256, 256), Values(256 * 256, 4));
        var recording = new RecordingBackend(new CpuBackend());
        var loss = recording.Mean(recording.MatMul(activations, weight));

        // The first pass warms everything up; the second is counted.
        recording.GradientsOf(loss, [weight]);

        var before = GC.GetAllocatedBytesForCurrentThread();

        recording.GradientsOf(loss, [weight]);

        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The mean's own way back makes two matrices of the product's size, the gradient is a third, and what a tensor costs is a little more:
        // not a fourth, the activations turned.
        Assert.InRange(allocated, 0, (3 * 256 * 256 * 4) + 16_384);
    }

    [Fact]
    public void AMatrixTurnedOnItsSide_IsMultipliedOnlyByOneWithAsManyRows_AndByNoneAtAll()
    {
        var turning = (IMultipliesTurnedLeft)new CpuBackend();
        var matrix = Tensor.Zeros(new Shape(3, 2));

        Assert.Throws<ArgumentException>(() => turning.MatMulTurnedLeft(matrix, Tensor.Zeros(new Shape(4, 2))));
        Assert.Throws<ArgumentException>(() => turning.MatMulTurnedLeft(matrix, Tensor.Zeros(new Shape(3))));
        Assert.Throws<ArgumentException>(() => turning.MatMulTurnedLeft(Tensor.Zeros(new Shape(3)), matrix));
        Assert.Throws<ArgumentNullException>(() => turning.MatMulTurnedLeft(null!, matrix));
        Assert.Throws<ArgumentNullException>(() => turning.MatMulTurnedLeft(matrix, null!));
    }

    private static (Tensor Left, Tensor Right) Gradients(ITensorBackend engine)
    {
        var left = Tensor.From(new Shape(32, 14), Values(32 * 14, 5));
        var right = Tensor.From(new Shape(14, 16), Values(14 * 16, 6));
        var recording = new RecordingBackend(engine);
        var loss = recording.Mean(recording.MatMul(left, right));
        var gradients = recording.GradientsOf(loss, [left, right]);

        return (gradients[left], gradients[right]);
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);

        return [.. Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() * 2) - 1))];
    }
}
