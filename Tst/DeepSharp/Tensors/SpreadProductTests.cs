// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A large product can be spread over the threads of the process a host says it may use. Every row of a product is added up on its
/// own, so a row worked out by one thread is the very bits it is worked out as by another, and a product spread over any number
/// of threads is the product worked out by one. The light engine starts no thread unless it is told how many it may use.
/// </summary>
public class SpreadProductTests
{
    [Theory]
    [InlineData(1, 1, 1, 8)]
    [InlineData(1, 40, 40, 8)]
    [InlineData(2, 9, 33, 8)]
    [InlineData(5, 17, 8, 2)]
    [InlineData(37, 16, 65, 3)]
    [InlineData(64, 31, 64, 4)]
    [InlineData(97, 70, 128, 7)]
    [InlineData(32, 0, 5, 4)]
    public void AProductSpreadOverThreads_IsTheBitsOfTheProductWorkedOutByOne_WhateverTheShapeAndTheCount(int rows, int inner, int columns, int threads)
    {
        var left = Tensor.From(new Shape(rows, inner), Values(rows * inner, 1));
        var right = Tensor.From(new Shape(inner, columns), Values(inner * columns, 2));
        var spread = new CpuBackend { Threads = threads, SpreadFrom = 1 }.MatMul(left, right);
        var released = ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape);

        Assert.True(released.AsSpan().SequenceEqual(spread.Values), $"A {rows}x{inner} by {inner}x{columns} product moved a bit when {threads} threads worked it out.");
    }

    [Fact]
    public void AProductOfTheLeftTurned_SpreadOverThreads_IsTheBitsOfTheOneWorkedOutByOne()
    {
        var left = Tensor.From(new Shape(48, 33), Values(48 * 33, 3));
        var right = Tensor.From(new Shape(48, 21), Values(48 * 21, 4));
        var one = (IMultipliesTurnedLeft)new CpuBackend();
        var many = (IMultipliesTurnedLeft)new CpuBackend { Threads = 5, SpreadFrom = 1 };

        Assert.True(one.MatMulTurnedLeft(left, right).Values.SequenceEqual(many.MatMulTurnedLeft(left, right).Values));
    }

    [Fact]
    public void AProductWithNoNumberInIt_IsTheBitsOfTheOneWorkedOutByOne_WhenSpread()
    {
        float[] special = [float.NaN, BitConverter.UInt32BitsToSingle(0x7FC00002), float.PositiveInfinity, 0f, -0f, 1e20f];
        var leftValues = Values(12 * 9, 5);
        var rightValues = Values(9 * 17, 6);

        for (var at = 0; at < special.Length; at++)
        {
            leftValues[(at * 7) % leftValues.Length] = special[at];
            rightValues[(at * 11) % rightValues.Length] = special[(at + 1) % special.Length];
        }

        var left = Tensor.From(new Shape(12, 9), leftValues);
        var right = Tensor.From(new Shape(9, 17), rightValues);
        var released = ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape);

        Assert.True(released.AsSpan().SequenceEqual(new CpuBackend { Threads = 4, SpreadFrom = 1 }.MatMul(left, right).Values));
    }

    [Fact]
    public void TwoProductsSpreadAtTheSameMomentOnOneEngine_ShareNothingTheyChange()
    {
        var engine = new CpuBackend { Threads = 4, SpreadFrom = 1 };
        var left = Tensor.From(new Shape(64, 40), Values(64 * 40, 7));
        var right = Tensor.From(new Shape(40, 64), Values(40 * 64, 8));
        var expected = ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape);
        var wrong = 0;

        Parallel.For(0, 16, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            if (!expected.AsSpan().SequenceEqual(engine.MatMul(left, right).Values))
            {
                Interlocked.Increment(ref wrong);
            }
        });

        Assert.Equal(0, wrong);
    }

    [Fact]
    public void AnEngineStartsNoThreadUnlessItIsToldHowManyItMayUse()
    {
        Assert.Equal(1, new CpuBackend().Threads);
        Assert.Equal(1L << 24, new CpuBackend().SpreadFrom);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-3)]
    public void ANumberOfThreadsBelowOne_IsRefused(int threads)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuBackend { Threads = threads });
    }

    [Fact]
    public void AThresholdBelowOne_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CpuBackend { SpreadFrom = 0 });
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);

        return [.. Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() * 2) - 1))];
    }
}
