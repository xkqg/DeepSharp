// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The rows of a matrix product, worked out as the light engine works every total out: a row at a time, each total kept in double
/// precision. The kernel is the one place that arithmetic is written, so a product, and later a product spread over several
/// ranges of rows, are the same bits whatever carries them; what it is held to, bit for bit, is <see cref="CpuBitOracleTests"/>.
/// </summary>
public class ProductRowsTests
{
    [Fact]
    public void TheRowsOfAnyRange_AreTheRowsOfTheWholeProduct_NoMatterWhereTheRangesCut()
    {
        var left = Matrix(13, 9, 1);
        var right = Matrix(9, 21, 2);
        var whole = new float[13 * 21];
        var cut = new float[13 * 21];
        var product = new ProductRows(left, right, new ProductShape(13, 9, 21));

        product.Work(whole, 0, 13);
        product.Work(cut, 0, 4);
        product.Work(cut, 4, 5);
        product.Work(cut, 5, 13);

        Assert.True(whole.AsSpan().SequenceEqual(cut));
    }

    [Fact]
    public void AKernelOverNoRows_WritesNothing()
    {
        var result = new float[3 * 4];
        var product = new ProductRows(Matrix(3, 2, 1), Matrix(2, 4, 2), new ProductShape(3, 2, 4));

        product.Work(result, 2, 2);

        Assert.All(result, value => Assert.Equal(0f, value));
    }

    [Fact]
    public void AProductAllocatesItsResultAndOneRowOfTotals_NoMoreWhateverItsLength()
    {
        var left = Matrix(64, 256, 1);
        var right = Matrix(256, 256, 2);
        var backend = new CpuBackend();
        var product = Tensor.From(new Shape(64, 256), left);
        var other = Tensor.From(new Shape(256, 256), right);

        backend.MatMul(product, other);

        // The least of several products: a product the runtime compiles again, or a probe a coverage run puts in its code for the first time, allocates
        // for itself on the thread that asks, and that is the runtime's allocation and not the product's: under coverage one run allocated 73,880 bytes where the bound is 68,096.
        var allocated = long.MaxValue;

        for (var repeat = 0; repeat < 5; repeat++)
        {
            var before = GC.GetAllocatedBytesForCurrentThread();

            backend.MatMul(product, other);

            allocated = Math.Min(allocated, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        // The result, one total for each column, and what a tensor and its shape cost: a few hundred bytes, never a row more.
        Assert.InRange(allocated, 64 * 256 * 4, (64 * 256 * 4) + (256 * 8) + 512);
    }

    private static float[] Matrix(int rows, int columns, int seed)
    {
        var random = new Random(seed);

        return [.. Enumerable.Range(0, rows * columns).Select(_ => (float)((random.NextDouble() * 2) - 1))];
    }
}
