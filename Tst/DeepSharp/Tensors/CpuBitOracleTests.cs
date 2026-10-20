// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Security.Cryptography;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// What a faster kernel is held to: the light engine answers every product and every log-softmax with the very bits 0.13.0
/// answered them with — by a copy of that arithmetic written out beside it, and by a digest of what it made, kept in this
/// file — over every shape that sits on either side of a vector's width, and over values that break a careless sum: a
/// negative nothing, a thousandth of a millionth, a number past the largest, infinities, and two not-a-numbers that differ
/// only in their payload.
/// </summary>
/// <remarks>
/// Every bit of every number is held. Of a result that is no number, only that it is none is: which of two not-a-numbers an addition
/// keeps is the instruction's to say, and the code a coverage run instruments is compiled with the operands in another order than the
/// same code run plainly (measured: 0xFFC00003 plain, 0x7FC00002 instrumented, in the very same cell), so a payload is not a contract
/// any test can hold the engine to.
/// </remarks>
public class CpuBitOracleTests
{
    // The digest of every answer the products below make, as the released arithmetic made them. A product that moved one bit of
    // one cell moves it.
    private const string ProductsDigest = "72B38750AD067468AA0AAC9097F87BEAD9764330516D54495D421D7E17FB61D2";

    private static readonly float[] Specials =
    [
        1f, -1f, 0f, -0f, 1e-20f, -1e-20f, 1e20f, -1e20f, 1e-40f, float.MaxValue, float.Epsilon,
        float.PositiveInfinity, float.NegativeInfinity,
        BitConverter.UInt32BitsToSingle(0x7FC00001), BitConverter.UInt32BitsToSingle(0x7FC00002), BitConverter.UInt32BitsToSingle(0xFFC00003),
    ];

    private readonly CpuBackend _backend = new();

    [Fact]
    public void TheMatrixProduct_IsBitForBitTheOne0130Released_OnEveryShape()
    {
        foreach (var (shape, left, right) in ProductCases())
        {
            AssertSameBits(Released(left, right), _backend.MatMul(left, right), $"a product of {left.Shape} by {right.Shape} ({shape})");
        }
    }

    [Fact]
    public void TheProductsTheReleaseMade_AreTheDigestKeptHere()
    {
        var digest = DigestOf(ProductCases().Select(made => Released(made.Left, made.Right)));

        Assert.True(digest == ProductsDigest, $"The released products digest to {digest}.");
    }

    [Fact]
    public void TheLogSoftmax_IsBitForBitTheOne0130Released_OnEveryShape()
    {
        foreach (var (shape, matrix) in SoftmaxCases())
        {
            AssertSameBits(Tensor.From(matrix.Shape, ReleasedCpuArithmetic.LogSoftmax(matrix.Values, matrix.Shape[1])), _backend.LogSoftmax(matrix), $"the log-softmax of {matrix.Shape} ({shape})");
        }
    }

    private static Tensor Released(Tensor left, Tensor right) =>
        Tensor.From(new Shape(left.Shape[0], right.Shape[1]), ReleasedCpuArithmetic.MatMul(left.Values, right.Values, left.Shape, right.Shape));

    private static IEnumerable<ProductCase> ProductCases()
    {
        var stream = new Stream(20261020);
        var cases = 0;
        int[] rows = [0, 1, 2, 3, 7, 8, 9, 17];
        int[] inners = [0, 1, 2, 3, 16, 17, 64];
        int[] columns = [.. Enumerable.Range(0, 18), 31, 32, 33, 64, 65];

        foreach (var row in rows)
        {
            foreach (var inner in inners)
            {
                foreach (var column in columns)
                {
                    var spoiled = ++cases % 5 == 0;

                    yield return new ProductCase(
                        spoiled ? "with special values" : "of ordinary ones",
                        stream.Matrix(row, inner, spoiled),
                        stream.Matrix(inner, column, spoiled));
                }
            }
        }

        // A product whose every term is the same size, so any change to the order the terms are added in changes the total.
        yield return new ProductCase("of terms that do not commute", Tensor.From(new Shape(1, 4), [1e8f, 1f, -1e8f, 1f]), Tensor.From(new Shape(4, 1), [1f, 1f, 1f, 1f]));
        yield return new ProductCase("of a negative nothing", Tensor.From(new Shape(1, 1), [-1f]), Tensor.From(new Shape(1, 1), [0f]));
    }

    private static IEnumerable<SoftmaxCase> SoftmaxCases()
    {
        var stream = new Stream(20261021);
        var cases = 0;

        foreach (var row in new[] { 0, 1, 2, 5 })
        {
            foreach (var column in Enumerable.Range(0, 20).Concat([31, 32, 33, 65]))
            {
                var spoiled = ++cases % 4 == 0;

                yield return new SoftmaxCase(spoiled ? "with special values" : "of ordinary ones", stream.Matrix(row, column, spoiled));
            }
        }

        yield return new SoftmaxCase("of a row of one value", Tensor.From(new Shape(1, 3), [5f, 5f, 5f]));
        yield return new SoftmaxCase("of a row with an infinity", Tensor.From(new Shape(1, 3), [float.NegativeInfinity, 0f, 1f]));
        yield return new SoftmaxCase("of a row whose largest is a nought of either sign", Tensor.From(new Shape(3, 4), [-3f, -0f, 0f, -1f, -0f, -0f, -5f, -2f, 0f, 0f, -0f, -7f]));
        yield return new SoftmaxCase("of a row of one nought", Tensor.From(new Shape(2, 1), [-0f, 0f]));
    }

    private static string DigestOf(IEnumerable<Tensor> products)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> word = stackalloc byte[4];

        foreach (var product in products)
        {
            foreach (var value in product.Values)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(word, float.IsNaN(value) ? 0x7FC00000u : BitConverter.SingleToUInt32Bits(value));
                hash.AppendData(word);
            }
        }

        return Convert.ToHexString(hash.GetHashAndReset());
    }

    private static void AssertSameBits(Tensor expected, Tensor actual, string what)
    {
        Assert.Equal(expected.Shape, actual.Shape);

        var wanted = expected.Values;
        var got = actual.Values;

        for (var at = 0; at < wanted.Length; at++)
        {
            var wantedBits = BitConverter.SingleToUInt32Bits(wanted[at]);
            var gotBits = BitConverter.SingleToUInt32Bits(got[at]);

            if (float.IsNaN(wanted[at]) && float.IsNaN(got[at]))
            {
                continue;
            }

            Assert.True(wantedBits == gotBits, $"{what}, cell {at}: 0.13.0 made 0x{wantedBits:X8}, this build makes 0x{gotBits:X8}.");
        }
    }

    private readonly record struct ProductCase(string Shape, Tensor Left, Tensor Right)
    {
        public void Deconstruct(out string shape, out Tensor left, out Tensor right) => (shape, left, right) = (Shape, Left, Right);
    }

    private readonly record struct SoftmaxCase(string Shape, Tensor Matrix)
    {
        public void Deconstruct(out string shape, out Tensor matrix) => (shape, matrix) = (Shape, Matrix);
    }

    // Numbers that are the same on every machine and every runtime: SplitMix64, and nothing the runtime owns.
    private sealed class Stream(ulong seed)
    {
        private ulong _state = seed;

        public Tensor Matrix(int rows, int columns, bool spoiled)
        {
            var values = new float[rows * columns];

            for (var at = 0; at < values.Length; at++)
            {
                var word = Next();

                values[at] = spoiled && word % 7 == 0
                    ? Specials[(int)((word >> 8) % (ulong)Specials.Length)]
                    : (float)(((int)(word >> 40) / 8388608.0 - 1.0) * Math.Pow(2, (int)(word % 21) - 10));
            }

            return Tensor.From(new Shape(rows, columns), values);
        }

        private ulong Next()
        {
            _state += 0x9E3779B97F4A7C15UL;
            var z = _state;

            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;

            return z ^ (z >> 31);
        }
    }
}
