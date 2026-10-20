// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The light engine's arithmetic exactly as 0.13.0 released it, written out once and never changed: the oracle every faster
/// kernel is held to, bit for bit, and what <c>tools/measure</c> times a kernel against, which links this file. A copy of the
/// code, not a call into it — what is under test may not be what it is held to.
/// </summary>
internal static class ReleasedCpuArithmetic
{
    /// <summary>The matrix product as 0.13.0 worked it out: a row at a time, each total kept in double precision.</summary>
    /// <param name="a">The left matrix, row-major.</param>
    /// <param name="b">The right matrix, row-major.</param>
    /// <param name="left">The left matrix's shape.</param>
    /// <param name="right">The right matrix's shape.</param>
    /// <returns>The product, row-major.</returns>
    public static float[] MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Shape left, Shape right)
    {
        var rows = left[0];
        var inner = left[1];
        var columns = right[1];
        var result = new float[rows * columns];
        var totals = new double[columns];

        for (var row = 0; row < rows; row++)
        {
            Array.Clear(totals);

            for (var step = 0; step < inner; step++)
            {
                var weight = (double)a[(row * inner) + step];
                var across = b.Slice(step * columns, columns);

                for (var column = 0; column < columns; column++)
                {
                    totals[column] += weight * across[column];
                }
            }

            for (var column = 0; column < columns; column++)
            {
                result[(row * columns) + column] = (float)totals[column];
            }
        }

        return result;
    }

    /// <summary>The log-softmax as 0.13.0 worked it out: each row's sum of exponentials kept in double precision.</summary>
    /// <param name="values">The matrix, row-major.</param>
    /// <param name="columns">How many values a row holds.</param>
    /// <returns>The log-softmax of every row, row-major.</returns>
    public static float[] LogSoftmax(ReadOnlySpan<float> values, int columns)
    {
        var result = new float[values.Length];

        for (var start = 0; start < values.Length; start += columns)
        {
            var row = values.Slice(start, columns);
            var largest = double.NegativeInfinity;

            foreach (var value in row)
            {
                largest = Math.Max(largest, value);
            }

            var total = 0d;

            foreach (var value in row)
            {
                total += Math.Exp(value - largest);
            }

            var shift = largest + Math.Log(total);

            for (var column = 0; column < columns; column++)
            {
                result[start + column] = (float)(row[column] - shift);
            }
        }

        return result;
    }
}
