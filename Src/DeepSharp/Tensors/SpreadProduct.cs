// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>A matrix product whose rows are worked out by several threads, each a stretch of them.</summary>
/// <remarks>
/// A row's totals are added up from that row of the left matrix and the whole of the right, and written to that row of the product
/// alone, so the stretches share nothing they change and a row is the very bits whichever thread works it out. Each stretch
/// is <see cref="ProductRows"/> over its rows, with the one array of totals it makes for itself.
/// </remarks>
internal sealed class SpreadProduct(float[] left, float[] right, ProductShape shape, bool turnedLeft)
{
    /// <summary>Works the whole product out into <paramref name="result"/>.</summary>
    /// <param name="result">The product, row-major.</param>
    /// <param name="threads">The most threads that may work on it; never more than the rows there are.</param>
    public void Work(float[] result, int threads)
    {
        var stretches = Math.Min(threads, shape.Rows);

        // Stretches of rows side by side, as even as the count allows, and a thread each: the calling thread takes one of them.
        Parallel.For(0, stretches, new ParallelOptions { MaxDegreeOfParallelism = stretches }, stretch =>
            new ProductRows(left, right, shape, turnedLeft).Work(result, (int)((long)shape.Rows * stretch / stretches), (int)((long)shape.Rows * (stretch + 1) / stretches)));
    }
}
