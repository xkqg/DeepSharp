// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>Rows as a pipeline hands them over — doubles, a row each — as a network takes them: one tensor of floats.</summary>
/// <remarks>
/// The one conversion between the two. The rows a network learns from and is judged by, the rows the report measures it
/// on and the rows it serves all go through it, so each reaches the network the same way.
/// </remarks>
internal static class HandedRowsExtensions
{
    extension(IReadOnlyList<double[]> rows)
    {
        /// <summary>The rows as one tensor of floats: row after row in their order, each value the float nearest it.</summary>
        /// <param name="width">How many values each row holds.</param>
        /// <returns>A tensor as tall as there are rows, and as wide as said.</returns>
        internal Tensor Floats(int width) =>
            Tensor.From(new Shape(rows.Count, width), [.. rows.SelectMany(row => row.Select(value => (float)value))]);
    }
}
