// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Learners.Networks;

/// <summary>Rows as a pipeline hands them over — doubles, a row each — as a network takes them: one tensor of floats, and two sets of answers made one.</summary>
/// <remarks>
/// The one conversion between the two. The rows a network learns from and is judged by, the rows the report measures it
/// on and the rows it serves all go through it, so each reaches the network the same way. A pair of networks whose answers
/// are added, or one taken from the truth, combines them here number by number.
/// </remarks>
internal static class HandedRowsExtensions
{
    extension(IReadOnlyList<double[]> rows)
    {
        /// <summary>The rows as one tensor of floats: row after row in their order, each value the float nearest it.</summary>
        /// <param name="width">How many values each row holds.</param>
        /// <returns>A tensor as tall as there are rows, and as wide as said.</returns>
        internal Tensor Floats(int width) => rows.Floats(width, null);

        /// <summary>The rows as one tensor of floats, each row laid out as one example of the given shape: an image, a sound.</summary>
        /// <param name="width">How many values each row holds.</param>
        /// <param name="example">The shape of one example, or nothing for a flat row.</param>
        /// <returns>A tensor as tall as there are rows, each row the shape of an example.</returns>
        internal Tensor Floats(int width, IReadOnlyList<int>? example) =>
            Tensor.From(example is null ? new Shape(rows.Count, width) : new Shape([rows.Count, .. example]), [.. rows.SelectMany(row => row.Select(value => (float)value))]);

        /// <summary>One set of answers made of two: each number of these rows with the matching number of the others.</summary>
        /// <param name="second">The rows to combine these with, a row for each of these, as wide.</param>
        /// <param name="combine">What becomes of a number of these rows and the matching number of the others.</param>
        /// <returns>The rows as these are shaped, each number what the operation made of the two.</returns>
        internal double[][] Combined(IReadOnlyList<double[]> second, Func<double, double, double> combine) =>
            [.. rows.Select((row, at) => (double[])[.. row.Select((value, column) => combine(value, second[at][column]))])];
    }
}
