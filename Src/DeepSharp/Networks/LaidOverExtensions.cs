// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A row laid over every row of a matrix, and a value laid across every column of its row: written in the backend's
/// own operations, so a layer that normalises needs no operation the seam does not have, and its gradient comes back
/// through the operations it was written in.
/// </summary>
/// <remarks>
/// The seam adds a tensor only to one of the same shape, so two shapes that differ by mistake are refused rather than
/// stretched; a layer that means to lay a row over a matrix says so, here.
/// </remarks>
internal static class LaidOverExtensions
{
    extension(ITensorBackend backend)
    {
        /// <summary>A matrix so many rows tall, each row the given one.</summary>
        internal Tensor RowsOf(Tensor row, int rows) => backend.AddRow(backend.Fill(new Shape(rows, row.Shape[0]), 0f), row);

        /// <summary>A matrix whose every row holds one value in every one of so many columns: row by row, the values given.</summary>
        internal Tensor ColumnsOf(Tensor values, int columns) =>
            backend.Transpose(backend.AddRow(backend.Fill(new Shape(columns, values.Shape[0]), 0f), values));

        /// <summary>One value, with no axes: what a tensor is scaled by.</summary>
        internal Tensor Scalar(double value) => backend.Fill(new Shape(), (float)value);
    }
}
