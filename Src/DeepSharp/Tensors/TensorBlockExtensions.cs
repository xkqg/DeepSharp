// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>Matrices stacked on their rows, multiplied block by block on any engine.</summary>
public static class TensorBlockExtensions
{
    extension(ITensorBackend backend)
    {
        /// <summary>Block <c>i</c> of the first operand, turned or not, times block <c>i</c> of the second, turned or not, for every block.</summary>
        /// <param name="left">A matrix stacked of <see cref="BlockProduct.Count"/> blocks one beneath the other.</param>
        /// <param name="right">A matrix stacked of as many blocks.</param>
        /// <param name="how">How many blocks there are, and which operands' blocks are turned.</param>
        /// <returns>The answers' blocks, one beneath the other.</returns>
        /// <remarks>
        /// An engine that has the product (<see cref="IMultipliesInBlocks"/>) is asked for it, and does it in a cost in proportion to the
        /// number of blocks. An engine that has not is asked for the products it has: each block picked out of its matrix by a product with a
        /// vector, multiplied, and put back by another, so the answer is the same and the cost is the square of the number of blocks.
        /// </remarks>
        /// <exception cref="ArgumentException">Either is not a matrix, the rows of either do not divide into the blocks, or the blocks do not meet in the middle.</exception>
        /// <exception cref="ArgumentOutOfRangeException">There are fewer than one block.</exception>
        public Tensor MatMulInBlocks(Tensor left, Tensor right, BlockProduct how)
        {
            ArgumentNullException.ThrowIfNull(backend);

            return backend is IMultipliesInBlocks own ? own.MatMulInBlocks(left, right, how) : Apart(backend, left, right, how);
        }
    }

    // The products every engine has: one block of each picked out by a product with a one in its place, multiplied, and put where it belongs by another.
    private static Tensor Apart(ITensorBackend backend, Tensor left, Tensor right, BlockProduct how)
    {
        var sizes = left.RequireBlockProduct(right, how);
        Tensor? stacked = null;

        for (var block = 0; block < how.Count; block++)
        {
            var place = new Place(block, how.Count);
            var first = Turned(backend, Picked(backend, left, sizes.Left, place), how.TurnLeft);
            var second = Turned(backend, Picked(backend, right, sizes.Right, place), how.TurnRight);
            var product = backend.MatMul(first, second);
            var placed = how.Count == 1 ? product : backend.MatMul(Selection(sizes.Rows, place, picks: false), product);

            stacked = stacked is null ? placed : backend.Add(stacked, placed);
        }

        return stacked!;
    }

    private static Tensor Picked(ITensorBackend backend, Tensor stack, BlockShape block, Place place) =>
        place.Count == 1 ? stack : backend.MatMul(Selection(block.Rows, place, picks: true), stack);

    private static Tensor Turned(ITensorBackend backend, Tensor block, bool turn) => turn ? backend.Transpose(block) : block;

    // The rows of a block out of the rows of the stack (picks), or the rows of the stack out of the rows of a block (places): a one where they meet.
    private static Tensor Selection(int rows, Place place, bool picks)
    {
        var (at, count) = (place.At, place.Count);
        var values = new float[rows * count * rows];
        var wide = picks ? count * rows : rows;

        for (var row = 0; row < rows; row++)
        {
            if (picks)
            {
                values[(row * wide) + (at * rows) + row] = 1f;
            }
            else
            {
                values[(((at * rows) + row) * wide) + row] = 1f;
            }
        }

        return Tensor.From(picks ? new Shape(rows, count * rows) : new Shape(count * rows, rows), values);
    }

    /// <summary>Which block of the stack is worked on, and how many there are.</summary>
    /// <param name="At">The block, counted from nought.</param>
    /// <param name="Count">How many blocks the matrices are stacked of.</param>
    private readonly record struct Place(int At, int Count);
}
