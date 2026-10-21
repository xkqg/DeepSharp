// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// An engine that multiplies matrices stacked on their rows block by block in one operation: what attention asks of a batch of series, one
/// product for every series.
/// </summary>
/// <remarks>
/// Not part of the seam: an engine written before it has never heard of it, and is asked for the products it has — one block at a time,
/// picked out and put back by products with a vector, as every engine always could be. An engine that has it makes block <c>i</c> of the
/// answer the product of block <c>i</c> of each, and for a batch of so many series that is a cost in proportion to the batch, where the
/// products it stands in for cost the square of it. An engine that wraps another and does not pass the product on is one that has not it. Ask through <see cref="TensorBlockExtensions.MatMulInBlocks"/>, which asks the engine
/// that has it and works the answer out for the one that has not.
/// </remarks>
public interface IMultipliesInBlocks
{
    /// <summary>Block <c>i</c> of the first operand, turned or not, times block <c>i</c> of the second, turned or not, for every block.</summary>
    /// <param name="left">A matrix stacked of <see cref="BlockProduct.Count"/> blocks one beneath the other.</param>
    /// <param name="right">A matrix stacked of as many blocks.</param>
    /// <param name="how">How many blocks there are, and which operands' blocks are turned.</param>
    /// <returns>The answers' blocks, one beneath the other.</returns>
    /// <exception cref="ArgumentException">Either is not a matrix, the rows of either do not divide into the blocks, or the blocks do not meet in the middle.</exception>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one block.</exception>
    Tensor MatMulInBlocks(Tensor left, Tensor right, BlockProduct how);
}
