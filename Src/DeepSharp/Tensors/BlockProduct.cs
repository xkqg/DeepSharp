// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>How matrices stacked on their rows as so many blocks are multiplied block by block: how many blocks, and whether either operand's blocks are turned on their sides first.</summary>
/// <param name="Count">How many blocks each operand is stacked of; the rows of each divide by it.</param>
public readonly record struct BlockProduct(int Count)
{
    /// <summary>Whether each block of the first operand is turned on its side before it is multiplied; it is not, unless said.</summary>
    public bool TurnLeft { get; init; }

    /// <summary>Whether each block of the second operand is turned on its side before it is multiplied; it is not, unless said.</summary>
    public bool TurnRight { get; init; }
}

/// <summary>The size of one block of a stacked operand.</summary>
/// <param name="Rows">The rows of one block.</param>
/// <param name="Columns">The columns of one block, which are the columns of the operand.</param>
public readonly record struct BlockShape(int Rows, int Columns);

/// <summary>The sizes a product of blocks works over, as <see cref="TensorOperandExtensions.RequireBlockProduct"/> worked them out.</summary>
/// <param name="Left">One block of the first operand, as it is stacked.</param>
/// <param name="Right">One block of the second operand, as it is stacked.</param>
/// <param name="How">How the blocks are multiplied.</param>
public readonly record struct BlockSizes(BlockShape Left, BlockShape Right, BlockProduct How)
{
    /// <summary>The rows of one block of the answer: those of the first block, or its columns where it is turned.</summary>
    public int Rows => How.TurnLeft ? Left.Columns : Left.Rows;

    /// <summary>What the two blocks meet in: the columns of the first block, or its rows where it is turned.</summary>
    public int Inner => How.TurnLeft ? Left.Rows : Left.Columns;

    /// <summary>The columns of one block of the answer: those of the second block, or its rows where it is turned.</summary>
    public int Columns => How.TurnRight ? Right.Rows : Right.Columns;
}
