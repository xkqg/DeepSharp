// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// Where a tensor's arithmetic actually happens.
/// </summary>
/// <remarks>
/// This seam exists from the first line for one reason: a model written against it does not know, and never
/// has to learn, whether its arithmetic runs on this machine's vector registers or somewhere else entirely.
/// A backend is handed to whatever needs one; nothing reaches for a shared instance of its own accord.
/// <para>
/// The operations are the ones a first network and its backward pass need, and no more: the backward pass of each
/// is written in these same operations, so working out a gradient never needs anything a backend does not offer.
/// </para>
/// </remarks>
public interface ITensorBackend
{
    /// <summary>Which backend this is, for a log line or an error message. Lowercase, one word.</summary>
    string Name { get; }

    /// <summary>Adds two tensors of the same shape, value by value.</summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">A tensor of the same shape.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Add(Tensor left, Tensor right);

    /// <summary>Subtracts one tensor from another of the same shape, value by value.</summary>
    /// <param name="left">The tensor subtracted from.</param>
    /// <param name="right">A tensor of the same shape, subtracted from it.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Subtract(Tensor left, Tensor right);

    /// <summary>Multiplies two tensors of the same shape, value by value.</summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">A tensor of the same shape.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Multiply(Tensor left, Tensor right);

    /// <summary>Multiplies two matrices: each row of the left against each column of the right.</summary>
    /// <param name="left">A matrix of so many rows, as wide as the right one is tall.</param>
    /// <param name="right">A matrix as tall as the left one is wide.</param>
    /// <returns>A new matrix as tall as the left one and as wide as the right one.</returns>
    /// <exception cref="ArgumentException">Either is not a matrix, or the left is not as wide as the right is tall.</exception>
    Tensor MatMul(Tensor left, Tensor right);

    /// <summary>Turns a matrix's rows into its columns.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <returns>A new matrix as tall as the given one is wide.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix.</exception>
    Tensor Transpose(Tensor matrix);

    /// <summary>Adds one row to every row of a matrix, as a layer adds its bias to every example of a batch.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <param name="row">A tensor of one axis, as long as the matrix is wide.</param>
    /// <returns>A new matrix of the same shape.</returns>
    /// <exception cref="ArgumentException">
    /// The first is not a matrix, the second is not a row, or the row is not as long as the matrix is wide.
    /// </exception>
    Tensor AddRow(Tensor matrix, Tensor row);

    /// <summary>Adds up the rows of a matrix: one total for each column.</summary>
    /// <param name="matrix">The matrix.</param>
    /// <returns>A new tensor of one axis, as long as the matrix is wide.</returns>
    /// <exception cref="ArgumentException">The tensor is not a matrix.</exception>
    Tensor SumRows(Tensor matrix);

    /// <summary>The average of every value, as one value with no axes: what a loss is.</summary>
    /// <param name="values">The values.</param>
    /// <returns>A new tensor with no axes.</returns>
    /// <exception cref="ArgumentException">The tensor holds no values.</exception>
    Tensor Mean(Tensor values);

    /// <summary>Multiplies every value by one value.</summary>
    /// <param name="values">The values.</param>
    /// <param name="factor">A tensor with no axes: the one value to multiply by.</param>
    /// <returns>A new tensor of the shape of the values.</returns>
    /// <exception cref="ArgumentException">The factor has axes.</exception>
    Tensor Scale(Tensor values, Tensor factor);

    /// <summary>A tensor of a shape with every value the same.</summary>
    /// <param name="shape">The shape.</param>
    /// <param name="value">The value every position holds.</param>
    /// <returns>A new tensor.</returns>
    Tensor Fill(Shape shape, float value);
}
