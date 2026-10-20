// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// An engine that can multiply a matrix turned on its side by another without making the turned copy first: the product a
/// weight's gradient is.
/// </summary>
/// <remarks>
/// Not part of the seam: an engine written outside the library has never heard of it and is asked for the turned copy and the
/// product, as every engine always was. It is a way for an engine that lays its values out where this library can read them
/// to leave out a copy, and the bits it makes are the bits the two operations make.
/// </remarks>
internal interface IMultipliesTurnedLeft
{
    /// <summary>The first matrix turned, times the second: the sum of each pair of columns' products, as the product of the turned copy.</summary>
    /// <param name="left">A matrix of as many rows as the second has.</param>
    /// <param name="right">A matrix.</param>
    /// <returns>A matrix of the first's columns by the second's columns.</returns>
    Tensor MatMulTurnedLeft(Tensor left, Tensor right);
}
