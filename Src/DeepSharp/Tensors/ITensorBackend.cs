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

    /// <summary>Multiplies two tensors of the same shape, value by value.</summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">A tensor of the same shape.</param>
    /// <returns>A new tensor; neither argument is changed.</returns>
    /// <exception cref="ArgumentException">The two shapes differ.</exception>
    Tensor Multiply(Tensor left, Tensor right);
}
