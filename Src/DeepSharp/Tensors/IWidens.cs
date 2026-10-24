// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// What an engine adds to <see cref="ITensorBackend"/> when it keeps what it works out in a narrower width than a single number: it
/// can hand a tensor back at full width, so the shares of a gradient that reach a number a network learns are added up wider than
/// each of them was made.
/// </summary>
/// <remarks>
/// Not a member of the seam, so an engine written before it compiles and is asked for nothing: the recording that works a gradient
/// out asks an engine whether it has this, as it asks whether an engine has the product over blocks, and an engine that has not is
/// handed shares to add as they were made. A wrapper that works for an engine must pass it on, or the engine it wraps has not got it.
/// </remarks>
public interface IWidens
{
    /// <summary>The same values at full width.</summary>
    /// <param name="values">A tensor, in whatever width the engine made or took it in.</param>
    /// <returns>A tensor of the same shape and values whose arithmetic from here on is at full width; the tensor itself when it is that already.</returns>
    Tensor Widened(Tensor values);
}
