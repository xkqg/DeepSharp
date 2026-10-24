// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// What an engine adds to <see cref="ITensorBackend"/> when it keeps what it works out in a narrower width than a single number: it
/// names the width, so a checkpoint of a run on it can say which width the run's every step was rounded in.
/// </summary>
/// <remarks>
/// Not a member of the seam, so an engine written before it compiles and names no width — which is how an engine at full width
/// names itself, and how every checkpoint written before this was written. A run goes on from a checkpoint only on an engine that
/// names the width the checkpoint names, since another width rounds every step otherwise.
/// </remarks>
public interface INamesItsPrecision
{
    /// <summary>
    /// The width the engine keeps what it works out in, as a lowercase word — <c>bfloat16</c>, <c>half</c> — or nothing for an engine
    /// at full width, which names no width.
    /// </summary>
    string? Precision { get; }
}
