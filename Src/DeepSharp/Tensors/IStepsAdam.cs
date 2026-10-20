// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tensors;

/// <summary>An engine that can take one of Adam's steps for a parameter in a single pass over its numbers.</summary>
/// <remarks>
/// Not part of the seam: an engine written outside the library has never heard of it and is asked for the twenty operations the
/// step is made of, as every engine always was. An engine that has it makes the very bits those operations make — the same
/// operations in the same order with the same roundings, one number at a time — or says it did not, and the twenty are asked
/// for after all.
/// </remarks>
internal interface IStepsAdam
{
    /// <summary>One step of Adam, or of AdamW from the shrunk parameter, for one parameter.</summary>
    /// <param name="gradient">The parameter's gradient.</param>
    /// <param name="before">What Adam remembered of the parameter before this step; nothing at its first.</param>
    /// <param name="stepping">The tensor the step is taken from, the rate, how the means are kept and the terms the step takes.</param>
    /// <returns>The tensor after the step and what Adam remembers now; nothing where the engine would not make the bits the operations make, which is wherever a number comes out as no number.</returns>
    AdamStepped? Stepped(Tensor gradient, AdamMoments? before, AdamStepping stepping);
}
