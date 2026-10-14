// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// The larger of two tensors, the smaller, and a tensor held between minus a bound and the bound: made of the operations
/// every engine already has, so the seam does not grow for them and every engine gives the same numbers.
/// </summary>
/// <remarks>
/// A value is chosen rather than computed: <c>Positive</c> says, value by value, where the second is above the first, and the
/// first and the second are each multiplied by a nought or a one before they are added — a number times one is itself and
/// times nought is nought, so the choice is exact where an arithmetic of rectified differences is not (the first plus the
/// rectified difference lands a float away from the second). They are defined for numbers: an infinity multiplied by the
/// nought that leaves it out is not a number, so whatever takes a gradient through them refuses one that is not finite
/// first. The result is the same 32 bits PyTorch's <c>maximum</c>, <c>minimum</c> and <c>clamp</c> give, but for the sign of
/// a nought, which a step against it does not tell apart.
/// </remarks>
internal static class TensorSelectionExtensions
{
    extension(ITensorBackend backend)
    {
        /// <summary>The larger of two tensors of one shape, value by value.</summary>
        /// <param name="first">The first.</param>
        /// <param name="second">The second.</param>
        /// <returns>The second where it is above the first, the first everywhere else.</returns>
        /// <exception cref="ArgumentException">The two are of other shapes.</exception>
        internal Tensor Larger(Tensor first, Tensor second) =>
            Chosen(backend, first, second, backend.Positive(backend.Subtract(second, first)));

        /// <summary>The smaller of two tensors of one shape, value by value.</summary>
        /// <param name="first">The first.</param>
        /// <param name="second">The second.</param>
        /// <returns>The second where it is below the first, the first everywhere else.</returns>
        /// <exception cref="ArgumentException">The two are of other shapes.</exception>
        internal Tensor Smaller(Tensor first, Tensor second) =>
            Chosen(backend, first, second, backend.Positive(backend.Subtract(first, second)));

        /// <summary>A tensor moved a share of the way from a start towards an end, as PyTorch's <c>lerp</c> moves it.</summary>
        /// <param name="start">The tensor moved.</param>
        /// <param name="end">The tensor it is moved towards.</param>
        /// <param name="weight">The share of the way: from nothing to one.</param>
        /// <returns>The start plus the weight times the way to the end — or, for a weight of a half or more, the end less the rest of the way times what is left — as PyTorch orders it.</returns>
        internal Tensor Lerped(Tensor start, Tensor end, double weight) =>
            weight < 0.5
                ? backend.Add(start, backend.Scale(backend.Subtract(end, start), backend.Fill(new Shape(), (float)weight)))
                : backend.Subtract(end, backend.Scale(backend.Subtract(end, start), backend.Fill(new Shape(), (float)(1 - weight))));

        /// <summary>A tensor with every value held between minus a bound and the bound.</summary>
        /// <param name="values">The tensor.</param>
        /// <param name="bound">How far either side of nothing a value may reach: a number above nothing.</param>
        /// <returns>The values, those beyond the bound at the bound.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The bound is not a finite number above nothing.</exception>
        internal Tensor Clamped(Tensor values, float bound)
        {
            ArgumentNullException.ThrowIfNull(values);

            if (!(bound > 0) || float.IsInfinity(bound))
            {
                throw new ArgumentOutOfRangeException(nameof(bound), bound, "A bound is a finite number above nothing.");
            }

            return backend.Smaller(backend.Larger(values, backend.Fill(values.Shape, -bound)), backend.Fill(values.Shape, bound));
        }
    }

    // The tensor a choice makes: the second where the choice says so, the first elsewhere — each multiplied by the one or the
    // nought that keeps it or leaves it out.
    private static Tensor Chosen(ITensorBackend backend, Tensor kept, Tensor taken, Tensor takes) =>
        backend.Add(backend.Multiply(kept, backend.Subtract(backend.Fill(kept.Shape, 1f), takes)), backend.Multiply(taken, takes));
}
