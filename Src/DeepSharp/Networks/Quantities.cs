// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// What a number that a setting is must be: a finite number above nothing, a finite number of at least nothing, a power of two.
/// </summary>
/// <remarks>
/// A learning rate, a spread, a penalty and a bound are each one of these, and the library asks the same question of each, in the
/// same words: what the setting is, and then what it must be. Every guard of a layer, a loss, a schedule, an optimizer and a run's
/// options asks here, and so does the reader of a network file, so the file and the code cannot disagree about what a number may be.
/// </remarks>
internal static class Quantities
{
    extension(double value)
    {
        /// <summary>Whether it is a finite number above nothing.</summary>
        /// <returns>Whether it is.</returns>
        public bool IsAboveNothing() => double.IsFinite(value) && value > 0;

        /// <summary>Whether it is a finite number of at least nothing.</summary>
        /// <returns>Whether it is.</returns>
        public bool IsAtLeastNothing() => double.IsFinite(value) && value >= 0;

        /// <summary>Whether it is a power of two: a half, one, two, four.</summary>
        /// <returns>Whether it is.</returns>
        public bool IsPowerOfTwo() => value.IsAboveNothing() && double.IsInteger(double.Log2(value));

        /// <summary>The number, when it is a finite number above nothing.</summary>
        /// <param name="name">The parameter it was handed in as.</param>
        /// <param name="what">What the setting is, as the refusal names it: the start of a sentence.</param>
        /// <returns>The number.</returns>
        /// <exception cref="ArgumentOutOfRangeException">It is nothing or below, or not a finite number.</exception>
        public double AboveNothing(string name, string what) =>
            value.IsAboveNothing() ? value : throw new ArgumentOutOfRangeException(name, value, $"{what} is a finite number above nothing.");

        /// <summary>The number, when it is a finite number of at least nothing.</summary>
        /// <param name="name">The parameter it was handed in as.</param>
        /// <param name="what">What the setting is, as the refusal names it: the start of a sentence.</param>
        /// <returns>The number.</returns>
        /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
        public double AtLeastNothing(string name, string what) =>
            value.IsAtLeastNothing() ? value : throw new ArgumentOutOfRangeException(name, value, $"{what} is a finite number of at least nothing.");

        /// <summary>The number, when it is a power of two.</summary>
        /// <param name="name">The parameter it was handed in as.</param>
        /// <param name="what">What the setting is, as the refusal names it: the start of a sentence.</param>
        /// <returns>The number.</returns>
        /// <exception cref="ArgumentOutOfRangeException">It is not a power of two.</exception>
        public double PowerOfTwo(string name, string what) =>
            value.IsPowerOfTwo() ? value : throw new ArgumentOutOfRangeException(name, value, $"{what} is a power of two.");
    }

    extension(double? value)
    {
        /// <summary>The number, when there is one and it is a finite number above nothing; nothing where nothing was said.</summary>
        /// <param name="name">The parameter it was handed in as.</param>
        /// <param name="what">What the setting is, as the refusal names it: the start of a sentence.</param>
        /// <returns>The number, or nothing.</returns>
        /// <exception cref="ArgumentOutOfRangeException">There is a number, and it is nothing or below, or not a finite number.</exception>
        public double? AboveNothing(string name, string what) => value?.AboveNothing(name, what);

        /// <summary>The number, when there is one and it is a finite number of at least nothing; nothing where nothing was said.</summary>
        /// <param name="name">The parameter it was handed in as.</param>
        /// <param name="what">What the setting is, as the refusal names it: the start of a sentence.</param>
        /// <returns>The number, or nothing.</returns>
        /// <exception cref="ArgumentOutOfRangeException">There is a number, and it is below nothing, or not a finite number.</exception>
        public double? AtLeastNothing(string name, string what) => value?.AtLeastNothing(name, what);
    }
}
