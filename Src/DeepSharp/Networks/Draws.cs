// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// The draws for one purpose at one point of a run. A <see cref="Draws"/> reads numbers off the Philox
/// blocks for one fixed counter prefix, in order, working out a fresh block of four words whenever the four
/// words it is holding have all been read. Two readers built from the same prefix and key read back the
/// same numbers in the same order; nothing about one reader depends on any other.
/// </summary>
public sealed class Draws
{
    private const float FloatScale = 1.0f / 16777216f;

    private const double DoubleScale = 1.0 / 9007199254740992.0;

    private readonly PhiloxKey _key;

    private readonly PhiloxCounter _prefix;

    private PhiloxCounter _block;

    private ulong _blockIndex;

    private int _wordPosition = 4;

    internal Draws(PhiloxKey key, PhiloxCounter prefix)
    {
        _key = key;
        _prefix = prefix;
    }

    /// <summary>A value at or above 0 and below 1, taken from the top 24 bits of the next 64-bit word.</summary>
    public float NextSingle() => (NextWord() >> 40) * FloatScale;

    /// <summary>
    /// A value at or above 0 and below 1, converted from the next 64-bit word the way numpy converts one:
    /// the top 53 bits, scaled by 2 to the power of minus 53.
    /// </summary>
    public double NextDouble() => (NextWord() >> 11) * DoubleScale;

    /// <summary>
    /// A whole number at or above 0 and below <paramref name="below"/>, every one of them exactly as likely
    /// as the others. Found by Lemire's method, over the top 32 bits of the next word: those bits, multiplied
    /// by the bound, place the result in range directly, and only the narrow band of outcomes that would
    /// land unevenly is ever redrawn.
    /// </summary>
    /// <param name="below">How many values to choose from, counting from zero.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="below"/> is less than one.</exception>
    public int NextInt(int below)
    {
        if (below < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(below), below, "There must be at least one value to choose below.");
        }

        var bound = (uint)below;
        var product = (NextWord() >> 32) * bound;

        if ((uint)product < bound)
        {
            var threshold = unchecked(0u - bound) % bound;
            while ((uint)product < threshold)
            {
                product = (NextWord() >> 32) * bound;
            }
        }

        return (int)(product >> 32);
    }

    /// <summary>A shuffled order of 0, 1, and up to but not including <paramref name="count"/>, by Fisher-Yates.</summary>
    /// <param name="count">How many indices to permute.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative.</exception>
    public int[] Permutation(int count)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "A permutation cannot cover a negative count.");
        }

        var values = new int[count];
        for (var index = 0; index < count; index++)
        {
            values[index] = index;
        }

        for (var index = count - 1; index > 0; index--)
        {
            var swapWith = NextInt(index + 1);
            (values[index], values[swapWith]) = (values[swapWith], values[index]);
        }

        return values;
    }

    private ulong NextWord()
    {
        if (_wordPosition == 4)
        {
            _block = Philox.Block(_prefix with { Word0 = _blockIndex }, _key);
            _blockIndex++;
            _wordPosition = 0;
        }

        var word = _wordPosition switch
        {
            0 => _block.Word0,
            1 => _block.Word1,
            2 => _block.Word2,
            _ => _block.Word3,
        };

        _wordPosition++;
        return word;
    }
}
