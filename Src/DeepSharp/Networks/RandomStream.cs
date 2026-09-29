// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;

namespace DeepSharp.Networks;

/// <summary>
/// Where every random number a network uses comes from, for one run. A stream holds nothing but the run's
/// seed: every batch of numbers it hands out is worked out fresh from the seed and from what the numbers are
/// for, rather than carried forward from whatever was drawn before. That is what lets a run resumed at an
/// epoch draw exactly what it would have drawn the first time, and what stops one layer's draws from
/// shifting when another layer starts asking for numbers too.
/// </summary>
/// <remarks>
/// <para>
/// Underneath, a stream is a counter-based generator — Philox4x64-10 (Salmon, Moraes, Dror and Shaw,
/// "Parallel Random Numbers: As Easy as 1, 2, 3", SC11; the same generator numpy ships as
/// <c>numpy.random.Philox</c>). A counter-based generator turns a counter and a key into a block of numbers
/// by running a fixed function over them, so the block for a given counter is always the same block,
/// however many times and in however many places it is asked for — there is no array to shuffle and no
/// state that steps forward one draw at a time.
/// </para>
/// <para>
/// The key is the seed, widened to 64 bits and paired with zero. The counter is four 64-bit words: which
/// block of a purpose's numbers this is, the step, the epoch, and a hash of the purpose string, in that
/// order. A purpose, an epoch and a step fix the last three of those words — the counter prefix a
/// <see cref="Draws"/> reads its blocks from — and the block number is the only part that moves as a
/// <see cref="Draws"/> is read further.
/// </para>
/// </remarks>
public sealed class RandomStream
{
    private readonly PhiloxKey _key;

    /// <summary>Creates a stream for one run.</summary>
    /// <param name="seed">The number every draw this stream ever hands out is worked out from.</param>
    public RandomStream(long seed)
    {
        Seed = seed;
        _key = new PhiloxKey(unchecked((ulong)seed), 0);
    }

    /// <summary>The seed this stream was created with.</summary>
    public long Seed { get; }

    /// <summary>
    /// The draws for one purpose at one point of a run. Calling this again with the same purpose, epoch and
    /// step, on a stream with the same seed, reads back exactly the same numbers in exactly the same order —
    /// nothing about the result depends on what this stream has been asked for before or since.
    /// </summary>
    /// <param name="purpose">
    /// What the draws are for: <c>"initialise:"</c> and the place a layer stands at, such as <c>"initialise:0"</c>, for
    /// what it starts at; <c>"shuffle"</c> for the order of an epoch's rows; <c>"dropout:"</c> and a dropout layer's path
    /// for what it leaves out. Two different purposes never share a draw.
    /// </param>
    /// <param name="epoch">Which epoch of the run the draws belong to.</param>
    /// <param name="step">Which step of the epoch the draws belong to.</param>
    /// <exception cref="ArgumentException"><paramref name="purpose"/> is null or empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="epoch"/> or <paramref name="step"/> is negative.
    /// </exception>
    public Draws Draw(string purpose, int epoch, int step)
    {
        if (string.IsNullOrEmpty(purpose))
        {
            throw new ArgumentException("A draw has to say what it is for.", nameof(purpose));
        }

        if (epoch < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epoch), epoch, "An epoch cannot be negative.");
        }

        if (step < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(step), step, "A step cannot be negative.");
        }

        var prefix = new PhiloxCounter(0, (ulong)step, (ulong)epoch, PurposeHash(purpose));

        return new Draws(_key, prefix);
    }

    // FNV-1a, 64-bit, over the UTF-8 bytes of the purpose. Chosen for being stable across processes and
    // machines: string.GetHashCode is randomised per process by design, which would turn the same purpose
    // into a different counter word every time the run started, and every draw with it.
    private static ulong PurposeHash(string purpose)
    {
        const ulong OffsetBasis = 0xcbf29ce484222325;
        const ulong Prime = 0x100000001b3;

        var hash = OffsetBasis;
        foreach (var value in Encoding.UTF8.GetBytes(purpose))
        {
            hash ^= value;
            hash = unchecked(hash * Prime);
        }

        return hash;
    }
}
