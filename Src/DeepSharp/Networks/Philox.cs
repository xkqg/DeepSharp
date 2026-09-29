// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>The four 64-bit words a Philox4x64 counter is made of.</summary>
/// <remarks>
/// In a <see cref="RandomStream"/>, <see cref="Word0"/> is which block of a purpose's numbers this is,
/// <see cref="Word1"/> is the step, <see cref="Word2"/> is the epoch, and <see cref="Word3"/> is a hash of
/// the purpose string.
/// </remarks>
internal readonly record struct PhiloxCounter(ulong Word0, ulong Word1, ulong Word2, ulong Word3);

/// <summary>The two 64-bit words a Philox4x64 key is made of.</summary>
internal readonly record struct PhiloxKey(ulong Word0, ulong Word1);

/// <summary>
/// The Philox4x64-10 block function (Salmon, Moraes, Dror and Shaw, "Parallel Random Numbers: As Easy as
/// 1, 2, 3", SC11 — the same generator numpy ships as <c>numpy.random.Philox</c>): ten rounds of
/// multiply-and-swap over a counter, mixed with a key that advances by a fixed amount every round. It keeps
/// no state of its own — the same counter and key always produce the same four words, on any machine, in
/// any process — which is what makes a draw replayable from nothing more than the numbers that went into
/// its counter.
/// </summary>
internal static class Philox
{
    // The two multipliers the round function's 64x64->128 multiplications use, and the two amounts the key
    // advances by every round (a Weyl sequence) — the constants the published algorithm specifies.
    private const ulong Multiplier0 = 0xD2E7470EE14C6C93;

    private const ulong Multiplier1 = 0xCA5A826395121157;

    private const ulong KeyIncrement0 = 0x9E3779B97F4A7C15;

    private const ulong KeyIncrement1 = 0xBB67AE8584CAA73B;

    private const int Rounds = 10;

    /// <summary>Runs the ten Philox4x64 rounds over one counter and key, producing four 64-bit words.</summary>
    /// <param name="counter">The counter for this block. Every distinct counter gives an independent block.</param>
    /// <param name="key">The key for this block. The same key and counter always give the same block.</param>
    internal static PhiloxCounter Block(PhiloxCounter counter, PhiloxKey key)
    {
        for (var round = 0; round < Rounds; round++)
        {
            var high0 = Math.BigMul(Multiplier0, counter.Word0, out var low0);
            var high1 = Math.BigMul(Multiplier1, counter.Word2, out var low1);

            counter = new PhiloxCounter(
                high1 ^ counter.Word1 ^ key.Word0,
                low1,
                high0 ^ counter.Word3 ^ key.Word1,
                low0);

            // Wrapped explicitly: the algorithm is defined modulo 2^64, and the key genuinely does carry
            // past ulong.MaxValue within a handful of rounds for most seeds — this is not an edge case.
            key = new PhiloxKey(unchecked(key.Word0 + KeyIncrement0), unchecked(key.Word1 + KeyIncrement1));
        }

        return counter;
    }
}
