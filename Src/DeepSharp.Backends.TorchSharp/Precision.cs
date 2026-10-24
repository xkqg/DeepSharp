// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using TorchSharp;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>The width an engine on libtorch keeps what it works out in: the width of a single number, or half of it.</summary>
/// <remarks>
/// A width is the engine's, never the model's or the file's: a network trained at one width is read back and served at another, and a
/// checkpoint of a run names the width it was on, since a run goes on from it only as the same run. The numbers a network learns are
/// kept at full width whatever the engine keeps its work in, and are taken in at the engine's width where a product reads them.
/// </remarks>
public enum Precision
{
    /// <summary>Single precision, as every engine kept its numbers before: nothing is rounded to a narrower width.</summary>
    Full = 0,

    /// <summary>
    /// The brain floating-point width: sixteen bits, eight of them the exponent a single number has, so it holds as large and as small a
    /// number as a single number does and keeps three digits of it. On a processor it is as fast as a processor makes half-width
    /// arithmetic; on a graphics card, faster than full width.
    /// </summary>
    BFloat16 = 1,

    /// <summary>
    /// The IEEE half-precision width: sixteen bits, five of them the exponent, so numbers beyond 65504 are lost and below 0.00006 are
    /// coarse — a loss scale exists for it — and it keeps more digits than <see cref="BFloat16"/>. A graphics card's width: a processor's
    /// libtorch has no hardware for it unless the processor has AVX512-FP16 or AVX-NE-CONVERT, and takes about six hundred times as long
    /// for a product as at full width without it.
    /// </summary>
    Half = 2,
}

/// <summary>What libtorch makes of the widths: which of its number types an engine keeps a width in.</summary>
internal static class PrecisionExtensions
{
    // The number type libtorch keeps each width in, in the order of the widths.
    private static readonly torch.ScalarType[] Widths = [torch.ScalarType.Float32, torch.ScalarType.BFloat16, torch.ScalarType.Float16];

    extension(Precision precision)
    {
        /// <summary>The number type libtorch keeps this width in.</summary>
        internal torch.ScalarType Kept => Widths[(int)precision];
    }
}
