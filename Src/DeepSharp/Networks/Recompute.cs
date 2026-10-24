// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// Which layers of a network have their workings thrown away after their forward pass and worked out again when the way back reaches
/// them, so that what they held is not held while the rest of the pass goes on and a larger batch fits in the same memory.
/// </summary>
/// <remarks>
/// It costs one more forward pass of those layers and nothing else: the draws of a dropout are counted rather than kept, so it drops
/// what it dropped, and a normalisation moves its running statistics once, in the pass that is kept. A run that recomputes is the run
/// that does not, to the bit. A layer holds everything below it, so asking for a block asks for its layers.
/// <para>
/// It pays for a layer whose workings are larger than what it makes — attention, a recurrent layer, an encoder block — and little for one
/// that holds little beyond its output: measured, four encoder blocks fitted 3.5 times the batch, and a stack of dense layers 1.15 times it
/// at a step 1.2 to 1.3 times as long.
/// </para>
/// </remarks>
public sealed class Recompute
{
    private Recompute(IReadOnlyList<string>? paths) => Paths = paths;

    /// <summary>Every layer of the network.</summary>
    public static Recompute Everything { get; } = new(null);

    /// <summary>The paths of the layers asked for; nothing when it is every layer.</summary>
    public IReadOnlyList<string>? Paths { get; }

    /// <summary>The layers that stand at these paths in the network — <c>0</c>, <c>encoder.self_attn</c>.</summary>
    /// <param name="paths">The path of each layer: at least one, none empty.</param>
    /// <returns>The layers asked for.</returns>
    /// <exception cref="ArgumentException">There is no path, or one is empty.</exception>
    public static Recompute Layers(params string[] paths)
    {
        ArgumentNullException.ThrowIfNull(paths);

        return paths.Length == 0 || paths.Any(string.IsNullOrWhiteSpace)
            ? throw new ArgumentException("A layer is asked for by the path it stands at, and at least one is asked for.", nameof(paths))
            : new Recompute([.. paths.Distinct(StringComparer.Ordinal)]);
    }

    /// <summary>Whether the layer that stands at a path is one of those asked for.</summary>
    /// <param name="path">The layer's path: not empty, which is the network itself.</param>
    /// <returns>Whether it is.</returns>
    internal bool Includes(string path) => path.Length > 0 && (Paths is null || Paths.Contains(path));
}
