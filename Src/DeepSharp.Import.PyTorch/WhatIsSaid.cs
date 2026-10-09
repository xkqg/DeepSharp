// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>What the caller says of the network the numbers go into: an example it takes, and what layers are handed.</summary>
/// <param name="Example">The shape of one example the network takes; nothing when none is said.</param>
/// <param name="Flattened">
/// What layers are handed, by their path: a stack's flatten, stated in place of an example; a linear layer or a
/// normalisation of a network written as code, whose order no example shows.
/// </param>
internal readonly record struct WhatIsSaid(Shape? Example, IReadOnlyDictionary<string, Shape> Flattened)
{
    /// <summary>
    /// What each flatten of a stack is handed, per example, by the flatten's path: found by running zeros of the example
    /// through the layers, or as stated; a flatten nothing is said of is not among them.
    /// </summary>
    /// <param name="layers">The network's layers, in the order they run.</param>
    /// <exception cref="ArgumentException">
    /// A statement states neither an image nor a row; an example is handed and a flatten's statement besides; or the example
    /// does not go through the network.
    /// </exception>
    /// <remarks>The zeros run on the light engine: only their shape is asked for, which no engine changes.</remarks>
    public IReadOnlyDictionary<string, Shape> HandedTo(IReadOnlyList<PlacedLayer> layers)
    {
        foreach (var (path, shape) in Flattened)
        {
            if (!shape.IsHandable)
            {
                throw new ArgumentException(
                    $"What the layer at {path} is handed is stated as {shape}, and what is stated is a row, a series — steps and channels —, an image — rows, columns and channels — or a volume — planes, rows, columns and channels.",
                    nameof(SafetensorsFile.Flattened));
            }
        }

        foreach (var (path, shape) in Flattened.Where(stated => stated.Value.Rank == 2))
        {
            if (!layers.EndsInASeries(path))
            {
                throw new ArgumentException(
                    $"What the layer at {path} is handed is stated as {shape}, and two axes are a series — steps and channels — only when the layer that walks axes nearest before it walks one axis.",
                    nameof(SafetensorsFile.Flattened));
            }
        }

        if (Example is not { } example)
        {
            return Flattened;
        }

        if (Flattened.Keys.Any(path => layers.Any(placed => placed.Path == path && placed.Flattens)))
        {
            throw new ArgumentException(
                "An example and what each flatten is handed say one thing twice: hand one of them.", nameof(SafetensorsFile.Flattened));
        }

        var handed = new Dictionary<string, Shape>(StringComparer.Ordinal);
        var pass = Pass.Evaluation(new CpuBackend());
        var reaching = Tensor.Zeros(new Shape([1, .. example.Axes]));

        foreach (var placed in layers)
        {
            if (placed.Flattens)
            {
                handed[placed.Path] = new Shape([.. reaching.Shape.Axes[1..]]);
            }

            try
            {
                reaching = placed.Layer.Forward(reaching, pass);
            }
            catch (ArgumentException refused)
            {
                throw new ArgumentException($"An example of {example} does not go through the network: {refused.Message}", nameof(SafetensorsFile.Example), refused);
            }
        }

        return handed;
    }
}
