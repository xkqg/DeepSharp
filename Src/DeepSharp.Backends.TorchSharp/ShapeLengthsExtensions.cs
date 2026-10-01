// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>A shape as libtorch is handed one.</summary>
internal static class ShapeLengthsExtensions
{
    /// <summary>The length of each axis, outermost first, as libtorch takes them: no axes at all for a single value.</summary>
    /// <param name="shape">The shape.</param>
    /// <returns>The lengths.</returns>
    public static long[] Lengths(this Shape shape)
    {
        var axes = shape.Axes;
        var lengths = new long[axes.Length];

        for (var axis = 0; axis < axes.Length; axis++)
        {
            lengths[axis] = axes[axis];
        }

        return lengths;
    }
}
