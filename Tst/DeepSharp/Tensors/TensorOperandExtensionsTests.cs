// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// What the seam refuses is said once, public, and every engine says it first. That each engine — the light one, one of the
/// tests' own and any other — refuses what the light one refuses, in its words, runs from the contract; here, that what the
/// seam takes, every refusal lets through.
/// </summary>
public class TensorOperandExtensionsTests
{
    private static readonly Window TwoByTwo = new(2, 2);

    [Fact]
    public void WhatTheSeamTakes_EveryRefusalLetsThrough()
    {
        var images = Zeros(1, 3, 3, 1);

        Zeros(2, 3).RequireSameShape(Zeros(2, 3), nameof(ITensorBackend.Add));
        Zeros(2, 3).RequireMatrixProduct(Zeros(3, 4));
        Zeros(2, 3).RequireMatrix(nameof(ITensorBackend.Transpose));
        Zeros(2, 3).RequireRow(Zeros(3));
        Zeros(1).RequireValues();
        Zeros(2, 3).RequireFactor(Zeros());
        Zeros(2, 3).RequireSameCount(new Shape(3, 2));
        images.RequireImagesFor(new Window(3, 3) { Padding = 1 });
        Zeros(4, 4).RequirePatchesOf(images.Shape, TwoByTwo);
    }

    private static Tensor Zeros(params int[] axes) => Tensor.Zeros(new Shape(axes));
}
