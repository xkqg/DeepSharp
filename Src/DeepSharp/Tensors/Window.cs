// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// The patch a convolution looks at, and how it walks over an image: so many rows by so many columns, moved so many
/// places at a time, over an image with so many rows and columns of nothing added around its edge.
/// </summary>
/// <param name="Height">How many rows of the image the window covers.</param>
/// <param name="Width">How many columns of the image it covers.</param>
/// <remarks>
/// A window that moves one place at a time over an image with no border — the stride and the padding as PyTorch and
/// Keras leave them — unless said: <c>new Window(3, 3) { Stride = 2, Padding = 1 }</c>. Where the window's last move
/// would leave part of it off the image, that move is not made, as PyTorch does not make it.
/// </remarks>
public readonly record struct Window(int Height, int Width)
{
    /// <summary>How many places the window moves between one patch and the next, down and across; one unless said.</summary>
    public int Stride { get; init; } = 1;

    /// <summary>How many rows and columns of nothing surround the image on every side; none unless said.</summary>
    public int Padding { get; init; }

    /// <summary>How many rows of patches the window makes of an image so many rows tall.</summary>
    /// <param name="height">How many rows the image has.</param>
    /// <returns>The rows of patches, nought or less when the window does not fit.</returns>
    public int RowsOver(int height) => PlacesAlong(height, Height);

    /// <summary>How many columns of patches the window makes of an image so many columns wide.</summary>
    /// <param name="width">How many columns the image has.</param>
    /// <returns>The columns of patches, nought or less when the window does not fit.</returns>
    public int ColumnsOver(int width) => PlacesAlong(width, Width);

    /// <inheritdoc />
    public override string ToString() => $"window {Height}x{Width} (stride {Stride}, padding {Padding})";

    // Where the window can stand along one side: from the first edge of the border to the last place it still fits whole.
    private int PlacesAlong(int length, int size) =>
        Stride < 1 || length + (2 * Padding) < size ? 0 : ((length + (2 * Padding) - size) / Stride) + 1;
}
