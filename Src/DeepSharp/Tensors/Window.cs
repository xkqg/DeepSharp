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
/// <para>
/// Its border is stated, the same number of rows and columns on every side, or worked out from each image as TensorFlow's
/// and Keras's <c>padding='same'</c> works it out: <c>new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same }</c>.
/// <see cref="BordersOver"/> is that one rule, and every engine's unfolding and folding reads its border from it.
/// </para>
/// <para>
/// What a window covers: one stride, down and across alike; a patch of neighbouring places, never one spread out with
/// gaps between them; every channel of a place at once, never channels split into groups. Nothing here pools. A
/// convolution another framework walks otherwise — a stride of its own for each axis, a dilated window, grouped channels —
/// has nothing here to be read into, and a network's file that says so is refused at the setting.
/// </para>
/// </remarks>
public readonly record struct Window(int Height, int Width)
{
    /// <summary>How many places the window moves between one patch and the next, down and across; one unless said.</summary>
    public int Stride { get; init; } = 1;

    /// <summary>How many rows and columns of nothing surround the image on every side; none unless said.</summary>
    /// <remarks>Read only when <see cref="PaddingMode"/> is <see cref="PaddingMode.Stated"/>: a window padded as 'same' works its border out, and is given none.</remarks>
    public int Padding { get; init; }

    /// <summary>How the border is had: the one <see cref="Padding"/> states for every side, unless said.</summary>
    public PaddingMode PaddingMode { get; init; }

    /// <summary>
    /// The rows and columns of nothing to pad with on each of the four sides, stated side by side; nothing, unless said. The
    /// way a one-dimensional or a three-dimensional window walks an engine that unfolds images: along one axis padded and
    /// along another not, which a border stated for every side cannot say.
    /// </summary>
    /// <remarks>Instead of <see cref="Padding"/> and of every mode but <see cref="PaddingMode.Stated"/>: a window given both is refused.</remarks>
    internal Borders? Border { get; init; }

    /// <summary>How many rows of patches the window makes of an image so many rows tall.</summary>
    /// <param name="height">How many rows the image has.</param>
    /// <returns>The rows of patches, nought or less when the window does not fit.</returns>
    public int RowsOver(int height) => PlacesAlong(height, Height, Down(height));

    /// <summary>How many columns of patches the window makes of an image so many columns wide.</summary>
    /// <param name="width">How many columns the image has.</param>
    /// <returns>The columns of patches, nought or less when the window does not fit.</returns>
    public int ColumnsOver(int width) => PlacesAlong(width, Width, Across(width));

    /// <summary>The rows and columns of nothing the window pads an image of so many rows and columns with, on each side.</summary>
    /// <param name="rows">How many rows the image has.</param>
    /// <param name="columns">How many columns it has.</param>
    /// <returns>
    /// The border: <see cref="Padding"/> on every side, when stated; padded as 'same', what TensorFlow pads with — along each
    /// axis, as many places as the stride fits into the image, the rows or columns those need beyond it split in two, the
    /// odd one after. None, for a window whose stride lets it stand nowhere.
    /// </returns>
    /// <remarks>
    /// The one rule for where a window stands: an engine that unfolds an image into patches, or folds them back, starts its
    /// first patch this far above and to the left of the image, and <see cref="RowsOver"/> and <see cref="ColumnsOver"/>
    /// count the places over the same border. Three rows at a stride of two over 28 are padded with none before and one
    /// after: fourteen places, which a border of one on every side gives too, at other places.
    /// </remarks>
    public Borders BordersOver(int rows, int columns)
    {
        var down = Down(rows);
        var across = Across(columns);

        return new Borders(down.Before, down.After, across.Before, across.After);
    }

    /// <inheritdoc />
    public override string ToString() =>
        Border is { } sides ? $"window {Height}x{Width} (stride {Stride}, padding {sides.Top} above, {sides.Bottom} below, {sides.Left} before, {sides.Right} after)"
        : PaddingMode == PaddingMode.Same ? $"window {Height}x{Width} (stride {Stride}, padding 'same')"
        : PaddingMode == PaddingMode.Causal ? $"window {Height}x{Width} (stride {Stride}, padding 'causal')"
        : $"window {Height}x{Width} (stride {Stride}, padding {Padding})";

    // Where the window can stand along one side: from the first edge of the border to the last place it still fits whole.
    private int PlacesAlong(int length, int size, Sides sides) =>
        Stride < 1 || length + sides.Total < size ? 0 : ((length + sides.Total - size) / Stride) + 1;

    // The border above and below an image so tall, and before and after one so wide.
    private Sides Down(int length) => Border is { } sides ? new Sides(sides.Top, sides.Bottom) : Along(length, Height);

    private Sides Across(int length) => Border is { } sides ? new Sides(sides.Left, sides.Right) : Along(length, Width);

    // The border along one side of an image so long, for a window so large along it: the stated padding on both ends; padded
    // as 'same', TensorFlow's — as many places as the stride fits, and the rows or columns they need beyond the image, never
    // fewer than none, the odd one after; padded as 'causal', the window's size less one before and nothing after, so the
    // last place a window stands at is the last of the image.
    private Sides Along(int length, int size)
    {
        if (PaddingMode == PaddingMode.Causal)
        {
            return new Sides(size - 1, 0);
        }

        if (PaddingMode != PaddingMode.Same)
        {
            return new Sides(Padding, Padding);
        }

        if (Stride < 1)
        {
            return default;
        }

        var places = (length + Stride - 1) / Stride;
        var needed = Math.Max(0, ((places - 1) * Stride) + size - length);

        return new Sides(needed / 2, needed - (needed / 2));
    }

    // The rows or columns of nothing before an image and after it, along one of its sides.
    private readonly record struct Sides(int Before, int After)
    {
        public int Total => Before + After;
    }
}

/// <summary>How a <see cref="Window"/> has its border.</summary>
public enum PaddingMode
{
    /// <summary>As its <see cref="Window.Padding"/> states: so many rows and columns of nothing on every side, none unless said.</summary>
    Stated,

    /// <summary>
    /// As TensorFlow's and Keras's <c>padding='same'</c>: along each axis the window stands at as many places as the stride
    /// fits into the image — every place, at a stride of one — and the rows or columns of nothing it needs for them are
    /// split in two, the odd one after the image. A window of two, or three at a stride of two over an even side, pads one
    /// after and none before, which no border stated for every side gives.
    /// </summary>
    Same,

    /// <summary>
    /// As Keras's <c>padding='causal'</c>: along each side the window pads its size less one before the image and nothing
    /// after it, so a place never sees what comes after its own. Along a side of one place there is no border. Made for a
    /// window one place tall over a series, where it is what a causal convolution is.
    /// </summary>
    Causal,
}

/// <summary>The rows and columns of nothing a window pads an image with, on each of its four sides.</summary>
/// <param name="Top">Rows above the image.</param>
/// <param name="Bottom">Rows below it.</param>
/// <param name="Left">Columns before it.</param>
/// <param name="Right">Columns after it.</param>
public readonly record struct Borders(int Top, int Bottom, int Left, int Right);
