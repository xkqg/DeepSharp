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
/// would leave part of it off the image, that move is not made, as PyTorch does not make it, unless
/// <see cref="RoundsUp"/> says to make it anyway.
/// <para>
/// Its border is stated, the same number of rows and columns on every side, or worked out from each image as TensorFlow's
/// and Keras's <c>padding='same'</c> works it out: <c>new Window(3, 3) { Stride = 2, PaddingMode = PaddingMode.Same }</c>.
/// <see cref="PlacingOver"/> is the one rule for where the window stands, and every engine's unfolding and folding reads
/// it: the border on each side, the stride and the dilation together, so none of the three can be read without the others.
/// </para>
/// <para>
/// A stride and a dilation are said for each axis, or once for every axis — <c>Stride = 2</c> is
/// <c>Stride = new Steps(2, 2)</c>. What a window covers is a patch of neighbouring places, unless it is dilated, and every
/// channel of a place at once; channels are split into groups by the convolution and never by the window.
/// </para>
/// </remarks>
public readonly record struct Window(int Height, int Width)
{
    /// <summary>How many places the window moves between one patch and the next, down and across; one unless said.</summary>
    public Steps Stride { get; init; } = 1;

    /// <summary>How many places apart the cells of a patch are, down and across; one — neighbouring places — unless said.</summary>
    /// <remarks>A window of three at a dilation of three covers three places and reaches over seven, as PyTorch's <c>dilation</c> does.</remarks>
    public Steps Dilation { get; init; } = 1;

    /// <summary>How many rows and columns of nothing surround the image on every side; none unless said.</summary>
    /// <remarks>Read only when <see cref="PaddingMode"/> is <see cref="PaddingMode.Stated"/>: a window padded as 'same' works its border out, and is given none.</remarks>
    public int Padding { get; init; }

    /// <summary>How the border is had: the one <see cref="Padding"/> states for every side, unless said.</summary>
    public PaddingMode PaddingMode { get; init; }

    /// <summary>
    /// Whether the window also stands at a last place whose patch overhangs the end of an axis, as PyTorch's
    /// <c>ceil_mode</c> has it; it stands only where it fits whole, unless said.
    /// </summary>
    /// <remarks>
    /// A place whose patch would begin in or past the border after the image is not stood at even so, which is PyTorch's own
    /// rule. What such a patch covers beyond the image is border, so a pooling leaves it out of an average and never makes it
    /// the largest.
    /// </remarks>
    public bool RoundsUp { get; init; }

    /// <summary>
    /// The rows and columns of nothing to pad with at each end of each axis, stated side by side; nothing, unless said. The
    /// way a one-dimensional or a three-dimensional window walks an engine that unfolds images: along one axis padded and
    /// along another not, which a border stated for every side cannot say.
    /// </summary>
    /// <remarks>Instead of <see cref="Padding"/> and of every mode but <see cref="PaddingMode.Stated"/>: a window given both is refused.</remarks>
    public Margins? Border { get; init; }

    /// <summary>How many rows of patches the window makes of an image so many rows tall.</summary>
    /// <param name="height">How many rows the image has.</param>
    /// <returns>The rows of patches, nought or less when the window does not fit.</returns>
    public int RowsOver(int height) => PlacesAlong(height, Downward, Down(height));

    /// <summary>How many columns of patches the window makes of an image so many columns wide.</summary>
    /// <param name="width">How many columns the image has.</param>
    /// <returns>The columns of patches, nought or less when the window does not fit.</returns>
    public int ColumnsOver(int width) => PlacesAlong(width, Crosswise, Across(width));

    /// <summary>Where the window stands over an image of so many rows and columns: its border, its stride and its dilation, together.</summary>
    /// <param name="rows">How many rows the image has.</param>
    /// <param name="columns">How many columns it has.</param>
    /// <returns>
    /// The border on each side — <see cref="Padding"/> on every side, when stated; padded as 'same', what TensorFlow pads
    /// with, along each axis as many places as the stride fits into the image, the rows or columns those need beyond it split
    /// in two, the odd one after — beside the stride and the dilation of each axis.
    /// </returns>
    /// <remarks>
    /// The one rule for where a window stands, and the one thing an engine reads of it: an engine that unfolds an image into
    /// patches, or folds them back, starts its first patch this far above and to the left of the image and takes each
    /// following cell a dilation further, and <see cref="RowsOver"/> and <see cref="ColumnsOver"/> count the places over the
    /// same border. The three arrive together so that none of them can be read while another is missed.
    /// </remarks>
    public Placing PlacingOver(int rows, int columns)
    {
        var down = Down(rows);
        var across = Across(columns);

        return new Placing(new Borders(down.Before, down.After, across.Before, across.After), Stride, Dilation);
    }

    /// <summary>The rows and columns of nothing the window pads an image of so many rows and columns with, on each side.</summary>
    /// <param name="rows">How many rows the image has.</param>
    /// <param name="columns">How many columns it has.</param>
    /// <returns>The border, as <see cref="PlacingOver"/> works it out.</returns>
    /// <remarks>What <see cref="PlacingOver"/> says of the border alone, for a caller that needs no more.</remarks>
    public Borders BordersOver(int rows, int columns) => PlacingOver(rows, columns).Border;

    /// <inheritdoc />
    public override string ToString()
    {
        var dilated = Dilation == (Steps)1 ? string.Empty : $", dilated by {Dilation}";
        var rounding = RoundsUp ? ", rounding up" : string.Empty;
        var border = Border is { } sides
            ? $"{sides.Down.Before} above, {sides.Down.After} below, {sides.Across.Before} before, {sides.Across.After} after"
            : PaddingMode.Said(Padding);

        return $"window {Height}x{Width} (stride {Stride}{dilated}{rounding}, padding {border})";
    }

    /// <summary>
    /// Whether any cell of any patch falls outside an image of so many rows and columns: because the window pads it, or
    /// because the window also stands at a last place whose patch overhangs the end.
    /// </summary>
    /// <param name="rows">How many rows the image has.</param>
    /// <param name="columns">How many columns it has.</param>
    /// <returns>Whether a patch covers anything that is not a value of the image.</returns>
    /// <remarks>What a pooling asks before it works out which cells of a patch are values of the example and which are not.</remarks>
    public bool PadsOver(int rows, int columns)
    {
        var border = BordersOver(rows, columns);

        return border != default
            || Beyond(rows, RowsOver(rows), Downward, border.Top)
            || Beyond(columns, ColumnsOver(columns), Crosswise, border.Left);
    }

    /// <summary>
    /// How many cells of each patch of each place are inside the image WITH its border, along each axis: the whole window,
    /// wherever it does not overhang the end.
    /// </summary>
    /// <param name="rows">How many rows the image has.</param>
    /// <param name="columns">How many columns it has.</param>
    /// <returns>A count for every row of patches, and one for every column of patches.</returns>
    /// <remarks>
    /// What an average that counts its border divides by: the border it was told to pad with counts, and the places beyond it
    /// that a window rounding up overhangs do not, which is PyTorch's own rule for <c>count_include_pad</c>.
    /// </remarks>
    internal (int[] Down, int[] Across) CellsWithin(int rows, int columns)
    {
        var border = BordersOver(rows, columns);

        return (Within(rows + border.Top + border.Bottom, RowsOver(rows), Downward),
                Within(columns + border.Left + border.Right, ColumnsOver(columns), Crosswise));
    }

    // Whether the patch of the last place reaches past the end of an axis, which it can only where the window rounds up.
    private static bool Beyond(int length, int places, Covering covering, int before) =>
        places > 0 && ((places - 1) * covering.Stride) - before + covering.Reach - 1 >= length;

    // How many cells of the patch at each place land within an axis so long, the border included.
    private static int[] Within(int padded, int places, Covering covering) =>
        [.. Enumerable.Range(0, Math.Max(0, places))
            .Select(place => Enumerable.Range(0, covering.Size).Count(cell => (place * covering.Stride) + (cell * covering.Dilation) < padded))];

    // Where the window can stand along one side: from the first edge of the border to the last place its patch still fits —
    // or, rounding up, one place further, where the patch overhangs the end; never a place whose patch would begin in or past
    // the border after the image, which is PyTorch's own rule. Measured against PyTorch over 1644 cases.
    private int PlacesAlong(int length, Covering covering, Sides sides)
    {
        if (covering.Stride < 1 || covering.Dilation < 1 || length + sides.Total < covering.Reach)
        {
            return 0;
        }

        var room = length + sides.Total - covering.Reach;
        var places = (RoundsUp ? (room + covering.Stride - 1) / covering.Stride : room / covering.Stride) + 1;

        return RoundsUp && (places - 1) * covering.Stride >= length + sides.Before ? places - 1 : places;
    }

    // How the window covers each axis, and moves along it.
    private Covering Downward => new(Height, Stride.Down, Dilation.Down);

    private Covering Crosswise => new(Width, Stride.Across, Dilation.Across);

    // The border above and below an image so tall, and before and after one so wide.
    private Sides Down(int length) => Border is { } sides ? sides.Down : Along(length, Downward);

    private Sides Across(int length) => Border is { } sides ? sides.Across : Along(length, Crosswise);

    // The border along one side of an image so long, for a window covering it so: the stated padding on both ends; padded
    // as 'same', TensorFlow's — as many places as the stride fits, and the rows or columns the window's reach needs beyond
    // the image, never fewer than none, the odd one after; padded as 'causal', the window's reach less one before and
    // nothing after, so the last place a window stands at is the last of the image.
    private Sides Along(int length, Covering covering)
    {
        if (PaddingMode == PaddingMode.Causal)
        {
            return new Sides(covering.Reach - 1, 0);
        }

        if (PaddingMode != PaddingMode.Same)
        {
            return new Sides(Padding, Padding);
        }

        if (covering.Stride < 1 || covering.Dilation < 1)
        {
            return default;
        }

        var places = (length + covering.Stride - 1) / covering.Stride;
        var needed = Math.Max(0, ((places - 1) * covering.Stride) + covering.Reach - length);

        return new Sides(needed / 2, needed - (needed / 2));
    }

    // How the window covers one axis and moves along it: how many places it covers, how far apart they are, and how far it
    // moves between one patch and the next.
    private readonly record struct Covering(int Size, int Stride, int Dilation)
    {
        // How many places of the axis the window spans, the gaps of a dilated window included.
        public int Reach => (Dilation * (Size - 1)) + 1;
    }
}

/// <summary>How far a window moves, or how far apart its cells are, along each axis of an image.</summary>
/// <param name="Down">Along the rows.</param>
/// <param name="Across">Along the columns.</param>
/// <remarks>The one number means both: <c>Stride = 2</c> is <c>new Steps(2, 2)</c>.</remarks>
public readonly record struct Steps(int Down, int Across)
{
    /// <summary>Whether both axes are the same, which is what a window says as one number.</summary>
    public bool Alike => Down == Across;

    /// <summary>The same number along both axes.</summary>
    /// <param name="alike">The number.</param>
    public static implicit operator Steps(int alike) => new(alike, alike);

    /// <inheritdoc />
    public override string ToString() => Alike ? $"{Down}" : $"{Down} down and {Across} across";
}

/// <summary>How far a window moves, or how far apart its cells are, along each axis of a volume.</summary>
/// <param name="Deep">Through the planes.</param>
/// <param name="Down">Along the rows of each plane.</param>
/// <param name="Across">Along the columns of each row.</param>
/// <remarks>The one number means all three: <c>Stride = 2</c> is <c>new Steps3D(2, 2, 2)</c>.</remarks>
public readonly record struct Steps3D(int Deep, int Down, int Across)
{
    /// <summary>Whether every axis is the same, which is what a window says as one number.</summary>
    public bool Alike => Deep == Down && Down == Across;

    /// <summary>The same number along every axis.</summary>
    /// <param name="alike">The number.</param>
    public static implicit operator Steps3D(int alike) => new(alike, alike, alike);

    /// <inheritdoc />
    public override string ToString() => Alike ? $"{Deep}" : $"{Deep} deep, {Down} down and {Across} across";
}

/// <summary>The places of nothing a window pads one axis with: so many before it, so many after it.</summary>
/// <param name="Before">Places before the axis.</param>
/// <param name="After">Places after it.</param>
public readonly record struct Sides(int Before, int After)
{
    /// <summary>The places of nothing both ends add up to.</summary>
    public int Total => Before + After;
}

/// <summary>The places of nothing a window pads each axis of an image with, at each end of that axis.</summary>
/// <param name="Down">Above the image and below it.</param>
/// <param name="Across">Before it and after it.</param>
public readonly record struct Margins(Sides Down, Sides Across);

/// <summary>The places of nothing a window pads each axis of a volume with, at each end of that axis.</summary>
/// <param name="Deep">Before the planes and after them.</param>
/// <param name="Down">Above each plane and below it.</param>
/// <param name="Across">Before each row and after it.</param>
public readonly record struct Margins3D(Sides Deep, Sides Down, Sides Across);

/// <summary>Where a window stands over an image: everything an engine reads of it, together.</summary>
/// <param name="Border">The rows and columns of nothing on each of the four sides.</param>
/// <param name="Stride">How far the window moves between one patch and the next, along each axis.</param>
/// <param name="Dilation">How far apart the cells of a patch are, along each axis.</param>
/// <remarks>
/// An engine's unfolding starts a patch at <c>place × stride − border</c> along each axis and takes its cells a dilation
/// apart. The three come as one value so that an engine cannot read the border and miss the stride, or read the stride and
/// miss the dilation: a window that says more than an engine reads would walk the wrong places and say nothing about it.
/// </remarks>
public readonly record struct Placing(Borders Border, Steps Stride, Steps Dilation);

/// <summary>How a <see cref="Window"/> has its border.</summary>
public enum PaddingMode
{
    /// <summary>As its <see cref="Window.Padding"/> states: so many rows and columns of nothing on every side, none unless said.</summary>
    Stated,

    /// <summary>
    /// As TensorFlow's and Keras's <c>padding='same'</c>: along each axis the window stands at as many places as the stride
    /// fits into the image — every place, at a stride of one — and the rows or columns of nothing its reach needs for them are
    /// split in two, the odd one after the image. A window of two, or three at a stride of two over an even side, pads one
    /// after and none before, which no border stated for every side gives.
    /// </summary>
    Same,

    /// <summary>
    /// As Keras's <c>padding='causal'</c>: along each side the window pads its reach less one before the image and nothing
    /// after it, so a place never sees what comes after its own. Along a side of one place it pads nothing, which is why it is
    /// the same word for every window. Made for a window one row tall over a series, where it is what a causal convolution is.
    /// </summary>
    Causal,
}

/// <summary>The rows and columns of nothing a window pads an image with, on each of its four sides.</summary>
/// <param name="Top">Rows above the image.</param>
/// <param name="Bottom">Rows below it.</param>
/// <param name="Left">Columns before it.</param>
/// <param name="Right">Columns after it.</param>
/// <remarks>What <see cref="Window.PlacingOver"/> works out for an image of a given size; a border a window is GIVEN is a <see cref="Margins"/>, a pair for each axis.</remarks>
public readonly record struct Borders(int Top, int Bottom, int Left, int Right);
