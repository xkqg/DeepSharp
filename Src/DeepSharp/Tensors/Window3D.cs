// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// The patch a three-dimensional convolution or pooling looks at, and how it walks through a volume: so many planes by so
/// many rows by so many columns, moved so many places at a time, over a volume with so many planes, rows and columns of
/// nothing added around it.
/// </summary>
/// <param name="Depth">How many planes of the volume the window covers.</param>
/// <param name="Height">How many rows of each plane it covers.</param>
/// <param name="Width">How many columns of each row it covers.</param>
/// <remarks>
/// A window that moves one place at a time through a volume with no border, as PyTorch's <c>Conv3d</c> and Keras's
/// <c>Conv3D</c> leave it, unless said: <c>new Window3D(3, 3, 3) { Stride = 2, Padding = 1 }</c>. Its border is worked out on
/// each axis alone by the two-dimensional <see cref="Window"/>'s rule — stated, or as TensorFlow's <c>padding='same'</c> — so a
/// volume is padded where a window stands as it is padded in two dimensions. What a window covers is what
/// <see cref="Window"/> covers: a block of neighbouring places unless it is dilated, every channel of a place at once. A
/// stride and a dilation are said for each axis, or once for every axis.
/// </remarks>
public readonly record struct Window3D(int Depth, int Height, int Width)
{
    /// <summary>How many places the window moves between one patch and the next, along every axis; one unless said.</summary>
    public Steps3D Stride { get; init; } = 1;

    /// <summary>How many places apart the cells of a patch are, along every axis; one — neighbouring places — unless said.</summary>
    public Steps3D Dilation { get; init; } = 1;

    /// <summary>How many planes, rows and columns of nothing surround the volume on every side; none unless said.</summary>
    /// <remarks>Read only when <see cref="PaddingMode"/> is <see cref="PaddingMode.Stated"/>.</remarks>
    public int Padding { get; init; }

    /// <summary>How the border is had: the one <see cref="Padding"/> states on every side, unless said.</summary>
    public PaddingMode PaddingMode { get; init; }

    /// <summary>Whether the window also stands at a last place whose patch overhangs the end of an axis, as PyTorch's <c>ceil_mode</c> has it.</summary>
    public bool RoundsUp { get; init; }

    /// <summary>The places of nothing at each end of each axis, stated apart; nothing, unless said.</summary>
    /// <remarks>Instead of <see cref="Padding"/> and of every mode but <see cref="PaddingMode.Stated"/>: a window given both is refused.</remarks>
    public Margins3D? Border { get; init; }

    /// <summary>How many places the window stands at along a volume so many planes deep.</summary>
    /// <param name="depth">How many planes the volume has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int DepthsOver(int depth) => Plane.RowsOver(depth);

    /// <summary>How many places the window stands at along a volume so many rows tall.</summary>
    /// <param name="height">How many rows each plane has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int RowsOver(int height) => Plane.ColumnsOver(height);

    /// <summary>How many places the window stands at along a volume so many columns wide.</summary>
    /// <param name="width">How many columns each row has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int ColumnsOver(int width) => Line.ColumnsOver(width);

    /// <inheritdoc />
    public override string ToString()
    {
        var dilated = Dilation == (Steps3D)1 ? string.Empty : $", dilated by {Dilation}";
        var rounding = RoundsUp ? ", rounding up" : string.Empty;
        var border = Border is { } sides
            ? $"{sides.Deep.Before} before and {sides.Deep.After} after the planes, {sides.Down.Before} above, {sides.Down.After} below, {sides.Across.Before} before, {sides.Across.After} after"
            : PaddingMode.Said(Padding);

        return $"window {Depth}x{Height}x{Width} (stride {Stride}{dilated}{rounding}, padding {border})";
    }

    /// <summary>Refuses a window that cannot stand anywhere, in the words the two-dimensional window refuses with.</summary>
    /// <exception cref="ArgumentException">
    /// A side or the stride is below one, the dilation below one, the border below nothing, or a border is given besides one
    /// worked out from the volume.
    /// </exception>
    internal void RequireStanding()
    {
        if (Depth < 1 || Height < 1 || Width < 1 || Stride.Deep < 1 || Stride.Down < 1 || Stride.Across < 1 || Padding < 0
            || (Border is { } sides && (sides.Deep.Before < 0 || sides.Deep.After < 0 || sides.Down.Before < 0 || sides.Down.After < 0 || sides.Across.Before < 0 || sides.Across.After < 0)))
        {
            throw new ArgumentException($"A {this} cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing.", "window");
        }

        if (Dilation.Deep < 1 || Dilation.Down < 1 || Dilation.Across < 1)
        {
            throw new ArgumentException($"A {this} cannot stand anywhere: its dilation is at least one.", "window");
        }

        if (Border is not null && (Padding != 0 || PaddingMode != PaddingMode.Stated))
        {
            throw new ArgumentException($"A {this} states its border at each end of each axis, and cannot be given a padding or a way to work one out besides.", "window");
        }

        if (PaddingMode != PaddingMode.Stated && Padding != 0)
        {
            throw new ArgumentException($"A {this} works its border out from each volume it stands on, and cannot be given one of {Padding} besides.", "window");
        }
    }

    /// <summary>
    /// The window as the engine's two-dimensional unfolding walks it over a volume, in two goes: the planes and rows first,
    /// each patch holding whole lines of columns, then the columns, each patch made of the lines the first held.
    /// </summary>
    /// <param name="width">How many columns each row of the volume has.</param>
    /// <returns>The two windows; the second is padded along the columns alone.</returns>
    internal WindowWalk3D Over(int width)
    {
        var along = Line.BordersOver(1, width);

        return new WindowWalk3D(
            Plane,
            new Window(Depth * Height, Width)
            {
                // The first unfolding laid the planes and rows it chose side by side, so the second walks whole lines: one
                // place down however it strides, and its cells a column's dilation apart and no more.
                Stride = new Steps(1, Stride.Across),
                Dilation = new Steps(1, Dilation.Across),
                RoundsUp = RoundsUp,
                Border = new Margins(new Sides(0, 0), new Sides(along.Left, along.Right)),
            });
    }

    // The window over the planes and rows: a two-dimensional window whose rows are the volume's planes and whose columns are
    // each plane's rows, so it carries those two axes' own strides, dilations and border.
    private Window Plane => new(Depth, Height)
    {
        Stride = new Steps(Stride.Deep, Stride.Down),
        Dilation = new Steps(Dilation.Deep, Dilation.Down),
        Padding = Padding,
        PaddingMode = PaddingMode,
        RoundsUp = RoundsUp,
        Border = Border is { } sides ? new Margins(sides.Deep, sides.Down) : null,
    };

    // The window along the columns alone, one row tall: what works out the border and the places of that one axis.
    private Window Line => new(1, Width)
    {
        Stride = new Steps(1, Stride.Across),
        Dilation = new Steps(1, Dilation.Across),
        Padding = Padding,
        PaddingMode = PaddingMode,
        RoundsUp = RoundsUp,
        Border = Border is { } sides ? new Margins(new Sides(0, 0), sides.Across) : null,
    };
}

/// <summary>How the engine's two-dimensional unfolding walks a volume: the planes and rows first, the columns after them.</summary>
/// <param name="Planes">The window over the planes and rows, each patch holding whole lines of columns.</param>
/// <param name="Lines">The window over the columns, as tall as the first made its patches' lines are.</param>
internal readonly record struct WindowWalk3D(Window Planes, Window Lines);
