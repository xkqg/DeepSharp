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
/// <see cref="Window"/> covers: one stride on every axis, a block of neighbouring places, every channel of a place at once.
/// </remarks>
public readonly record struct Window3D(int Depth, int Height, int Width)
{
    /// <summary>How many places the window moves between one patch and the next, along every axis; one unless said.</summary>
    public int Stride { get; init; } = 1;

    /// <summary>How many planes, rows and columns of nothing surround the volume on every side; none unless said.</summary>
    /// <remarks>Read only when <see cref="PaddingMode"/> is <see cref="PaddingMode.Stated"/>.</remarks>
    public int Padding { get; init; }

    /// <summary>How the border is had: the one <see cref="Padding"/> states on every side, unless said.</summary>
    public PaddingMode PaddingMode { get; init; }

    /// <summary>How many places the window stands at along a volume so many planes deep.</summary>
    /// <param name="depth">How many planes the volume has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int DepthsOver(int depth) => Like(Depth, Height).RowsOver(depth);

    /// <summary>How many places the window stands at along a volume so many rows tall.</summary>
    /// <param name="height">How many rows each plane has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int RowsOver(int height) => Like(Depth, Height).ColumnsOver(height);

    /// <summary>How many places the window stands at along a volume so many columns wide.</summary>
    /// <param name="width">How many columns each row has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int ColumnsOver(int width) => Like(1, Width).ColumnsOver(width);

    /// <inheritdoc />
    public override string ToString() => $"window {Depth}x{Height}x{Width} (stride {Stride}, padding {PaddingMode.Said(Padding)})";

    /// <summary>Refuses a window that cannot stand anywhere, in the words the two-dimensional window refuses with.</summary>
    /// <exception cref="ArgumentException">
    /// A side or the stride is below one, the border below nothing, or a border is given besides one worked out from the volume.
    /// </exception>
    internal void RequireStanding()
    {
        if (Depth < 1 || Height < 1 || Width < 1 || Stride < 1 || Padding < 0)
        {
            throw new ArgumentException($"A {this} cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing.", "window");
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
        var along = Like(1, Width).BordersOver(1, width);

        return new WindowWalk3D(
            Like(Depth, Height),
            new Window(Depth * Height, Width) { Stride = Stride, Border = new Borders(0, 0, along.Left, along.Right) });
    }

    // A two-dimensional window of the given sides with this window's stride and way of padding: what works out a border and a
    // number of places along two axes, or along one when a side is a single place.
    private Window Like(int rows, int columns) => new(rows, columns) { Stride = Stride, Padding = Padding, PaddingMode = PaddingMode };
}

/// <summary>How the engine's two-dimensional unfolding walks a volume: the planes and rows first, the columns after them.</summary>
/// <param name="Planes">The window over the planes and rows, each patch holding whole lines of columns.</param>
/// <param name="Lines">The window over the columns, as tall as the first made its patches' lines are.</param>
internal readonly record struct WindowWalk3D(Window Planes, Window Lines);
