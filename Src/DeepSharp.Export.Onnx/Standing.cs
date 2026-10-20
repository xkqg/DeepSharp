// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A window as ONNX states it for the examples that reach it: its length along each axis, its stride and dilation, and the places
/// of nothing at each end of each axis, worked out from the examples' lengths where the window works its border out from them.
/// </summary>
/// <param name="Sizes">The window's length along each axis.</param>
/// <param name="Strides">How far it moves between one place and the next, along each axis.</param>
/// <param name="Dilations">How far apart its cells are, along each axis.</param>
/// <param name="Pads">The places before each axis, then the places after each: ONNX's <c>pads</c>.</param>
internal sealed record Standing(int[] Sizes, int[] Strides, int[] Dilations, int[] Pads)
{
    /// <summary>Whether the window also stands at a last place whose patch overhangs the end: ONNX's <c>ceil_mode</c>.</summary>
    public bool RoundsUp { get; init; }

    /// <summary>Whether any axis is dilated.</summary>
    public bool Dilated => Dilations.Any(each => each != 1);

    /// <summary>The window along a series.</summary>
    /// <param name="window">The window.</param>
    /// <param name="extents">How long the series is.</param>
    public static Standing Of(Window1D window, ReadOnlySpan<int> extents)
    {
        var along = Along(new Border(window.PaddingMode, window.Padding, window.Border, window.RoundsUp), new Axis(window.Length, window.Stride, window.Dilation, extents[0]));

        return new([window.Length], [window.Stride], [window.Dilation], [along.Before, along.After]) { RoundsUp = window.RoundsUp };
    }

    /// <summary>The window over an image.</summary>
    /// <param name="window">The window.</param>
    /// <param name="extents">How many rows and columns the image has.</param>
    public static Standing Of(Window window, ReadOnlySpan<int> extents)
    {
        var borders = window.BordersOver(extents[0], extents[1]);

        return new(
            [window.Height, window.Width],
            [window.Stride.Down, window.Stride.Across],
            [window.Dilation.Down, window.Dilation.Across],
            [borders.Top, borders.Left, borders.Bottom, borders.Right]) { RoundsUp = window.RoundsUp };
    }

    /// <summary>The window through a volume.</summary>
    /// <param name="window">The window.</param>
    /// <param name="extents">How many planes, rows and columns the volume has.</param>
    public static Standing Of(Window3D window, ReadOnlySpan<int> extents)
    {
        var sides = window.Border;
        var deep = Along(new Border(window.PaddingMode, window.Padding, sides?.Deep, window.RoundsUp), new Axis(window.Depth, window.Stride.Deep, window.Dilation.Deep, extents[0]));
        var down = Along(new Border(window.PaddingMode, window.Padding, sides?.Down, window.RoundsUp), new Axis(window.Height, window.Stride.Down, window.Dilation.Down, extents[1]));
        var across = Along(new Border(window.PaddingMode, window.Padding, sides?.Across, window.RoundsUp), new Axis(window.Width, window.Stride.Across, window.Dilation.Across, extents[2]));

        return new(
            [window.Depth, window.Height, window.Width],
            [window.Stride.Deep, window.Stride.Down, window.Stride.Across],
            [window.Dilation.Deep, window.Dilation.Down, window.Dilation.Across],
            [deep.Before, down.Before, across.Before, deep.After, down.After, across.After]) { RoundsUp = window.RoundsUp };
    }

    /// <summary>The window of a transposed convolution along a series, whose border is the places taken off each end of what it gives back.</summary>
    /// <param name="window">The window.</param>
    public static Standing GivenBack(Window1D window)
    {
        var along = Stated(window.Padding, window.Border);

        return new([window.Length], [window.Stride], [window.Dilation], [along.Before, along.After]);
    }

    /// <summary>The window of a transposed convolution over an image, whose border is the places taken off each end of what it gives back.</summary>
    /// <param name="window">The window.</param>
    public static Standing GivenBack(Window window)
    {
        var down = Stated(window.Padding, window.Border?.Down);
        var across = Stated(window.Padding, window.Border?.Across);

        return new(
            [window.Height, window.Width],
            [window.Stride.Down, window.Stride.Across],
            [window.Dilation.Down, window.Dilation.Across],
            [down.Before, across.Before, down.After, across.After]);
    }

    /// <summary>The window of a transposed convolution through a volume, whose border is the places taken off each end of what it gives back.</summary>
    /// <param name="window">The window.</param>
    public static Standing GivenBack(Window3D window)
    {
        var deep = Stated(window.Padding, window.Border?.Deep);
        var down = Stated(window.Padding, window.Border?.Down);
        var across = Stated(window.Padding, window.Border?.Across);

        return new(
            [window.Depth, window.Height, window.Width],
            [window.Stride.Deep, window.Stride.Down, window.Stride.Across],
            [window.Dilation.Deep, window.Dilation.Down, window.Dilation.Across],
            [deep.Before, down.Before, across.Before, deep.After, down.After, across.After]);
    }

    // The places a window states at the ends of one axis: a transposed convolution cannot read a border worked out from the examples.
    private static Sides Stated(int padding, Sides? border) => border ?? new Sides(padding, padding);

    /// <summary>States the window on a node: the lengths, the stride, the border and, where it is not one, the dilation.</summary>
    /// <param name="node">The node.</param>
    public void State(NodeProto node)
    {
        node.States("kernel_shape", Sizes.Select(each => (long)each).ToArray());
        node.States("strides", Strides.Select(each => (long)each).ToArray());
        node.States("pads", Pads.Select(each => (long)each).ToArray());

        if (Dilated)
        {
            node.States("dilations", Dilations.Select(each => (long)each).ToArray());
        }
    }

    // The places before and after one axis, worked out by the window one row tall that walks it.
    private static Sides Along(Border how, Axis axis)
    {
        var line = new Window(1, axis.Size)
        {
            Stride = new Steps(1, axis.Stride),
            Dilation = new Steps(1, axis.Dilation),
            Padding = how.Stated is null ? how.Padding : 0,
            PaddingMode = how.Stated is null ? how.Mode : PaddingMode.Stated,
            RoundsUp = how.RoundsUp,
            Border = how.Stated is { } sides ? new Margins(new Sides(0, 0), sides) : null,
        };
        var worked = line.BordersOver(1, axis.Extent);

        return new Sides(worked.Left, worked.Right);
    }

    // How a window gets its border, the same along every axis of it.
    private readonly record struct Border(PaddingMode Mode, int Padding, Sides? Stated, bool RoundsUp);

    // What a window is along one axis, and how long the examples are along it.
    private readonly record struct Axis(int Size, int Stride, int Dilation, int Extent);
}
