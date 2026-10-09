// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A window walking a series, an image or a volume — what a convolution and a pooling along one, two or three axes have in
/// common: how many axes, how many values a patch holds, the patches themselves, and where the window stands.
/// </summary>
/// <remarks>
/// Every walk stands on the engine's two-dimensional unfolding, so every engine unfolds along one axis and three as it
/// does along two, and a window along any of them stands at the same places on every engine.
/// </remarks>
internal interface ISpatialWalk
{
    /// <summary>How many axes the window walks: one, two or three.</summary>
    int Rank { get; }

    /// <summary>How many values a patch holds of each channel.</summary>
    int Volume { get; }

    /// <summary>The patches the window takes of a batch laid out with its channels last: a row for every example and every place.</summary>
    /// <param name="backend">The engine that unfolds.</param>
    /// <param name="batch">The batch: an example by its axes by its channel.</param>
    /// <returns>A matrix with a row for every example and every place — an example first, then its places in order — and in each row the patch's values.</returns>
    Tensor Unfold(ITensorBackend backend, Tensor batch);

    /// <summary>How many places the window stands at along each axis of an example so many places long; none or less where it does not fit.</summary>
    /// <param name="extents">How long each axis of an example is.</param>
    int[] PlacesOver(ReadOnlySpan<int> extents);

    /// <summary>Whether the window pads an example of these lengths at all.</summary>
    /// <param name="extents">How long each axis of an example is.</param>
    bool PadsOver(ReadOnlySpan<int> extents);

    /// <summary>The window as a refusal says it.</summary>
    string Describe();
}

/// <summary>What a batch of series, images or volumes is, and the refusals every layer that walks one makes alike.</summary>
internal static class SpatialWalkExtensions
{
    extension(Shape batch)
    {
        /// <summary>What the batch is a batch of, singular, as a refusal says it: <c>series</c>, <c>image</c> or <c>volume</c>; by its axes.</summary>
        internal string Thing() => batch.Rank switch { 3 => "series", 4 => "image", _ => "volume" };

        /// <summary>
        /// Refuses a batch that is not a batch of series, images or volumes with the given number of axes to walk, or that holds
        /// another number of channels than the layer reads.
        /// </summary>
        /// <param name="rank">How many axes are walked: one, two or three.</param>
        /// <param name="channels">How many channels it must hold, when the layer reads a given number.</param>
        /// <param name="layer">The layer's words for what it is: <c>A convolution of 3 channels</c>, say.</param>
        /// <exception cref="ArgumentException">The batch has other axes, or another number of channels.</exception>
        internal void RequireSpatial(int rank, int? channels, string layer)
        {
            if (batch.Rank != rank + 2 || (channels is { } wanted && batch[batch.Rank - 1] != wanted))
            {
                var things = rank switch { 1 => "series", 2 => "images", _ => "volumes" };
                var layout = rank switch
                {
                    1 => "series by step by channel",
                    2 => "image by row by column by channel",
                    _ => "volume by plane by row by column by channel",
                };

                throw new ArgumentException($"{layer} takes a batch of {things}, {layout}, and was handed a {batch} one.", "input");
            }
        }
    }

    extension(ISpatialWalk walk)
    {
        /// <summary>The lengths of the axes an example has, between its batch's axis and its channel's.</summary>
        internal int[] ExtentsOf(Shape batch) => [.. batch.Axes[1..^1]];

        /// <summary>Refuses a batch that is not a batch of series, images or volumes the window walks, or whose examples the window does not fit.</summary>
        /// <param name="batch">The shape of the batch.</param>
        /// <param name="channels">How many channels it must hold, when the layer reads a given number.</param>
        /// <param name="layer">The layer's words for what it is: <c>A convolution of 3 channels</c>, say.</param>
        /// <exception cref="ArgumentException">The batch has other axes, or another number of channels, or the window does not fit an example.</exception>
        internal void RequireBatch(Shape batch, int? channels, string layer)
        {
            batch.RequireSpatial(walk.Rank, channels, layer);

            if (walk.PlacesOver(walk.ExtentsOf(batch)).Any(places => places < 1))
            {
                throw new ArgumentException($"A {walk.Describe()} is larger than a {batch} {batch.Thing()} with its border.", "input");
            }
        }
    }
}

/// <summary>A window along a series: laid out one row tall for the engine's two-dimensional unfolding.</summary>
internal sealed class LineWalk : ISpatialWalk
{
    private readonly Window1D _window;

    public LineWalk(Window1D window)
    {
        window.RequireStanding();
        _window = window;
    }

    public int Rank => 1;

    public int Volume => _window.Length;

    public Tensor Unfold(ITensorBackend backend, Tensor batch) =>
        backend.Unfold(backend.Reshape(batch, new Shape(batch.Shape[0], 1, batch.Shape[1], batch.Shape[2])), _window.Over(batch.Shape[1]));

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.PlacesOver(extents[0])];

    public bool PadsOver(ReadOnlySpan<int> extents)
    {
        var border = _window.Over(extents[0]).BordersOver(1, extents[0]);

        return border.Left > 0 || border.Right > 0;
    }

    public string Describe() => _window.ToString();
}

/// <summary>A window over an image: the engine's own two-dimensional unfolding.</summary>
internal sealed class PlaneWalk : ISpatialWalk
{
    private readonly Window _window;

    public PlaneWalk(Window window)
    {
        window.RequireStanding();
        _window = window;
    }

    public int Rank => 2;

    public int Volume => _window.Height * _window.Width;

    public Tensor Unfold(ITensorBackend backend, Tensor batch) => backend.Unfold(batch, _window);

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.RowsOver(extents[0]), _window.ColumnsOver(extents[1])];

    public bool PadsOver(ReadOnlySpan<int> extents)
    {
        var border = _window.BordersOver(extents[0], extents[1]);

        return border.Top > 0 || border.Bottom > 0 || border.Left > 0 || border.Right > 0;
    }

    public string Describe() => _window.ToString();
}

/// <summary>
/// A window through a volume, in two goes of the engine's two-dimensional unfolding. The first walks the planes and rows of
/// the volume laid out as an image of planes by rows whose channels are each row's columns and channels together, so each
/// patch holds whole lines of columns; the second walks the columns of those lines, as an image as tall as the window's
/// planes and rows are, whose patches are the volume's.
/// </summary>
internal sealed class VolumeWalk : ISpatialWalk
{
    private readonly Window3D _window;

    public VolumeWalk(Window3D window)
    {
        window.RequireStanding();
        _window = window;
    }

    public int Rank => 3;

    public int Volume => _window.Depth * _window.Height * _window.Width;

    public Tensor Unfold(ITensorBackend backend, Tensor batch)
    {
        var shape = batch.Shape;
        var width = shape[3];
        var walk = _window.Over(width);

        // Every plane an image whose channels hold a row's columns and channels together: the patches are lines of columns.
        var lines = backend.Unfold(backend.Reshape(batch, new Shape(shape[0], shape[1], shape[2], width * shape[4])), walk.Planes);

        // Each line an image as tall as the window's planes and rows, whose one patch row per column is the volume's patch.
        return backend.Unfold(backend.Reshape(lines, new Shape(lines.Shape[0], _window.Depth * _window.Height, width, shape[4])), walk.Lines);
    }

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.DepthsOver(extents[0]), _window.RowsOver(extents[1]), _window.ColumnsOver(extents[2])];

    public bool PadsOver(ReadOnlySpan<int> extents)
    {
        var planes = _window.Over(extents[2]).Planes.BordersOver(extents[0], extents[1]);
        var lines = _window.Over(extents[2]).Lines.BordersOver(_window.Depth * _window.Height, extents[2]);

        return planes.Top > 0 || planes.Bottom > 0 || planes.Left > 0 || planes.Right > 0 || lines.Left > 0 || lines.Right > 0;
    }

    public string Describe() => _window.ToString();
}
