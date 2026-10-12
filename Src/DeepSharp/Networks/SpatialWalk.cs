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

    /// <summary>
    /// The patches put back where the window took them from, adding what lands on the same place: the unfolding turned
    /// around, for a layer that gives back the places a convolution took away.
    /// </summary>
    /// <param name="backend">The engine that folds.</param>
    /// <param name="patches">The patches, laid out as <see cref="Unfold"/> lays them out.</param>
    /// <param name="extents">How long each axis of the example they are put back onto is.</param>
    /// <returns>A batch of that shape, channels last; what fell on the border is dropped.</returns>
    Tensor Fold(ITensorBackend backend, Tensor patches, ReadOnlySpan<int> extents);

    /// <summary>How many places the window stands at along each axis of an example so many places long; none or less where it does not fit.</summary>
    /// <param name="extents">How long each axis of an example is.</param>
    int[] PlacesOver(ReadOnlySpan<int> extents);

    /// <summary>Whether any cell of any patch falls outside an example of these lengths: the window pads it, or it overhangs the end.</summary>
    /// <param name="extents">How long each axis of an example is.</param>
    bool PadsOver(ReadOnlySpan<int> extents);

    /// <summary>
    /// How many cells of each place's patch are inside the example WITH its border, along each axis the window walks: the
    /// whole window, wherever it does not overhang the end.
    /// </summary>
    /// <param name="extents">How long each axis of an example is.</param>
    /// <returns>One count for every place of every axis, the axes in their own order.</returns>
    IReadOnlyList<int[]> CellsWithin(ReadOnlySpan<int> extents);

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

        /// <summary>
        /// Each example with the places of nothing its window's border states added around it, and nothing else done to it:
        /// what a layer that only pads gives, and what a transposed convolution adds at the end of an axis.
        /// </summary>
        /// <param name="backend">The engine that unfolds.</param>
        /// <param name="batch">The batch: an example by its axes by its channel.</param>
        /// <param name="value">What the places added hold.</param>
        /// <returns>The batch, every axis as long as the border makes it.</returns>
        /// <remarks>
        /// A window one place wide, with the border stated, unfolds an example into a row for every place of the padded
        /// example: the padded example itself, read back as the batch it is. Where the value is not nought, the places the
        /// example does not reach are given it through the same mask that tells a value of the example from its border, so
        /// nothing the example holds is touched and nothing is rounded.
        /// </remarks>
        internal Tensor Padded(ITensorBackend backend, Tensor batch, float value)
        {
            var places = walk.PlacesOver(walk.ExtentsOf(batch.Shape));
            var shape = new Shape([batch.Shape[0], .. places, batch.Shape[batch.Shape.Rank - 1]]);
            var padded = backend.Reshape(walk.Unfold(backend, batch), shape);

            if (value is 0f)
            {
                return padded;
            }

            var real = backend.Reshape(walk.Unfold(backend, backend.Fill(batch.Shape, 1f)), shape);

            return backend.Add(
                padded,
                backend.Scale(backend.Subtract(backend.Fill(shape, 1f), real), backend.Fill(new Shape(), value)));
        }

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

    public Tensor Fold(ITensorBackend backend, Tensor patches, ReadOnlySpan<int> extents)
    {
        var channels = patches.Shape[1] / Volume;
        var count = patches.Shape[0] / _window.PlacesOver(extents[0]);

        return backend.Reshape(
            backend.Fold(patches, new Shape(count, 1, extents[0], channels), _window.Over(extents[0])),
            new Shape(count, extents[0], channels));
    }

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.PlacesOver(extents[0])];

    public bool PadsOver(ReadOnlySpan<int> extents) => _window.Over(extents[0]).PadsOver(1, extents[0]);

    public IReadOnlyList<int[]> CellsWithin(ReadOnlySpan<int> extents) => [_window.Over(extents[0]).CellsWithin(1, extents[0]).Across];

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

    public Tensor Fold(ITensorBackend backend, Tensor patches, ReadOnlySpan<int> extents)
    {
        var channels = patches.Shape[1] / Volume;
        var count = patches.Shape[0] / (_window.RowsOver(extents[0]) * _window.ColumnsOver(extents[1]));

        return backend.Fold(patches, new Shape(count, extents[0], extents[1], channels), _window);
    }

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.RowsOver(extents[0]), _window.ColumnsOver(extents[1])];

    public bool PadsOver(ReadOnlySpan<int> extents) => _window.PadsOver(extents[0], extents[1]);

    public IReadOnlyList<int[]> CellsWithin(ReadOnlySpan<int> extents)
    {
        var within = _window.CellsWithin(extents[0], extents[1]);

        return [within.Down, within.Across];
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

    // The two unfoldings turned around, the columns first and the planes and rows after them: what the first laid side by
    // side the second puts back, and the volume comes out of the same two windows the walk went over.
    public Tensor Fold(ITensorBackend backend, Tensor patches, ReadOnlySpan<int> extents)
    {
        var walk = _window.Over(extents[2]);
        var channels = patches.Shape[1] / Volume;
        var lines = _window.Depth * _window.Height;
        var holding = patches.Shape[0] / _window.ColumnsOver(extents[2]);

        var columns = backend.Fold(patches, new Shape(holding, lines, extents[2], channels), walk.Lines);
        var planes = backend.Fold(
            backend.Reshape(columns, new Shape(holding, lines * extents[2] * channels)),
            new Shape(holding / (_window.DepthsOver(extents[0]) * _window.RowsOver(extents[1])), extents[0], extents[1], extents[2] * channels),
            walk.Planes);

        return backend.Reshape(planes, new Shape([planes.Shape[0], extents[0], extents[1], extents[2], channels]));
    }

    public int[] PlacesOver(ReadOnlySpan<int> extents) => [_window.DepthsOver(extents[0]), _window.RowsOver(extents[1]), _window.ColumnsOver(extents[2])];

    public bool PadsOver(ReadOnlySpan<int> extents)
    {
        var walk = _window.Over(extents[2]);

        return walk.Planes.PadsOver(extents[0], extents[1]) || walk.Lines.PadsOver(_window.Depth * _window.Height, extents[2]);
    }

    // The planes and rows come from the first unfolding and the columns from the second; the second walk's own rows are one
    // place of whole lines, which the first has already counted, so only its columns are read here.
    public IReadOnlyList<int[]> CellsWithin(ReadOnlySpan<int> extents)
    {
        var walk = _window.Over(extents[2]);
        var planes = walk.Planes.CellsWithin(extents[0], extents[1]);

        return [planes.Down, planes.Across, walk.Lines.CellsWithin(_window.Depth * _window.Height, extents[2]).Across];
    }

    public string Describe() => _window.ToString();
}
