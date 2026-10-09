// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

// The words of a description that walk a series, an image or a volume: convolutions, poolings and the dropouts of whole channels.
public sealed partial class Sequential
{
    /// <summary>A convolution along a series: a window slid along each, making so many channels at every place it stands.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential Conv1D(int filters, Window1D window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);

        return Add(new ConvolutionWord(
            "conv1d", "a series of steps and channels", filters, new LineWalk(window), (channels, draws) => new Conv1D(channels, filters, window, draws)));
    }

    /// <summary>A convolution through a volume: a block slid through each, making so many channels at every place it stands.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential Conv3D(int filters, Window3D window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);

        return Add(new ConvolutionWord(
            "conv3d", "a volume of planes, rows, columns and channels", filters, new VolumeWalk(window), (channels, draws) => new Conv3D(channels, filters, window, draws)));
    }

    /// <summary>The largest value a window covers along a series, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential MaxPool1D(Window1D window) =>
        Add(new PoolingWord("maxpool1d", "a series of steps and channels", new LineWalk(window), () => new MaxPool1D(window)));

    /// <summary>The largest value a window covers over an image, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential MaxPool2D(Window window) =>
        Add(new PoolingWord("maxpool2d", "an image of rows, columns and channels", new PlaneWalk(window), () => new MaxPool2D(window)));

    /// <summary>The largest value a window covers through a volume, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential MaxPool3D(Window3D window) =>
        Add(new PoolingWord("maxpool3d", "a volume of planes, rows, columns and channels", new VolumeWalk(window), () => new MaxPool3D(window)));

    /// <summary>The largest value of each run of so many steps of a series, for each channel, as Keras's <c>MaxPooling1D</c> takes it.</summary>
    /// <param name="size">How many steps each run holds, and how far the window moves: the runs do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential MaxPool1D(int size) => MaxPool1D(new Window1D(size) { Stride = size });

    /// <summary>The largest value of each square of so many rows and columns of an image, for each channel, as Keras's <c>MaxPooling2D</c> takes it.</summary>
    /// <param name="size">How many rows and columns each square holds, and how far the window moves: the squares do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential MaxPool2D(int size) => MaxPool2D(new Window(size, size) { Stride = size });

    /// <summary>The largest value of each cube of so many planes, rows and columns of a volume, for each channel, as Keras's <c>MaxPooling3D</c> takes it.</summary>
    /// <param name="size">How many planes, rows and columns each cube holds, and how far the window moves: the cubes do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential MaxPool3D(int size) => MaxPool3D(new Window3D(size, size, size) { Stride = size });

    /// <summary>The average of the values a window covers along a series, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="countsPadding">Whether a border the window pads with counts as values of nought in each average; it is left out, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential AvgPool1D(Window1D window, bool countsPadding = false) =>
        Add(new PoolingWord("avgpool1d", "a series of steps and channels", new LineWalk(window), () => new AvgPool1D(window) { CountsPadding = countsPadding }));

    /// <summary>The average of the values a window covers over an image, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="countsPadding">Whether a border the window pads with counts as values of nought in each average; it is left out, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential AvgPool2D(Window window, bool countsPadding = false) =>
        Add(new PoolingWord("avgpool2d", "an image of rows, columns and channels", new PlaneWalk(window), () => new AvgPool2D(window) { CountsPadding = countsPadding }));

    /// <summary>The average of the values a window covers through a volume, for each channel.</summary>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="countsPadding">Whether a border the window pads with counts as values of nought in each average; it is left out, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential AvgPool3D(Window3D window, bool countsPadding = false) =>
        Add(new PoolingWord("avgpool3d", "a volume of planes, rows, columns and channels", new VolumeWalk(window), () => new AvgPool3D(window) { CountsPadding = countsPadding }));

    /// <summary>The average of each run of so many steps of a series, for each channel, as Keras's <c>AveragePooling1D</c> takes it.</summary>
    /// <param name="size">How many steps each run holds, and how far the window moves: the runs do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential AvgPool1D(int size) => AvgPool1D(new Window1D(size) { Stride = size });

    /// <summary>The average of each square of so many rows and columns of an image, for each channel, as Keras's <c>AveragePooling2D</c> takes it.</summary>
    /// <param name="size">How many rows and columns each square holds, and how far the window moves: the squares do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential AvgPool2D(int size) => AvgPool2D(new Window(size, size) { Stride = size });

    /// <summary>The average of each cube of so many planes, rows and columns of a volume, for each channel, as Keras's <c>AveragePooling3D</c> takes it.</summary>
    /// <param name="size">How many planes, rows and columns each cube holds, and how far the window moves: the cubes do not overlap.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The size is below one.</exception>
    public Sequential AvgPool3D(int size) => AvgPool3D(new Window3D(size, size, size) { Stride = size });

    /// <summary>The largest value of each channel of a series, all its steps pooled: a row of channels for each series.</summary>
    /// <param name="keepsAxes">Whether the axis pooled is kept, as an axis of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalMaxPool1D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalmaxpool1d", "a series of steps and channels", 1, keepsAxes, () => new GlobalMaxPool1D { KeepsAxes = keepsAxes }));

    /// <summary>The largest value of each channel of an image, all its places pooled: a row of channels for each image.</summary>
    /// <param name="keepsAxes">Whether the axes pooled are kept, as axes of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalMaxPool2D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalmaxpool2d", "an image of rows, columns and channels", 2, keepsAxes, () => new GlobalMaxPool2D { KeepsAxes = keepsAxes }));

    /// <summary>The largest value of each channel of a volume, all its places pooled: a row of channels for each volume.</summary>
    /// <param name="keepsAxes">Whether the axes pooled are kept, as axes of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalMaxPool3D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalmaxpool3d", "a volume of planes, rows, columns and channels", 3, keepsAxes, () => new GlobalMaxPool3D { KeepsAxes = keepsAxes }));

    /// <summary>The average of each channel of a series, all its steps pooled: a row of channels for each series.</summary>
    /// <param name="keepsAxes">Whether the axis pooled is kept, as an axis of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalAvgPool1D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalavgpool1d", "a series of steps and channels", 1, keepsAxes, () => new GlobalAvgPool1D { KeepsAxes = keepsAxes }));

    /// <summary>The average of each channel of an image, all its places pooled: a row of channels for each image.</summary>
    /// <param name="keepsAxes">Whether the axes pooled are kept, as axes of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalAvgPool2D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalavgpool2d", "an image of rows, columns and channels", 2, keepsAxes, () => new GlobalAvgPool2D { KeepsAxes = keepsAxes }));

    /// <summary>The average of each channel of a volume, all its places pooled: a row of channels for each volume.</summary>
    /// <param name="keepsAxes">Whether the axes pooled are kept, as axes of one place; dropped, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential GlobalAvgPool3D(bool keepsAxes = false) =>
        Add(new GlobalPoolingWord("globalavgpool3d", "a volume of planes, rows, columns and channels", 3, keepsAxes, () => new GlobalAvgPool3D { KeepsAxes = keepsAxes }));

    /// <summary>Leaves whole channels of each series out while training, each afresh on every pass.</summary>
    /// <param name="rate">The share of the channels left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential SpatialDropout1D(double rate) =>
        Add(new SpatialDropoutWord("spatialdropout1d", "a series of steps and channels", 1, global::DeepSharp.Networks.Dropout.RequireRate(rate), () => new SpatialDropout1D(rate)));

    /// <summary>Leaves whole channels of each image out while training, each afresh on every pass.</summary>
    /// <param name="rate">The share of the channels left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential SpatialDropout2D(double rate) =>
        Add(new SpatialDropoutWord("spatialdropout2d", "an image of rows, columns and channels", 2, global::DeepSharp.Networks.Dropout.RequireRate(rate), () => new SpatialDropout2D(rate)));

    /// <summary>Leaves whole channels of each volume out while training, each afresh on every pass.</summary>
    /// <param name="rate">The share of the channels left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential SpatialDropout3D(double rate) =>
        Add(new SpatialDropoutWord("spatialdropout3d", "a volume of planes, rows, columns and channels", 3, global::DeepSharp.Networks.Dropout.RequireRate(rate), () => new SpatialDropout3D(rate)));

    // The refusal every word of a description that walks makes of an example with other axes than it walks.
    private static string TakesOnly(string word, string takes, Shape each) => $"{word}, takes each example as {takes}, and each reaching it is {each}.";

    /// <summary>A convolution along some axes, making as many channels as it is asked for.</summary>
    private sealed record ConvolutionWord(string Word, string Takes, int Filters, ISpatialWalk Walk, Func<int, Draws, Layer> Build) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != Walk.Rank + 1 ? TakesOnly(Word, Takes, each)
            : Walk.PlacesOver(each.Axes[..^1]).Any(places => places < 1) ? $"{Word}, slides a {Walk.Describe()} over each example, and an example {each} is smaller than it."
            : null;

        public Shape After(Shape each) => new([.. Walk.PlacesOver(each.Axes[..^1]), Filters]);

        public Layer Make(Shape each, Draws draws) => Build(each[each.Rank - 1], draws);
    }

    /// <summary>A pooling along some axes: it keeps the channels it is handed.</summary>
    private sealed record PoolingWord(string Word, string Takes, ISpatialWalk Walk, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != Walk.Rank + 1 ? TakesOnly(Word, Takes, each)
            : Walk.PlacesOver(each.Axes[..^1]).Any(places => places < 1) ? $"{Word}, slides a {Walk.Describe()} over each example, and an example {each} is smaller than it."
            : null;

        public Shape After(Shape each) => new([.. Walk.PlacesOver(each.Axes[..^1]), each[each.Rank - 1]]);

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }

    /// <summary>A pooling of every place of an example into one: a row of channels, or the axes kept as axes of one place.</summary>
    private sealed record GlobalPoolingWord(string Word, string Takes, int Rank, bool KeepsAxes, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == Rank + 1 ? null : TakesOnly(Word, Takes, each);

        public Shape After(Shape each) => KeepsAxes ? new([.. Enumerable.Repeat(1, Rank), each[each.Rank - 1]]) : new Shape(each[each.Rank - 1]);

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }

    /// <summary>A dropout of whole channels: it hands on the shape it is handed.</summary>
    private sealed record SpatialDropoutWord(string Word, string Takes, int Rank, double Rate, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == Rank + 1 ? null : TakesOnly(Word, Takes, each);

        public Shape After(Shape each) => each;

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }
}
