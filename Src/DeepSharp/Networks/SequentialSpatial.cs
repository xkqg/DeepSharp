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
    public Sequential Conv1D(int filters, Window1D window) => Conv1D(filters, window, 1, new Beginning());

    /// <summary>The same with the channels in groups: each group of channels in reaching its own group of channels out.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Sequential Conv1D(int filters, Window1D window, int groups) => Conv1D(filters, window, groups, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv1D(int filters, Window1D window, Beginning beginning) => Conv1D(filters, window, 1, beginning);

    /// <summary>The same with the channels in groups and a say in how the layer starts, or that it has no bias.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv1D(int filters, Window1D window, int groups, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        RequireUnseeded(beginning);

        return Add(new ConvolutionWord(
            "conv1d",
            "a series of steps and channels",
            filters,
            new LineWalk(window),
            (channels, draws) => new Conv1D(new Channels { In = channels, Out = filters, Groups = groups }, window, beginning with { Draws = draws })));
    }

    /// <summary>A convolution through a volume: a block slid through each, making so many channels at every place it stands.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential Conv3D(int filters, Window3D window) => Conv3D(filters, window, 1, new Beginning());

    /// <summary>The same with the channels in groups: each group of channels in reaching its own group of channels out.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Sequential Conv3D(int filters, Window3D window, int groups) => Conv3D(filters, window, groups, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv3D(int filters, Window3D window, Beginning beginning) => Conv3D(filters, window, 1, beginning);

    /// <summary>The same with the channels in groups and a say in how the layer starts, or that it has no bias.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv3D(int filters, Window3D window, int groups, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        RequireUnseeded(beginning);

        return Add(new ConvolutionWord(
            "conv3d",
            "a volume of planes, rows, columns and channels",
            filters,
            new VolumeWalk(window),
            (channels, draws) => new Conv3D(new Channels { In = channels, Out = filters, Groups = groups }, window, beginning with { Draws = draws })));
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

    /// <summary>A convolution with a group for every channel of a series: each channel convolved by a kernel of its own.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential DepthwiseConv1D(int multiplier, Window1D window) => DepthwiseConv1D(multiplier, window, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential DepthwiseConv1D(int multiplier, Window1D window, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(multiplier, 1);
        RequireUnseeded(beginning);

        return Add(new DepthwiseWord(
            "conv1d",
            "a series of steps and channels",
            multiplier,
            new LineWalk(window),
            (channels, draws) => new Conv1D(new Channels { In = channels, Out = channels * multiplier, Groups = channels }, window, beginning with { Draws = draws })));
    }

    /// <summary>A convolution with a group for every channel of an image: each channel convolved by a kernel of its own.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential DepthwiseConv2D(int multiplier, Window window) => DepthwiseConv2D(multiplier, window, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential DepthwiseConv2D(int multiplier, Window window, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(multiplier, 1);
        RequireUnseeded(beginning);

        return Add(new DepthwiseWord(
            "conv2d",
            "an image of rows, columns and channels",
            multiplier,
            new PlaneWalk(window),
            (channels, draws) => new Conv2D(new Channels { In = channels, Out = channels * multiplier, Groups = channels }, window, beginning with { Draws = draws })));
    }

    /// <summary>A convolution with a group for every channel of a volume: each channel convolved by a kernel of its own.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential DepthwiseConv3D(int multiplier, Window3D window) => DepthwiseConv3D(multiplier, window, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="multiplier">How many channels it makes of each channel it is handed.</param>
    /// <param name="window">The block, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel of each.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential DepthwiseConv3D(int multiplier, Window3D window, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(multiplier, 1);
        RequireUnseeded(beginning);

        return Add(new DepthwiseWord(
            "conv3d",
            "a volume of planes, rows, columns and channels",
            multiplier,
            new VolumeWalk(window),
            (channels, draws) => new Conv3D(new Channels { In = channels, Out = channels * multiplier, Groups = channels }, window, beginning with { Draws = draws })));
    }

    /// <summary>A convolution along a series that gives back the steps an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks the series it makes; its border is stated, never worked out.</param>
    /// <param name="outputPadding">How many steps of nothing to keep at the end of the series; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv1DTranspose(int filters, Window1D window, int outputPadding = 0) => Conv1DTranspose(filters, window, new Beginning(), outputPadding);

    /// <summary>A convolution along a series that gives back the steps an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks the series it makes; its border is stated, never worked out.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <param name="outputPadding">How many steps of nothing to keep at the end of the series; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts; or The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv1DTranspose(int filters, Window1D window, Beginning beginning, int outputPadding = 0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        RequireUnseeded(beginning);

        var stated = window.Border ?? new Sides(window.Padding, window.Padding);
        var settled = TransposedConvolution.Settling(stated, outputPadding);

        return Add(new TransposedWord(
            "conv1dtranspose",
            "a series of steps and channels",
            1,
            each =>
            [
                TransposedConvolution.GivesBack(each[0], window.Stride, (window.Dilation * (window.Length - 1)) + 1, settled.Dropped) + settled.Added,
                filters,
            ],
            (channels, draws) => new Conv1DTranspose(new Channels { In = channels, Out = filters }, window, beginning with { Draws = draws }, outputPadding)));
    }

    /// <summary>A convolution over an image that gives back the rows and columns an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks the image it makes; its border is stated, never worked out.</param>
    /// <param name="outputPadding">How many rows and columns of nothing to keep at the end of each axis; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv2DTranspose(int filters, Window window, Steps outputPadding = default) => Conv2DTranspose(filters, window, new Beginning(), outputPadding);

    /// <summary>A convolution over an image that gives back the rows and columns an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks the image it makes; its border is stated, never worked out.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <param name="outputPadding">How many rows and columns of nothing to keep at the end of each axis; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts; or The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv2DTranspose(int filters, Window window, Beginning beginning, Steps outputPadding = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        RequireUnseeded(beginning);

        var even = new Sides(window.Padding, window.Padding);
        var down = TransposedConvolution.Settling(window.Border is { } sides ? sides.Down : even, outputPadding.Down);
        var across = TransposedConvolution.Settling(window.Border is { } also ? also.Across : even, outputPadding.Across);

        return Add(new TransposedWord(
            "conv2dtranspose",
            "an image of rows, columns and channels",
            2,
            each =>
            [
                TransposedConvolution.GivesBack(each[0], window.Stride.Down, (window.Dilation.Down * (window.Height - 1)) + 1, down.Dropped) + down.Added,
                TransposedConvolution.GivesBack(each[1], window.Stride.Across, (window.Dilation.Across * (window.Width - 1)) + 1, across.Dropped) + across.Added,
                filters,
            ],
            (channels, draws) => new Conv2DTranspose(new Channels { In = channels, Out = filters }, window, beginning with { Draws = draws }, outputPadding)));
    }

    /// <summary>A convolution through a volume that gives back the places an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The block, and how it walks the volume it makes; its border is stated, never worked out.</param>
    /// <param name="outputPadding">How many places of nothing to keep at the end of each axis; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv3DTranspose(int filters, Window3D window, Steps3D outputPadding = default) => Conv3DTranspose(filters, window, new Beginning(), outputPadding);

    /// <summary>A convolution through a volume that gives back the places an ordinary one takes away.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The block, and how it walks the volume it makes; its border is stated, never worked out.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <param name="outputPadding">How many places of nothing to keep at the end of each axis; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts; or The window cannot stand anywhere or works its border out, or the output padding reaches as far as the window does.</exception>
    public Sequential Conv3DTranspose(int filters, Window3D window, Beginning beginning, Steps3D outputPadding = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        RequireUnseeded(beginning);

        var even = new Sides(window.Padding, window.Padding);
        var stated = window.Border ?? new Margins3D(even, even, even);
        var deep = TransposedConvolution.Settling(stated.Deep, outputPadding.Deep);
        var down = TransposedConvolution.Settling(stated.Down, outputPadding.Down);
        var across = TransposedConvolution.Settling(stated.Across, outputPadding.Across);

        return Add(new TransposedWord(
            "conv3dtranspose",
            "a volume of planes, rows, columns and channels",
            3,
            each =>
            [
                TransposedConvolution.GivesBack(each[0], window.Stride.Deep, (window.Dilation.Deep * (window.Depth - 1)) + 1, deep.Dropped) + deep.Added,
                TransposedConvolution.GivesBack(each[1], window.Stride.Down, (window.Dilation.Down * (window.Height - 1)) + 1, down.Dropped) + down.Added,
                TransposedConvolution.GivesBack(each[2], window.Stride.Across, (window.Dilation.Across * (window.Width - 1)) + 1, across.Dropped) + across.Added,
                filters,
            ],
            (channels, draws) => new Conv3DTranspose(new Channels { In = channels, Out = filters }, window, beginning with { Draws = draws }, outputPadding)));
    }

    /// <summary>Adds steps of nothing at each end of every series, and nothing else.</summary>
    /// <param name="border">The steps before the series and after it.</param>
    /// <param name="value">What those steps hold; nothing, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of steps is below nothing.</exception>
    public Sequential Pad1D(Sides border, float value = 0f) =>
        Add(new PaddingWord("pad1d", "a series of steps and channels", 1, [border], () => new Pad1D(border, value)));

    /// <summary>Adds steps at each end of every series, filled from the series itself, and nothing else.</summary>
    /// <param name="border">The steps before the series and after it.</param>
    /// <param name="filling">What fills them: the series mirrored back, its edge repeated, or the series gone round.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of steps is below nothing.</exception>
    public Sequential Pad1D(Sides border, Filling filling) =>
        Add(new PaddingWord("pad1d", "a series of steps and channels", 1, [border], () => new Pad1D(border, filling)));

    /// <summary>Adds rows and columns of nothing around every image, and nothing else.</summary>
    /// <param name="border">The rows above each image and below it, and the columns before it and after it.</param>
    /// <param name="value">What those rows and columns hold; nothing, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of rows or columns is below nothing.</exception>
    public Sequential Pad2D(Margins border, float value = 0f) =>
        Add(new PaddingWord("pad2d", "an image of rows, columns and channels", 2, [border.Down, border.Across], () => new Pad2D(border, value)));

    /// <summary>Adds rows and columns around every image, filled from the image itself, and nothing else.</summary>
    /// <param name="border">The rows above each image and below it, and the columns before it and after it.</param>
    /// <param name="filling">What fills them: the image mirrored back, its edge repeated, or the image gone round.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of rows or columns is below nothing.</exception>
    public Sequential Pad2D(Margins border, Filling filling) =>
        Add(new PaddingWord("pad2d", "an image of rows, columns and channels", 2, [border.Down, border.Across], () => new Pad2D(border, filling)));

    /// <summary>Adds planes, rows and columns of nothing around every volume, and nothing else.</summary>
    /// <param name="border">The places at each end of each axis.</param>
    /// <param name="value">What those places hold; nothing, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is below nothing.</exception>
    public Sequential Pad3D(Margins3D border, float value = 0f) =>
        Add(new PaddingWord(
            "pad3d",
            "a volume of planes, rows, columns and channels",
            3,
            [border.Deep, border.Down, border.Across],
            () => new Pad3D(border, value)));

    /// <summary>Adds planes, rows and columns around every volume, filled from the volume itself, and nothing else.</summary>
    /// <param name="border">The places at each end of each axis.</param>
    /// <param name="filling">What fills them: the volume mirrored back, its edge repeated, or the volume gone round.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is below nothing.</exception>
    public Sequential Pad3D(Margins3D border, Filling filling) =>
        Add(new PaddingWord(
            "pad3d",
            "a volume of planes, rows, columns and channels",
            3,
            [border.Deep, border.Down, border.Across],
            () => new Pad3D(border, filling)));

    /// <summary>Pools every series to so many places, each the average of the places it is made of.</summary>
    /// <param name="places">How many places each series is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The places are fewer than one.</exception>
    public Sequential AdaptiveAvgPool1D(int places) =>
        Add(new AdaptiveWord("adaptiveavgpool1d", "a series of steps and channels", new Shape(places), () => new AdaptiveAvgPool1D(new Shape(places))));

    /// <summary>Pools every image to so many rows and columns, each the average of the places it is made of.</summary>
    /// <param name="rows">How many rows each image is pooled to.</param>
    /// <param name="columns">How many columns it is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is fewer than one.</exception>
    public Sequential AdaptiveAvgPool2D(int rows, int columns) =>
        Add(new AdaptiveWord(
            "adaptiveavgpool2d",
            "an image of rows, columns and channels",
            new Shape(rows, columns),
            () => new AdaptiveAvgPool2D(new Shape(rows, columns))));

    /// <summary>Pools every volume to so many planes, rows and columns, each the average of the places it is made of.</summary>
    /// <param name="planes">How many planes each volume is pooled to.</param>
    /// <param name="rows">How many rows it is pooled to.</param>
    /// <param name="columns">How many columns it is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is fewer than one.</exception>
    public Sequential AdaptiveAvgPool3D(int planes, int rows, int columns) =>
        Add(new AdaptiveWord(
            "adaptiveavgpool3d",
            "a volume of planes, rows, columns and channels",
            new Shape(planes, rows, columns),
            () => new AdaptiveAvgPool3D(new Shape(planes, rows, columns))));

    /// <summary>Pools every series to so many places, each the largest of the places it is made of.</summary>
    /// <param name="places">How many places each series is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The places are fewer than one.</exception>
    public Sequential AdaptiveMaxPool1D(int places) =>
        Add(new AdaptiveWord("adaptivemaxpool1d", "a series of steps and channels", new Shape(places), () => new AdaptiveMaxPool1D(new Shape(places))));

    /// <summary>Pools every image to so many rows and columns, each the largest of the places it is made of.</summary>
    /// <param name="rows">How many rows each image is pooled to.</param>
    /// <param name="columns">How many columns it is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is fewer than one.</exception>
    public Sequential AdaptiveMaxPool2D(int rows, int columns) =>
        Add(new AdaptiveWord(
            "adaptivemaxpool2d",
            "an image of rows, columns and channels",
            new Shape(rows, columns),
            () => new AdaptiveMaxPool2D(new Shape(rows, columns))));

    /// <summary>Pools every volume to so many planes, rows and columns, each the largest of the places it is made of.</summary>
    /// <param name="planes">How many planes each volume is pooled to.</param>
    /// <param name="rows">How many rows it is pooled to.</param>
    /// <param name="columns">How many columns it is pooled to.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">A number of places is fewer than one.</exception>
    public Sequential AdaptiveMaxPool3D(int planes, int rows, int columns) =>
        Add(new AdaptiveWord(
            "adaptivemaxpool3d",
            "a volume of planes, rows, columns and channels",
            new Shape(planes, rows, columns),
            () => new AdaptiveMaxPool3D(new Shape(planes, rows, columns))));

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

    /// <summary>A convolution with a group for every channel: it makes so many channels of each it is handed.</summary>
    private sealed record DepthwiseWord(string Word, string Takes, int Multiplier, ISpatialWalk Walk, Func<int, Draws, Layer> Build) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != Walk.Rank + 1 ? TakesOnly(Word, Takes, each)
            : Walk.PlacesOver(each.Axes[..^1]).Any(places => places < 1) ? $"{Word}, slides a {Walk.Describe()} over each example, and an example {each} is smaller than it."
            : null;

        public Shape After(Shape each) => new([.. Walk.PlacesOver(each.Axes[..^1]), each[each.Rank - 1] * Multiplier]);

        Layer IWord.Make(Shape each, Draws draws) => Build(each[each.Rank - 1], draws);
    }

    /// <summary>A convolution that gives the places back: every axis comes out as long as its window gives back.</summary>
    /// <remarks>
    /// The places are worked out as plain numbers, because a border can take more than the window gives and a shape holds no
    /// length below one: the word says so in its own words rather than letting a shape refuse what it was asked to judge.
    /// </remarks>
    private sealed record TransposedWord(string Word, string Takes, int Rank, Func<Shape, int[]> Made, Func<int, Draws, Layer> Build) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != Rank + 1 ? TakesOnly(Word, Takes, each)
            : Made(each)[..^1].Any(places => places < 1)
                ? $"{Word}, gives back the places its window takes away, and an example {each} is left with none of them."
                : null;

        public Shape After(Shape each) => new(Made(each));

        Layer IWord.Make(Shape each, Draws draws) => Build(each[each.Rank - 1], draws);
    }

    /// <summary>A dropout of whole channels: it hands on the shape it is handed.</summary>
    private sealed record SpatialDropoutWord(string Word, string Takes, int Rank, double Rate, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == Rank + 1 ? null : TakesOnly(Word, Takes, each);

        public Shape After(Shape each) => each;

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }

    /// <summary>A layer that only pads: every axis comes out as long as the places added at its ends make it.</summary>
    private sealed record PaddingWord(string Word, string Takes, int Rank, IReadOnlyList<Sides> Border, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == Rank + 1 ? null : TakesOnly(Word, Takes, each);

        public Shape After(Shape each) =>
            new([.. each.Axes[..^1].ToArray().Select((length, axis) => length + Border[axis].Total), each[each.Rank - 1]]);

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }

    /// <summary>A pooling to a size of its own: every axis comes out as long as it was asked for, the channels unchanged.</summary>
    /// <remarks>
    /// An example that holds no place along an axis is refused where the description is handed its shape, so what is left for
    /// the word to refuse is an example of other axes than it pools; the layer itself refuses a batch that holds none.
    /// </remarks>
    private sealed record AdaptiveWord(string Word, string Takes, Shape Sizes, Func<Layer> Build) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == Sizes.Rank + 1 ? null : TakesOnly(Word, Takes, each);

        public Shape After(Shape each) => new([.. Sizes.Axes, each[each.Rank - 1]]);

        Layer IWord.Make(Shape each, Draws draws) => Build();
    }
}
