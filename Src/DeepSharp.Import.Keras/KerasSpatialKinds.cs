// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>
/// Where a Keras layer's window stands and how it walks, as the layer says it: the length of each side, one stride on every
/// axis, and how the border is had. The window of a series, an image or a volume is made of it as the layer walks one, two
/// or three axes.
/// </summary>
/// <param name="Sides">The length of each side of the window, outermost first.</param>
/// <param name="Stride">How many places it moves between one patch and the next, along every axis.</param>
/// <param name="Padding">How its border is had.</param>
internal readonly record struct Footprint(int[] Sides, int Stride, PaddingMode Padding)
{
    /// <summary>The window along a series.</summary>
    public Window1D Line() => new(Sides[0]) { Stride = Stride, PaddingMode = Padding };

    /// <summary>The window over an image.</summary>
    public Window Plane() => new(Sides[0], Sides[1]) { Stride = Stride, PaddingMode = Padding };

    /// <summary>The window through a volume.</summary>
    public Window3D Volume() => new(Sides[0], Sides[1], Sides[2]) { Stride = Stride, PaddingMode = Padding };
}

/// <summary>
/// Keras's <c>Conv1D</c>, <c>Conv2D</c> and <c>Conv3D</c>: a convolution whose window walks one stride along every axis,
/// padded as 'valid' or 'same' — and, along a series, as 'causal'.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes the window walks: one, two or three.</param>
/// <param name="words">The words that make the convolution, from the channels it makes and the window it walks.</param>
internal sealed class ConvolutionKind(string name, int axes, Func<int, Footprint, Action<Sequential>> words) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var filters = layer.Setting<int>("filters");
        var sides = layer.Axes("kernel_size", axes);
        var strides = layer.Axes("strides", axes);
        var dilation = layer.Axes("dilation_rate", axes);
        var groups = layer.Setting("groups", 1);
        var padding = layer.Setting("padding", "valid");

        if (strides.Differ())
        {
            layer.Refuse($"it strides {strides.Spread()}, and a window here walks one stride {strides.Directions()} alike.");
        }

        if (dilation.Any(rate => rate != 1))
        {
            layer.Refuse($"its window is dilated by {dilation.Spread()}, and a window here covers neighbouring places.");
        }

        if (groups != 1)
        {
            layer.Refuse($"it splits its channels into {groups} groups, and a convolution here takes every channel of a place at once.");
        }

        if (padding is not ("valid" or "same" or "causal") || (padding == "causal" && axes != 1))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as {(axes == 1 ? "'valid', 'same' or 'causal'" : "'valid' or 'same'")}.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));
        RequireBias(layer);

        var mode = padding switch { "same" => PaddingMode.Same, "causal" => PaddingMode.Causal, _ => PaddingMode.Stated };

        return new KerasLayer(words(filters, new Footprint(sides, strides[0], mode)), ActivationOf(layer)) { Slots = ["weight", "bias"], Sizes = sides };
    }
}

/// <summary>
/// Keras's <c>MaxPooling1D</c>, <c>MaxPooling2D</c>, <c>MaxPooling3D</c> and the three <c>AveragePooling</c> layers: a pooling
/// whose window walks one stride along every axis, padded as 'valid' or 'same', its stride the size of its window when it says none.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes the window walks: one, two or three.</param>
/// <param name="words">The words that make the pooling, from the window it walks.</param>
internal sealed class PoolingKind(string name, int axes, Func<Footprint, Action<Sequential>> words) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var sides = layer.Alike("pool_size", axes);
        var strides = layer.Says("strides") ? layer.Alike("strides", axes) : sides;
        var padding = layer.Setting("padding", "valid");

        if (strides.Differ())
        {
            layer.Refuse($"it strides {strides.Spread()}, and a window here walks one stride {strides.Directions()} alike.");
        }

        if (padding is not ("valid" or "same"))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as 'valid' or 'same'.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));

        return new KerasLayer(words(new Footprint(sides, strides[0], padding == "same" ? PaddingMode.Same : PaddingMode.Stated)), "linear");
    }
}

/// <summary>
/// Keras's <c>GlobalMaxPooling1D</c>, <c>GlobalMaxPooling2D</c>, <c>GlobalMaxPooling3D</c> and the three
/// <c>GlobalAveragePooling</c> layers: every place of an example pooled into one, the axes dropped or, as Keras's
/// <c>keepdims</c> says, kept as axes of one place.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes it pools: one, two or three.</param>
/// <param name="words">The words that make the pooling, from whether it keeps its axes.</param>
internal sealed class GlobalPoolingKind(string name, int axes, Func<bool, Action<Sequential>> words) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireChannelsLast(layer, ThingsWalkedAlong(axes));

        return new KerasLayer(words(layer.Setting("keepdims", false)), "linear");
    }
}

/// <summary>
/// Keras's <c>SpatialDropout1D</c>, <c>SpatialDropout2D</c> and <c>SpatialDropout3D</c>, leaving out whole channels.
/// </summary>
/// <remarks>
/// A spatial dropout works its noise shape out itself, one value for each channel, and has no setting for one: a file that
/// says a noise shape beside it is read as Keras reads it, the shape unused.
/// </remarks>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes it walks: one, two or three.</param>
/// <param name="words">The words that make the dropout, from the share of channels it leaves out.</param>
internal sealed class SpatialDropoutKind(string name, int axes, Func<double, Action<Sequential>> words) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var rate = layer.Setting<double>("rate");
        RequireChannelsLast(layer, ThingsWalkedAlong(axes));

        return new KerasLayer(words(rate), "linear");
    }
}

/// <summary>How the lengths Keras writes for a window's axes are said in a fault.</summary>
internal static class WindowLengthsExtensions
{
    extension(int[] lengths)
    {
        /// <summary>Whether the lengths are not all alike.</summary>
        internal bool Differ() => lengths.Any(length => length != lengths[0]);

        /// <summary>The lengths each with the direction it runs, outermost first: <c>2 down and 1 across</c>; a series' one length alone.</summary>
        internal string Spread() => lengths.Length == 1 ? $"{lengths[0]}" : Told(lengths.Zip(DirectionsOf(lengths.Length), (length, direction) => $"{length} {direction}"));

        /// <summary>The directions the lengths run, outermost first: <c>down and across</c>, <c>deep, down and across</c>.</summary>
        internal string Directions() => Told(DirectionsOf(lengths.Length));
    }

    // The directions of the axes of an image or a volume, outermost first.
    private static string[] DirectionsOf(int axes) => axes == 2 ? ["down", "across"] : ["deep", "down", "across"];

    // The parts told in a row: commas between them, and 'and' before the last.
    private static string Told(IEnumerable<string> parts)
    {
        var all = parts.ToArray();

        return $"{string.Join(", ", all[..^1])} and {all[^1]}";
    }
}
