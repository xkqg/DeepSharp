// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>
/// Where a Keras layer's window stands and how it walks, as the layer says it: the length of each side, one stride on every
/// axis, and how the border is had. The window of a series, an image or a volume is made of it as the layer walks one, two
/// or three axes.
/// </summary>
/// <param name="Sides">The length of each side of the window, outermost first.</param>
/// <param name="Stride">How many places it moves between one patch and the next, along each axis.</param>
/// <param name="Dilation">How many places apart its cells are, along each axis.</param>
/// <param name="Padding">How its border is had.</param>
internal readonly record struct Footprint(int[] Sides, int[] Stride, int[] Dilation, PaddingMode Padding)
{
    /// <summary>How many groups the channels are split into; one unless the layer says otherwise.</summary>
    public int Groups { get; init; } = 1;

    /// <summary>The window along a series.</summary>
    public Window1D Line() => new(Sides[0]) { Stride = Stride[0], Dilation = Dilation[0], PaddingMode = Padding };

    /// <summary>The window over an image.</summary>
    public Window Plane() => new(Sides[0], Sides[1])
    {
        Stride = new Steps(Stride[0], Stride[1]),
        Dilation = new Steps(Dilation[0], Dilation[1]),
        PaddingMode = Padding,
    };

    /// <summary>The window through a volume.</summary>
    public Window3D Volume() => new(Sides[0], Sides[1], Sides[2])
    {
        Stride = new Steps3D(Stride[0], Stride[1], Stride[2]),
        Dilation = new Steps3D(Dilation[0], Dilation[1], Dilation[2]),
        PaddingMode = Padding,
    };
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

        if (padding is not ("valid" or "same" or "causal") || (padding == "causal" && axes != 1))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as {(axes == 1 ? "'valid', 'same' or 'causal'" : "'valid' or 'same'")}.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));
        RequireBias(layer);

        var mode = padding switch { "same" => PaddingMode.Same, "causal" => PaddingMode.Causal, _ => PaddingMode.Stated };

        return new KerasLayer(words(filters, new Footprint(sides, strides, dilation, mode) { Groups = groups }), ActivationOf(layer))
        {
            Slots = ["weight", "bias"],
            Sizes = sides,
        };
    }
}

/// <summary>
/// Keras's <c>Conv1DTranspose</c>, <c>Conv2DTranspose</c> and <c>Conv3DTranspose</c>: a convolution that gives back the
/// places an ordinary one takes away.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes the window walks: one, two or three.</param>
/// <remarks>
/// Keras states no border but the words 'valid' and 'same', and a layer that MAKES its own example cannot have one worked
/// out from it: so the border is worked out here, from the sizes the layer itself makes. Padded as 'valid' it drops nothing
/// and the example comes out as long as the window gives back; padded as 'same' it comes out the input's length times the
/// stride, and what the window gives back beyond that is dropped — half before and the rest after, which is TensorFlow's own
/// rule and which no border stated alike can say.
/// </remarks>
internal sealed class TransposedConvolutionKind(string name, int axes) : KerasKind
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
        var padding = layer.Setting("padding", "valid");
        var grown = layer.Says("output_padding") ? layer.Axes("output_padding", axes) : null;

        if (padding is not ("valid" or "same"))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as 'valid' or 'same'.");
        }

        if (padding == "same" && grown is not null)
        {
            layer.Refuse("it pads as 'same' and states how many places to keep at the end of each axis, and the two say different sizes; Keras reads the first and leaves the second.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));
        RequireBias(layer);

        var footprint = new Footprint(sides, strides, dilation, padding == "same" ? PaddingMode.Same : PaddingMode.Stated);

        return new KerasLayer(Words(filters, footprint, grown), ActivationOf(layer))
        {
            Slots = [new("weight", 0, sides) { Kept = KerasKeeping.GivenBack }, "bias"],
            Sizes = sides,
        };
    }

    // The words that make it, with the border worked out from the sizes the layer itself makes.
    private Action<Sequential> Words(int filters, Footprint footprint, int[]? grown)
    {
        var (sides, strides, dilation, _) = footprint;
        var pads = Dropped(footprint);
        var step = grown ?? new int[axes];

        return axes switch
        {
            1 => description => description.Conv1DTranspose(
                filters,
                new Window1D(sides[0]) { Stride = strides[0], Dilation = dilation[0], Border = new Sides(pads[0], pads[1]) },
                step[0]),
            2 => description => description.Conv2DTranspose(
                filters,
                new Window(sides[0], sides[1])
                {
                    Stride = new Steps(strides[0], strides[1]),
                    Dilation = new Steps(dilation[0], dilation[1]),
                    Border = new Margins(new Sides(pads[0], pads[1]), new Sides(pads[2], pads[3])),
                },
                new Steps(step[0], step[1])),
            _ => description => description.Conv3DTranspose(
                filters,
                new Window3D(sides[0], sides[1], sides[2])
                {
                    Stride = new Steps3D(strides[0], strides[1], strides[2]),
                    Dilation = new Steps3D(dilation[0], dilation[1], dilation[2]),
                    Border = new Margins3D(new Sides(pads[0], pads[1]), new Sides(pads[2], pads[3]), new Sides(pads[4], pads[5])),
                },
                new Steps3D(step[0], step[1], step[2])),
        };
    }

    // What the folding drops at each end of each axis: nothing, where the layer pads as 'valid' and the example comes out as
    // long as the window gives back. Padded as 'same' Keras makes an example the input's length times the stride, and what the
    // window gives back beyond that is dropped, half before and the rest after — TensorFlow's own rule read backwards.
    private int[] Dropped(Footprint footprint)
    {
        var pads = new int[2 * axes];

        if (footprint.Padding != PaddingMode.Same)
        {
            return pads;
        }

        for (var axis = 0; axis < axes; axis++)
        {
            var reach = (footprint.Dilation[axis] * (footprint.Sides[axis] - 1)) + 1;
            var total = Math.Max(0, reach - footprint.Stride[axis]);

            pads[2 * axis] = total / 2;
            pads[(2 * axis) + 1] = total - (total / 2);
        }

        return pads;
    }
}


/// <summary>
/// Keras's <c>DepthwiseConv1D</c> and <c>DepthwiseConv2D</c>: a convolution with a group for every channel it is handed,
/// making <c>depth_multiplier</c> channels of each.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes the window walks: one or two.</param>
internal sealed class DepthwiseConvolutionKind(string name, int axes) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var footprint = Walked(layer, axes, out var multiplier);

        return new KerasLayer(Words(footprint, multiplier, axes), ActivationOf(layer))
        {
            Slots = [new("weight", 0, footprint.Sides) { Kept = KerasKeeping.Apart }, "bias"],
            Sizes = footprint.Sides,
        };
    }

    // The window and the multiplier a depthwise convolution says, each setting of it that no window here can refused at the layer.
    internal static Footprint Walked(KerasSettings layer, int axes, out int multiplier)
    {
        var sides = layer.Axes("kernel_size", axes);
        var strides = layer.Axes("strides", axes);
        var dilation = layer.Axes("dilation_rate", axes);
        var padding = layer.Setting("padding", "valid");

        multiplier = layer.Setting("depth_multiplier", 1);

        if (padding is not ("valid" or "same"))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as 'valid' or 'same'.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));
        RequireBias(layer);

        return new Footprint(sides, strides, dilation, padding == "same" ? PaddingMode.Same : PaddingMode.Stated);
    }

    // A convolution with a group for every channel it is handed: what a depthwise convolution is.
    internal static Action<Sequential> Words(Footprint footprint, int multiplier, int axes) =>
        axes == 1
            ? description => description.DepthwiseConv1D(multiplier, footprint.Line())
            : description => description.DepthwiseConv2D(multiplier, footprint.Plane());
}

/// <summary>
/// Keras's <c>SeparableConv1D</c> and <c>SeparableConv2D</c>: a depthwise convolution and then a convolution of a window of
/// one place, which are two layers here. Keras adds its bias once, after both, so the first is given one of noughts.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes the window walks: one or two.</param>
internal sealed class SeparableConvolutionKind(string name, int axes) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var filters = layer.Setting<int>("filters");
        var footprint = DepthwiseConvolutionKind.Walked(layer, axes, out var multiplier);
        var ones = new int[axes];

        Array.Fill(ones, 1);

        var pointwise = new Footprint(ones, ones, ones, PaddingMode.Stated);

        return new KerasLayer(
            description =>
            {
                DepthwiseConvolutionKind.Words(footprint, multiplier, axes)(description);

                if (axes == 1)
                {
                    description.Conv1D(filters, pointwise.Line());
                }
                else
                {
                    description.Conv2D(filters, pointwise.Plane());
                }
            },
            ActivationOf(layer))
        {
            // Keras writes the depthwise kernel, the pointwise kernel and the one bias, in that order; here they are the
            // first layer's kernel, the second's, and the second's bias, and the first's bias is noughts.
            Slots =
            [
                new("weight", 0, footprint.Sides) { Kept = KerasKeeping.Apart },
                new("weight", 1, ones),
                new("bias", 1, []),
            ],
            Noughts = [new("bias", 0, [])],
            Places = 2,
        };
    }
}


/// <summary>
/// Keras's <c>ZeroPadding1D</c>, <c>ZeroPadding2D</c> and <c>ZeroPadding3D</c>: a layer that adds places of nothing at each
/// end of each axis and does nothing else. Keras writes the places as one number for every end; along a series, as the pair
/// of its two ends; and over an image or through a volume, as one number or a pair for each axis.
/// </summary>
/// <param name="name">The class name Keras writes the kind under.</param>
/// <param name="axes">How many axes it pads: one, two or three.</param>
internal sealed class PaddingKind(string name, int axes) : KerasKind
{
    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        RequireChannelsLast(layer, ThingsWalkedAlong(axes));

        return new KerasLayer(Padding(Sided(layer)), "linear");
    }

    // The places before and after each axis, from what Keras writes: one number for every end of every axis; the two ends of
    // a single axis as the pair itself, which is what Keras writes along a series; or, over an image or through a volume, a
    // number or a pair for each axis. A place below nothing is no padding at all and is refused where the layer says it.
    private int[] Sided(KerasSettings layer)
    {
        var written = layer.Setting("padding", default(JsonElement));
        var sides = new int[2 * axes];

        if (Whole(written, out var every))
        {
            Array.Fill(sides, every);
        }
        else if (written.ValueKind != JsonValueKind.Array || !Ends(written, sides))
        {
            layer.Refuse(
                $"its 'padding' is written as {layer.Written("padding")?.Quoted() ?? "nothing"}, and Keras writes "
                + $"{(axes == 1 ? "one number for both ends, or the pair of them" : "one number for every end, one for each axis, or a pair for each axis")} there.");

            return sides;
        }

        if (sides.Any(places => places < 0))
        {
            layer.Refuse($"it pads {layer.Written("padding")!.Quoted()} places, and a padding here adds places and never takes them away.");
        }

        return sides;
    }

    // The ends Keras writes as a list: the pair of a single axis, or a number or a pair for each of two or three axes; whether
    // what it writes is one of those.
    private bool Ends(JsonElement written, int[] sides)
    {
        var each = written.EnumerateArray().ToArray();

        if (axes == 1)
        {
            return Pair(each, sides, 0);
        }

        if (each.Length != axes)
        {
            return false;
        }

        for (var axis = 0; axis < axes; axis++)
        {
            if (Whole(each[axis], out var both))
            {
                sides[2 * axis] = both;
                sides[(2 * axis) + 1] = both;
            }
            else if (each[axis].ValueKind != JsonValueKind.Array || !Pair([.. each[axis].EnumerateArray()], sides, axis))
            {
                return false;
            }
        }

        return true;
    }

    // The two ends of one axis, as Keras writes them: whether they are a pair of whole numbers.
    private static bool Pair(JsonElement[] written, int[] sides, int axis)
    {
        if (written.Length != 2 || !Whole(written[0], out var before) || !Whole(written[1], out var after))
        {
            return false;
        }

        sides[2 * axis] = before;
        sides[(2 * axis) + 1] = after;

        return true;
    }

    // One whole number, as Keras writes a count of places.
    private static bool Whole(JsonElement written, out int number)
    {
        number = 0;

        return written.ValueKind == JsonValueKind.Number && written.TryGetInt32(out number);
    }

    // The word of the padding for the places it adds along each axis.
    private Action<Sequential> Padding(int[] sides) => axes switch
    {
        1 => description => description.Pad1D(new Sides(sides[0], sides[1])),
        2 => description => description.Pad2D(new Margins(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3]))),
        _ => description => description.Pad3D(
            new Margins3D(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3]), new Sides(sides[4], sides[5]))),
    };
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

        if (padding is not ("valid" or "same"))
        {
            layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as 'valid' or 'same'.");
        }

        RequireChannelsLast(layer, ThingsWalkedAlong(axes));

        return new KerasLayer(
            words(new Footprint(sides, strides, [.. Enumerable.Repeat(1, axes)], padding == "same" ? PaddingMode.Same : PaddingMode.Stated)),
            "linear");
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
