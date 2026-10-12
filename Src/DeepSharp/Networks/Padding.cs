// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Adds places of nothing around each example and does nothing else: so many before and after each axis of a series, an
/// image or a volume — Keras's <c>ZeroPadding</c>, PyTorch's <c>ConstantPad</c>, ONNX's <c>Pad</c> of a constant.
/// </summary>
/// <remarks>
/// A convolution's own window pads the example it walks, so a layer that only pads is for what a window cannot say: a border
/// a later layer must see as values, one that differs from side to side where the layers after it pad alike, or the places a
/// transposed convolution adds at the end of an axis alone. It holds no numbers and learns nothing.
/// <para>
/// The padding is the unfolding of a window one place wide whose border is the places asked for, which every engine already
/// does: the patches it hands back are the padded example, a row for every place of it. Where the value padded with is not
/// nought, the places the example does not reach are given it by the same mask a pooling uses to tell a value of the example
/// from its border, so the example's own values are not touched at all.
/// </para>
/// </remarks>
public abstract class Padding : Layer
{
    private readonly ISpatialWalk _walk;
    private readonly int _rank;

    private protected Padding(ISpatialWalk walk, int rank, float value)
    {
        _walk = walk;
        _rank = rank;
        Value = value;
    }

    /// <summary>What the places it adds hold; nothing, unless said.</summary>
    public float Value { get; }

    /// <summary>Writes the value padded with, when it is not nought: a padding of nothing has nothing to say.</summary>
    private protected void WriteValue(Utf8JsonWriter writer)
    {
        if (Value != 0f)
        {
            writer.WriteNumber("value", Value);
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of series, images or volumes of the rank this padding pads.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        input.Shape.RequireSpatial(_rank, null, "A padding");

        return _walk.Padded(pass.Backend, input, Value);
    }
}

/// <summary>Adds steps of nothing at each end of a series.</summary>
public sealed class Pad1D : Padding, ISaved<Pad1D>
{
    /// <summary>A padding that adds the given steps at each end of a series.</summary>
    /// <param name="border">The steps before the series and after it.</param>
    /// <param name="value">What those steps hold; nothing, unless said.</param>
    /// <exception cref="ArgumentException">A number of steps is below nothing.</exception>
    public Pad1D(Sides border, float value = 0f)
        : base(new LineWalk(new Window1D(1) { Border = border }), 1, value) => Border = border;

    /// <summary>The steps before the series and after it.</summary>
    public Sides Border { get; }

    /// <inheritdoc />
    public static string Name => "pad1d";

    /// <inheritdoc />
    public static Pad1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 1);

        return new(new Sides(border[0], border[1]), (float)rebuilding.Number(settings, "value", 0));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBorder([Border.Before, Border.After]);
        WriteValue(writer);
    }
}

/// <summary>Adds rows and columns of nothing around each image.</summary>
public sealed class Pad2D : Padding, ISaved<Pad2D>
{
    /// <summary>A padding that adds the given rows and columns around each image.</summary>
    /// <param name="border">The rows above the image and below it, and the columns before it and after it.</param>
    /// <param name="value">What those rows and columns hold; nothing, unless said.</param>
    /// <exception cref="ArgumentException">A number of rows or columns is below nothing.</exception>
    public Pad2D(Margins border, float value = 0f)
        : base(new PlaneWalk(new Window(1, 1) { Border = border }), 2, value) => Border = border;

    /// <summary>The rows and columns added around each image.</summary>
    public Margins Border { get; }

    /// <inheritdoc />
    public static string Name => "pad2d";

    /// <inheritdoc />
    public static Pad2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 2);

        return new(
            new Margins(new Sides(border[0], border[1]), new Sides(border[2], border[3])),
            (float)rebuilding.Number(settings, "value", 0));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBorder([Border.Down.Before, Border.Down.After, Border.Across.Before, Border.Across.After]);
        WriteValue(writer);
    }
}

/// <summary>Adds planes, rows and columns of nothing around each volume.</summary>
public sealed class Pad3D : Padding, ISaved<Pad3D>
{
    /// <summary>A padding that adds the given planes, rows and columns around each volume.</summary>
    /// <param name="border">The places at each end of each axis.</param>
    /// <param name="value">What those places hold; nothing, unless said.</param>
    /// <exception cref="ArgumentException">A number of places is below nothing.</exception>
    public Pad3D(Margins3D border, float value = 0f)
        : base(new VolumeWalk(new Window3D(1, 1, 1) { Border = border }), 3, value) => Border = border;

    /// <summary>The planes, rows and columns added around each volume.</summary>
    public Margins3D Border { get; }

    /// <inheritdoc />
    public static string Name => "pad3d";

    /// <inheritdoc />
    public static Pad3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 3);

        return new(
            new Margins3D(new Sides(border[0], border[1]), new Sides(border[2], border[3]), new Sides(border[4], border[5])),
            (float)rebuilding.Number(settings, "value", 0));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBorder([
            Border.Deep.Before, Border.Deep.After,
            Border.Down.Before, Border.Down.After,
            Border.Across.Before, Border.Across.After,
        ]);
        WriteValue(writer);
    }
}
