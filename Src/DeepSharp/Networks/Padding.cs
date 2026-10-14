// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What the places a padding adds hold.</summary>
public enum Filling
{
    /// <summary>One value in every place: nothing, unless a value is said — Keras's <c>ZeroPadding</c>, PyTorch's <c>ConstantPad</c>, ONNX's <c>Pad</c> of a constant.</summary>
    Constant,

    /// <summary>
    /// The example mirrored back from its edge, the edge itself not repeated: <c>0 1 2 3</c> padded by two at each end is
    /// <c>2 1 0 1 2 3 2 1</c> — PyTorch's reflection padding, ONNX's <c>reflect</c>.
    /// </summary>
    Mirror,

    /// <summary>
    /// The edge repeated: <c>0 1 2 3</c> padded by two at each end is <c>0 0 0 1 2 3 3 3</c> — PyTorch's replication padding,
    /// ONNX's <c>edge</c>.
    /// </summary>
    Repeat,

    /// <summary>
    /// The example gone round: <c>0 1 2 3</c> padded by two at each end is <c>2 3 0 1 2 3 0 1</c> — PyTorch's circular padding,
    /// ONNX's <c>wrap</c>.
    /// </summary>
    Wrap,
}

/// <summary>
/// Adds places around each example and does nothing else: so many before and after each axis of a series, an image or a
/// volume, holding one value — Keras's <c>ZeroPadding</c>, PyTorch's <c>ConstantPad</c>, ONNX's <c>Pad</c> of a constant —
/// or the example's own values, mirrored back, repeated from the edge or gone round (<see cref="Networks.Filling"/>).
/// </summary>
/// <remarks>
/// A convolution's own window pads the example it walks, so a layer that only pads is for what a window cannot say: a border
/// a later layer must see as values, one that differs from side to side where the layers after it pad alike, a border that
/// is the example's own values where a window can only pad with nought, or the places a transposed convolution adds at the
/// end of an axis alone. It holds no numbers and learns nothing.
/// <para>
/// The padding is the unfolding of a window one place wide whose border is the places asked for, which every engine already
/// does: the patches it hands back are the padded example, a row for every place of it. Where the value padded with is not
/// nought, the places the example does not reach are given it by the same mask a pooling uses to tell a value of the example
/// from its border, so the example's own values are not touched at all.
/// </para>
/// <para>
/// Places filled from the example are each the example's value at the place they mirror, repeat or wrap to, picked by a
/// matrix of ones and noughts along one axis after another, so every value is the example's own, rounded not at all, and a
/// gradient goes back to the place each was taken from, added where several were. A mirror needs the axis to be longer than
/// the places it adds at an end, and a wrap needs it to be at least as long; both are refused by name when it is not.
/// </para>
/// </remarks>
public abstract class Padding : Layer
{
    private readonly ISpatialWalk _walk;
    private readonly FillPlan _fill;

    private protected Padding(ISpatialWalk walk, float value, Filling filling, int[] sides)
    {
        _walk = walk;
        Value = value;
        Filling = filling;
        _fill = new FillPlan(filling, sides);
    }

    /// <summary>What the places it adds hold, when they hold one value; nothing, unless said.</summary>
    public float Value { get; }

    /// <summary>What fills the places it adds: one value, or the example's own values.</summary>
    public Filling Filling { get; }

    /// <summary>
    /// The oldest version of a network's part that says this padding: the second, unless it fills its places from the example,
    /// which no library before this release could read.
    /// </summary>
    public int FileVersion => Filling == Filling.Constant ? NetworkDocument.SecondVersion : NetworkDocument.FifthVersion;

    /// <summary>Writes how the places are filled, when it is from the example: a padding of one value has nothing to say of it.</summary>
    private protected void WriteFilling(Utf8JsonWriter writer)
    {
        if (Filling != Filling.Constant)
        {
            writer.WriteString("border", Filling switch { Filling.Mirror => "mirror", Filling.Repeat => "repeat", _ => "wrap" });
        }
    }

    /// <summary>How a file says the places are filled, or nothing for one value.</summary>
    /// <param name="settings">The object the padding was written as.</param>
    /// <param name="rebuilding">What the file is read with.</param>
    /// <returns>What fills the places.</returns>
    /// <exception cref="FormatException">The file names another way to fill them, or a value beside one that fills them from the example.</exception>
    private protected static Filling FillingIn(JsonElement settings, Rebuilding rebuilding)
    {
        var filling = rebuilding.SaysOneOf(settings, "border", "'mirror', 'repeat' or 'wrap'", "mirror", "repeat", "wrap") switch
        {
            "mirror" => Filling.Mirror,
            "repeat" => Filling.Repeat,
            "wrap" => Filling.Wrap,
            _ => Filling.Constant,
        };

        if (filling != Filling.Constant && rebuilding.Number(settings, "value", 0) is not 0d)
        {
            throw rebuilding.Refuse("value", "A padding filled from the example holds no value of its own.");
        }

        return filling;
    }

    /// <summary>Writes the value padded with, when it is not nought: a padding of nothing has nothing to say.</summary>
    private protected void WriteValue(Utf8JsonWriter writer)
    {
        if (Value is not 0f)
        {
            writer.WriteNumber("value", Value);
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of series, images or volumes of the rank this padding pads.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        input.Shape.RequireSpatial(_walk.Rank, null, "A padding");

        return Filling == Filling.Constant
            ? _walk.Padded(pass.Backend, input, Value)
            : _walk.FilledFrom(pass.Backend, input, _fill);
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
        : base(new LineWalk(new Window1D(1) { Border = border }), value, Filling.Constant, [border.Before, border.After]) => Border = border;

    /// <summary>A padding that fills the steps it adds from the series itself.</summary>
    /// <param name="border">The steps before the series and after it.</param>
    /// <param name="filling">What fills them.</param>
    /// <exception cref="ArgumentException">A number of steps is below nothing.</exception>
    public Pad1D(Sides border, Filling filling)
        : base(new LineWalk(new Window1D(1) { Border = border }), 0f, filling, [border.Before, border.After]) => Border = border;

    /// <summary>The steps before the series and after it.</summary>
    public Sides Border { get; }

    /// <inheritdoc />
    public static string Name => "pad1d";

    /// <inheritdoc />
    public static Pad1D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 1);

        var filling = FillingIn(settings, rebuilding);

        return filling == Filling.Constant
            ? new(new Sides(border[0], border[1]), (float)rebuilding.Number(settings, "value", 0))
            : new(new Sides(border[0], border[1]), filling);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBorder([Border.Before, Border.After]);
        WriteValue(writer);
        WriteFilling(writer);
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
        : base(new PlaneWalk(new Window(1, 1) { Border = border }), value, Filling.Constant, [border.Down.Before, border.Down.After, border.Across.Before, border.Across.After]) => Border = border;

    /// <summary>A padding that fills the rows and columns it adds from the image itself.</summary>
    /// <param name="border">The rows above the image and below it, and the columns before it and after it.</param>
    /// <param name="filling">What fills them.</param>
    /// <exception cref="ArgumentException">A number of rows or columns is below nothing.</exception>
    public Pad2D(Margins border, Filling filling)
        : base(new PlaneWalk(new Window(1, 1) { Border = border }), 0f, filling, [border.Down.Before, border.Down.After, border.Across.Before, border.Across.After]) => Border = border;

    /// <summary>The rows and columns added around each image.</summary>
    public Margins Border { get; }

    /// <inheritdoc />
    public static string Name => "pad2d";

    /// <inheritdoc />
    public static Pad2D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 2);
        var margins = new Margins(new Sides(border[0], border[1]), new Sides(border[2], border[3]));
        var filling = FillingIn(settings, rebuilding);

        return filling == Filling.Constant ? new(margins, (float)rebuilding.Number(settings, "value", 0)) : new(margins, filling);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBorder([Border.Down.Before, Border.Down.After, Border.Across.Before, Border.Across.After]);
        WriteValue(writer);
        WriteFilling(writer);
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
        : base(new VolumeWalk(new Window3D(1, 1, 1) { Border = border }), value, Filling.Constant, Sided(border)) => Border = border;

    /// <summary>A padding that fills the planes, rows and columns it adds from the volume itself.</summary>
    /// <param name="border">The places at each end of each axis.</param>
    /// <param name="filling">What fills them.</param>
    /// <exception cref="ArgumentException">A number of places is below nothing.</exception>
    public Pad3D(Margins3D border, Filling filling)
        : base(new VolumeWalk(new Window3D(1, 1, 1) { Border = border }), 0f, filling, Sided(border)) => Border = border;

    /// <summary>The planes, rows and columns added around each volume.</summary>
    public Margins3D Border { get; }

    /// <inheritdoc />
    public static string Name => "pad3d";

    /// <inheritdoc />
    public static Pad3D Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var border = rebuilding.BorderIn(settings, 3);
        var margins = new Margins3D(new Sides(border[0], border[1]), new Sides(border[2], border[3]), new Sides(border[4], border[5]));
        var filling = FillingIn(settings, rebuilding);

        return filling == Filling.Constant ? new(margins, (float)rebuilding.Number(settings, "value", 0)) : new(margins, filling);
    }

    // The places at each end of each axis, a pair an axis in the axes' own order.
    private static int[] Sided(Margins3D border) =>
        [border.Deep.Before, border.Deep.After, border.Down.Before, border.Down.After, border.Across.Before, border.Across.After];

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
        WriteFilling(writer);
    }
}
