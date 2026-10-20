// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A variation of the rows a network learns from, and of those alone: while it trains each batch is varied afresh, and while it answers
/// nothing is.
/// </summary>
/// <remarks>
/// <para>
/// The variation is drawn from a seed of its own, which keys a stream of its own at the layer's place in the network, the epoch and the
/// step. The run's seed moves none of it, so the same variation is drawn by any run, a run resumed at an epoch draws what it would have
/// drawn the first time, and training a network again gives the same scores. Nothing is kept: a variation is made of a batch as it
/// passes and is gone, so the rows the pipeline hands over are the rows, whole, and what a feature held at one value still holds it.
/// </para>
/// <para>
/// It stands first, before any layer that learns: a variation is of the rows themselves, made on the values as the batch carries
/// them, and a gradient is not carried through it.
/// </para>
/// </remarks>
public abstract class Variation : Layer
{
    private readonly string _purpose;

    private protected Variation(string purpose, long seed)
    {
        _purpose = purpose;
        Seed = seed;
    }

    /// <summary>The number the variation is drawn from.</summary>
    public long Seed { get; }

    /// <inheritdoc />
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(pass);

        if (pass.Mode == PassMode.Evaluation)
        {
            return input;
        }

        return Varied(input, new RandomStream(Seed).Draw($"{_purpose}:{Path}", pass.Epoch, pass.Step));
    }

    /// <summary>A batch varied, drawing from the variation's stream.</summary>
    /// <param name="input">The batch.</param>
    /// <param name="draws">The draws of this layer, this epoch and this step.</param>
    /// <returns>The batch, varied.</returns>
    private protected abstract Tensor Varied(Tensor input, Draws draws);
}

/// <summary>Mirrors images at random while a network trains: each image across its columns, its rows, both or neither.</summary>
/// <remarks>
/// Images are laid out with their channels last, as the convolutions here take them. Each image is flipped across each axis the layer
/// is told to with a chance of a half, by a coin of its own.
/// </remarks>
public sealed class RandomFlip : Variation, ISaved<RandomFlip>
{
    /// <summary>A flip across the given axes.</summary>
    /// <param name="horizontal">Whether an image is mirrored across its columns, left for right.</param>
    /// <param name="vertical">Whether an image is mirrored across its rows, top for bottom.</param>
    /// <param name="seed">The number the flips are drawn from.</param>
    public RandomFlip(bool horizontal, bool vertical, long seed)
        : base("randomFlip", seed)
    {
        Horizontal = horizontal;
        Vertical = vertical;
    }

    /// <summary>Whether an image is mirrored across its columns.</summary>
    public bool Horizontal { get; }

    /// <summary>Whether an image is mirrored across its rows.</summary>
    public bool Vertical { get; }

    /// <inheritdoc />
    public static string Name => "randomFlip";

    /// <inheritdoc />
    public static RandomFlip Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "horizontal") != 0, rebuilding.Whole(settings, "vertical") != 0, rebuilding.Whole(settings, "seed"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("horizontal", Horizontal ? 1 : 0);
        writer.WriteNumber("vertical", Vertical ? 1 : 0);
        writer.WriteNumber("seed", Seed);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of images.</exception>
    private protected override Tensor Varied(Tensor input, Draws draws)
    {
        input.Shape.RequireSpatial(2, null, "A random flip");

        var rows = input.Shape[1];
        var columns = input.Shape[2];
        var channels = input.Shape[3];
        var each = rows * columns * channels;
        var from = input.Values;
        var made = new float[from.Length];

        for (var image = 0; image < input.Shape[0]; image++)
        {
            // Both coins are drawn for every image, so which axes are on moves no other image's draws.
            var across = draws.NextDouble() < 0.5 && Horizontal;
            var down = draws.NextDouble() < 0.5 && Vertical;

            for (var row = 0; row < rows; row++)
            {
                var fromRow = down ? rows - 1 - row : row;

                for (var column = 0; column < columns; column++)
                {
                    var fromColumn = across ? columns - 1 - column : column;

                    from.Slice((image * each) + (((fromRow * columns) + fromColumn) * channels), channels)
                        .CopyTo(made.AsSpan((image * each) + (((row * columns) + column) * channels), channels));
                }
            }
        }

        return Tensor.From(input.Shape, made);
    }
}

/// <summary>Shifts images at random while a network trains, by padding each with noughts and cutting out a window of its own size.</summary>
/// <remarks>
/// The shape is kept: each image is padded by a border on every side and a window of the image's own size is cut out of it at a place
/// drawn for the image, so it moves by up to the border either way and what moves in from outside is nought. Images are laid out with
/// their channels last.
/// </remarks>
public sealed class RandomCrop : Variation, ISaved<RandomCrop>
{
    /// <summary>A crop that shifts an image by up to the given border.</summary>
    /// <param name="pad">How many places an image is padded by on every side, and so how far it can move: nought for none.</param>
    /// <param name="seed">The number the shifts are drawn from.</param>
    /// <exception cref="ArgumentOutOfRangeException">The border is below nought.</exception>
    public RandomCrop(int pad, long seed)
        : base("randomCrop", seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(pad);

        Pad = pad;
    }

    /// <summary>How many places an image is padded by on every side.</summary>
    public int Pad { get; }

    /// <inheritdoc />
    public static string Name => "randomCrop";

    /// <inheritdoc />
    public static RandomCrop Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "pad"), rebuilding.Whole(settings, "seed"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("pad", Pad);
        writer.WriteNumber("seed", Seed);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of images.</exception>
    private protected override Tensor Varied(Tensor input, Draws draws)
    {
        input.Shape.RequireSpatial(2, null, "A random crop");

        if (Pad == 0)
        {
            return input;
        }

        var rows = input.Shape[1];
        var columns = input.Shape[2];
        var channels = input.Shape[3];
        var each = rows * columns * channels;
        var from = input.Values;
        var made = new float[from.Length];
        var places = (2 * Pad) + 1;

        for (var image = 0; image < input.Shape[0]; image++)
        {
            var down = (int)(draws.NextDouble() * places) - Pad;
            var right = (int)(draws.NextDouble() * places) - Pad;

            for (var row = 0; row < rows; row++)
            {
                var fromRow = row + down;

                for (var column = 0; column < columns; column++)
                {
                    var fromColumn = column + right;

                    if (fromRow is >= 0 && fromRow < rows && fromColumn is >= 0 && fromColumn < columns)
                    {
                        from.Slice((image * each) + (((fromRow * columns) + fromColumn) * channels), channels)
                            .CopyTo(made.AsSpan((image * each) + (((row * columns) + column) * channels), channels));
                    }
                }
            }
        }

        return Tensor.From(input.Shape, made);
    }
}

/// <summary>Adds noise to the values of a batch while a network trains: normal, of the spread it is told, and independent value to value.</summary>
/// <remarks>Keras's <c>GaussianNoise</c>, of any shape of batch.</remarks>
public sealed class GaussianNoise : Variation, ISaved<GaussianNoise>
{
    /// <summary>A noise of the given spread.</summary>
    /// <param name="spread">The standard deviation of the noise added to every value; nought for none.</param>
    /// <param name="seed">The number the noise is drawn from.</param>
    /// <exception cref="ArgumentOutOfRangeException">The spread is not a number of nought or more.</exception>
    public GaussianNoise(double spread, long seed)
        : base("gaussianNoise", seed)
    {
        if (spread < 0 || !double.IsFinite(spread))
        {
            throw new ArgumentOutOfRangeException(nameof(spread), spread, "The spread of a noise is a number of nought or more.");
        }

        Spread = spread;
    }

    /// <summary>The standard deviation of the noise.</summary>
    public double Spread { get; }

    /// <inheritdoc />
    public static string Name => "gaussianNoise";

    /// <inheritdoc />
    public static GaussianNoise Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, "spread"), rebuilding.Whole(settings, "seed"));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("spread", Spread);
        writer.WriteNumber("seed", Seed);
    }

    /// <inheritdoc />
    private protected override Tensor Varied(Tensor input, Draws draws)
    {
        if (Spread is 0)
        {
            return input;
        }

        var from = input.Values;
        var made = new float[from.Length];

        for (var at = 0; at < made.Length; at++)
        {
            made[at] = from[at] + (float)(Spread * draws.NextNormal());
        }

        return Tensor.From(input.Shape, made);
    }
}
