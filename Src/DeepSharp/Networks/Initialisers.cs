// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How many values a layer's numbers are read against and how many they feed: what an initialiser scales by.</summary>
/// <param name="In">How many inputs reach each output: a linear layer's inputs, a convolution's window times its channels in.</param>
/// <param name="Out">How many outputs each input reaches: a linear layer's outputs, a convolution's window times its channels out.</param>
public readonly record struct Fans(int In, int Out);

/// <summary>What a layer's numbers start at before it has learned anything.</summary>
/// <remarks>
/// Drawn from the draws a layer is handed — out of the run's stream, for a network described in Keras's words and built at
/// its first fit; out of whatever stream a person draws from, for one written as code — each layer's for the place it
/// stands at, so the same seed starts the same network and adding a layer never moves another layer's start. The linear layer and the convolution start as PyTorch starts them —
/// <see cref="KaimingUniform"/> for the weights, <see cref="FanInUniform"/> for the bias — and any other can be named.
/// </remarks>
public abstract class Initialiser
{
    /// <summary>Draws a starting tensor.</summary>
    /// <param name="shape">The slot's shape.</param>
    /// <param name="fans">How many values the slot is read against and feeds.</param>
    /// <param name="draws">The draws for this slot.</param>
    /// <returns>The tensor to start at.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A fan is below one.</exception>
    public Tensor Draw(Shape shape, Fans fans, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(fans.In, 1, nameof(fans));
        ArgumentOutOfRangeException.ThrowIfLessThan(fans.Out, 1, nameof(fans));

        return Drawn(shape, fans, draws);
    }

    /// <summary>Draws a starting tensor, with fans that are one.</summary>
    /// <param name="shape">The slot's shape.</param>
    /// <param name="fans">How many values the slot is read against and feeds, each at least one.</param>
    /// <param name="draws">The draws for this slot.</param>
    /// <returns>The tensor to start at.</returns>
    protected abstract Tensor Drawn(Shape shape, Fans fans, Draws draws);

    /// <summary>A tensor of values, each made by the same draw.</summary>
    /// <param name="shape">The shape.</param>
    /// <param name="draw">What makes one value.</param>
    /// <returns>The tensor.</returns>
    private protected static Tensor Filled(Shape shape, Func<double> draw)
    {
        var values = new float[shape.Count];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = (float)draw();
        }

        return Tensor.From(shape, values);
    }

    /// <summary>A spread: a finite number above nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not.</exception>
    private protected static double RequireDeviation(double deviation) =>
        double.IsFinite(deviation) && deviation > 0
            ? deviation
            : throw new ArgumentOutOfRangeException(nameof(deviation), deviation, "A spread is a finite number above nothing.");

    /// <summary>A mean, or a value: a finite number.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not.</exception>
    private protected static double RequireMean(double mean) =>
        double.IsFinite(mean) ? mean : throw new ArgumentOutOfRangeException(nameof(mean), mean, "A mean is a finite number.");

    /// <summary>A tensor of values spread evenly between minus a bound and the bound.</summary>
    /// <param name="shape">The shape.</param>
    /// <param name="bound">How far either side of nothing the values reach.</param>
    /// <param name="draws">Where the values come from.</param>
    /// <returns>The tensor.</returns>
    private protected static Tensor Uniform(Shape shape, double bound, Draws draws)
    {
        var values = new float[shape.Count];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = (float)(bound * ((2 * draws.NextDouble()) - 1));
        }

        return Tensor.From(shape, values);
    }
}

/// <summary>
/// He's uniform initialisation, for a slope it is written for: evenly between minus and plus the root of six over one plus
/// the slope squared, over the inputs.
/// </summary>
/// <remarks>
/// With PyTorch's slope of the root of five — the start it gives every linear layer and convolution — the bound is one over
/// the root of the inputs; with a slope of nothing, the root of six over the inputs, He's bound for a layer a rectifier
/// follows.
/// </remarks>
public sealed class KaimingUniform : Initialiser
{
    /// <summary>He's initialisation for a leaky rectifier of the given slope.</summary>
    /// <param name="negativeSlope">The slope below nothing: the root of five, unless said, as PyTorch's linear layer has it.</param>
    /// <exception cref="ArgumentOutOfRangeException">The slope is not a number.</exception>
    public KaimingUniform(double negativeSlope = 2.23606797749979)
    {
        if (!double.IsFinite(negativeSlope))
        {
            throw new ArgumentOutOfRangeException(nameof(negativeSlope), negativeSlope, "A slope is a number.");
        }

        NegativeSlope = negativeSlope;
    }

    /// <summary>The slope below nothing it is written for.</summary>
    public double NegativeSlope { get; }

    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) =>
        Uniform(shape, Math.Sqrt(6 / ((1 + (NegativeSlope * NegativeSlope)) * fans.In)), draws);
}

/// <summary>Evenly between minus and plus one over the root of the inputs: how PyTorch starts a linear layer's or a convolution's bias.</summary>
public sealed class FanInUniform : Initialiser
{
    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Uniform(shape, 1 / Math.Sqrt(fans.In), draws);
}

/// <summary>Glorot's uniform initialisation: evenly between minus and plus the root of six over the inputs and outputs together, as Keras starts a layer.</summary>
public sealed class GlorotUniform : Initialiser
{
    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Uniform(shape, Math.Sqrt(6.0 / (fans.In + fans.Out)), draws);
}

/// <summary>Nothing, everywhere.</summary>
public sealed class Zeros : Initialiser
{
    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Tensor.Zeros(shape);
}

/// <summary>Numbers from a normal distribution: Keras's <c>RandomNormal</c>, a mean of nought and a spread of a twentieth unless said.</summary>
/// <remarks>
/// Drawn through <see cref="Draws.NextNormal"/>, so a layer started this way is started again from the same seed to within the
/// last bit of the machine's logarithm and cosine.
/// </remarks>
public sealed class Normal : Initialiser
{
    /// <summary>A normal start of the given mean and spread.</summary>
    /// <param name="deviation">The spread: a number above nothing; a twentieth, unless said, as Keras leaves it.</param>
    /// <param name="mean">The mean; nought, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The spread is not a finite number above nothing, or the mean is not a finite number.</exception>
    public Normal(double deviation = 0.05, double mean = 0)
    {
        Deviation = RequireDeviation(deviation);
        Mean = RequireMean(mean);
    }

    /// <summary>The spread.</summary>
    public double Deviation { get; }

    /// <summary>The mean.</summary>
    public double Mean { get; }

    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Filled(shape, () => Mean + (Deviation * draws.NextNormal()));
}

/// <summary>
/// Numbers from a normal distribution cut two spreads either side of its mean: Keras's <c>TruncatedNormal</c>, a mean of nought
/// and a spread of a twentieth unless said.
/// </summary>
/// <remarks>
/// A number beyond two spreads from the mean is drawn again, as Keras draws it, so the cut is relative to the spread given —
/// not the absolute bounds PyTorch's <c>trunc_normal_</c> takes. The spread of what is left is 0.8796 of the spread given.
/// </remarks>
public sealed class TruncatedNormal : Initialiser
{
    /// <summary>A truncated normal start of the given mean and spread.</summary>
    /// <param name="deviation">The spread before the cut: a number above nothing; a twentieth, unless said.</param>
    /// <param name="mean">The mean; nought, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The spread is not a finite number above nothing, or the mean is not a finite number.</exception>
    public TruncatedNormal(double deviation = 0.05, double mean = 0)
    {
        Deviation = RequireDeviation(deviation);
        Mean = RequireMean(mean);
    }

    /// <summary>The spread before the cut.</summary>
    public double Deviation { get; }

    /// <summary>The mean.</summary>
    public double Mean { get; }

    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Filled(shape, () =>
    {
        double standard;

        do
        {
            standard = draws.NextNormal();
        }
        while (Math.Abs(standard) > 2);

        return Mean + (Deviation * standard);
    });
}

/// <summary>
/// A matrix whose shorter side is orthonormal, times a gain: Keras's <c>Orthogonal</c>. The columns are orthonormal where the
/// matrix has more rows than columns, the rows where it has more columns.
/// </summary>
/// <remarks>
/// The matrix is the one a QR factorisation of normal numbers gives with the signs of its diagonal put right — here worked out
/// by Gram–Schmidt over the columns, twice, in double precision — so that it is the same matrix whichever way it is worked out
/// and its columns stay orthogonal to within a millionth. Only a slot of two axes has one.
/// </remarks>
public sealed class Orthogonal : Initialiser
{
    /// <summary>An orthogonal start scaled by a gain.</summary>
    /// <param name="gain">What the matrix is multiplied by: a number above nothing; one, unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The gain is not a finite number above nothing.</exception>
    public Orthogonal(double gain = 1)
    {
        Gain = double.IsFinite(gain) && gain > 0
            ? gain
            : throw new ArgumentOutOfRangeException(nameof(gain), gain, "A gain is a finite number above nothing.");
    }

    /// <summary>What the matrix is multiplied by.</summary>
    public double Gain { get; }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The slot has not two axes.</exception>
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws)
    {
        if (shape.Rank != 2)
        {
            throw new ArgumentException($"An orthogonal start is a matrix, and a slot of {shape} is not one.", nameof(shape));
        }

        int rows = shape[0], columns = shape[1];
        int length = Math.Max(rows, columns), width = Math.Min(rows, columns);
        var vectors = new double[width][];

        // The columns of a matrix of normal numbers, as many rows long as the longer side.
        for (var column = 0; column < width; column++)
        {
            vectors[column] = new double[length];
        }

        for (var row = 0; row < length; row++)
        {
            for (var column = 0; column < width; column++)
            {
                vectors[column][row] = draws.NextNormal();
            }
        }

        for (var column = 0; column < width; column++)
        {
            // Twice is enough: one pass leaves what rounding left of the columns before it, and a second takes it out.
            for (var pass = 0; pass < 2; pass++)
            {
                for (var before = 0; before < column; before++)
                {
                    var dot = 0d;

                    for (var at = 0; at < length; at++)
                    {
                        dot += vectors[column][at] * vectors[before][at];
                    }

                    for (var at = 0; at < length; at++)
                    {
                        vectors[column][at] -= dot * vectors[before][at];
                    }
                }
            }

            var norm = Math.Sqrt(vectors[column].Sum(value => value * value));

            for (var at = 0; at < length; at++)
            {
                vectors[column][at] /= norm;
            }
        }

        var values = new float[rows * columns];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                // More rows than columns: the matrix as it is. More columns than rows: its transpose.
                values[(row * columns) + column] = (float)(Gain * (rows >= columns ? vectors[column][row] : vectors[row][column]));
            }
        }

        return Tensor.From(shape, values);
    }
}

/// <summary>One, everywhere.</summary>
public sealed class Ones : Initialiser
{
    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Filled(shape, () => 1d);
}

/// <summary>The same number, everywhere: Keras's <c>Constant</c>.</summary>
public sealed class Constant : Initialiser
{
    /// <summary>A start at the given number.</summary>
    /// <param name="value">The number.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number.</exception>
    public Constant(double value) => Value = RequireMean(value);

    /// <summary>The number.</summary>
    public double Value { get; }

    /// <inheritdoc />
    protected override Tensor Drawn(Shape shape, Fans fans, Draws draws) => Filled(shape, () => Value);
}
