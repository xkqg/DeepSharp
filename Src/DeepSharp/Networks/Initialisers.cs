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
