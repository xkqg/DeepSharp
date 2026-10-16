// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>Keeps every value above nothing and makes the rest nothing: the rectifier.</summary>
public sealed class Relu : Layer, ISaved<Relu>
{
    /// <inheritdoc />
    public static string Name => "relu";

    /// <inheritdoc />
    public static Relu Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Relu(input);
}

/// <summary>Bends every value between minus one and one along the hyperbolic tangent.</summary>
public sealed class Tanh : Layer, ISaved<Tanh>
{
    /// <inheritdoc />
    public static string Name => "tanh";

    /// <inheritdoc />
    public static Tanh Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Tanh(input);
}

/// <summary>Bends every value between nothing and one along the logistic curve.</summary>
/// <remarks>
/// A network trained with <see cref="BinaryCrossEntropy"/> ends without one: the loss takes the logits and applies the
/// curve itself, which is both steadier and what PyTorch does, and a prediction goes through it on its way out. A stack
/// that ends in one is refused where it is compiled with such a loss, and a description in Keras's words that ends in one
/// leaves it out as it is compiled.
/// </remarks>
public sealed class Sigmoid : Layer, ISaved<Sigmoid>
{
    /// <inheritdoc />
    public static string Name => "sigmoid";

    /// <inheritdoc />
    public static Sigmoid Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Sigmoid(input);
}

/// <summary>
/// Weighs every value by the share of a normal curve below it: the Gaussian error linear unit, smooth where the rectifier bends.
/// </summary>
/// <remarks>
/// Exactly, the value times half of one plus the error function of the value over the root of two — PyTorch's and Keras's GELU
/// unless either is told to approximate it. Told to, it is the hyperbolic tangent's approximation both write: half the value
/// times one plus the tangent of the root of two over π times the value plus 0.044715 of its cube. Which one a layer is, is
/// written into its file, so a network read back makes the numbers it made.
/// </remarks>
public sealed class Gelu : Layer, ISaved<Gelu>
{
    private const string ApproximateKey = "approximate";

    /// <summary>The GELU, exactly or by the hyperbolic tangent's approximation.</summary>
    /// <param name="approximate">Whether it is the hyperbolic tangent's approximation; it is exact, unless said.</param>
    public Gelu(bool approximate = false) => Approximate = approximate;

    /// <summary>Whether it is the hyperbolic tangent's approximation rather than the exact curve.</summary>
    public bool Approximate { get; }

    /// <inheritdoc />
    public static string Name => "gelu";

    /// <inheritdoc />
    public static Gelu Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Holds(settings, ApproximateKey));
    }

    /// <inheritdoc />
    /// <remarks>Whether it approximates, always: the exact curve and its approximation make different numbers.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBoolean(ApproximateKey, Approximate);
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        var backend = pass.Backend;
        var half = backend.Scale(input, backend.Scalar(0.5));

        if (!Approximate)
        {
            return backend.Add(half, backend.Multiply(half, backend.Erf(backend.Scale(input, backend.Scalar(1 / Math.Sqrt(2))))));
        }

        var cubed = backend.Multiply(backend.Multiply(input, input), input);
        var inner = backend.Scale(backend.Add(input, backend.Scale(cubed, backend.Scalar(0.044715))), backend.Scalar(Math.Sqrt(2 / Math.PI)));

        return backend.Add(half, backend.Multiply(half, backend.Tanh(inner)));
    }
}

/// <summary>Weighs every value by its own logistic curve: the sigmoid linear unit, also called swish.</summary>
/// <remarks>PyTorch's <c>SiLU</c> and Keras's <c>silu</c>, the value times the sigmoid of it.</remarks>
public sealed class Silu : Layer, ISaved<Silu>
{
    /// <inheritdoc />
    public static string Name => "silu";

    /// <inheritdoc />
    public static Silu Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Multiply(input, pass.Backend.Sigmoid(input));
}

/// <summary>Weighs every value by the hyperbolic tangent of its softplus: Misra's mish.</summary>
/// <remarks>PyTorch's <c>Mish</c> and Keras's <c>mish</c>, the value times the tangent of the logarithm of one plus its exponential.</remarks>
public sealed class Mish : Layer, ISaved<Mish>
{
    /// <inheritdoc />
    public static string Name => "mish";

    /// <inheritdoc />
    public static Mish Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Multiply(input, pass.Backend.Tanh(pass.Backend.Softplus(input)));
}

/// <summary>Keeps every value above nothing, and lets a share of every other through: the leaky rectifier.</summary>
/// <remarks>
/// At nought it slopes as it does below it, as PyTorch's does, so a gradient reaching a value of exactly nought is the slope's
/// share of it. A hundredth unless said, as PyTorch leaves it; Keras's layer starts at three tenths and its activation word at
/// two, and a description in Keras's words says which.
/// </remarks>
public sealed class LeakyRelu : Layer, ISaved<LeakyRelu>
{
    private const string SlopeKey = "negativeSlope";

    /// <summary>The leaky rectifier with the given slope below nothing.</summary>
    /// <param name="negativeSlope">The share of a value below nothing let through: nothing at least, a hundredth unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">The slope is not a finite number of nothing or more.</exception>
    public LeakyRelu(double negativeSlope = 0.01) =>
        NegativeSlope = double.IsFinite(negativeSlope) && negativeSlope >= 0
            ? negativeSlope
            : throw new ArgumentOutOfRangeException(nameof(negativeSlope), negativeSlope, "The share of a value below nothing that is let through is a number of nothing or more.");

    /// <summary>The share of a value below nothing let through.</summary>
    public double NegativeSlope { get; }

    /// <inheritdoc />
    public static string Name => "leakyRelu";

    /// <inheritdoc />
    public static LeakyRelu Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, SlopeKey));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(SlopeKey, NegativeSlope);
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        var backend = pass.Backend;

        // The value times the slope where it is at or below nothing and one above it: the step sends nothing back, so the
        // gradient at nought is the slope, as PyTorch's is.
        return backend.Multiply(input, backend.Add(backend.Scale(backend.Positive(input), backend.Scalar(1 - NegativeSlope)), backend.Fill(input.Shape, (float)NegativeSlope)));
    }
}

/// <summary>
/// Keeps every value above nothing, and lets a share of every other through by a number it learns: the parametric rectifier.
/// </summary>
/// <remarks>
/// PyTorch's <c>PReLU</c>: one number for every value, or one for each channel — the last axis, here where the channels come last.
/// Its slot is PyTorch's, <c>weight</c>, and starts at a quarter, as PyTorch starts it. At nought it slopes as it does below it.
/// </remarks>
public sealed class PRelu : Layer, ISaved<PRelu>
{
    private const string ParametersKey = "parameters";

    /// <summary>The parametric rectifier, its shares starting at a quarter.</summary>
    /// <param name="parameters">How many shares it learns: one for every value, or as many as the channels of the last axis.</param>
    /// <exception cref="ArgumentOutOfRangeException">It learns fewer than one.</exception>
    public PRelu(int parameters = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(parameters, 1);

        Weight = AddParameter("weight", Tensor.From(new Shape(parameters), [.. Enumerable.Repeat(0.25f, parameters)]));
    }

    /// <summary>The shares of a value below nothing let through: one, or one for each channel.</summary>
    public Parameter Weight { get; }

    /// <summary>How many shares it learns.</summary>
    public int Count => Weight.Value.Shape[0];

    /// <inheritdoc />
    public static string Name => "prelu";

    /// <inheritdoc />
    public static PRelu Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, ParametersKey));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(ParametersKey, Count);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">It learns a share for each channel, and the last axis holds another number of them.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (Count > 1 && (input.Shape.Rank < 2 || input.Shape[input.Shape.Rank - 1] != Count))
        {
            throw new ArgumentException($"A parametric rectifier of {Count} shares takes a batch whose last axis holds that many channels, and was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var rows = backend.Reshape(input, new Shape(input.Shape.Count / Count, Count));
        var above = backend.Positive(rows);
        var shares = backend.RowsOf(Weight.Value, rows.Shape[0]);

        // One where the value is above nothing, its share where it is not: the step sends nothing back, so a share learns from
        // the values at or below nothing alone, and the gradient at nought is the share, as PyTorch's is.
        var slopes = backend.Add(above, backend.Multiply(backend.Subtract(backend.Fill(rows.Shape, 1f), above), shares));

        return backend.Reshape(backend.Multiply(rows, slopes), input.Shape);
    }
}

/// <summary>Keeps every value above nothing, and bends every other towards minus alpha along an exponential: the exponential linear unit.</summary>
/// <remarks>
/// PyTorch's <c>ELU</c> and Keras's: alpha times one less than the exponential of a value at or below nothing. The exponential is
/// only taken of what is at or below nothing, so a large value overflows nothing, and at nought it slopes by alpha, as PyTorch's does.
/// </remarks>
public sealed class Elu : Layer, ISaved<Elu>
{
    private const string AlphaKey = "alpha";

    /// <summary>The exponential linear unit with the given alpha.</summary>
    /// <param name="alpha">What the values below nothing bend towards, below nothing: a finite number of nothing or more, one unless said.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not one.</exception>
    public Elu(double alpha = 1) =>
        Alpha = double.IsFinite(alpha) && alpha >= 0
            ? alpha
            : throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "An exponential linear unit bends towards minus alpha, and alpha is a number of nothing or more.");

    /// <summary>What the values below nothing bend towards, below nothing.</summary>
    public double Alpha { get; }

    /// <inheritdoc />
    public static string Name => "elu";

    /// <inheritdoc />
    public static Elu Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, AlphaKey));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(AlphaKey, Alpha);
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => Exponentially(pass.Backend, input, Alpha, 1);

    /// <summary>
    /// The value where it is above nothing, and alpha times one less than its exponential where it is not, all times a scale:
    /// the step sends nothing back, and the exponential is of the value less its rectified self — never above nothing, so
    /// nothing overflows — whose slope at nought is one, so the gradient there is the scale times alpha, as PyTorch's is.
    /// </summary>
    internal static Tensor Exponentially(ITensorBackend backend, Tensor input, double alpha, double scale)
    {
        var above = backend.Positive(input);
        var bent = backend.Scale(backend.Subtract(backend.Exp(backend.Subtract(input, backend.Relu(input))), backend.Fill(input.Shape, 1f)), backend.Scalar(alpha));
        var unit = backend.Add(backend.Multiply(above, input), backend.Multiply(backend.Subtract(backend.Fill(input.Shape, 1f), above), bent));

        return scale is 1d ? unit : backend.Scale(unit, backend.Scalar(scale));
    }
}

/// <summary>The exponential linear unit at the alpha and the scale that keep a network's values normalised: the self-normalising one.</summary>
/// <remarks>
/// Klambauer et al.'s SELU, at their constants as PyTorch and Keras write them: a scale of 1.0507009873554805 and an alpha of
/// 1.6732632423543772. Neither is a setting, so it has none.
/// </remarks>
public sealed class Selu : Layer, ISaved<Selu>
{
    private const double Alpha = 1.6732632423543772;

    private const double Scale = 1.0507009873554805;

    /// <inheritdoc />
    public static string Name => "selu";

    /// <inheritdoc />
    public static Selu Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => Elu.Exponentially(pass.Backend, input, Alpha, Scale);
}

/// <summary>Turns the last axis of every example into shares of one whole: the softmax, as a layer.</summary>
/// <remarks>
/// The exponential of the log-softmax, as a prediction of a loss of classes is made. A network compiled with a loss that applies
/// the softmax itself — <see cref="CrossEntropy"/> — does not end in one: a stack that does is refused where it is compiled, and a
/// description in Keras's words that ends in one leaves it out, as it does a sigmoid before a binary cross-entropy.
/// </remarks>
public sealed class Softmax : Layer, ISaved<Softmax>
{
    /// <inheritdoc />
    public static string Name => "softmax";

    /// <inheritdoc />
    public static Softmax Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is no batch: it has no axis beside its examples.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank < 2)
        {
            throw new ArgumentException($"A softmax shares out the last axis of every example of a batch, and a {input.Shape} tensor is not a batch of any.", nameof(input));
        }

        var backend = pass.Backend;
        var width = input.Shape[input.Shape.Rank - 1];
        var rows = backend.Reshape(input, new Shape(input.Shape.Count / width, width));

        return backend.Reshape(backend.Exp(backend.LogSoftmax(rows)), input.Shape);
    }
}
