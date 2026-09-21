// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// A shape and the values that fill it.
/// </summary>
/// <remarks>
/// A tensor knows nothing about arithmetic. Everything a network does to one is done by an
/// <see cref="ITensorBackend"/>, which is what lets the same model run its heavy work somewhere else later
/// without a line of the model changing. The values are laid out row-major — the last axis moves fastest —
/// and a tensor never changes once it exists, so handing one to two layers is safe.
/// </remarks>
public sealed class Tensor
{
    private readonly float[] _values;

    private Tensor(Shape shape, float[] values)
    {
        Shape = shape;
        _values = values;
    }

    /// <summary>The axes this tensor is laid out along.</summary>
    public Shape Shape { get; }

    /// <summary>The values, row-major: the last axis moves fastest.</summary>
    public ReadOnlySpan<float> Values => _values;

    /// <summary>A tensor of the given shape with every value at zero.</summary>
    /// <param name="shape">The axes to lay the values out along.</param>
    public static Tensor Zeros(Shape shape) => new(shape, new float[shape.Count]);

    /// <summary>A tensor of the given shape holding a copy of the given values, row-major.</summary>
    /// <param name="shape">The axes to lay the values out along.</param>
    /// <param name="values">Exactly as many values as the shape holds.</param>
    /// <exception cref="ArgumentException">There are more or fewer values than the shape holds.</exception>
    public static Tensor From(Shape shape, ReadOnlySpan<float> values)
    {
        if (values.Length != shape.Count)
        {
            throw new ArgumentException(
                $"A {shape} tensor holds {shape.Count} values and {values.Length} were given.", nameof(values));
        }

        // Copied, not kept: a caller who reuses a scratch buffer would otherwise rewrite a tensor that has
        // already been handed to a layer, and that shows up as a wrong answer with no trace back to here.
        return new Tensor(shape, values.ToArray());
    }

    /// <summary>Builds a tensor from values already laid out row-major, without copying them.</summary>
    /// <remarks>For a backend that has just produced a buffer nobody else holds.</remarks>
    internal static Tensor Wrap(Shape shape, float[] values) => new(shape, values);

    /// <summary>The tensor as it is spoken: <c>Tensor 2x3</c>.</summary>
    public override string ToString() => $"Tensor {Shape}";
}
