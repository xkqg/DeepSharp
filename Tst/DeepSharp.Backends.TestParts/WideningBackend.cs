// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// An engine outside the library that keeps its work in a narrower width than a single number, as far as a test can see it: it hands
/// every operation to the engine it wraps and says, through <see cref="IWidens"/>, that it widens — counting each tensor it is asked
/// to widen, so a test sees which of the shares of a gradient are taken to full width and which are not.
/// </summary>
/// <param name="inner">The engine that works everything out.</param>
/// <param name="precision">The width it says it keeps its work in; nothing for an engine that says none, as one at full width does.</param>
public sealed class WideningBackend(ITensorBackend inner, string? precision = null) : ITensorBackend, IWidens, INamesItsPrecision
{
    private readonly List<Shape> _widened = [];
    private readonly List<float> _filled = [];

    /// <inheritdoc />
    public string Name => inner.Name;

    /// <inheritdoc />
    string? INamesItsPrecision.Precision => precision;

    /// <summary>The value of every tensor of one value it was asked to fill, in the order it was asked.</summary>
    public IReadOnlyList<float> Filled => _filled;

    /// <summary>The shape of every tensor it was asked to widen, in the order it was asked.</summary>
    public IReadOnlyList<Shape> Widened => _widened;

    /// <inheritdoc />
    Tensor IWidens.Widened(Tensor values)
    {
        _widened.Add(values.Shape);

        return values;
    }

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => inner.Add(left, right);

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) => inner.Subtract(left, right);

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) => inner.Multiply(left, right);

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right) => inner.MatMul(left, right);

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix) => inner.Transpose(matrix);

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row) => inner.AddRow(matrix, row);

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix) => inner.SumRows(matrix);

    /// <inheritdoc />
    public Tensor Mean(Tensor values) => inner.Mean(values);

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor) => inner.Scale(values, factor);

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        if (shape.Rank == 0)
        {
            _filled.Add(value);
        }

        return inner.Fill(shape, value);
    }

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => inner.Relu(values);

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => inner.Positive(values);

    /// <inheritdoc />
    public Tensor FirstLargest(Tensor matrix) => inner.FirstLargest(matrix);

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => inner.Tanh(values);

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => inner.Sigmoid(values);

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => inner.Exp(values);

    /// <inheritdoc />
    public Tensor Log(Tensor values) => inner.Log(values);

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => inner.Sqrt(values);

    /// <inheritdoc />
    public Tensor Softplus(Tensor values) => inner.Softplus(values);

    /// <inheritdoc />
    public Tensor Sin(Tensor values) => inner.Sin(values);

    /// <inheritdoc />
    public Tensor Erf(Tensor values) => inner.Erf(values);

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right) => inner.Divide(left, right);

    /// <inheritdoc />
    public Tensor LogSoftmax(Tensor matrix) => inner.LogSoftmax(matrix);

    /// <inheritdoc />
    public Tensor Reshape(Tensor values, Shape shape) => inner.Reshape(values, shape);

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window) => inner.Unfold(images, window);

    /// <inheritdoc />
    public Tensor Fold(Tensor patches, Shape images, Window window) => inner.Fold(patches, images, window);
}
