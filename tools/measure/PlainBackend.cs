// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Measure;

/// <summary>
/// The light engine as an engine of somebody else's is asked for everything: every operation one call, and none of the shortcuts
/// the light engine takes for itself — a step of Adam in one pass, a weight's gradient without the turned copy. The same
/// arithmetic, so a run on it is the bits a run on the light engine makes, and what a run costs on one and on the other is
/// what the shortcuts are worth.
/// </summary>
/// <param name="inner">The engine the arithmetic runs on.</param>
internal sealed class PlainBackend(ITensorBackend inner) : ITensorBackend
{
    public string Name => "plain";

    public Tensor Add(Tensor left, Tensor right) => inner.Add(left, right);

    public Tensor Subtract(Tensor left, Tensor right) => inner.Subtract(left, right);

    public Tensor Multiply(Tensor left, Tensor right) => inner.Multiply(left, right);

    public Tensor MatMul(Tensor left, Tensor right) => inner.MatMul(left, right);

    public Tensor Transpose(Tensor matrix) => inner.Transpose(matrix);

    public Tensor AddRow(Tensor matrix, Tensor row) => inner.AddRow(matrix, row);

    public Tensor SumRows(Tensor matrix) => inner.SumRows(matrix);

    public Tensor Mean(Tensor values) => inner.Mean(values);

    public Tensor Scale(Tensor values, Tensor factor) => inner.Scale(values, factor);

    public Tensor Fill(Shape shape, float value) => inner.Fill(shape, value);

    public Tensor Relu(Tensor values) => inner.Relu(values);

    public Tensor Positive(Tensor values) => inner.Positive(values);

    public Tensor FirstLargest(Tensor matrix) => inner.FirstLargest(matrix);

    public Tensor Tanh(Tensor values) => inner.Tanh(values);

    public Tensor Sigmoid(Tensor values) => inner.Sigmoid(values);

    public Tensor Exp(Tensor values) => inner.Exp(values);

    public Tensor Log(Tensor values) => inner.Log(values);

    public Tensor Sqrt(Tensor values) => inner.Sqrt(values);

    public Tensor Softplus(Tensor values) => inner.Softplus(values);

    public Tensor Sin(Tensor values) => inner.Sin(values);

    public Tensor Erf(Tensor values) => inner.Erf(values);

    public Tensor Divide(Tensor left, Tensor right) => inner.Divide(left, right);

    public Tensor LogSoftmax(Tensor matrix) => inner.LogSoftmax(matrix);

    public Tensor Reshape(Tensor values, Shape shape) => inner.Reshape(values, shape);

    public Tensor Unfold(Tensor images, Window window) => inner.Unfold(images, window);

    public Tensor Fold(Tensor patches, Shape images, Window window) => inner.Fold(patches, images, window);
}
