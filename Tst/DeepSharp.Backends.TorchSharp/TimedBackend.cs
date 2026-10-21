// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Torch;

// An engine asked for everything, with the time of what it is asked for kept apart: a product with a vector, any other product, and the
// turning and laying out of what it holds.
internal sealed class TimedBackend(ITensorBackend inner) : ITensorBackend, IMultipliesInBlocks
{
    public string Name => inner.Name;

    public double Picks { get; private set; }

    /// <summary>How many products the engine was asked for.</summary>
    public int Multiplications { get; private set; }

    public double Products { get; private set; }

    public double Moves { get; private set; }

    public Tensor Add(Tensor left, Tensor right) => inner.Add(left, right);

    public Tensor Subtract(Tensor left, Tensor right) => inner.Subtract(left, right);

    public Tensor Multiply(Tensor left, Tensor right) => inner.Multiply(left, right);

    public Tensor MatMul(Tensor left, Tensor right)
    {
        Multiplications++;

        var started = Stopwatch.GetTimestamp();
        var made = inner.MatMul(left, right);
        var took = Stopwatch.GetElapsedTime(started).TotalMilliseconds;

        if (left.Shape[0] == 1 || left.Shape[1] == 1 || right.Shape[1] == 1)
        {
            Picks += took;
        }
        else
        {
            Products += took;
        }

        return made;
    }

    // A product over blocks is passed on to the engine that has it, or worked out there from the products it has: wrapped, it would otherwise be
    // asked for the products every engine has and cost the square of its blocks.
    Tensor IMultipliesInBlocks.MatMulInBlocks(Tensor left, Tensor right, BlockProduct how)
    {
        Multiplications++;

        var started = Stopwatch.GetTimestamp();
        var made = inner.MatMulInBlocks(left, right, how);

        Products += Stopwatch.GetElapsedTime(started).TotalMilliseconds;

        return made;
    }

    public Tensor Transpose(Tensor matrix) => Moved(() => inner.Transpose(matrix));

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

    public Tensor Reshape(Tensor values, Shape shape) => Moved(() => inner.Reshape(values, shape));

    public Tensor Unfold(Tensor images, Window window) => inner.Unfold(images, window);

    public Tensor Fold(Tensor patches, Shape images, Window window) => inner.Fold(patches, images, window);

    private Tensor Moved(Func<Tensor> move)
    {
        var started = Stopwatch.GetTimestamp();
        var made = move();

        Moves += Stopwatch.GetElapsedTime(started).TotalMilliseconds;

        return made;
    }
}
