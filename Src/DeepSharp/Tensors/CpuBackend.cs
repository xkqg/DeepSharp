// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;

namespace DeepSharp.Tensors;

/// <summary>
/// The backend that ships: this machine's own vector registers, through .NET's tensor primitives.
/// </summary>
/// <remarks>
/// There is no native library behind this and nothing to install. That is the whole point — a model built
/// here runs inside an ordinary .NET application, on any operating system the runtime reaches, without a
/// hundred megabytes of platform-specific binaries travelling with it.
/// <para>
/// The tensor primitives hold no matrix product, so that one is written here. Every total — a product's inner sum, a
/// column's sum, a mean — is kept in double precision while it is added up: a single-precision total stops taking small
/// values in once it is large, and ten million tenths came to a mean of 0.1087937 rather than a tenth.
/// </para>
/// </remarks>
public sealed class CpuBackend : ITensorBackend
{
    /// <inheritdoc />
    public string Name => "cpu";

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Elementwise(left, right, TensorPrimitives.Add, nameof(Add));

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) =>
        Elementwise(left, right, TensorPrimitives.Subtract, nameof(Subtract));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) =>
        Elementwise(left, right, TensorPrimitives.Multiply, nameof(Multiply));

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right)
    {
        RequireMatrix(left, nameof(MatMul), nameof(left));
        RequireMatrix(right, nameof(MatMul), nameof(right));

        var rows = left.Shape[0];
        var inner = left.Shape[1];
        var columns = right.Shape[1];

        if (right.Shape[0] != inner)
        {
            throw new ArgumentException(
                $"MatMul needs the left matrix as wide as the right one is tall, and was given {left.Shape} and {right.Shape}.",
                nameof(right));
        }

        var a = left.Values;
        var b = right.Values;
        var result = new float[rows * columns];
        var totals = new double[columns];

        for (var row = 0; row < rows; row++)
        {
            Array.Clear(totals);

            // Row by row of the right, so both are read in the order they are laid out.
            for (var step = 0; step < inner; step++)
            {
                var weight = (double)a[(row * inner) + step];
                var across = b.Slice(step * columns, columns);

                for (var column = 0; column < columns; column++)
                {
                    totals[column] += weight * across[column];
                }
            }

            for (var column = 0; column < columns; column++)
            {
                result[(row * columns) + column] = (float)totals[column];
            }
        }

        return Tensor.Wrap(new Shape(rows, columns), result);
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        RequireMatrix(matrix, nameof(Transpose), nameof(matrix));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var values = matrix.Values;
        var result = new float[values.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                result[(column * rows) + row] = values[(row * columns) + column];
            }
        }

        return Tensor.Wrap(new Shape(columns, rows), result);
    }

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        RequireMatrix(matrix, nameof(AddRow), nameof(matrix));
        ArgumentNullException.ThrowIfNull(row);

        if (row.Shape.Rank != 1)
        {
            throw new ArgumentException($"AddRow needs a row, a tensor of one axis, and was given a {row.Shape} one.", nameof(row));
        }

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];

        if (row.Shape[0] != columns)
        {
            throw new ArgumentException(
                $"AddRow needs a row as long as the matrix is wide, and was given {matrix.Shape} and {row.Shape}.", nameof(row));
        }

        var result = new float[matrix.Values.Length];

        for (var at = 0; at < rows; at++)
        {
            TensorPrimitives.Add(matrix.Values.Slice(at * columns, columns), row.Values, result.AsSpan(at * columns, columns));
        }

        return Tensor.Wrap(matrix.Shape, result);
    }

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix)
    {
        RequireMatrix(matrix, nameof(SumRows), nameof(matrix));

        var rows = matrix.Shape[0];
        var columns = matrix.Shape[1];
        var values = matrix.Values;
        var totals = new double[columns];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                totals[column] += values[(row * columns) + column];
            }
        }

        return Tensor.Wrap(new Shape(columns), [.. totals.Select(total => (float)total)]);
    }

    /// <inheritdoc />
    public Tensor Mean(Tensor values)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Shape.Count == 0)
        {
            throw new ArgumentException($"A mean needs at least one value, and a {values.Shape} tensor holds none.", nameof(values));
        }

        var total = 0d;

        foreach (var value in values.Values)
        {
            total += value;
        }

        return Tensor.Wrap(new Shape(), [(float)(total / values.Shape.Count)]);
    }

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(factor);

        if (factor.Shape.Rank != 0)
        {
            throw new ArgumentException(
                $"Scale multiplies by one value, a tensor with no axes, and was given a {factor.Shape} one.", nameof(factor));
        }

        var result = new float[values.Values.Length];
        TensorPrimitives.Multiply(values.Values, factor.Values[0], result);

        return Tensor.Wrap(values.Shape, result);
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value)
    {
        var result = new float[shape.Count];
        Array.Fill(result, value);

        return Tensor.Wrap(shape, result);
    }

    /// <summary>The shape check the value-by-value operations share, and the buffer they write into.</summary>
    private static Tensor Elementwise(
        Tensor left,
        Tensor right,
        ElementwiseOperation operation,
        string name)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Shape != right.Shape)
        {
            throw new ArgumentException(
                $"{name} needs two tensors of the same shape, and was given {left.Shape} and {right.Shape}.",
                nameof(right));
        }

        var result = new float[left.Shape.Count];
        operation(left.Values, right.Values, result);
        return Tensor.Wrap(left.Shape, result);
    }

    /// <summary>Refuses a tensor that is not a matrix, in the words of the operation that needs one.</summary>
    private static void RequireMatrix(Tensor tensor, string operation, string parameter)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameter);

        if (tensor.Shape.Rank != 2)
        {
            throw new ArgumentException($"{operation} works on matrices, and was given a {tensor.Shape} tensor.", parameter);
        }
    }

    /// <summary>One of the tensor primitives that reads two spans and writes a third.</summary>
    private delegate void ElementwiseOperation(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination);
}
