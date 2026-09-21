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
/// </remarks>
public sealed class CpuBackend : ITensorBackend
{
    /// <inheritdoc />
    public string Name => "cpu";

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Elementwise(left, right, TensorPrimitives.Add, nameof(Add));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) =>
        Elementwise(left, right, TensorPrimitives.Multiply, nameof(Multiply));

    /// <summary>The shape check both operations share, and the buffer both write into.</summary>
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

    /// <summary>One of the tensor primitives that reads two spans and writes a third.</summary>
    private delegate void ElementwiseOperation(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> destination);
}
