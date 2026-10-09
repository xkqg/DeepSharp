// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// An engine outside the library, as one of your own would be: it reaches only what DeepSharp publishes. It refuses what
/// the seam refuses before anything else, works its arithmetic out as the light engine does, and keeps every tensor it
/// makes on a storage of its own — so a test sees the tensors an engine keeps elsewhere travel through a whole run, and how
/// often each is copied out.
/// </summary>
public sealed class CopyCountingBackend : ITensorBackend
{
    private readonly CpuBackend _arithmetic = new();
    private readonly List<CopyCountingStorage> _made = [];

    /// <inheritdoc />
    public string Name => "copycounting";

    /// <summary>The storage of every tensor it has made, in the order it made them.</summary>
    public IReadOnlyList<CopyCountingStorage> Made => _made;

    /// <summary>How many tensors it has been handed that stood on a storage other than its own.</summary>
    public int TakenIn { get; private set; }

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Add));

        return Kept(_arithmetic.Add(Taken(left), Taken(right)));
    }

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Subtract));

        return Kept(_arithmetic.Subtract(Taken(left), Taken(right)));
    }

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Multiply));

        return Kept(_arithmetic.Multiply(Taken(left), Taken(right)));
    }

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right)
    {
        left.RequireMatrixProduct(right);

        return Kept(_arithmetic.MatMul(Taken(left), Taken(right)));
    }

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(Transpose));

        return Kept(_arithmetic.Transpose(Taken(matrix)));
    }

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row)
    {
        matrix.RequireRow(row);

        return Kept(_arithmetic.AddRow(Taken(matrix), Taken(row)));
    }

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(SumRows));

        return Kept(_arithmetic.SumRows(Taken(matrix)));
    }

    /// <inheritdoc />
    public Tensor Mean(Tensor values)
    {
        values.RequireValues();

        return Kept(_arithmetic.Mean(Taken(values)));
    }

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor)
    {
        values.RequireFactor(factor);

        return Kept(_arithmetic.Scale(Taken(values), Taken(factor)));
    }

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value) => Kept(_arithmetic.Fill(shape, value));

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Kept(_arithmetic.Relu(Taken(values)));

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Kept(_arithmetic.Positive(Taken(values)));

    /// <inheritdoc />
    public Tensor FirstLargest(Tensor matrix)
    {
        matrix.RequireRowsToPickFrom(nameof(FirstLargest));

        return Kept(_arithmetic.FirstLargest(Taken(matrix)));
    }

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Kept(_arithmetic.Tanh(Taken(values)));

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Kept(_arithmetic.Sigmoid(Taken(values)));

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => Kept(_arithmetic.Exp(Taken(values)));

    /// <inheritdoc />
    public Tensor Log(Tensor values) => Kept(_arithmetic.Log(Taken(values)));

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => Kept(_arithmetic.Sqrt(Taken(values)));

    /// <inheritdoc />
    public Tensor Softplus(Tensor values) => Kept(_arithmetic.Softplus(Taken(values)));

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right)
    {
        left.RequireSameShape(right, nameof(Divide));

        return Kept(_arithmetic.Divide(Taken(left), Taken(right)));
    }

    /// <inheritdoc />
    public Tensor LogSoftmax(Tensor matrix)
    {
        matrix.RequireMatrix(nameof(LogSoftmax));

        return Kept(_arithmetic.LogSoftmax(Taken(matrix)));
    }

    /// <inheritdoc />
    public Tensor Reshape(Tensor values, Shape shape)
    {
        values.RequireSameCount(shape);

        return Kept(_arithmetic.Reshape(Taken(values), shape));
    }

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window)
    {
        images.RequireImagesFor(window);

        return Kept(_arithmetic.Unfold(Taken(images), window));
    }

    /// <inheritdoc />
    public Tensor Fold(Tensor patches, Shape images, Window window)
    {
        patches.RequirePatchesOf(images, window);

        return Kept(_arithmetic.Fold(Taken(patches), images, window));
    }

    // A tensor on a storage other than this engine's is taken in where it is handed over, and counted; no tensor at all is
    // refused by the name the light engine gives it.
    private Tensor Taken(Tensor tensor, [CallerArgumentExpression(nameof(tensor))] string? name = null)
    {
        ArgumentNullException.ThrowIfNull(tensor, name);

        if (tensor.Storage is not CopyCountingStorage)
        {
            TakenIn++;
        }

        return tensor;
    }

    // What the arithmetic made, kept on a storage of this engine's own.
    private Tensor Kept(Tensor made)
    {
        var storage = new CopyCountingStorage(made.Values);
        _made.Add(storage);

        return Tensor.On(made.Shape, storage);
    }
}
