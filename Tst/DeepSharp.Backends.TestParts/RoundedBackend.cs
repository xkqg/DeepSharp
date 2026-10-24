// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Parts;

/// <summary>
/// An engine outside the library that keeps its work in half the width of a single number the way the libtorch engine does — by
/// the same table, in plain arithmetic: what a product makes, and what follows a product, is rounded to the narrow width (a value
/// beyond what half holds is lost to infinity, one below what it holds to nought), what is added to or multiplied by a tensor of full
/// width, and everything that must not lose its digits, is made at full width. A test sees on it what a narrow width does to a
/// training step without a graphics card or a libtorch.
/// </summary>
/// <param name="kept">The width: <see cref="Width.Half"/> or <see cref="Width.BFloat16"/>.</param>
public sealed class RoundedBackend(RoundedBackend.Width kept) : ITensorBackend, IWidens, INamesItsPrecision
{
    private readonly CpuBackend _arithmetic = new();
    private readonly ConditionalWeakTable<Tensor, object> _narrow = new();

    /// <summary>The narrow widths this engine can keep its work in.</summary>
    public enum Width
    {
        /// <summary>Five bits of exponent: nothing above 65504, little below 0.00006.</summary>
        Half,

        /// <summary>Eight bits of exponent and a significand of eight bits.</summary>
        BFloat16,
    }

    /// <inheritdoc />
    public string Name => "rounded";

    /// <inheritdoc />
    string? INamesItsPrecision.Precision => kept == Width.Half ? "half" : "bfloat16";

    /// <summary>How many tensors it has rounded to the narrow width, so a test sees that the table reached them.</summary>
    public int Rounded { get; private set; }

    /// <summary>Whether a tensor this engine made is kept in the narrow width.</summary>
    /// <param name="values">The tensor.</param>
    /// <returns>Whether it is.</returns>
    public bool IsNarrow(Tensor values) => _narrow.TryGetValue(values, out _);

    /// <inheritdoc />
    Tensor IWidens.Widened(Tensor values) => IsNarrow(values) ? _arithmetic.Reshape(values, values.Shape) : values;

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Meeting(left, right, _arithmetic.Add(left, right));

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) => Meeting(left, right, _arithmetic.Subtract(left, right));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) => Meeting(left, right, _arithmetic.Multiply(left, right));

    /// <inheritdoc />
    public Tensor Divide(Tensor left, Tensor right) => _arithmetic.Divide(left, right);

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right) => Narrowed(_arithmetic.MatMul(left, right));

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix) => Following(matrix, _arithmetic.Transpose(matrix));

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row) => Following(matrix, _arithmetic.AddRow(matrix, row));

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix) => _arithmetic.SumRows(matrix);

    /// <inheritdoc />
    public Tensor Mean(Tensor values) => _arithmetic.Mean(values);

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor) => Following(values, _arithmetic.Scale(values, factor));

    /// <inheritdoc />
    public Tensor Fill(Shape shape, float value) => _arithmetic.Fill(shape, value);

    /// <inheritdoc />
    public Tensor Relu(Tensor values) => Following(values, _arithmetic.Relu(values));

    /// <inheritdoc />
    public Tensor Positive(Tensor values) => Following(values, _arithmetic.Positive(values));

    /// <inheritdoc />
    public Tensor FirstLargest(Tensor matrix) => Following(matrix, _arithmetic.FirstLargest(matrix));

    /// <inheritdoc />
    public Tensor Tanh(Tensor values) => Following(values, _arithmetic.Tanh(values));

    /// <inheritdoc />
    public Tensor Sigmoid(Tensor values) => Following(values, _arithmetic.Sigmoid(values));

    /// <inheritdoc />
    public Tensor Exp(Tensor values) => _arithmetic.Exp(values);

    /// <inheritdoc />
    public Tensor Log(Tensor values) => _arithmetic.Log(values);

    /// <inheritdoc />
    public Tensor Sqrt(Tensor values) => _arithmetic.Sqrt(values);

    /// <inheritdoc />
    public Tensor Softplus(Tensor values) => _arithmetic.Softplus(values);

    /// <inheritdoc />
    public Tensor Sin(Tensor values) => _arithmetic.Sin(values);

    /// <inheritdoc />
    public Tensor Erf(Tensor values) => _arithmetic.Erf(values);

    /// <inheritdoc />
    public Tensor LogSoftmax(Tensor matrix) => _arithmetic.LogSoftmax(matrix);

    /// <inheritdoc />
    public Tensor Reshape(Tensor values, Shape shape) => Following(values, _arithmetic.Reshape(values, shape), rounds: false);

    /// <inheritdoc />
    public Tensor Unfold(Tensor images, Window window) => Following(images, _arithmetic.Unfold(images, window));

    /// <inheritdoc />
    public Tensor Fold(Tensor patches, Shape images, Window window) => Following(patches, _arithmetic.Fold(patches, images, window));

    // Two tensors that meet: made at the narrow width when both are in it, and at full width when they are not.
    private Tensor Meeting(Tensor left, Tensor right, Tensor made) => IsNarrow(left) && IsNarrow(right) ? Narrowed(made) : made;

    // What follows a tensor leaves it where it was: a narrow tensor's follower is narrow, a full one's is not.
    private Tensor Following(Tensor from, Tensor made, bool rounds = true) =>
        IsNarrow(from) ? (rounds ? Narrowed(made) : Marked(made)) : made;

    private Tensor Marked(Tensor made)
    {
        _narrow.AddOrUpdate(made, this);

        return made;
    }

    // A tensor rounded to the narrow width, value by value, and kept there.
    private Tensor Narrowed(Tensor made)
    {
        var values = made.Values.ToArray();

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = Round(values[at]);
        }

        Rounded++;

        return Marked(Tensor.From(made.Shape, values));
    }

    private float Round(float value)
    {
        if (kept == Width.Half)
        {
            return (float)(Half)value;
        }

        if (!float.IsFinite(value))
        {
            return value;
        }

        // Round to nearest even on the sixteen bits that stay.
        var bits = BitConverter.SingleToInt32Bits(value);
        var rounded = bits + 0x7FFF + ((bits >> 16) & 1);

        return BitConverter.Int32BitsToSingle(rounded & unchecked((int)0xFFFF0000));
    }
}
