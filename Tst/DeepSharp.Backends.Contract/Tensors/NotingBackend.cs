// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// An engine that runs every operation on the engine it wraps and notes each one it is asked for, by name, in the order it
/// was asked: so a test sees what a pass, its way back or a whole run asks of the engine under test.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
internal class NotingBackend(ITensorBackend engine) : ITensorBackend
{
    private readonly List<string> _asked = [];

    public virtual string Name => engine.Name;

    /// <summary>Every operation asked for so far, by name, in order.</summary>
    public IReadOnlyList<string> Asked => _asked;

    /// <summary>How many times the operation of that name was asked for, from the given place in <see cref="Asked"/> on.</summary>
    public int Count(string operation, int from = 0) => _asked.Skip(from).Count(asked => asked == operation);

    public Tensor Add(Tensor left, Tensor right) => Noted(nameof(Add), engine.Add(left, right));

    public Tensor Subtract(Tensor left, Tensor right) => Noted(nameof(Subtract), engine.Subtract(left, right));

    public Tensor Multiply(Tensor left, Tensor right) => Noted(nameof(Multiply), engine.Multiply(left, right));

    public Tensor MatMul(Tensor left, Tensor right) => Noted(nameof(MatMul), engine.MatMul(left, right));

    public Tensor Transpose(Tensor matrix) => Noted(nameof(Transpose), engine.Transpose(matrix));

    public Tensor AddRow(Tensor matrix, Tensor row) => Noted(nameof(AddRow), engine.AddRow(matrix, row));

    public Tensor SumRows(Tensor matrix) => Noted(nameof(SumRows), engine.SumRows(matrix));

    public Tensor Mean(Tensor values) => Noted(nameof(Mean), engine.Mean(values));

    public Tensor Scale(Tensor values, Tensor factor) => Noted(nameof(Scale), engine.Scale(values, factor));

    public Tensor Fill(Shape shape, float value) => Noted(nameof(Fill), engine.Fill(shape, value));

    public Tensor Relu(Tensor values) => Noted(nameof(Relu), engine.Relu(values));

    public Tensor Positive(Tensor values) => Noted(nameof(Positive), engine.Positive(values));

    public Tensor Tanh(Tensor values) => Noted(nameof(Tanh), engine.Tanh(values));

    public Tensor Sigmoid(Tensor values) => Noted(nameof(Sigmoid), engine.Sigmoid(values));

    public Tensor Exp(Tensor values) => Noted(nameof(Exp), engine.Exp(values));

    public Tensor Log(Tensor values) => Noted(nameof(Log), engine.Log(values));

    public Tensor Sqrt(Tensor values) => Noted(nameof(Sqrt), engine.Sqrt(values));

    public Tensor Softplus(Tensor values) => Noted(nameof(Softplus), engine.Softplus(values));

    public Tensor Divide(Tensor left, Tensor right) => Noted(nameof(Divide), engine.Divide(left, right));

    public Tensor LogSoftmax(Tensor matrix) => Noted(nameof(LogSoftmax), engine.LogSoftmax(matrix));

    public Tensor Reshape(Tensor values, Shape shape) => Noted(nameof(Reshape), engine.Reshape(values, shape));

    public Tensor Unfold(Tensor images, Window window) => Noted(nameof(Unfold), engine.Unfold(images, window));

    public Tensor Fold(Tensor patches, Shape images, Window window) => Noted(nameof(Fold), engine.Fold(patches, images, window));

    private Tensor Noted(string operation, Tensor made)
    {
        _asked.Add(operation);

        return made;
    }
}
