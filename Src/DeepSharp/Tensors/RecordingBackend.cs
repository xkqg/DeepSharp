// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// A backend that writes down every operation of one pass as it runs it on the backend it wraps, and then works back
/// from a loss to how much each tensor the pass read moved it.
/// </summary>
/// <param name="inner">The backend the arithmetic runs on.</param>
/// <remarks>
/// Made for one pass and handed to what runs it, like any backend, and dropped once the gradients are taken: nothing
/// holds one, shares one or reaches for one, so a pass that wants gradients says so by being handed one. What it keeps
/// is a record of this pass's arithmetic, not a second description of the model — the model is whatever code ran.
/// <para>
/// Every operation the seam offers is written down with the rule that sends a gradient back through it, so an operation
/// added to the seam without such a rule does not compile. The way back runs on the wrapped backend's own operations
/// and reads no tensor's values, so it works wherever the arithmetic does. A tensor read more than once gets the sum
/// of what every reading sends back.
/// </para>
/// </remarks>
public sealed class RecordingBackend(ITensorBackend inner) : ITensorBackend
{
    private readonly ITensorBackend _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    private readonly List<Recorded> _steps = [];

    /// <inheritdoc />
    public string Name => _inner.Name;

    /// <inheritdoc />
    public Tensor Add(Tensor left, Tensor right) => Kept(new Added(left, right, _inner.Add(left, right)));

    /// <inheritdoc />
    public Tensor Subtract(Tensor left, Tensor right) => Kept(new Subtracted(left, right, _inner.Subtract(left, right)));

    /// <inheritdoc />
    public Tensor Multiply(Tensor left, Tensor right) => Kept(new Multiplied(left, right, _inner.Multiply(left, right)));

    /// <inheritdoc />
    public Tensor MatMul(Tensor left, Tensor right) => Kept(new MatrixProduct(left, right, _inner.MatMul(left, right)));

    /// <inheritdoc />
    public Tensor Transpose(Tensor matrix) => Kept(new Transposed(matrix, _inner.Transpose(matrix)));

    /// <inheritdoc />
    public Tensor AddRow(Tensor matrix, Tensor row) => Kept(new RowAdded(matrix, row, _inner.AddRow(matrix, row)));

    /// <inheritdoc />
    public Tensor SumRows(Tensor matrix) => Kept(new RowsSummed(matrix, _inner.SumRows(matrix)));

    /// <inheritdoc />
    public Tensor Mean(Tensor values) => Kept(new Averaged(values, _inner.Mean(values)));

    /// <inheritdoc />
    public Tensor Scale(Tensor values, Tensor factor) => Kept(new Scaled(values, factor, _inner.Scale(values, factor)));

    /// <inheritdoc />
    /// <remarks>A filled tensor reads nothing, so there is nothing to send a gradient back to.</remarks>
    public Tensor Fill(Shape shape, float value) => _inner.Fill(shape, value);

    /// <summary>How much each of the given tensors moved a loss this pass worked out.</summary>
    /// <param name="loss">The loss: one value, with no axes, worked out in this pass.</param>
    /// <param name="parameters">The tensors whose gradients are wanted — the numbers a network learns — each one the pass read.</param>
    /// <returns>A gradient for each, of its own shape; nothing, as zeros, for one the pass read without it reaching the loss.</returns>
    /// <exception cref="ArgumentException">
    /// The loss has axes or was not worked out in this pass, or a tensor asked for is one the pass never read.
    /// </exception>
    /// <remarks>
    /// The loss's own gradient is one, and each operation from the loss back to the start sends what it received on to
    /// what it read. A tensor the pass never read is refused rather than given a gradient of nothing: asking for it is
    /// asking about a number the pass had no way to move, and that is a mistake in the asking.
    /// </remarks>
    public Gradients GradientsOf(Tensor loss, IEnumerable<Tensor> parameters)
    {
        ArgumentNullException.ThrowIfNull(loss);
        ArgumentNullException.ThrowIfNull(parameters);

        IReadOnlyList<Tensor> asked = [.. parameters];

        if (loss.Shape.Rank != 0)
        {
            throw new ArgumentException(
                $"A gradient is taken of one value, a loss with no axes, and this loss is {loss.Shape}.", nameof(loss));
        }

        var last = _steps.FindLastIndex(step => ReferenceEquals(step.Output, loss));

        if (last < 0)
        {
            throw new ArgumentException(
                "The loss was not worked out in this pass, so there is nothing to work back through.", nameof(loss));
        }

        var read = new HashSet<Tensor>(_steps.Take(last + 1).SelectMany(step => step.Inputs), ReferenceEqualityComparer.Instance);

        if (asked.FirstOrDefault(parameter => !read.Contains(parameter)) is { } unread)
        {
            throw new ArgumentException(
                $"A {unread.Shape} tensor was asked for that this pass never read before its loss, so the loss cannot have moved with it.",
                nameof(parameters));
        }

        var received = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance) { [loss] = _inner.Fill(new Shape(), 1f) };

        for (var at = last; at >= 0; at--)
        {
            if (!received.TryGetValue(_steps[at].Output, out var gradient))
            {
                continue;
            }

            foreach (var share in _steps[at].Back(_inner, gradient))
            {
                received[share.To] = received.TryGetValue(share.To, out var sofar) ? _inner.Add(sofar, share.Gradient) : share.Gradient;
            }
        }

        var gradients = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);

        foreach (var parameter in asked)
        {
            gradients[parameter] = received.TryGetValue(parameter, out var gradient) ? gradient : _inner.Fill(parameter.Shape, 0f);
        }

        return new Gradients(gradients);
    }

    private Tensor Kept(Recorded step)
    {
        _steps.Add(step);

        return step.Output;
    }

    // One operation of a pass: what it read, what it made, and the rule that sends a gradient back through it.
    private abstract record Recorded(Tensor Output)
    {
        public abstract IEnumerable<Tensor> Inputs { get; }

        public abstract IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient);
    }

    private sealed record Added(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Left, Right];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Left, gradient), new(Right, gradient)];
    }

    private sealed record Subtracted(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Left, Right];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Left, gradient), new(Right, backend.Scale(gradient, backend.Fill(new Shape(), -1f)))];
    }

    private sealed record Multiplied(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Left, Right];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Left, backend.Multiply(gradient, Right)), new(Right, backend.Multiply(gradient, Left))];
    }

    private sealed record MatrixProduct(Tensor Left, Tensor Right, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Left, Right];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
        [
            new(Left, backend.MatMul(gradient, backend.Transpose(Right))),
            new(Right, backend.MatMul(backend.Transpose(Left), gradient)),
        ];
    }

    private sealed record Transposed(Tensor Matrix, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Matrix];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Matrix, backend.Transpose(gradient))];
    }

    private sealed record RowAdded(Tensor Matrix, Tensor Row, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Matrix, Row];

        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Matrix, gradient), new(Row, backend.SumRows(gradient))];
    }

    private sealed record RowsSummed(Tensor Matrix, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Matrix];

        // Every row added into the total, so every row is sent the total's gradient.
        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Matrix, backend.AddRow(backend.Fill(Matrix.Shape, 0f), gradient))];
    }

    private sealed record Averaged(Tensor Values, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Values];

        // A mean refuses no values, so a share is always one of so many.
        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
            [new(Values, backend.Scale(backend.Fill(Values.Shape, 1f / Values.Shape.Count), gradient))];
    }

    private sealed record Scaled(Tensor Values, Tensor Factor, Tensor Output) : Recorded(Output)
    {
        public override IEnumerable<Tensor> Inputs => [Values, Factor];

        // The factor moved every value it scaled, so it is sent the total of the gradient times each value: the mean
        // taken so many times, and nothing when it scaled nothing.
        public override IEnumerable<GradientShare> Back(ITensorBackend backend, Tensor gradient) =>
        [
            new(Values, backend.Scale(gradient, Factor)),
            new(Factor, Values.Shape.Count == 0
                ? backend.Fill(new Shape(), 0f)
                : backend.Scale(backend.Mean(backend.Multiply(gradient, Values)), backend.Fill(new Shape(), Values.Shape.Count))),
        ];
    }

    // What one operation sends back to one tensor it read.
    private readonly record struct GradientShare(Tensor To, Tensor Gradient);
}

/// <summary>How much each tensor asked for moved a pass's loss: one gradient each, of that tensor's own shape.</summary>
/// <remarks>Each tensor is known by which one it is, never by the values it holds, so two tensors that happen to hold the same numbers keep their own gradients.</remarks>
public sealed class Gradients
{
    private readonly Dictionary<Tensor, Tensor> _of;

    internal Gradients(Dictionary<Tensor, Tensor> of) => _of = of;

    /// <summary>The gradient of one of the tensors asked for.</summary>
    /// <param name="parameter">The tensor, as it was handed to the pass.</param>
    /// <returns>Its gradient, of its own shape.</returns>
    /// <exception cref="ArgumentException">The tensor was not among those asked for.</exception>
    public Tensor this[Tensor parameter]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(parameter);

            return _of.TryGetValue(parameter, out var gradient)
                ? gradient
                : throw new ArgumentException($"A gradient was not asked for this {parameter.Shape} tensor.", nameof(parameter));
        }
    }
}
