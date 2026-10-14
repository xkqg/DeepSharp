// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A place in a layer where one of its numbers is kept: its name and the tensor it holds now.
/// </summary>
/// <remarks>
/// A tensor never changes, so a slot is what changes: learning puts a new tensor into it, of the same shape, and the slot
/// stays the one it was — which is how an optimizer keeps what it knows about a number from one step to the next, and how
/// a snapshot can hold every number of a network by holding the tensors that were in its slots.
/// </remarks>
public abstract class Slot
{
    private protected Slot(string name, Tensor initial)
    {
        ArgumentNullException.ThrowIfNull(initial);

        Name = name;
        Value = initial;
    }

    /// <summary>Its name in the layer that keeps it.</summary>
    public string Name { get; }

    /// <summary>The tensor it holds now.</summary>
    public Tensor Value { get; private set; }

    /// <summary>Puts another tensor of the same shape into the slot.</summary>
    /// <param name="value">The tensor.</param>
    /// <exception cref="ArgumentException">The tensor is of another shape.</exception>
    internal void Replace(Tensor value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Shape != Value.Shape)
        {
            throw new ArgumentException($"'{Name}' holds a {Value.Shape} tensor, and was handed a {value.Shape} one.", nameof(value));
        }

        Value = value;
    }
}

/// <summary>A number a layer learns: what an optimizer moves, a step at a time.</summary>
/// <remarks>
/// Its tensor is replaced by an optimizer, by bringing back a snapshot, and by loading numbers read from a file — a network's
/// own, or one another framework saved, through <see cref="Network.Load"/> — by nothing else.
/// </remarks>
public sealed class Parameter : Slot
{
    internal Parameter(string name, Tensor initial)
        : base(name, initial)
    {
    }

    /// <summary>What the learning does to its numbers beside stepping them; nothing, unless said.</summary>
    /// <remarks>
    /// A penalty the loss adds, a bound the numbers are held under after every step, and whether they are held still. A network's
    /// file keeps them beside the numbers, and a run that goes on from a checkpoint is refused when they have changed.
    /// </remarks>
    public SlotRules Rules { get; set; }

    /// <summary>Whether a step moves it: it does, unless its rules hold it still.</summary>
    public bool Learns => Rules.Learns;

    /// <summary>The penalty the loss adds for its numbers, worked out on a backend; nothing when its rules say none.</summary>
    /// <param name="backend">The backend the arithmetic runs on: a recording one, when the penalty is to be differentiated.</param>
    /// <returns>The penalty, a single value.</returns>
    /// <remarks>
    /// Keras's regularizers: the absolute values summed times <see cref="SlotRules.L1"/> and the squares summed times
    /// <see cref="SlotRules.L2"/>. The sign of nothing is nothing, as it is for Keras, so a number at nothing is pushed neither way
    /// by the first; the sums are worked out through the backend's own mean, counted back up.
    /// </remarks>
    internal Tensor? PenaltyOver(ITensorBackend backend)
    {
        if (!Rules.Penalises)
        {
            return null;
        }

        var count = Value.Shape.Count;
        Tensor? total = null;

        if (Rules.L1 > 0)
        {
            var negative = backend.Subtract(backend.Fill(Value.Shape, 0f), Value);
            var sign = backend.Subtract(backend.Positive(Value), backend.Positive(negative));
            var sum = backend.Scale(backend.Mean(backend.Multiply(Value, sign)), backend.Fill(new Shape(), count));

            total = backend.Scale(sum, backend.Fill(new Shape(), (float)Rules.L1));
        }

        if (Rules.L2 > 0)
        {
            var sum = backend.Scale(backend.Mean(backend.Multiply(Value, Value)), backend.Fill(new Shape(), count));
            var squares = backend.Scale(sum, backend.Fill(new Shape(), (float)Rules.L2));

            total = total is null ? squares : backend.Add(total, squares);
        }

        return total;
    }

    /// <summary>The numbers a step moved this parameter to, held under the bound its rules say: Keras's <c>MaxNorm</c>.</summary>
    /// <param name="moved">The tensor the step made.</param>
    /// <param name="backend">The backend the arithmetic runs on, which records nothing.</param>
    /// <returns>
    /// The tensor with every unit's numbers — a matrix's column, a vector as a whole — scaled by <c>min(n, m) / (1e-7 + n)</c>, with
    /// <c>n</c> their norm and <c>m</c> the bound; those under the bound by a hair under one.
    /// </returns>
    internal Tensor Bounded(Tensor moved, ITensorBackend backend)
    {
        if (Rules.MaxNorm is not { } bound)
        {
            return moved;
        }

        var squares = backend.Multiply(moved, moved);

        if (moved.Shape.Rank == 2)
        {
            var norms = backend.Sqrt(backend.SumRows(squares));
            var ratio = backend.Divide(backend.Smaller(norms, backend.Fill(norms.Shape, (float)bound)), backend.Add(norms, backend.Fill(norms.Shape, 1e-7f)));

            // The row of ratios laid under every row of the matrix: a column of ones times the row, which multiplies by one exactly.
            return backend.Multiply(moved, backend.MatMul(backend.Fill(new Shape(moved.Shape[0], 1), 1f), backend.Reshape(ratio, new Shape(1, moved.Shape[1]))));
        }

        var norm = backend.Sqrt(backend.Scale(backend.Mean(squares), backend.Fill(new Shape(), moved.Shape.Count)));

        return backend.Scale(moved, backend.Divide(backend.Smaller(norm, backend.Fill(new Shape(), (float)bound)), backend.Add(norm, backend.Fill(new Shape(), 1e-7f))));
    }
}

/// <summary>
/// A number a layer measures while it trains and uses unchanged afterwards: a normalisation's mean and variance.
/// </summary>
/// <remarks>
/// Measured on the training rows alone, the way the pipeline learns everything it learns, so the rows a network is measured
/// on never move it. No optimizer touches it: only a training pass does, through <see cref="Update"/>.
/// </remarks>
public sealed class RunningStatistic : Slot
{
    internal RunningStatistic(string name, Tensor initial)
        : base(name, initial)
    {
    }

    /// <summary>Puts what a training pass measured into the statistic.</summary>
    /// <param name="value">The statistic as it stands after this pass, of the same shape.</param>
    /// <param name="pass">The pass that measured it: a training pass.</param>
    /// <exception cref="InvalidOperationException">The pass does not train: an evaluation pass uses a statistic and never moves it.</exception>
    /// <exception cref="ArgumentException">The tensor is of another shape.</exception>
    public void Update(Tensor value, Pass pass)
    {
        ArgumentNullException.ThrowIfNull(pass);

        if (pass.Mode != PassMode.Training)
        {
            throw new InvalidOperationException(
                $"'{Name}' is measured on the rows a network trains on, and an evaluation pass only uses it.");
        }

        Replace(value);
    }
}
