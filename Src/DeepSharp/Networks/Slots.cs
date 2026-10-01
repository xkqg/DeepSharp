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
