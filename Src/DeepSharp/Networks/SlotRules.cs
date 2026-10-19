// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// What the learning does to the numbers of a parameter beside stepping them: a penalty the loss adds for them, a bound they are
/// held under after every step, and whether they are moved at all.
/// </summary>
/// <remarks>
/// A penalty is Keras's regularizer: <see cref="L1"/> times the sum of the absolute values plus <see cref="L2"/> times the sum of the
/// squares, with no half, added to the loss the run reports — in training and in validation alike — so its gradient is what
/// moves the numbers. A bound is Keras's <c>MaxNorm</c>: after every step each unit's numbers — a dense layer's column, a
/// convolution's filter, a vector as a whole — are scaled by <c>min(n, m) / (1e-7 + n)</c> when their norm <c>n</c> is above the
/// bound <c>m</c>. A parameter that is held is moved by nothing: it is no part of what the loss is differentiated by, an
/// optimizer remembers nothing of it, and a normalisation all of whose numbers are held measures nothing of its batches. They
/// are the parameter's own, written beside its numbers in a network's file when they say anything; none of them is a setting of
/// the layer that keeps it, nor of the run.
/// </remarks>
public readonly record struct SlotRules
{
    /// <summary>How much of the sum of the absolute values of the numbers the loss adds; nothing, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    public double L1
    {
        get;
        init => field = double.IsFinite(value) && value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "A penalty is a finite number of at least nothing.");
    }

    /// <summary>How much of the sum of the squares of the numbers the loss adds; nothing, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    public double L2
    {
        get;
        init => field = double.IsFinite(value) && value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "A penalty is a finite number of at least nothing.");
    }

    /// <summary>The most norm each unit's numbers may have after a step; no bound, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    public double? MaxNorm
    {
        get;
        init => field = value is null || (double.IsFinite(value.Value) && value.Value > 0)
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "A bound is a finite number above nothing.");
    }

    /// <summary>Whether the numbers are held still: moved by no step, and out of the loss's gradient; they learn, unless said.</summary>
    public bool Held { get; init; }

    /// <summary>
    /// The share of the run's rate the numbers move at, so a part of a network can learn faster or slower than the rest: the
    /// whole rate, unless said. PyTorch's learning rate of a parameter group, said of the parameter it is about.
    /// </summary>
    /// <remarks>
    /// A step moves the numbers by the rate the schedule gives the epoch times this share, and what an optimizer remembers of
    /// them is worked out from their gradients alone, so the share changes how far a step goes and never what is remembered.
    /// A share of nothing would move nothing: that is what <see cref="Held"/> says, and it is refused here in its favour.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    public double? RateShare
    {
        get;
        init => field = value is null || (double.IsFinite(value.Value) && value.Value > 0)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value),
                value,
                "A share of the rate is a finite number above nothing; numbers that no step should move are held still, with Held.");
    }

    /// <summary>Whether the numbers learn.</summary>
    public bool Learns => !Held;

    /// <summary>Whether it says a penalty.</summary>
    public bool Penalises => L1 > 0 || L2 > 0;

    /// <summary>Whether it says nothing: the rules of a parameter that learns as it always did.</summary>
    internal bool IsDefault => this == default;
}
