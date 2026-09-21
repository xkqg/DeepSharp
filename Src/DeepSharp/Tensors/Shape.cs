// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// How many axes a tensor has and how long each one is — <c>2x3</c> is two rows of three.
/// </summary>
/// <remarks>
/// A shape is a value: two shapes with the same axes are the same shape, and one can be compared, stored in a
/// dictionary and printed. A shape with no axes at all is a single value, which is what a loss is.
/// </remarks>
public readonly struct Shape : IEquatable<Shape>
{
    private static readonly int[] None = [];

    private readonly int[]? _axes;
    private readonly int _count;

    /// <summary>Creates a shape from the length of each axis, outermost first.</summary>
    /// <param name="axes">The length of each axis. No axes at all means a single value.</param>
    /// <exception cref="ArgumentOutOfRangeException">An axis is negative, or the axes together describe more
    /// values than can be counted.</exception>
    public Shape(params int[] axes)
    {
        ArgumentNullException.ThrowIfNull(axes);

        long count = 1;
        foreach (int length in axes)
        {
            if (length < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(axes), length, "An axis cannot be shorter than nothing.");
            }

            // Counted as a long on purpose: an int product wraps around silently, and a wrapped count buys a
            // buffer far too small for the values that are about to be written into it.
            count *= length;
            if (count > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(axes), string.Join('x', axes),
                    "These axes describe more values than a single tensor can hold.");
            }
        }

        _axes = axes.Length == 0 ? None : [.. axes];
        _count = (int)count;
    }

    /// <summary>How many axes this shape has. Zero means a single value.</summary>
    public int Rank => Axes.Length;

    /// <summary>How many values a tensor of this shape holds. A shape with no axes holds one.</summary>
    /// <remarks>Read rather than stored for one case: a struct can always be brought into existence without
    /// running a constructor — <c>default(Shape)</c>, an array of them, a field nobody assigned — and such a
    /// shape has no axes, so it is a single value and must say so rather than claiming to hold nothing.</remarks>
    public int Count => _axes is null ? 1 : _count;

    /// <summary>The length of each axis, outermost first.</summary>
    public ReadOnlySpan<int> Axes => _axes ?? None;

    /// <summary>The length of one axis.</summary>
    /// <param name="axis">Which axis, counted from the outermost.</param>
    /// <exception cref="ArgumentOutOfRangeException">There is no such axis.</exception>
    public int this[int axis]
    {
        get
        {
            if ((uint)axis >= (uint)Rank)
            {
                throw new ArgumentOutOfRangeException(nameof(axis), axis, $"A {this} shape has no axis {axis}.");
            }

            return Axes[axis];
        }
    }

    /// <inheritdoc />
    public bool Equals(Shape other) => Axes.SequenceEqual(other.Axes);

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Shape other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = default(HashCode);
        foreach (int length in Axes)
        {
            hash.Add(length);
        }

        return hash.ToHashCode();
    }

    /// <summary>The shape as it is spoken: <c>2x3</c>, or <c>scalar</c> when it has no axes.</summary>
    public override string ToString() => Rank == 0 ? "scalar" : string.Join('x', _axes!);

    /// <summary>Whether two shapes describe the same axes.</summary>
    public static bool operator ==(Shape left, Shape right) => left.Equals(right);

    /// <summary>Whether two shapes describe different axes.</summary>
    public static bool operator !=(Shape left, Shape right) => !left.Equals(right);
}
