// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.ObjectModel;

namespace DeepSharp.Pipelines;

/// <summary>
/// A list that is equal to another of the same items in the same order, and hashes by them.
/// </summary>
/// <typeparam name="T">What the list holds.</typeparam>
/// <remarks>
/// A step is a record, and a record compares each member with that member's own equality. A list does not compare by what it
/// holds, so a step holding one would call two steps of one file different. The parameter kinds, which every list a step
/// holds passes through, hand it back as one of these, typed as the read-only list the step shows, so the equality a record
/// writes for itself is the right one and no step writes it by hand. It cannot be changed, through the list it is typed as or
/// through any other it is cast to.
/// </remarks>
internal sealed class ValueList<T> : ReadOnlyCollection<T>, IEquatable<ValueList<T>>
{
    /// <summary>Holds a copy of these items.</summary>
    /// <param name="items">The items, in order.</param>
    /// <exception cref="ArgumentNullException">There are no items.</exception>
    public ValueList(IEnumerable<T> items)
        : base(Copy(items))
    {
    }

    /// <summary>The list of nothing.</summary>
    public static ValueList<T> None { get; } = new([]);

    /// <inheritdoc />
    public bool Equals(ValueList<T>? other) => other is not null && Items.SequenceEqual(other.Items);

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as ValueList<T>);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var item in Items)
        {
            hash.Add(item);
        }

        return hash.ToHashCode();
    }

    private static T[] Copy(IEnumerable<T> items)
    {
        ArgumentNullException.ThrowIfNull(items);

        return [.. items];
    }
}

/// <summary>
/// Makes a list one that is equal by what it holds.
/// </summary>
internal static class ValueLists
{
    extension<T>(IEnumerable<T> items)
    {
        /// <summary>This list as one that is equal to another of the same items.</summary>
        /// <returns>The list itself when it is one already, and a copy of it otherwise.</returns>
        /// <exception cref="ArgumentNullException">There are no items.</exception>
        public ValueList<T> Valued() => items as ValueList<T> ?? new ValueList<T>(items);
    }
}
