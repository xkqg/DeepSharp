// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Globalization;
using DeepSharp.Networks;
using Onnxify.Safetensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>A dict or an OrderedDict a pickle builds: its items in the order they were first set, as Python keeps them.</summary>
/// <param name="ordered">Whether it is an OrderedDict, as a state dictionary is.</param>
internal sealed class PickledDict(bool ordered)
{
    private readonly List<KeyValuePair<object?, object?>> _entries = [];
    private readonly Dictionary<PickledKey, int> _places = [];

    /// <summary>Whether it is an OrderedDict.</summary>
    public bool Ordered => ordered;

    /// <summary>Its items, in the order they were first set.</summary>
    public IReadOnlyList<KeyValuePair<object?, object?>> Entries => _entries;

    /// <summary>Sets the value under a key: in the key's place when it holds one already, as Python does, and after the last otherwise.</summary>
    public void Set(object? key, object? value)
    {
        if (_places.TryGetValue(new PickledKey(key), out var place))
        {
            _entries[place] = new(_entries[place].Key, value);

            return;
        }

        _places.Add(new PickledKey(key), _entries.Count);
        _entries.Add(new(key, value));
    }

    /// <summary>What it is: <c>an OrderedDict</c> or <c>a dict</c>.</summary>
    public override string ToString() => ordered ? "an OrderedDict" : "a dict";
}

/// <summary>A key of a dict a pickle builds, equal to another as Python's are: by value, a tuple by what it holds.</summary>
/// <param name="Value">The key: None, a truth, a number, text, or a tuple of those.</param>
/// <remarks>
/// Its hash is the runtime's own hash of a text only equal keys share, seeded afresh in every process: the hash the runtime
/// gives a number or a tuple is the same for values a file can choose by the thousand, and a dict of them would take
/// minutes to fill; this one no file can aim at.
/// </remarks>
internal readonly record struct PickledKey(object? Value)
{
    /// <summary>Whether two keys are the same key.</summary>
    public bool Equals(PickledKey other) => StructuralComparisons.StructuralEqualityComparer.Equals(Value, other.Value);

    /// <summary>A hash that two keys that are the same key share, and that a file cannot choose.</summary>
    public override int GetHashCode() => Encoded(Value).GetHashCode(StringComparison.Ordinal);

    // A text of the key that equal keys share: a tag for its kind, then what it is — nought and minus nought alike, every
    // not-a-number alike, as the runtime compares them — text and tuples with their lengths first, so no two run together.
    private static string Encoded(object? key) => key switch
    {
        null => "N",
        bool truth => truth ? "T" : "F",
        double number => string.Create(CultureInfo.InvariantCulture, $"D{(number == 0 ? 0 : double.IsNaN(number) ? double.NaN : number):R};"),
        IFormattable whole => string.Create(CultureInfo.InvariantCulture, $"I{whole};"),
        string text => string.Create(CultureInfo.InvariantCulture, $"S{text.Length}:{text}"),
        _ => string.Create(CultureInfo.InvariantCulture, $"U{((object?[])key).Length}:{string.Concat(((object?[])key).Select(Encoded))}"),
    };
}

/// <summary>A storage a pickle loads by its persistent id: the file's record <c>data/{key}</c>, read once the pickle is.</summary>
/// <param name="Key">Its key: the name of its record.</param>
/// <param name="Kind">The kind of number it holds.</param>
/// <param name="Count">How many numbers it holds.</param>
/// <param name="At">The byte of the pickle that first loads it.</param>
internal sealed record PickledStorage(string Key, StorageKind Kind, long Count, int At)
{
    /// <summary>What it is: <c>storage '0'</c>.</summary>
    public override string ToString() => $"storage '{Key.Quoted()}'";
}

/// <summary>A tensor a pickle rebuilds: a view of a storage, from an offset into it, each axis with its length and stride.</summary>
/// <param name="storage">The storage it views.</param>
/// <param name="offset">Where in the storage its first number is.</param>
/// <param name="lengths">The length of each axis.</param>
/// <param name="strides">How far apart in the storage two numbers one step along each axis are.</param>
internal sealed class PickledTensor(PickledStorage storage, long offset, long[] lengths, long[] strides)
{
    /// <summary>The storage it views.</summary>
    public PickledStorage Storage => storage;

    /// <summary>
    /// The tensor as the file holds it: its kind of number, its lengths, and its numbers gathered row by row out of its
    /// storage's bytes, each turned round when the file was written on a machine of the other byte order.
    /// </summary>
    /// <param name="stored">The storage's bytes, as its record holds them.</param>
    /// <param name="bigEndian">Whether the file was written on a machine that keeps a number's bytes the other way round.</param>
    public TensorView Numbers(byte[] stored, bool bigEndian)
    {
        var size = storage.Kind.Size;
        var count = lengths.Aggregate(1L, (product, length) => product * length);
        var numbers = new byte[count * size];
        var index = new long[lengths.Length];
        var from = offset;

        for (var at = 0L; at < count; at++)
        {
            stored.AsSpan((int)(from * size), size).CopyTo(numbers.AsSpan((int)(at * size)));

            // One step on along the last axis; past its end, back to its start and one step on along the axis before it.
            for (var axis = lengths.Length - 1; axis >= 0; axis--)
            {
                from += strides[axis];

                if (++index[axis] < lengths[axis])
                {
                    break;
                }

                from -= strides[axis] * lengths[axis];
                index[axis] = 0;
            }
        }

        for (var at = 0; bigEndian && at < numbers.Length; at += storage.Kind.Swapped)
        {
            numbers.AsSpan(at, storage.Kind.Swapped).Reverse();
        }

        return new TensorView(storage.Kind.Kind, lengths.Select(length => (ulong)length), numbers);
    }

    /// <summary>What it is: <c>a tensor</c>.</summary>
    public override string ToString() => "a tensor";
}
