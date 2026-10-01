// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Numbers handed for a network's slots, each held to the slot its path names: the one rule a file's numbers go into a
/// network by, whichever file — a network's own, or one another framework saved — they were read from.
/// </summary>
/// <remarks>
/// A number is for a slot the network has, handed once, of that slot's shape and finite; and every slot is handed one. A
/// number that breaks the rule is noted where its file holds it and kept out, the reading goes on, and every fault is named
/// once the reading is over — so the numbers go in together, or none of them does.
/// </remarks>
internal sealed class SlotLoad
{
    private readonly NamedSlot[] _slots;
    private readonly Dictionary<string, Slot> _byPath;
    private readonly HashSet<string> _handed = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> _values = new(StringComparer.Ordinal);
    private readonly List<SlotLoadFault> _faults = [];

    /// <summary>A load into the given slots: a network's, in the order it lists them.</summary>
    public SlotLoad(IEnumerable<NamedSlot> slots)
    {
        _slots = [.. slots];
        _byPath = _slots.ToDictionary(named => named.Path, named => named.Slot, StringComparer.Ordinal);
    }

    /// <summary>The network's slots, by path.</summary>
    public IReadOnlyDictionary<string, Slot> Slots => _byPath;

    /// <summary>The numbers that keep to the rule, by the path of their slot.</summary>
    public IReadOnlyDictionary<string, Tensor> Values => _values;

    /// <summary>Every fault: those noted as the numbers were handed, in that order, then each slot nothing was handed for.</summary>
    public IReadOnlyList<SlotLoadFault> Faults =>
    [
        .. _faults,
        .. _slots.Where(named => !_handed.Contains(named.Path))
            .Select(named => new SlotLoadFault(null, named.Path, $"'{named.Path}' is missing: every slot of the network is written.")),
    ];

    /// <summary>Takes a number handed for a slot: kept when it keeps to the rule, its fault noted when it does not.</summary>
    /// <exception cref="ArgumentException">The entry leaves out its path, its tensor or where its file holds it: the reader's fault, not the file's.</exception>
    public void Take(SlotEntry entry)
    {
        if (entry.Path is null || entry.Value is null || entry.Source is null)
        {
            throw new ArgumentException(
                "A number for a slot names the path of the slot, the tensor to put into it and where its file holds it, and this one leaves one out.",
                nameof(entry));
        }

        if (Handed(entry.Path, entry.Source) is not null)
        {
            Keep(entry);
        }
    }

    /// <summary>
    /// The slot a path is handed for, the first time it is; nothing, with its fault noted, when the network has no slot at the
    /// path or it was handed already.
    /// </summary>
    public Slot? Handed(string path, string source)
    {
        if (!_byPath.TryGetValue(path, out var slot))
        {
            _faults.Add(new SlotLoadFault(source, path, $"'{path.Quoted()}' is no slot of this network."));

            return null;
        }

        if (!_handed.Add(path))
        {
            _faults.Add(new SlotLoadFault(source, path, $"'{path.Quoted()}' is written twice here, and only one of the two would be read."));

            return null;
        }

        return slot;
    }

    /// <summary>Keeps the number handed for a slot when it is of the slot's shape and every value is finite; notes its fault otherwise.</summary>
    /// <remarks>Only for a path <see cref="Handed"/> has given a slot for.</remarks>
    public void Keep(SlotEntry entry)
    {
        var expected = _byPath[entry.Path].Value.Shape;

        if (expected.Unlike([.. entry.Value.Shape.Axes.ToArray().Select(length => (int?)length)], entry.Path) is { } unlike)
        {
            _faults.Add(new SlotLoadFault(entry.Source, entry.Path, unlike));

            return;
        }

        var values = entry.Value.Values;

        for (var at = 0; at < values.Length; at++)
        {
            if (!float.IsFinite(values[at]))
            {
                _faults.Add(new SlotLoadFault(
                    entry.Source,
                    entry.Path,
                    string.Create(CultureInfo.InvariantCulture, $"'{entry.Path}' holds finite numbers, and its value at {at} is {values[at]}.")));

                return;
            }
        }

        _values[entry.Path] = entry.Value;
    }

    /// <summary>Puts every number kept into its slot: asked only once no fault was found, so the slots take all of them or none.</summary>
    public void Apply()
    {
        foreach (var (path, value) in _values)
        {
            _byPath[path].Replace(value);
        }
    }
}

/// <summary>What a tensor written for a place of a known shape is refused with when it is of another.</summary>
internal static class WrittenShapeExtensions
{
    extension(Shape expected)
    {
        /// <summary>
        /// Why a tensor written with these lengths is not of this shape, in the words a refusal names it with; nothing when it
        /// is. A length that could not be read as a whole number is unknown, and never matches.
        /// </summary>
        /// <param name="written">The length of each axis as written, outermost first; nothing for one that is no whole number.</param>
        /// <param name="saying">What the tensor is, as the refusal names it: the path of its slot.</param>
        internal string? Unlike(IReadOnlyList<int?> written, string saying)
        {
            if (written.SequenceEqual(expected.Axes.ToArray().Select(length => (int?)length)))
            {
                return null;
            }

            var said = written.Count == 0 ? "scalar" : string.Join('x', written.Select(length => length?.ToString(CultureInfo.InvariantCulture) ?? "?"));

            return $"'{saying}' is a {expected} slot here, and is written as {said}.";
        }
    }
}
