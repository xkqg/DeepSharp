// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>Numbers read for a network's slots that cannot go into it, with every fault found among them, each where the file holds it.</summary>
/// <remarks>
/// Every fault at once, so a file is mended in one sitting, and each at the file's own address — a tensor's name, a dataset's
/// path — rather than a line and a column, which a file of numbers does not have. Refused, the numbers change nothing: the
/// network holds what it held.
/// </remarks>
public sealed class SlotLoadException : FormatException
{
    /// <summary>A refusal of the numbers read for a network's slots.</summary>
    /// <param name="faults">Every fault found among them.</param>
    public SlotLoadException(IReadOnlyList<SlotLoadFault> faults)
        : base(string.Join(Environment.NewLine, faults ?? throw new ArgumentNullException(nameof(faults))))
    {
        Faults = faults;
    }

    /// <summary>Every fault found: those of the numbers in the order they were read, then each slot the file holds no number for.</summary>
    public IReadOnlyList<SlotLoadFault> Faults { get; }
}

/// <summary>One fault among the numbers read for a network's slots.</summary>
/// <param name="Source">Where the file holds the number, in the file's own words; nothing for a slot the file holds no number for.</param>
/// <param name="Slot">The path the number was read for: a slot of the network, or a path it has no slot at.</param>
/// <param name="Message">What is wrong, naming the slot.</param>
public readonly record struct SlotLoadFault(string? Source, string Slot, string Message)
{
    /// <summary>
    /// The fault where the file holds it: <c>line 3: '1.weight' …</c>, the place shown as <see cref="FileTextExtensions.Quoted"/>
    /// shows a file's text; the message alone for a number the file does not hold.
    /// </summary>
    public override string ToString() => Source is null ? Message : $"{Source.Quoted()}: {Message}";
}
