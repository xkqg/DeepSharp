// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A field of a cell's properties panel.</summary>
/// <param name="Name">What changing it names.</param>
/// <param name="Label">What it says.</param>
/// <param name="Kind">What it takes.</param>
/// <param name="Value">What it holds now: text, a number, on or off, or a list of words.</param>
/// <param name="Description">What it is for; nothing when it says nothing.</param>
/// <param name="Options">The choices it offers, for a field that offers any.</param>
/// <param name="IsReadOnly">Whether it can be changed.</param>
/// <remarks>Two looks at a field are equal while it holds and offers the same, whichever list each came in.</remarks>
public readonly record struct HostedField(
    string Name, string Label, FieldKind Kind, object? Value, string? Description, IReadOnlyList<HostedOption> Options, bool IsReadOnly)
{
    /// <inheritdoc />
    public bool Equals(HostedField other) =>
        Name == other.Name && Label == other.Label && Kind == other.Kind && SameValue(Value, other.Value)
        && Description == other.Description && Options.SequenceEqual(other.Options) && IsReadOnly == other.IsReadOnly;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Name, Label, Kind, Description, Options.Count, IsReadOnly);

    // A list of words is the same while it holds the same words; any other value is itself.
    private static bool SameValue(object? value, object? other) =>
        value is IEnumerable<string> words && other is IEnumerable<string> others ? words.SequenceEqual(others) : Equals(value, other);
}
