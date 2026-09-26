// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A section of a cell's properties panel, from the part that has one for the cell.</summary>
/// <param name="Part">The part it came from, which a change to one of its fields goes to.</param>
/// <param name="Title">What it is called.</param>
/// <param name="Description">What it says above its fields; nothing when it says nothing.</param>
/// <param name="Fields">Its fields, in order.</param>
/// <remarks>Two looks at a section are equal while it holds the same, whichever list its fields came in.</remarks>
public readonly record struct HostedSection(string Part, string Title, string? Description, IReadOnlyList<HostedField> Fields)
{
    /// <inheritdoc />
    public bool Equals(HostedSection other) =>
        Part == other.Part && Title == other.Title && Description == other.Description && Fields.SequenceEqual(other.Fields);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Part, Title, Description, Fields.Count);
}
