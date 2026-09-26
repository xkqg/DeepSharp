// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A cell of an open notebook as it stands: what it is, what it holds, what it shows, and how it last ran.</summary>
/// <param name="Id">The cell's id; a change that rewrites a block replaces it.</param>
/// <param name="Type">The cell's type: a DeepSharp block, C# code, markdown and so on.</param>
/// <param name="Language">The language its text is written in, when it has one.</param>
/// <param name="Source">Its text.</param>
/// <param name="Outputs">What it shows, in order.</param>
/// <param name="Metadata">
/// What is kept with it, each value written as JSON: how it is shown among them — whether its output is hidden or cut
/// short, whether its text is folded away — as Verso's properties panel sets it.
/// </param>
/// <param name="ExecutionCount">How many times it has run; nothing when it has not.</param>
/// <param name="LastStatus">How its last run ended, as the engine says it: Success, Failed or Cancelled; nothing when it has not run.</param>
/// <param name="LastElapsed">How long its last run took; nothing when it has not run.</param>
/// <remarks>Two looks at a cell are equal while it holds and shows the same, whichever list or dictionary each came in.</remarks>
public readonly record struct HostedCell(
    Guid Id,
    string Type,
    string? Language,
    string Source,
    IReadOnlyList<HostedOutput> Outputs,
    IReadOnlyDictionary<string, string> Metadata,
    int? ExecutionCount,
    string? LastStatus,
    TimeSpan? LastElapsed)
{
    /// <inheritdoc />
    public bool Equals(HostedCell other) =>
        Id == other.Id && Type == other.Type && Language == other.Language && Source == other.Source && Outputs.SequenceEqual(other.Outputs)
        && Metadata.Count == other.Metadata.Count && Metadata.All(pair => other.Metadata.TryGetValue(pair.Key, out var value) && value == pair.Value)
        && ExecutionCount == other.ExecutionCount && LastStatus == other.LastStatus && LastElapsed == other.LastElapsed;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Id, Type, Language, Source, Outputs.Count, Metadata.Count, ExecutionCount, LastStatus);
}
