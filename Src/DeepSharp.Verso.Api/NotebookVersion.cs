// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>An open notebook as of one version: every cell in order, and the run under way.</summary>
/// <param name="Version">Its number: the notebook opens at 0, and each change to it makes the next.</param>
/// <param name="Cells">Its cells, in order.</param>
/// <param name="Running">The run under way at this version; nothing when none is.</param>
/// <remarks>Two looks at a version are equal while they hold the same, whichever list the cells came in.</remarks>
public readonly record struct NotebookVersion(long Version, IReadOnlyList<HostedCell> Cells, HostedRun? Running)
{
    /// <inheritdoc />
    public bool Equals(NotebookVersion other) => Version == other.Version && Cells.SequenceEqual(other.Cells) && Running == other.Running;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Version, Cells.Count, Running);
}
