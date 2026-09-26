// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What changed in an open notebook up to one version: each cell that changed, as it stands at that version.</summary>
/// <param name="Version">The version it brings the notebook to.</param>
/// <param name="Order">Every cell's id in order, when cells came, went or moved; nothing when none did.</param>
/// <param name="Cells">Every cell that came or changed, as it stands at <paramref name="Version"/>.</param>
/// <param name="Running">The run under way at <paramref name="Version"/>; nothing when none is.</param>
/// <param name="Layout">The layout the notebook is shown in at <paramref name="Version"/>, and what it lets a person do.</param>
/// <remarks>
/// A change holds each cell as it stands rather than what was done to it, so applying one that repeats what a view already
/// has changes nothing. A cell a click rewrote is a new cell, under a new id, in the old one's place. Two looks at a
/// change are equal while they hold the same, whichever list each came in.
/// </remarks>
public readonly record struct NotebookChange(
    long Version, IReadOnlyList<Guid>? Order, IReadOnlyList<HostedCell> Cells, HostedRun? Running, HostedLayout Layout)
{
    /// <inheritdoc />
    public bool Equals(NotebookChange other) =>
        Version == other.Version && (Order is null ? other.Order is null : other.Order is not null && Order.SequenceEqual(other.Order))
        && Cells.SequenceEqual(other.Cells) && Running == other.Running && Layout == other.Layout;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Version, Cells.Count, Running, Layout);

    /// <summary>This change and a later one, taken as one: the later look at each cell, and only cells that still stand.</summary>
    /// <param name="later">The later change.</param>
    /// <returns>Both, as one change up to the later one's version.</returns>
    internal NotebookChange Then(NotebookChange later)
    {
        var order = later.Order ?? Order;
        var again = later.Cells.Select(cell => cell.Id).ToHashSet();
        var standing = order?.ToHashSet();
        IEnumerable<HostedCell> cells = [.. Cells.Where(cell => !again.Contains(cell.Id)), .. later.Cells];

        return new NotebookChange(later.Version, order, [.. cells.Where(cell => standing?.Contains(cell.Id) ?? true)], later.Running, later.Layout);
    }
}
