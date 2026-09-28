// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What changed in an open notebook up to one version: each cell that changed, as it stands at that version.</summary>
/// <param name="Version">The version it brings the notebook to.</param>
/// <param name="Order">Every cell's id in order, when cells came, went or moved; nothing when none did.</param>
/// <param name="Cells">Every cell that came or changed, as it stands at <paramref name="Version"/>.</param>
/// <param name="Running">The run under way at <paramref name="Version"/>; nothing when none is.</param>
/// <param name="Executing">
/// What the engine runs at <paramref name="Version"/> that no run owns — a block a change runs, or what a stop left
/// behind — which a view shows running, with nothing to stop, in the order each began.
/// </param>
/// <param name="Layout">The layout the notebook is shown in at <paramref name="Version"/>, and what it lets a person do.</param>
/// <param name="Buttons">
/// Every toolbar button as it stands at <paramref name="Version"/>, when any says something else than before; nothing when
/// none does.
/// </param>
/// <param name="Kernels">What became of the notebook's kernels up to <paramref name="Version"/>.</param>
/// <param name="Unsaved">Whether the notebook differs at <paramref name="Version"/> from the file it was last saved to.</param>
/// <param name="ThemeId">The theme the notebook chose at <paramref name="Version"/>; nothing while it chose none.</param>
/// <param name="Arrangement">
/// What the layout the notebook is shown in draws of its own at <paramref name="Version"/>, when it draws something else
/// than before; nothing when it does not.
/// </param>
/// <param name="Metadata">
/// What the notebook says of itself at <paramref name="Version"/> — title, default kernel, dates, format — when it says
/// something else than before; nothing when it does not.
/// </param>
/// <remarks>
/// A change holds each cell as it stands rather than what was done to it, so applying one that repeats what a view already
/// has changes nothing. A cell a click rewrote is a new cell, under a new id, in the old one's place. Two looks at a
/// change are equal while they hold the same, whichever list each came in.
/// </remarks>
public readonly record struct NotebookChange(
    long Version,
    IReadOnlyList<Guid>? Order,
    IReadOnlyList<HostedCell> Cells,
    HostedRun? Running,
    IReadOnlyList<HostedExecution> Executing,
    HostedLayout Layout,
    IReadOnlyList<HostedToolbarAction>? Buttons,
    HostedKernels Kernels,
    bool Unsaved,
    string? ThemeId,
    HostedArrangement? Arrangement,
    HostedMetadata? Metadata)
{
    /// <inheritdoc />
    public bool Equals(NotebookChange other) =>
        Version == other.Version && (Order is null ? other.Order is null : other.Order is not null && Order.SequenceEqual(other.Order))
        && Cells.SequenceEqual(other.Cells) && Running == other.Running && Executing.SequenceEqual(other.Executing) && Layout == other.Layout
        && (Buttons is null ? other.Buttons is null : other.Buttons is not null && Buttons.SequenceEqual(other.Buttons))
        && Kernels == other.Kernels && Unsaved == other.Unsaved && ThemeId == other.ThemeId && Arrangement == other.Arrangement
        && Metadata == other.Metadata;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Version, Cells.Count, Running, Executing.Count, Layout, Kernels, HashCode.Combine(Unsaved, ThemeId, Metadata), Arrangement);

    /// <summary>
    /// This change and a later one, taken as one: the later look at each cell, only cells that still stand, what runs, the
    /// kernels, whether the notebook is unsaved and the theme it chose as the later one tells it, and the latest look at the
    /// buttons, at the arrangement and at what the notebook says of itself either had.
    /// </summary>
    /// <param name="later">The later change.</param>
    /// <returns>Both, as one change up to the later one's version.</returns>
    internal NotebookChange Then(NotebookChange later)
    {
        var order = later.Order ?? Order;
        var again = later.Cells.Select(cell => cell.Id).ToHashSet();
        var standing = order?.ToHashSet();
        IEnumerable<HostedCell> cells = [.. Cells.Where(cell => !again.Contains(cell.Id)), .. later.Cells];

        return new NotebookChange(
            later.Version,
            order,
            [.. cells.Where(cell => standing?.Contains(cell.Id) ?? true)],
            later.Running,
            later.Executing,
            later.Layout,
            later.Buttons ?? Buttons,
            later.Kernels,
            later.Unsaved,
            later.ThemeId,
            later.Arrangement ?? Arrangement,
            later.Metadata ?? Metadata);
    }
}
