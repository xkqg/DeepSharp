// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// An open notebook as of one version: every cell in order, the run under way, what the engine runs that no run owns, the
/// layout it is shown in and what that layout draws of its own, every toolbar button, each saying whether it can be pressed
/// then, what became of its kernels, whether it differs from its file, and what it says of itself.
/// </summary>
/// <param name="Version">Its number: the notebook opens at 0, and each change to it makes the next.</param>
/// <param name="Cells">Its cells, in order.</param>
/// <param name="Running">The run under way at this version; nothing when none is.</param>
/// <param name="Executing">
/// What the engine runs at this version that no run owns — a block a change runs, or what a stop left behind — which a
/// view shows running, with nothing to stop, in the order each began.
/// </param>
/// <param name="Layout">The layout it is shown in at this version, and what that lets a person do to its cells.</param>
/// <param name="Buttons">
/// Every toolbar button the engine has, by place and then in their order, each asked at this version whether it can be
/// pressed — as a page draws them, with nothing to ask apart.
/// </param>
/// <param name="Kernels">What became of its kernels up to this version.</param>
/// <param name="Unsaved">
/// Whether it differs at this version from the file it was last saved to, as Verso's own comparison of two notebooks finds
/// it — what a save would change in the file.
/// </param>
/// <param name="ThemeId">
/// The theme the notebook chose, one of <see cref="NotebookHost.Themes"/>; nothing while it chose none, when a view draws it
/// in its own look.
/// </param>
/// <param name="Arrangement">
/// What the layout it is shown in draws of its own at this version — Verso's dashboard, its presentation — with a slot for
/// each cell it shows; nothing for the notebook's own layout, its list of cells.
/// </param>
/// <param name="Metadata">What it says of itself at this version: its title, default kernel, dates and format.</param>
/// <remarks>Two looks at a version are equal while they hold the same, whichever list the cells came in.</remarks>
public readonly record struct NotebookVersion(
    long Version,
    IReadOnlyList<HostedCell> Cells,
    HostedRun? Running,
    IReadOnlyList<HostedExecution> Executing,
    HostedLayout Layout,
    IReadOnlyList<HostedToolbarAction> Buttons,
    HostedKernels Kernels,
    bool Unsaved,
    string? ThemeId,
    HostedArrangement Arrangement,
    HostedMetadata Metadata)
{
    /// <inheritdoc />
    public bool Equals(NotebookVersion other) =>
        Version == other.Version && Cells.SequenceEqual(other.Cells) && Running == other.Running && Executing.SequenceEqual(other.Executing)
        && Layout == other.Layout && Buttons.SequenceEqual(other.Buttons) && Kernels == other.Kernels && Unsaved == other.Unsaved && ThemeId == other.ThemeId
        && Arrangement == other.Arrangement && Metadata == other.Metadata;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Version, Cells.Count, Running, Executing.Count, Layout, Kernels, HashCode.Combine(Unsaved, ThemeId, Metadata), Arrangement);
}
