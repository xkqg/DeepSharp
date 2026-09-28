// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// What became of a notebook's kernels, as the engine says it: how many times one was started afresh, how many starts are
/// under way, why the last one failed, and how many starts afresh ever began. One value, made anew for each word and
/// swapped whole, so a version never tells half of one.
/// </summary>
/// <param name="count">How many times a kernel was started afresh.</param>
/// <param name="underway">How many starts afresh are under way.</param>
/// <param name="fault">Why the last start afresh failed; nothing when none did, or once the kernels answer again.</param>
/// <param name="begun">How many starts afresh ever began, told the moment the engine says one begins.</param>
internal sealed class Restarts(long count, int underway, string? fault, long begun)
{
    /// <summary>A notebook's kernels as it opens: none started afresh.</summary>
    public static readonly Restarts None = new(0, 0, null, 0);

    /// <summary>As every view is told it.</summary>
    public HostedKernels Hosted => new(count, underway > 0, fault);

    /// <summary>Whether a kernel is being started afresh now.</summary>
    public bool Restarting => underway > 0;

    /// <summary>How many starts afresh ever began: a read that sees it move overlapped one.</summary>
    public long Begun => begun;

    /// <summary>A kernel begins to start afresh.</summary>
    /// <returns>The kernels after.</returns>
    public Restarts Began() => new(count, underway + 1, fault, begun + 1);

    /// <summary>A kernel started afresh: it is counted, and a start that failed before is past.</summary>
    /// <returns>The kernels after.</returns>
    public Restarts Ended() => new(count + 1, underway - 1, null, begun);

    /// <summary>A kernel failed to start afresh.</summary>
    /// <param name="why">Why.</param>
    /// <returns>The kernels after.</returns>
    public Restarts Failed(string why) => new(count, underway - 1, why, begun);

    /// <summary>A cell began: the kernels answer again, as Verso's editors stop showing a failed start once a cell runs.</summary>
    /// <returns>The kernels after; these, when no start had failed.</returns>
    public Restarts Answered() => fault is null ? this : new(count, underway, null, begun);
}
