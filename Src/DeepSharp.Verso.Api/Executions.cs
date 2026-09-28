// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;

namespace DeepSharp.Verso.Api;

/// <summary>
/// Everything the engine runs in a notebook that no run owns: a cell it began while no run was under way — a block a
/// change runs — until it says that cell ended; and whatever a run a stop left behind still runs, until the engine says
/// it ended inside that run's own work. Each is kept as the engine said it, so neither overwrites the other: a cell a
/// change begins leaves a cell left behind told, and the end of either takes back only its own.
/// </summary>
internal sealed class Executions
{
    // The cells begun while no run was under way, and since when: the engine runs such a cell once at a time.
    private ImmutableDictionary<Guid, DateTimeOffset> _unowned = ImmutableDictionary<Guid, DateTimeOffset>.Empty;

    // The runs a stop left behind, each of which says itself what it still runs.
    private ImmutableList<Run> _leftBehind = ImmutableList<Run>.Empty;

    /// <summary>The engine began a cell while no run was under way.</summary>
    /// <param name="cell">The cell.</param>
    public void Began(Guid cell) => ImmutableInterlocked.Update(ref _unowned, unowned => unowned.SetItem(cell, DateTimeOffset.UtcNow));

    /// <summary>The engine says a cell it began while no run was under way ended.</summary>
    /// <param name="cell">The cell.</param>
    public void Ended(Guid cell) => ImmutableInterlocked.Update(ref _unowned, unowned => unowned.Remove(cell));

    /// <summary>A stop left a run behind: whatever it runs goes on without it, until the engine says it ended.</summary>
    /// <param name="run">The run.</param>
    public void LeftBehind(Run run) => ImmutableInterlocked.Update(ref _leftBehind, left => left.Add(run));

    /// <summary>Forgets every run left behind whose work has ended.</summary>
    public void Forget() => ImmutableInterlocked.Update(ref _leftBehind, left => left.RemoveAll(run => run.Now is null));

    /// <summary>
    /// What the engine runs now that no run owns, in the order each began. The run under way is not among them, even once
    /// stopped: it is told as the run under way until its turn ends.
    /// </summary>
    /// <param name="running">The run under way; nothing when none is.</param>
    /// <returns>What runs.</returns>
    public IReadOnlyList<HostedExecution> Now(Run? running) =>
    [
        .. Volatile.Read(ref _unowned).Select(unowned => new HostedExecution(unowned.Key, unowned.Value, LeftBehind: false))
            .Concat(Volatile.Read(ref _leftBehind).Where(run => run != running).Select(run => run.Now).OfType<Underway>()
                .Select(left => new HostedExecution(left.Cell, left.Since, LeftBehind: true)))
            .OrderBy(each => each.Since),
    ];
}
