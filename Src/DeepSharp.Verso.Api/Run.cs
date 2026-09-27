// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;

namespace DeepSharp.Verso.Api;

/// <summary>
/// A run, from the moment it is asked: its number, the cell it runs, what it runs now, and whether it has yet to start,
/// runs, or was stopped before it ran. Each change of that is one exchange, so a stop and the start that arrive together
/// agree on which came first, and a run stopped before it started never starts.
/// </summary>
/// <param name="number">Which run it is, counted from one for each open notebook.</param>
/// <param name="cell">The cell it runs; none for a button's run.</param>
/// <param name="takesTheCSharpTurn">Whether it takes its turn among the process's C# runs.</param>
internal sealed class Run(long number, Guid? cell, bool takesTheCSharpTurn)
{
    private const int Before = 0;
    private const int Going = 1;
    private const int StoppedFirst = 2;

    private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);

    // Marked by the stop before anyone waiting on it is woken, so whatever the run still asks sees it stopped first.
    private readonly CancellationTokenSource _stop = new();

    private int _state = Before;

    // What runs now — a cell the engine began, or code with no cell — until it ends.
    private StrongBox<Underway>? _underway;

    public long Number => number;

    public DateTimeOffset Asked { get; } = DateTimeOffset.UtcNow;

    public Task Stopped => _stopped.Task;

    // Marked when the run is stopped: what the run asks of the notebook, and the engine running it, look at it.
    public CancellationToken Token => _stop.Token;

    // Whether it takes its turn among the process's C# runs.
    public bool TakesTheCSharpTurn => takesTheCSharpTurn;

    // Whether it still waits for the C# turn.
    public bool Waits => takesTheCSharpTurn && Volatile.Read(ref _state) == Before;

    // Whether a stop came before it ran: then it never runs.
    public bool StoppedBeforeItRan => Volatile.Read(ref _state) == StoppedFirst;

    // What runs now; nothing before the first cell, between two cells, and once what ran has ended.
    public Underway? Now => Volatile.Read(ref _underway)?.Value;

    // The run as every view is told it: its number, the cell that runs now — none for code with no cell — or else the cell
    // it was asked for, and since when.
    public HostedRun Hosted => Now is { } now
        ? new HostedRun(number, now.Cell, now.Since, Waits)
        : new HostedRun(number, cell, Asked, Waits);

    // It may start now: it runs, unless a stop came first.
    public bool Starts() => Interlocked.CompareExchange(ref _state, Going, Before) == Before;

    // Stops it: a run that has yet to start never runs; one that runs is ended by whoever runs it. Whether this stop was
    // the first.
    public bool Stop()
    {
        Interlocked.CompareExchange(ref _state, StoppedFirst, Before);

        // Marked at once; what listens to the mark is told elsewhere, so a stop never waits for it.
        _ = _stop.CancelAsync();

        return _stopped.TrySetResult();
    }

    // A cell began, or code with no cell: it is what runs now, in its kernel, since now.
    public void Began(Guid? began, string? kernel) =>
        Volatile.Write(ref _underway, new StrongBox<Underway>(new Underway(began, kernel, DateTimeOffset.UtcNow)));

    // What ran ended — a cell, by its id, or code with no cell, by none — when that is what runs now.
    public void Ended(Guid? ended)
    {
        var now = Volatile.Read(ref _underway);

        if (now is not null && now.Value.Cell == ended)
        {
            Interlocked.CompareExchange(ref _underway, null, now);
        }
    }
}

/// <summary>What a run runs now: a cell, or code with no cell, the kernel that runs it, and since when.</summary>
/// <param name="Cell">The cell; none for code.</param>
/// <param name="Kernel">The kernel that runs it, which a stop starts afresh; none when no kernel runs it — a cell that only draws.</param>
/// <param name="Since">When it began.</param>
internal readonly record struct Underway(Guid? Cell, string? Kernel, DateTimeOffset Since);
