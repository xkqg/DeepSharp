// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// A run, from the moment it is asked: its number, the cell it runs, and one value — whether it has yet to start, runs,
/// was stopped or ended by itself, what it runs now, and the verbs on their way through it — which every change replaces
/// whole in one exchange, so a stop and a start, an end or a cell the engine begins, arriving together, agree on which
/// came first.
/// </summary>
/// <param name="number">Which run it is, counted from one for each open notebook.</param>
/// <param name="cell">The cell it runs; none for a button's run.</param>
/// <param name="takesTheCSharpTurn">Whether it takes its turn among the process's C# runs.</param>
/// <param name="told">
/// Asked to tell every view whenever what the run runs now changes — code it runs in no cell among it, of which the engine
/// says nothing — so a run that runs on after it was stopped is told as it ends.
/// </param>
/// <param name="tell">
/// Tells the notebook of the run's stop, in the step that stops it, so what the run asks for from then on writes nothing;
/// it hands back what ends once every write let through before the stop has landed.
/// </param>
/// <remarks>
/// A stop is one step, in this order: it takes the run's end, unless the run already ended by itself; it marks the run,
/// which the engine and every part the run acts through read; it tells the notebook; and only then does it take what runs
/// now as the kernel to start afresh. So what the engine lets begin after the stop began before it, and the decision is
/// one value handed to the run's own flow whole, never read again once what runs has moved on.
/// </remarks>
internal sealed class Run(long number, Guid? cell, bool takesTheCSharpTurn, Action told, Func<Task> tell)
{
    // The stop's decision, set once by the stop that takes the run.
    private readonly TaskCompletionSource<RunStop> _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);

    // Ended once a stop took the run and no verb let through before it is still on its way.
    private readonly TaskCompletionSource _landed = new(TaskCreationOptions.RunContinuationsAsynchronously);

    // Marked by the stop before anyone waiting on it is woken, so whatever the run still asks sees it stopped first.
    private readonly CancellationTokenSource _stop = new();

    private State _state = new(Phase.Before, null, 0, Resetting: false);

    // Where a run stands.
    private enum Phase
    {
        // Asked, and not yet running: it may still wait for the C# turn.
        Before,

        // Running.
        Going,

        // Taken by a stop while it ran, the stop's step not yet done.
        Stopping,

        // Taken by a stop while it ran.
        Stopped,

        // Taken by a stop before it ran: it never runs.
        StoppedFirst,

        // Ended by itself: no stop takes it any more.
        Ended,
    }

    public long Number => number;

    public DateTimeOffset Asked { get; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// The stop's decision, once a stop took the run: whether it came before the run ran, what ran at the stop, and what
    /// ends once everything let through before it has landed.
    /// </summary>
    public Task<RunStop> Stopped => _stopped.Task;

    // Marked when the run is stopped: what the run asks of the notebook, and the engine running it, look at it.
    public CancellationToken Token => _stop.Token;

    // Whether it takes its turn among the process's C# runs.
    public bool TakesTheCSharpTurn => takesTheCSharpTurn;

    // Whether it still waits for the C# turn.
    public bool Waits => takesTheCSharpTurn && Current().Phase == Phase.Before;

    // Whether a stop took the run: what it asks is refused, and what it would answer or hand over is dropped.
    public bool Claimed => Current().Phase is Phase.Stopping or Phase.Stopped or Phase.StoppedFirst;

    // What runs now; nothing before the first cell, between two cells, and once what ran has ended.
    public Underway? Now => Current().Underway;

    // The run as every view is told it: its number, the cell that runs now — none for code with no cell — or else the cell
    // it was asked for, and since when.
    public HostedRun Hosted
    {
        get
        {
            var now = Current();
            var waits = takesTheCSharpTurn && now.Phase == Phase.Before;

            return now.Underway is { } underway
                ? new HostedRun(number, underway.Cell, underway.Since, waits)
                : new HostedRun(number, cell, Asked, waits);
        }
    }

    // It may start now: it runs, unless a stop came first.
    public bool Starts() => Exchanged(state => state.Phase == Phase.Before ? state with { Phase = Phase.Going } : null) is not null;

    // Its own flow says its work ended: the run ended by itself, unless a stop took its end first.
    public bool Ends() => Exchanged(state => state.Phase == Phase.Going ? state with { Phase = Phase.Ended } : null) is not null;

    // Stops it, in one step: takes its end, marks it, tells the notebook, and takes what runs now. Whether this stop took the
    // run; a run already stopped, or one that ended by itself, is not stopped again and nothing is told.
    public bool Stop()
    {
        var taken = Exchanged(state => state.Phase switch
        {
            Phase.Before => state with { Phase = Phase.StoppedFirst },
            Phase.Going => state with { Phase = Phase.Stopping },
            _ => null,
        });

        if (taken is null)
        {
            return false;
        }

        // Marked at once; what listens to the mark is told elsewhere, so a stop never waits for it.
        _ = _stop.CancelAsync();

        var written = tell();
        Underway? ran = null;

        if (taken.Phase == Phase.Stopping)
        {
            // Read in the exchange that ends the step: whatever the engine began before it is what runs.
            Exchanged(state =>
            {
                ran = state.Underway;

                return state with { Phase = Phase.Stopped };
            });
        }

        LandedIfDone(Current());
        _stopped.TrySetResult(new RunStop(taken.Phase == Phase.StoppedFirst, ran, LandedAsync(written)));

        return true;
    }

    // Refuses a verb that runs cells or code once a stop took the run. It is not counted as on its way: a stop ends what it
    // runs by starting a kernel afresh, never by waiting for it.
    public void ThrowIfStopped()
    {
        if (Claimed)
        {
            throw new OperationCanceledException(Token);
        }
    }

    // Lets through a verb the run's work asks of the notebook — clearing, adding, moving, switching — unless a stop took the
    // run, and counts it until it has landed, so the stop's start afresh waits for it. A run that ended by itself lets it
    // through uncounted: no stop can wait for it any more.
    public Admission Admit() => Admitted(untilBegan: false);

    // Lets Run All through, counted until the first cell begins: its reset puts every kernel away and clears the notebook's
    // variables before any cell begins, so a stop during it waits for the reset to end.
    public Admission AdmitUntilBegan() => Admitted(untilBegan: true);

    // A cell began, or code with no cell: it is what runs now, in its kernel, since now; a reset on its way has ended.
    public void Began(Guid? began, string? kernel)
    {
        var underway = new Underway(began, kernel, DateTimeOffset.UtcNow);

        LandedIfDone(Exchanged(state => state.Resetting
            ? state with { Underway = underway, InFlight = state.InFlight - 1, Resetting = false }
            : state with { Underway = underway })!);
        told();
    }

    // What ran ended — a cell, by its id, or code with no cell, by none — when that is what runs now.
    public void Ended(Guid? ended)
    {
        if (Exchanged(state => state.Underway is { } now && now.Cell == ended ? state with { Underway = null } : null) is not null)
        {
            told();
        }
    }

    // A verb let through has landed.
    internal void Landed(bool untilBegan)
    {
        if (Exchanged(state => !untilBegan
            ? state with { InFlight = state.InFlight - 1 }
            : state.Resetting ? state with { InFlight = state.InFlight - 1, Resetting = false } : null) is { } after)
        {
            LandedIfDone(after);
        }
    }

    private Admission Admitted(bool untilBegan)
    {
        var counted = false;

        var after = Exchanged(state =>
        {
            counted = state.Phase is Phase.Before or Phase.Going;

            return state.Phase switch
            {
                Phase.Before or Phase.Going when untilBegan => state with { InFlight = state.InFlight + 1, Resetting = true },
                Phase.Before or Phase.Going => state with { InFlight = state.InFlight + 1 },
                Phase.Ended => state,
                _ => null,
            };
        });

        return after is null ? throw new OperationCanceledException(Token) : new Admission(counted ? this : null, untilBegan);
    }

    // Once a stop took the run and nothing let through is on its way, the verbs have landed.
    private void LandedIfDone(State now)
    {
        if (now.Phase is Phase.Stopping or Phase.Stopped or Phase.StoppedFirst && now.InFlight == 0)
        {
            _landed.TrySetResult();
        }
    }

    // What the stop's flow waits for before it starts a kernel afresh: the notebook's writes and the run's verbs let through
    // before the stop.
    private async Task LandedAsync(Task written)
    {
        await written;
        await _landed.Task;
    }

    private State Current() => Volatile.Read(ref _state);

    // Makes the next value from the one there is, and puts it in place whole; nothing when the transition refuses.
    private State? Exchanged(Func<State, State?> transition)
    {
        while (true)
        {
            var before = Current();

            if (transition(before) is not { } after)
            {
                return null;
            }

            if (ReferenceEquals(Interlocked.CompareExchange(ref _state, after, before), before))
            {
                return after;
            }
        }
    }

    // The run at one moment.
    private sealed record State(Phase Phase, Underway? Underway, int InFlight, bool Resetting);
}

/// <summary>What a run runs now: a cell, or code with no cell, the kernel that runs it, and since when.</summary>
/// <param name="Cell">The cell; none for code.</param>
/// <param name="Kernel">The kernel that runs it, which a stop starts afresh; none when no kernel runs it — a cell that only draws.</param>
/// <param name="Since">When it began.</param>
internal readonly record struct Underway(Guid? Cell, string? Kernel, DateTimeOffset Since);

/// <summary>A stop's decision, taken in the step that stopped the run.</summary>
/// <param name="BeforeItRan">Whether the run was stopped before it ran: it never runs.</param>
/// <param name="Ran">What ran at the stop — the kernel to start afresh; nothing when nothing ran.</param>
/// <param name="Drained">
/// Ends once every write the notebook let through before the stop, and every verb the run let through before it, has landed.
/// </param>
internal readonly record struct RunStop(bool BeforeItRan, Underway? Ran, Task Drained);

/// <summary>A verb let through a run, counted until it has landed; nothing to count for a run no stop can take any more.</summary>
/// <param name="Run">The run it was counted on; nothing when it was not counted.</param>
/// <param name="UntilBegan">Whether it is on its way only until the run's first cell begins: Run All's reset.</param>
internal readonly record struct Admission(Run? Run, bool UntilBegan) : IDisposable
{
    public void Dispose() => Run?.Landed(UntilBegan);
}
