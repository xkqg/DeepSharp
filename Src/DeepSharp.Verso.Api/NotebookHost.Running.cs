// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Verso;

namespace DeepSharp.Verso.Api;

// What runs, and what the engine says about it: a run of one cell, the run under way, what runs that no run owns, what
// became of the kernels, and the stop that ends a run. The engine speaks from inside the work that runs, so what it
// says is the run whose flow carries it.
public sealed partial class NotebookHost
{
    /// <summary>Runs a cell, as pressing its run button does.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The cell as it stands after the run, or after <see cref="Stop"/> ended it.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <remarks>
    /// The cell runs where the engine runs it: in its type's kernel, or none when its type only draws; else in the kernel
    /// its language names; else in none when a renderer claims its type; else in the notebook's default kernel. A C# run
    /// takes its turn among every C# run in the process, because a C# kernel takes over the process's console while it
    /// runs, and two at once would print into each other; while it waits for that turn it is the run under way all the
    /// same, told to every view and stopped like any other. A cell that only draws takes no C# turn.
    /// </remarks>
    public Task<HostedCell> RunAsync(Guid cell) => TurnAsync(async () =>
    {
        var running = Standing(cell);
        var run = RunFor(running.Id, running.KernelIn(Scaffold, Extensions).IsNamed(EngineExtensions.CSharp));

        // The engine is handed the run's token, as Verso's browser editor hands it its own.
        await RunUntilStoppedAsync(run, () => Scaffold.ExecuteCellAsync(running.Id, run.Token));

        return (await PublishedAsync()).Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// Stops a run — a cell's, or a toolbar button's — whether it still waits for another notebook's C# run or runs. One
    /// that waits never runs: its wait ends, and when the C# turn it waited for comes, nothing starts. One that runs is
    /// stopped the only way Verso's engine stops a run that does not end, with a fresh kernel — the kernel of what ran at
    /// the stop, a cell or code a button runs in no cell — so what that kernel held, the notebook's variables and the
    /// pipeline handed to C# cells among them, is gone. The stop is one step: unless the run already ended by itself, it
    /// takes the run's end, marks the run, tells the notebook — so what the run left behind asks for from then on writes
    /// nothing, and the notebook takes its next change at once — and only then takes what runs as the kernel to start
    /// afresh, which happens once everything let through before the stop has landed. A run that never ends goes on in the
    /// background until the application does. A button that runs cells is stopped cell by cell: the cell under way is left
    /// behind, no cell it would still run begins, nothing else it asks of the notebook is done, and a file it hands over
    /// goes to nobody; a stop between two cells starts no kernel afresh.
    /// </summary>
    /// <param name="run">The run, by its number — so a stop sent again after that run ended stops no other.</param>
    /// <returns>Whether it stopped that run; nothing is stopped when no run of that number is under way.</returns>
    public bool Stop(long run) => Volatile.Read(ref _running) is { } under && under.Number == run && under.Stop();

    /// <summary>
    /// The run under way as it stands now — the cell that runs or waits, since when, and the number a stop names — or
    /// nothing; every view is told it with each version, from the run's start. A button's run names the cell it runs
    /// now, and none before its first cell, between two, or while it runs code in no cell.
    /// </summary>
    public HostedRun? Running => Volatile.Read(ref _running)?.Hosted;

    /// <summary>
    /// What the engine runs now that no run owns, in the order each began: a block a change runs — a click, a changed
    /// field — and what a stop left behind, each from the engine's word it began to its word it ended; every view is told
    /// it with each version, as running with nothing to stop.
    /// </summary>
    public IReadOnlyList<HostedExecution> Executing => _executions.Now(Volatile.Read(ref _running));

    /// <summary>
    /// What became of the notebook's kernels: how many times one was started afresh — by a stop, or by Verso's Restart
    /// Kernel — whether one is being started afresh now, and why the last start failed; every view is told it with each
    /// version.
    /// </summary>
    public HostedKernels Kernels => Volatile.Read(ref _restarts).Hosted;

    // What a kernel is asked while nothing else holds it still: answered only when no start afresh overlapped the asking —
    // none under way as it began, none begun while it read, as the engine says it begins one before it puts the kernel
    // away. A read that overlapped one answers nothing, whatever the kernel said, and so does one the kernel failed while
    // it was put away: what it would have said belongs to a kernel no longer there.
    private async Task<T> KernelReadAsync<T>(Func<Task<T>> read, T nothing)
    {
        var before = Volatile.Read(ref _restarts);

        if (before.Restarting)
        {
            return nothing;
        }

        try
        {
            var answer = await read();

            return Volatile.Read(ref _restarts).Begun == before.Begun ? answer : nothing;
        }
        catch (Exception gone) when (gone is ObjectDisposedException or InvalidOperationException && Volatile.Read(ref _restarts).Begun != before.Begun)
        {
            return nothing;
        }
    }

    // The engine says a cell began: it is what the run whose work the engine is in runs now. A cell the engine begins
    // outside every run — a block a change runs — is no run's: every view is told it runs, with nothing to stop. Either
    // way the kernels answer again, so a start afresh that failed before is past.
    private void Began(Guid cell)
    {
        ImmutableInterlocked.Update(ref _restarts, restarts => restarts.Answered());

        // The engine found the cell a moment ago, and nothing else changes the notebook while a run holds its turn.
        if (_raising.Value is { } run)
        {
            run.Began(cell, Scaffold.Cells.First(each => each.Id == cell).KernelIn(Scaffold, Extensions));
        }
        else
        {
            _executions.Began(cell);
        }

        Said();
    }

    // The engine says a cell ended: the run whose work that was runs nothing now, unless something else began since — a
    // run a stop left behind among them, which is then told running no more; a cell no run owns is told running no more.
    private void Ended(Guid cell)
    {
        if (_raising.Value is { } run)
        {
            run.Ended(cell);
        }
        else
        {
            _executions.Ended(cell);
        }

        Said();
    }

    // The engine says a cell showed something.
    private void Showed(Guid cell) => Said();

    // The engine begins to start a kernel afresh: every view is told at once, since a start takes a while.
    private void Restarting(string? kernel)
    {
        ImmutableInterlocked.Update(ref _restarts, restarts => restarts.Began());
        Said();
    }

    // The engine started a kernel afresh: it is counted, and what the kernel held — the notebook's variables — is gone.
    private void Restarted(string? kernel)
    {
        ImmutableInterlocked.Update(ref _restarts, restarts => restarts.Ended());
        Said();
    }

    // The engine failed to start a kernel afresh: every view is told why, until one starts afresh or a cell begins.
    private void RestartFailed(string? kernel, Exception why)
    {
        ImmutableInterlocked.Update(ref _restarts, restarts => restarts.Failed(why.Message));
        Said();
    }

    /// <summary>A run, numbered for the notebook, taking the C# turn when it runs C#.</summary>
    /// <param name="cell">The cell it runs; none when it runs several, or code in no cell.</param>
    /// <param name="runsCSharp">Whether it runs C#.</param>
    /// <returns>The run.</returns>
    /// <remarks>Its stop tells the notebook in the step that stops it, so what it asks for from then on writes nothing.</remarks>
    internal Run RunFor(Guid? cell, bool runsCSharp) => new(Interlocked.Increment(ref _runs), cell, takesTheCSharpTurn: runsCSharp, this);

    // A run tells every view whenever what it runs now changes.
    void IRunListener.Moved() => Said();

    // A run's stop tells the notebook's blocks, in the step that stops it.
    Task IRunListener.StoppedAsync() => _blocks.StoppedAsync();

    /// <summary>
    /// A run a change asks for, run as every run is — told to every view from its ask, in the C# turn when it runs C#, and
    /// stopped as any run is — with a press's work: through the run's port, so a stop reaches every cell it would still run.
    /// </summary>
    /// <param name="run">The run.</param>
    /// <param name="work">What it does through the run's port.</param>
    /// <returns>A task that ends when the run ends, or is stopped.</returns>
    internal Task RunForChangeAsync(Run run, Func<RunPort, Task> work) => RunUntilStoppedAsync(run, () => work(new RunPort(Scaffold, run)));

    /// <summary>Whether a cell runs in the C# kernel, by the kernel rule.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>Whether it does; not for a cell the notebook does not have.</returns>
    internal bool RunsCSharp(Guid cell) =>
        Scaffold.GetCell(cell) is { } found && found.KernelIn(Scaffold, Extensions).IsNamed(EngineExtensions.CSharp);

    /// <summary>Whether code in a language — the notebook's default kernel when it names none — runs in the C# kernel.</summary>
    /// <param name="language">The language.</param>
    /// <returns>Whether it does.</returns>
    internal bool RunsCSharp(string? language) => (language ?? Scaffold.DefaultKernelId).IsNamed(EngineExtensions.CSharp);

    // Runs until it ends, or until Stop. The run is under way from here, so a Stop finds it even while it waits: a C#
    // run first waits its turn among the process's C# runs, and when that turn is not free every view is told the run
    // waits. A Stop then ends the wait at once, and the turn, when it comes, starts nothing. Only the run under way can
    // be stopped, so the slot holds one, and a Stop after it ended meets a run nobody waits for any more. A close marks
    // the notebook closed before it looks for the run under way, and the run takes its slot before it looks for a close,
    // each by an exchange, so one always finds the other: a run whose turn began as the notebook shut stops itself. A run
    // stopped before it ran waits for what the notebook let through before the stop, as every stopped run does.
    private async Task<bool> RunUntilStoppedAsync(Run run, Func<Task> start)
    {
        Interlocked.Exchange(ref _running, run);

        try
        {
            if (Volatile.Read(ref _audience).Closed)
            {
                run.Stop();
            }

            if (!run.TakesTheCSharpTurn)
            {
                return run.Starts() ? await GoAsync(run, start) : await StoppedBeforeItRanAsync(run);
            }

            var turned = CSharpRuns.TakeTurnAsync(() => run.Starts() ? GoAsync(run, start) : Task.FromResult(false));

            // The turn was free when it began at once; otherwise the run waits, and every view is told.
            if (run.Waits)
            {
                await PublishedAsync();
            }

            await Task.WhenAny(turned, run.Stopped);

            return run.Stopped.IsCompleted && (await run.Stopped).BeforeItRan ? await StoppedBeforeItRanAsync(run) : await turned;
        }
        finally
        {
            Volatile.Write(ref _running, null);
        }
    }

    // A run stopped before it ran never runs; its flow ends once what was let through before the stop has landed.
    private static async Task<bool> StoppedBeforeItRanAsync(Run run)
    {
        await (await run.Stopped).Drained;

        return false;
    }

    // A run under way, until it ends or a stop ends it; every view is told it from its start. It is not asked to stop,
    // since one that does not end does not listen either (measured: a C# loop that awaits goes on with its token
    // cancelled); a fresh kernel ends the turn, and the run is left behind. While it runs, its turn tells every view what
    // the engine says it shows, so a view sees it before the run ends. Whichever comes first — the work's end or a stop —
    // only wakes the flow: the run's end is decided by one exchange on the run, and a stop hands the flow its decision
    // whole, so nothing here reads again what runs once it has moved on.
    private async Task<bool> GoAsync(Run run, Func<Task> start)
    {
        await PublishedAsync();

        // What the engine says from inside this work is this run's.
        _raising.Value = run;

        var ran = start();

        await Task.WhenAny(ran, run.Stopped);

        if (ran.IsCompleted && run.Ends())
        {
            // Ended by itself: what the work met, a fault among it, is its own and reaches whoever asked for it.
            await ran;

            return true;
        }

        // Stopped: the notebook was told in the stop's own step. Whatever the run still runs goes on without it, and every
        // view is told so until the engine says it ended. Only once what the notebook and the run let through before the
        // stop has landed is the kernel that ran at the stop started afresh — none when nothing ran, before the first cell,
        // between two cells, while Run All resets the kernels, or while a cell only draws, so what the kernels hold stays.
        var stop = await run.Stopped;

        _executions.LeftBehind(run);
        await stop.Drained;

        if (stop.Ran is { Kernel: { } kernel })
        {
            await Scaffold.RestartKernelAsync(kernel);
        }

        return true;
    }
}
