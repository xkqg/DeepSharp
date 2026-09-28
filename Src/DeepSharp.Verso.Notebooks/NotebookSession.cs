// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What is known of the bytes a notebook's source holds now.</summary>
/// <param name="Known">Whether they were read at all: a gesture is handed no file, so it knows nothing of them.</param>
/// <param name="Fingerprint">A SHA-256 of the bytes, when they were read; nothing when they could not be.</param>
internal readonly record struct SourceBytes(bool Known, string? Fingerprint)
{
    /// <summary>Nothing read: what a gesture knows, which is handed no file.</summary>
    public static SourceBytes Unknown { get; } = new(false, null);

    /// <summary>Read, and gone: the file could not be opened, so nothing learned from it holds.</summary>
    public static SourceBytes Unreadable { get; } = new(true, null);

    /// <summary>Read, with this fingerprint.</summary>
    /// <param name="fingerprint">A SHA-256 of the bytes.</param>
    /// <returns>What is known.</returns>
    public static SourceBytes Of(string fingerprint) => new(true, fingerprint);

    /// <summary>Whether a fit learned from bytes with this fingerprint may still stand.</summary>
    /// <param name="learnedFrom">The fingerprint of the bytes it was learned from.</param>
    /// <returns><see langword="true"/> unless the bytes were read and are other ones.</returns>
    public bool Allow(string learnedFrom) => !Known || Fingerprint == learnedFrom;
}

/// <summary>
/// One change's turn on a notebook, as it began: how many runs had been stopped by then, and the mark of the run the
/// change belongs to.
/// </summary>
/// <param name="Stops">How many runs had been stopped when the change began.</param>
/// <param name="Mark">Marked when the run the change belongs to is stopped; never, for a change no run owns.</param>
/// <remarks>
/// A stop that comes after the turn began voids what the change asks for from then on: a run it started goes on by
/// itself, and writes nothing more — no grid, and nothing handed to C# cells. A host marks a run before it tells the
/// notebook of its stop, so a write that finds the count unchanged but the mark set is refused all the same.
/// </remarks>
internal readonly record struct NotebookTurn(int Stops, CancellationToken Mark);

/// <summary>
/// What only holds between the calls on one notebook: which block shows what, what a gesture asked a block for, what
/// the notebook hands to C# cells, and that one change on it runs at a time.
/// </summary>
/// <remarks>
/// Kept by the block type Verso loaded, the one object every part of a notebook reaches; nothing here is saved, and
/// nothing is shared with another notebook. No host promises one call at a time, so everything the session knows is
/// one value: each change makes the next value from the one there is and puts it in place whole, and a reader takes
/// the value there is. A reader never sees half a change, and no change is lost to another made beside it.
/// <para>
/// One view is kept: the last one worked out, under the key of the steps it is worked out from and the fingerprint of
/// the bytes it was read from. A view is those two and nothing else, so while both stay it is the same view, and
/// another page of it, or the same view shown again, runs no step. One, because a view holds every row at its block;
/// the rows as read are kept apart, by the source's own cache.
/// </para>
/// <para>
/// The pipeline handed to C# cells goes through here and nowhere else, always into the variables the caller was
/// handed: the declaration the blocks make, or — only while the blocks declare the same steps and the source holds
/// the same bytes — what this notebook's own last run learned from them. Whenever that may no longer hold, it is
/// taken back, and a C# cell sees none rather than one the blocks no longer make.
/// </para>
/// </remarks>
internal sealed class NotebookSession
{
    // Everything the session knows, as one value that each change replaces whole.
    private State _state = new();

    // The last change's turn: the next one waits for it, so they take the lane in the order they came.
    private Task _lane = Task.CompletedTask;

    /// <summary>The rows the notebook's source opened last, kept for the next view.</summary>
    public SourceCache Sources { get; } = new();

    /// <summary>
    /// How long a gesture that can change the blocks waits before it reads them: longer than the quarter of a second VS
    /// Code holds a keystroke before it sends it, which nothing tells the host is on its way.
    /// </summary>
    public static TimeSpan SettleTime { get; } = TimeSpan.FromMilliseconds(300);

    /// <summary>The pipeline the blocks made at the last gesture, when there was one.</summary>
    public NotebookPipeline? Assembled => Now.Assembled;

    /// <summary>How many views were worked out from the source up: the number keeping the last one exists to keep down.</summary>
    public int ViewsRun => Now.ViewsRun;

    /// <summary>How many times a run of the whole pipeline was fitted and handed over.</summary>
    public int RunsFitted => Now.RunsFitted;

    /// <summary>The blocks that show data, each with the key of the steps its rows were worked out from.</summary>
    public IReadOnlyDictionary<Guid, string> Shown => Now.Shown.ToDictionary(each => each.Key, each => each.Value.Key);

    // The value there is now: taken once, it holds still while it is read.
    private State Now => Volatile.Read(ref _state);

    /// <summary>
    /// Runs one change on the notebook — a gesture, a change in a block's form, a toolbar button — once every change made
    /// before it has finished.
    /// </summary>
    /// <typeparam name="T">What the change answers.</typeparam>
    /// <param name="mark">
    /// The mark of the run the change belongs to, as the host hands it the part; never marked, for a change no run owns.
    /// </param>
    /// <param name="change">
    /// The change, handed its turn: a gesture may ask a block for something with it, through <see cref="AskAsync"/>, and
    /// what it writes it writes through <see cref="LetThroughAsync(NotebookTurn, Func{Task})"/>.
    /// </param>
    /// <returns>Its answer.</returns>
    /// <exception cref="OperationCanceledException">
    /// The run the change belongs to was stopped before the change began: it is born stopped, does nothing, and lets the
    /// next change in.
    /// </exception>
    /// <remarks>
    /// A gesture leaves a request and then runs the block that takes it; two gestures that interleave would each take
    /// the other's, and a change in a form written while a gesture rewrites the blocks lands on a block that is going
    /// away. One host sends one request at a time, another runs them side by side, and this holds for both. A change
    /// handed on from inside another never runs: it waits for the one it is inside. A stop gives the lane back at once:
    /// the change it stopped goes on by itself, and what it asks for from then on writes nothing. The turn begins in the
    /// same step that gives the change the lane, and its run's mark is read only after, so a stop comes either before it
    /// — and the change never begins — or after, and gives its lane back.
    /// </remarks>
    public async Task<T> OneAtATimeAsync<T>(CancellationToken mark, Func<NotebookTurn, Task<T>> change)
    {
        var mine = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var before = Interlocked.Exchange(ref _lane, mine.Task);

        await before;

        var turn = default(NotebookTurn);

        Change(state =>
        {
            turn = new NotebookTurn(state.Stops, mark);

            return state with { Holder = mine };
        });

        try
        {
            mark.ThrowIfCancellationRequested();

            return await change(turn);
        }
        finally
        {
            // A change a stop gave the lane back for no longer holds it: the lane may be the next change's by now.
            Change(state => state.Holder == mine ? state with { Holder = null } : state);
            mine.TrySetResult();
        }
    }

    /// <summary>
    /// Counts a stop of the run under way, and gives the notebook back at once: the change that run holds lets the next
    /// one in, and what the run asks for from then on writes nothing — no grid, and nothing handed to C# cells — while
    /// the run itself goes on by itself.
    /// </summary>
    /// <returns>
    /// A task that ends once every write let through before the stop has landed; at once when none is on its way. It
    /// waits for no write let through after the stop.
    /// </returns>
    /// <remarks>
    /// The count, the lane given back and the writes the stop waits for are one step: a write is let through in the step
    /// that reads the count, so it is either counted here or refused.
    /// </remarks>
    public Task Stopped()
    {
        TaskCompletionSource? holding = null;
        TaskCompletionSource? waiting = null;

        Change(state =>
        {
            var stops = state.Stops + 1;

            holding = state.Holder;
            waiting = state.InFlight.IsEmpty ? null : new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

            return state with
            {
                Stops = stops,
                Holder = null,
                Drains = waiting is null ? state.Drains : state.Drains.Add(new Drain(stops, waiting)),
            };
        });

        holding?.TrySetResult();

        return waiting?.Task ?? Task.CompletedTask;
    }

    /// <summary>Whether a run was stopped since a change began: what the change asks for then writes nothing.</summary>
    /// <param name="turn">The change's turn.</param>
    /// <returns><see langword="true"/> once a stop came after the turn began, or its run was marked stopped.</returns>
    public bool StoppedSince(NotebookTurn turn) => Now.Stops != turn.Stops || turn.Mark.IsCancellationRequested;

    /// <summary>
    /// Lets one whole write of a change through — what it writes into the notebook, its cells or the files beside it —
    /// unless a stop came since the change began; worked out first, so nothing waits for a stop but the writing.
    /// </summary>
    /// <param name="turn">The change's turn.</param>
    /// <param name="write">The write, whole.</param>
    /// <returns><see langword="true"/> when it was let through and written; <see langword="false"/> when it was refused.</returns>
    /// <remarks>
    /// Let through in the same step that reads the count of stops, and counted until it has landed: a stop made while it
    /// is on its way waits for it, and one made before refuses it. Nothing else waits for anything, so no write ever
    /// waits for another.
    /// </remarks>
    public Task<bool> LetThroughAsync(NotebookTurn turn, Func<Task> write) => LetThroughAsync(turn, CancellationToken.None, write);

    /// <summary>
    /// Lets one whole write through for a change and the block's run it asked for, unless a stop came since the change
    /// began, or either run was marked stopped.
    /// </summary>
    /// <param name="turn">The change's turn.</param>
    /// <param name="also">The mark of the block's run, which the engine hands the block.</param>
    /// <param name="write">The write, whole.</param>
    /// <returns><see langword="true"/> when it was let through and written; <see langword="false"/> when it was refused.</returns>
    public async Task<bool> LetThroughAsync(NotebookTurn turn, CancellationToken also, Func<Task> write)
    {
        ArgumentNullException.ThrowIfNull(write);

        if (!Admitted(turn, also))
        {
            return false;
        }

        try
        {
            await write();
        }
        finally
        {
            Landed(turn.Stops);
        }

        return true;
    }

    /// <summary>Lets one whole write of a change through that ends where it begins, unless a stop came since the change began.</summary>
    /// <param name="turn">The change's turn.</param>
    /// <param name="write">The write, whole.</param>
    /// <returns><see langword="true"/> when it was let through and written; <see langword="false"/> when it was refused.</returns>
    public bool LetThrough(NotebookTurn turn, Action write) => LetThrough(turn, CancellationToken.None, write);

    /// <summary>
    /// Lets one whole write that ends where it begins through for a change and the block's run it asked for, unless a stop
    /// came since the change began, or either run was marked stopped.
    /// </summary>
    /// <param name="turn">The change's turn.</param>
    /// <param name="also">The mark of the block's run, which the engine hands the block.</param>
    /// <param name="write">The write, whole.</param>
    /// <returns><see langword="true"/> when it was let through and written; <see langword="false"/> when it was refused.</returns>
    public bool LetThrough(NotebookTurn turn, CancellationToken also, Action write)
    {
        ArgumentNullException.ThrowIfNull(write);

        if (!Admitted(turn, also))
        {
            return false;
        }

        try
        {
            write();
        }
        finally
        {
            Landed(turn.Stops);
        }

        return true;
    }

    // A write is let through while no stop came since its turn began and no run it belongs to is marked, and counted as on
    // its way under that count of stops, in one step.
    private bool Admitted(NotebookTurn turn, CancellationToken also)
    {
        var admitted = false;

        Change(state =>
        {
            admitted = state.Stops == turn.Stops && !turn.Mark.IsCancellationRequested && !also.IsCancellationRequested;

            return admitted ? state with { InFlight = state.InFlight.SetItem(turn.Stops, state.InFlight.GetValueOrDefault(turn.Stops) + 1) } : state;
        });

        return admitted;
    }

    // A write let through under this count of stops landed: every stop that waited for nothing else ends.
    private void Landed(int stops)
    {
        var ended = ImmutableList<Drain>.Empty;

        Change(state =>
        {
            var left = state.InFlight[stops] - 1;
            var inFlight = left == 0 ? state.InFlight.Remove(stops) : state.InFlight.SetItem(stops, left);

            ended = state.Drains.RemoveAll(drain => inFlight.Keys.Any(each => each < drain.Below));

            return state with { InFlight = inFlight, Drains = state.Drains.RemoveRange(ended) };
        });

        foreach (var drain in ended)
        {
            drain.Done.TrySetResult();
        }
    }

    /// <summary>
    /// The blocks whose views the blocks as they are now no longer make, for the caller to clear before the notebook
    /// catches up: blocks still, never a cell that stopped being one, since what such a cell shows is its own.
    /// </summary>
    /// <param name="now">The pipeline the cells make now.</param>
    /// <param name="except">The block a gesture was made on, which that gesture shows again itself; nothing otherwise.</param>
    /// <returns>The blocks.</returns>
    public IReadOnlyList<Guid> StaleIn(NotebookPipeline now, Guid? except)
    {
        ArgumentNullException.ThrowIfNull(now);

        return [.. Now.Shown.Where(each => each.Key != except && !each.Value.StillHolds(now, each.Key) && now.PositionOf(each.Key) >= 0).Select(each => each.Key)];
    }

    /// <summary>
    /// Catches up with blocks that changed where no gesture saw it — a block inserted, taken away or moved, a cell turned
    /// into another kind, text typed into one: forgets every view they no longer make, keeps the pipeline the cells make
    /// now, and takes back what was handed to C# cells unless the blocks still make it.
    /// </summary>
    /// <param name="now">The pipeline the cells make now.</param>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    /// <param name="except">The block a gesture was made on, which that gesture shows again itself; nothing otherwise.</param>
    /// <remarks>
    /// The caller clears what <see cref="StaleIn"/> names first and catches up after, in one write let through for its
    /// turn: a view is forgotten only once it is cleared, so a view a stop left on the screen is still known, and the next
    /// catch-up clears it. It hands nothing over: only a gesture or the toolbar's run does.
    /// </remarks>
    public void CaughtUp(NotebookPipeline now, IVariableStore variables, Guid? except)
    {
        ArgumentNullException.ThrowIfNull(now);
        ArgumentNullException.ThrowIfNull(variables);

        ForgetStale(now, except);
        Publish(now);

        if (variables.TryGet<string>(StepKernel.HandOver, out var handed) && handed != EnvelopeFor(variables, now, SourceBytes.Unknown))
        {
            Withdraw(variables);
        }
    }

    /// <summary>Keeps the pipeline a gesture assembled, for the parts that ask what is around a block.</summary>
    /// <param name="assembled">The pipeline.</param>
    public void Publish(NotebookPipeline assembled) => Change(state => state with { Assembled = assembled });

    /// <summary>Hands C# cells the pipeline the blocks make, knowing nothing of the source's bytes.</summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    public void HandOver(IVariableStore variables, NotebookPipeline assembled) =>
        HandOver(variables, assembled, SourceBytes.Unknown);

    /// <summary>Hands C# cells the pipeline the blocks make, knowing what the source's bytes are now.</summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    /// <param name="bytes">What is known of the source's bytes.</param>
    public void HandOver(IVariableStore variables, NotebookPipeline assembled, SourceBytes bytes)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (EnvelopeFor(variables, assembled, bytes) is { } envelope)
        {
            variables.Set(StepKernel.HandOver, envelope);
        }
        else
        {
            variables.Remove(StepKernel.HandOver);
        }
    }

    /// <summary>
    /// What a notebook hands on as its pipeline: what its own last run learned, while the blocks declare those steps,
    /// the bytes are the ones it learned from and it is still what C# cells were handed; otherwise the declaration.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    /// <param name="bytes">What is known of the source's bytes.</param>
    /// <returns>The pipeline as text, or nothing while the blocks make none.</returns>
    public string? EnvelopeFor(IVariableStore variables, NotebookPipeline assembled, SourceBytes bytes)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (!assembled.Whole)
        {
            return null;
        }

        return Now.Run is { } stamped
               && stamped.Key == KeyOf(assembled.Readable)
               && bytes.Allow(stamped.Fingerprint)
               && variables.TryGet<string>(StepKernel.HandOver, out var handed)
               && handed == stamped.Envelope
            ? stamped.Envelope
            : assembled.Readable.ToJson();
    }

    /// <summary>
    /// Hands C# cells what a run of the whole pipeline learned, beside its declaration — only when the run is of the
    /// steps the blocks declare now, since blocks can change while a run is on its way.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="prepared">The run.</param>
    /// <param name="fingerprint">The fingerprint of the bytes it read.</param>
    /// <param name="turn">The turn of the change that asked for the run: a stop since voids it.</param>
    /// <param name="also">The mark of the block's run, which the engine hands the block.</param>
    /// <returns><see langword="true"/> when it was handed over.</returns>
    /// <remarks>
    /// What it learned is written as text first; recording the run and handing it over are then one write, let through
    /// whole or not at all.
    /// </remarks>
    public bool HandOverFit(IVariableStore variables, PreparedData prepared, string fingerprint, NotebookTurn turn, CancellationToken also)
    {
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(prepared);

        var envelope = prepared.ToJson();
        var stamp = new RunStamp(KeyOf(prepared.Declaration), fingerprint, envelope);
        var fitted = false;

        LetThrough(turn, also, () =>
        {
            Change(state =>
            {
                fitted = state.Assembled?.Readable.Equals(prepared.Declaration) == true;

                return fitted ? state with { Run = stamp, RunsFitted = state.RunsFitted + 1 } : state;
            });

            if (fitted)
            {
                variables.Set(StepKernel.HandOver, envelope);
            }
        });

        return fitted;
    }

    /// <summary>
    /// Hands over again what the last run learned, when it was a run of these steps over these bytes: a run asked
    /// for again then runs nothing.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="declaration">The steps asked to run.</param>
    /// <param name="fingerprint">The fingerprint of the bytes they would read.</param>
    /// <param name="turn">The turn of the change that asked for the run: a stop since voids it.</param>
    /// <param name="also">The mark of the block's run, which the engine hands the block.</param>
    /// <returns>
    /// <see langword="true"/> when the last run was that run — handed over again, unless a stop came since — so nothing is
    /// fitted again.
    /// </returns>
    public bool HandOverFitAgain(IVariableStore variables, PipelineDeclaration declaration, string fingerprint, NotebookTurn turn, CancellationToken also)
    {
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(declaration);

        if (Now.Run is not { } stamped || stamped.Fingerprint != fingerprint || stamped.Key != KeyOf(declaration))
        {
            return false;
        }

        LetThrough(turn, also, () => variables.Set(StepKernel.HandOver, stamped.Envelope));

        return true;
    }

    /// <summary>
    /// Takes back what was handed to C# cells: the blocks changed where no gesture saw it, so what was handed may no
    /// longer be what they make.
    /// </summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    public static void Withdraw(IVariableStore variables)
    {
        ArgumentNullException.ThrowIfNull(variables);

        variables.Remove(StepKernel.HandOver);
    }

    /// <summary>Hands C# cells the folder a relative path in the pipeline is read from, or takes it back.</summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="folder">The notebook's folder, or nothing for a notebook never saved, which reads from the working directory.</param>
    public static void HandOverFolder(IVariableStore variables, string? folder)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (folder is null)
        {
            variables.Remove(StepKernel.Folder);
        }
        else
        {
            variables.Set(StepKernel.Folder, folder);
        }
    }

    /// <summary>Remembers why a change made in a block's form was not made, while its text stays as it was.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="source">The block's text when the change was refused.</param>
    /// <param name="why">Why.</param>
    public void Refused(Guid cell, string source, string why)
    {
        var refusal = new Refusal(source, why);

        Change(state => state with { Refusals = state.Refusals.SetItem(cell, refusal) });
    }

    /// <summary>Forgets a refusal: a change to the block's form was made, or asked for what already was.</summary>
    /// <param name="cell">The block's cell.</param>
    public void Accepted(Guid cell) => Change(state => state with { Refusals = state.Refusals.Remove(cell) });

    /// <summary>Why the last change to a block's form was not made, while the block's text is what it was then.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="source">The block's text now.</param>
    /// <returns>Why, or nothing when no change was refused or the block has changed since.</returns>
    public string? RefusalFor(Guid cell, string source) =>
        Now.Refusals.TryGetValue(cell, out var refusal) && refusal.Source == source ? refusal.Why : null;

    /// <summary>
    /// Remembers the last change a select that commits made: which select, the steps its walk started from, and the
    /// blocks and bytes the change left. One, and only in memory: it replaces the one before.
    /// </summary>
    /// <param name="control">The select: its gesture and what it carries, as it was drawn.</param>
    /// <param name="drawn">The steps its list was drawn over, which every value of its walk picks from.</param>
    /// <param name="keyLeft">The key of the whole declaration the change left.</param>
    /// <param name="fingerprintLeft">The fingerprint of the source's bytes the change was made over.</param>
    public void SelectCommitted(string control, IReadOnlyList<IPipelineStep> drawn, string keyLeft, string fingerprintLeft)
    {
        var select = new SelectCommit(control, drawn, keyLeft, fingerprintLeft);

        Change(state => state with { Select = select });
    }

    /// <summary>
    /// The steps a select's walk started from, when a send goes on from the last change that select made: the same
    /// select, over the blocks and bytes that change left.
    /// </summary>
    /// <param name="control">The select that sent it.</param>
    /// <param name="key">The key of the whole declaration now.</param>
    /// <param name="fingerprint">The fingerprint of the source's bytes now, as this session read them.</param>
    /// <returns>The steps, or nothing when the send goes on from no change of its own: something else changed since.</returns>
    public IReadOnlyList<IPipelineStep>? ContinuationOf(string control, string key, string fingerprint) =>
        Now.Select is { } last && last.Control == control && last.KeyLeft == key && last.FingerprintLeft == fingerprint ? last.Drawn : null;

    /// <summary>
    /// Asks a block to show something, and runs it: leaves the block the request, which its kernel takes once, and runs
    /// the block through the notebook — unless a stop came since the change that asks began, which then leaves nothing
    /// and runs nothing.
    /// </summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="request">What to show.</param>
    /// <param name="turn">The turn of the change that asks.</param>
    /// <param name="notebook">What can be done to the notebook: the block is run through it.</param>
    /// <returns>A task that ends when the block has run, or at once for a change that was stopped.</returns>
    /// <remarks>
    /// The request carries the turn, so a stop that comes while the block runs voids it too. The ask owns its request: one
    /// its block never took — the block was refused, or the run was stopped before the block began — is gone once the ask
    /// ends, so the block's next run never runs it.
    /// </remarks>
    public async Task AskAsync(Guid cell, ViewRequest request, NotebookTurn turn, INotebookOperations notebook)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        var asked = request with { Turn = turn };
        var left = false;

        Change(state =>
        {
            left = state.Stops == turn.Stops && !turn.Mark.IsCancellationRequested;

            return left ? state with { Requests = state.Requests.SetItem(cell, asked) } : state;
        });

        if (!left)
        {
            return;
        }

        try
        {
            await notebook.ExecuteCellAsync(cell);
        }
        finally
        {
            // Only this ask's own request: a later change's for the same block stays.
            Change(state => state.Requests.TryGetValue(cell, out var standing) && standing.Equals(asked) ? state with { Requests = state.Requests.Remove(cell) } : state);
        }
    }

    /// <summary>The ticket of a block's run as it begins: the count of stops by then, and the mark of the run.</summary>
    /// <param name="mark">The mark the engine hands the block's run.</param>
    /// <returns>The ticket, which what the run writes is let through with.</returns>
    public NotebookTurn Enter(CancellationToken mark) => new(Now.Stops, mark);

    /// <summary>
    /// Takes what a block was asked to show, once, for a run of the block begun under the same count of stops as the
    /// change that asked: a request a stop voided is dropped, and a later change's request is left to that change's run.
    /// </summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="executor">The ticket of the block's run.</param>
    /// <returns>The request, or nothing when the block runs because somebody ran it, or for no request of its own.</returns>
    public ViewRequest? Take(Guid cell, NotebookTurn executor)
    {
        ViewRequest? taken = null;

        Change(state =>
        {
            taken = null;

            if (!state.Requests.TryGetValue(cell, out var request))
            {
                return state;
            }

            // A stopped change's request is never run: whoever looks for one drops it.
            if (request.Turn.Stops != state.Stops || request.Turn.Mark.IsCancellationRequested)
            {
                return state with { Requests = state.Requests.Remove(cell) };
            }

            // A run begun before a stop — one a stop left behind — leaves a later change's request to that change's run.
            if (request.Turn.Stops != executor.Stops)
            {
                return state;
            }

            taken = request;

            return state with { Requests = state.Requests.Remove(cell) };
        });

        return taken;
    }

    /// <summary>The view kept last, when it is the one these steps work out over these bytes.</summary>
    /// <param name="key">The key of the steps the view is worked out from.</param>
    /// <param name="fingerprint">The fingerprint of the source's bytes.</param>
    /// <returns>The view, or nothing when it has to be worked out.</returns>
    public PipelineView? ViewFor(string key, string fingerprint) =>
        Now.View is { } kept && kept.Key == key && kept.Fingerprint == fingerprint ? kept.View : null;

    /// <summary>Counts a view worked out just now, and keeps it in place of the one kept before.</summary>
    /// <param name="key">The key of the steps it was worked out from.</param>
    /// <param name="fingerprint">The fingerprint of the bytes it was read from.</param>
    /// <param name="view">The view.</param>
    /// <returns>The same view.</returns>
    public PipelineView Keep(string key, string fingerprint, PipelineView view)
    {
        var kept = new KeptView(key, fingerprint, view);

        Change(state => state with { ViewsRun = state.ViewsRun + 1, View = kept });

        return view;
    }

    /// <summary>Remembers that a block shows data: what its rows were worked out from, and what its grid offers.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="key">The key of the steps the rows at the block were worked out from.</param>
    /// <param name="header">The columns the grid shows, and what its header offers for each.</param>
    public void Showing(Guid cell, string key, GridHeader header)
    {
        var shown = new ShownGrid(key, header);

        Change(state => state with { Shown = state.Shown.SetItem(cell, shown) });
    }

    /// <summary>Remembers that a block shows the list of the source's columns, and the blocks it was drawn for.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="key">The key of the whole declaration the list was drawn for.</param>
    public void Listing(Guid cell, string key)
    {
        var shown = new ShownList(key);

        Change(state => state with { Shown = state.Shown.SetItem(cell, shown) });
    }

    /// <summary>
    /// Forgets a block a change took away — what it showed, and why a change in its form was not made — and remembers
    /// that it is gone, rewritten as another block or taken out, so a change still on its way to it is not made.
    /// </summary>
    /// <param name="cell">The block's cell.</param>
    public void Removed(Guid cell) =>
        Change(state => state with { Shown = state.Shown.Remove(cell), Refusals = state.Refusals.Remove(cell), Gone = state.Gone.Add(cell) });

    /// <summary>Whether a block still stands: no change took it away.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <returns><see langword="true"/> unless a change took the block away.</returns>
    public bool Stands(Guid cell) => !Now.Gone.Contains(cell);

    /// <summary>Forgets that a block shows data: it shows its card alone, or nothing.</summary>
    /// <param name="cell">The block's cell.</param>
    public void Hidden(Guid cell) => Change(state => state with { Shown = state.Shown.Remove(cell) });

    /// <summary>
    /// Forgets every view the blocks as they are now no longer show: its rows are worked out from other steps, its grid
    /// offers what no longer holds, or its list was drawn for other blocks — and says which, for the caller to clear.
    /// </summary>
    /// <param name="now">The pipeline the blocks make now.</param>
    /// <param name="except">The block a gesture was made on, which that gesture shows again itself.</param>
    /// <returns>The blocks whose views were forgotten.</returns>
    public IReadOnlyList<Guid> ForgetStale(NotebookPipeline now, Guid? except)
    {
        ArgumentNullException.ThrowIfNull(now);

        Guid[] stale = [];

        Change(state =>
        {
            stale = [.. state.Shown.Where(each => each.Key != except && !each.Value.StillHolds(now, each.Key)).Select(each => each.Key)];

            return state with { Shown = state.Shown.RemoveRange(stale) };
        });

        return stale;
    }

    /// <summary>The key of a whole declaration: that of its last step, which covers every step above it.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <returns>The key; empty for a declaration without steps.</returns>
    internal static string KeyOf(PipelineDeclaration declaration) =>
        declaration.Steps.Count == 0 ? string.Empty : declaration.KeyAt(declaration.Steps.Count - 1);

    // Makes the next value from the one there is and puts it in place whole. A change made beside it in the meantime
    // has the next value made again from the one that change left, so the transition is worked out from nothing but it.
    private void Change(Func<State, State> transition) => ImmutableInterlocked.Update(ref _state, transition);

    /// <summary>Everything the session knows at one moment.</summary>
    private sealed record State
    {
        /// <summary>What each block was asked to show the next time it runs.</summary>
        public ImmutableDictionary<Guid, ViewRequest> Requests { get; init; } = ImmutableDictionary<Guid, ViewRequest>.Empty;

        /// <summary>What each block shows, under the key it was drawn for.</summary>
        public ImmutableDictionary<Guid, ShownView> Shown { get; init; } = ImmutableDictionary<Guid, ShownView>.Empty;

        /// <summary>Why the last change to each block's form was not made.</summary>
        public ImmutableDictionary<Guid, Refusal> Refusals { get; init; } = ImmutableDictionary<Guid, Refusal>.Empty;

        /// <summary>The blocks a change took away.</summary>
        public ImmutableHashSet<Guid> Gone { get; init; } = [];

        /// <summary>The pipeline the blocks made at the last gesture.</summary>
        public NotebookPipeline? Assembled { get; init; }

        /// <summary>The last change a select that commits made.</summary>
        public SelectCommit? Select { get; init; }

        /// <summary>The one view kept.</summary>
        public KeptView? View { get; init; }

        /// <summary>What the notebook's own last run learned.</summary>
        public RunStamp? Run { get; init; }

        /// <summary>How many views were worked out from the source up.</summary>
        public int ViewsRun { get; init; }

        /// <summary>How many runs of the whole pipeline were fitted and handed over.</summary>
        public int RunsFitted { get; init; }

        /// <summary>How many runs were stopped.</summary>
        public int Stops { get; init; }

        /// <summary>The change that holds the lane now: a stop gives it back.</summary>
        public TaskCompletionSource? Holder { get; init; }

        /// <summary>How many writes are on their way, under the count of stops each was let through at.</summary>
        public ImmutableDictionary<int, int> InFlight { get; init; } = ImmutableDictionary<int, int>.Empty;

        /// <summary>The stops still waiting for writes let through before them.</summary>
        public ImmutableList<Drain> Drains { get; init; } = [];
    }

    /// <summary>A stop waiting for the writes let through before it.</summary>
    /// <param name="Below">The count of stops it made: it waits for every write let through under a smaller one.</param>
    /// <param name="Done">Ended once none is on its way.</param>
    private readonly record struct Drain(int Below, TaskCompletionSource Done);

    /// <summary>Why a change was not made, and the text it was refused on.</summary>
    /// <param name="Source">The block's text.</param>
    /// <param name="Why">Why.</param>
    private sealed record Refusal(string Source, string Why);

    /// <summary>A view, and what it was worked out from.</summary>
    /// <param name="Key">The key of the steps.</param>
    /// <param name="Fingerprint">The fingerprint of the source's bytes.</param>
    /// <param name="View">The rows at the block, and where each stands.</param>
    private sealed record KeptView(string Key, string Fingerprint, PipelineView View);

    /// <summary>What a block shows, under the key it was drawn for.</summary>
    /// <param name="Key">The key.</param>
    private abstract record ShownView(string Key)
    {
        /// <summary>Whether the blocks as they are now still show what this showed.</summary>
        /// <param name="now">The pipeline the blocks make now.</param>
        /// <param name="cell">The block it is shown on.</param>
        /// <returns><see langword="true"/> while it says what holds.</returns>
        public abstract bool StillHolds(NotebookPipeline now, Guid cell);
    }

    /// <summary>A grid: the key its rows were worked out under, and what its header offered.</summary>
    /// <param name="Key">The view's key.</param>
    /// <param name="Header">The grid's header.</param>
    private sealed record ShownGrid(string Key, GridHeader Header) : ShownView(Key)
    {
        /// <inheritdoc />
        public override bool StillHolds(NotebookPipeline now, Guid cell) => now.ViewKeyOf(cell) == Key && Header.StillHolds(now.Readable);
    }

    /// <summary>The list of the source's columns: the key of the whole declaration it was drawn for.</summary>
    /// <param name="Key">That key.</param>
    private sealed record ShownList(string Key) : ShownView(Key)
    {
        /// <inheritdoc />
        public override bool StillHolds(NotebookPipeline now, Guid cell) => KeyOf(now.Readable) == Key;
    }

    /// <summary>The last change a select that commits made.</summary>
    /// <param name="Control">The select, as it was drawn.</param>
    /// <param name="Drawn">The steps its walk started from.</param>
    /// <param name="KeyLeft">The key of the whole declaration the change left.</param>
    /// <param name="FingerprintLeft">The fingerprint of the bytes it was made over.</param>
    private sealed record SelectCommit(string Control, IReadOnlyList<IPipelineStep> Drawn, string KeyLeft, string FingerprintLeft);

    /// <summary>What the notebook's own last run learned, and from which steps and bytes.</summary>
    /// <param name="Key">The key of the steps it ran.</param>
    /// <param name="Fingerprint">The fingerprint of the bytes it read.</param>
    /// <param name="Envelope">The pipeline it handed over: the declaration with what it learned.</param>
    private sealed record RunStamp(string Key, string Fingerprint, string Envelope);
}
