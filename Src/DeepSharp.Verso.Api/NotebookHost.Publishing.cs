// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using Verso;

namespace DeepSharp.Verso.Api;

// The notebook's versions, and who is told them: one version at a time, made by whoever holds the turn, and offered to
// every view. What the engine says while a turn is under way is gathered for a moment and published as one version,
// and an ask to publish at once is published at once.
public sealed partial class NotebookHost
{
    /// <summary>
    /// Begins a view of the notebook: the notebook as it stands, then each change after it. It never waits for anything
    /// done to the notebook, so a view that begins while a cell runs is told at once which cell runs, and since when.
    /// </summary>
    /// <returns>The view; disposing it ends it.</returns>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    public NotebookSubscription Subscribe()
    {
        var view = new NotebookSubscription(this);

        // The view joins first and takes the notebook as it stands after, so no change falls between the two.
        ImmutableInterlocked.Update(ref _audience, audience => audience.Closed ? throw new ObjectDisposedException(GetType().FullName) : audience.With(view));
        view.Snapshot = Current;

        return view;
    }

    /// <summary>Ends a view: it is told nothing more.</summary>
    /// <param name="view">The view.</param>
    internal void Leave(NotebookSubscription view) => ImmutableInterlocked.Update(ref _audience, audience => audience.Without(view));

    // The end of a turn: the notebook is published as it stands, now that nothing else acts on it.
    private Task<NotebookVersion> EndTurnAsync() => PublishAsync(whole: true);

    // A turn's work, what it does told while it does it, and then, whatever the work did, the notebook published as it
    // stands. The turn is the notebook's one publisher, so no two versions are ever made at once: what the engine says is
    // gathered for a moment first, as a burst of output is one version, and an ask to publish at once is published at once.
    private async Task<T> PublishingAsync<T>(Func<Task<T>> change)
    {
        try
        {
            var said = Listen();
            var work = change();

            while (await Task.WhenAny(work, said) == said)
            {
                var atOnce = await said;

                said = Listen();

                if (!atOnce)
                {
                    await Task.WhenAny(work, Task.Delay(Gathering));
                }

                await PublishAsync(whole: false);
            }

            return await work;
        }
        finally
        {
            await EndTurnAsync();
        }
    }

    // Asks the turn under way to publish the notebook at once — a run that waits or starts, a change done — and waits for
    // that version, so what comes next is told after it. Only a turn's own work asks, so a turn is there to answer; two
    // that ask at once are told the same version.
    private Task<NotebookVersion> PublishedAsync()
    {
        var asked = _next.AskAsync();

        Volatile.Read(ref _said).TrySetResult(true);

        return asked;
    }

    // Publishes the notebook as it stands, and tells it to whoever asked for the next version: whole when nothing else acts
    // on the notebook, and otherwise as a turn's work leaves it meanwhile.
    private Task<NotebookVersion> PublishAsync(bool whole) => _next.TellAsync(() => NextAsync(whole));

    // The notebook as it stands as the next version, unless it is the last one again: as the current version, and to every
    // view, each told the cells that came or changed, and the buttons when any says something else. A cell caught half
    // written by a run keeps what it showed at the last version; the run's next word about it, or the end of its turn,
    // publishes the rest.
    private async Task<NotebookVersion> NextAsync(bool whole)
    {
        var buttons = await Extensions.ButtonsAsync(Scaffold);
        var last = Current;

        // Drawn before what is unsaved is taken, since drawing places what the layout had not placed yet.
        var arrangement = await Scaffold.ArrangementAsync();
        var unsaved = await _file.UnsavedAsync(takeBack: whole);
        var before = last.Cells.ToDictionary(cell => cell.Id);

        HostedCell[] cells = [.. Scaffold.Cells.Select(cell => cell.Hosted(before.TryGetValue(cell.Id, out var was) ? was.Outputs : []))];
        HostedCell[] changed = [.. cells.Where(cell => !before.TryGetValue(cell.Id, out var was) || was != cell)];
        var order = cells.Select(cell => cell.Id).SequenceEqual(last.Cells.Select(cell => cell.Id)) ? null : cells.Select(cell => cell.Id).ToArray();

        // The run under way is read once, so it is told either as the run or among what runs with no run, never both.
        var under = Volatile.Read(ref _running);
        var running = under?.Hosted;

        _executions.Forget();

        var executing = _executions.Now(under);
        var kernels = Kernels;
        var layout = Scaffold.Layout;
        var theme = Scaffold.ThemeId;

        var pressable = buttons.SequenceEqual(last.Buttons) ? null : buttons;
        HostedArrangement? arranged = arrangement == last.Arrangement ? null : arrangement;
        var metadata = Scaffold.SaysOfItself;
        HostedMetadata? said = metadata == last.Metadata ? null : metadata;

        if (order is null && changed.Length == 0 && running == last.Running && executing.SequenceEqual(last.Executing) && layout == last.Layout && pressable is null
            && kernels == last.Kernels && unsaved == last.Unsaved && theme == last.ThemeId && arranged is null && said is null)
        {
            return last;
        }

        var next = new NotebookVersion(last.Version + 1, cells, running, executing, layout, buttons, kernels, unsaved, theme, arrangement, metadata);

        Volatile.Write(ref _published, new StrongBox<NotebookVersion>(next));

        foreach (var view in Volatile.Read(ref _audience).Views)
        {
            view.Offer(new NotebookChange(next.Version, order, changed, running, executing, layout, pressable, kernels, unsaved, theme, arranged, said));
        }

        return next;
    }

    // The engine said something of a cell or a kernel, or a run what it runs now. It says so from inside the run, and the
    // run waits for what it calls — a view doing its own work here held a click twice as long (measured) — so nothing is
    // done here but asking for it to be published: the turn under way is woken to publish it, and when no turn to tell it
    // is queued already, one is, for what is said outside any turn, such as a cell's background task showing more after
    // its run, or what a stop left behind ending.
    private void Said()
    {
        Volatile.Read(ref _said).TrySetResult(false);

        if (Interlocked.Exchange(ref _telling, 1) == 0)
        {
            _ = Task.Run(() => _turns.TakeTurnAsync(() =>
            {
                Volatile.Write(ref _telling, 0);

                // A closed notebook is published no more: its engine is gone, whatever a stop left behind still says.
                return Volatile.Read(ref _audience).Closed ? Task.FromResult(Current) : PublishAsync(whole: true);
            }));
        }
    }

    // A fresh wait for the next word that the notebook is to be published: whether to publish it at once.
    private Task<bool> Listen()
    {
        var said = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        Volatile.Write(ref _said, said);

        return said.Task;
    }

    // Who views the notebook, since when nobody has, and whether it is closed.
    private sealed class Audience(ImmutableArray<NotebookSubscription> views, bool closed, long aloneSince)
    {
        public static readonly Audience Gone = new([], closed: true, aloneSince: 0);

        public ImmutableArray<NotebookSubscription> Views => views;

        public bool Closed => closed;

        // How long nobody has viewed the notebook; the first view begins after it opens.
        public TimeSpan AloneFor => TimeSpan.FromMilliseconds(Environment.TickCount64 - aloneSince);

        public static Audience Opened() => new([], closed: false, Environment.TickCount64);

        public Audience With(NotebookSubscription view) => new(views.Add(view), closed, aloneSince);

        // The view that leaves last leaves the notebook alone from now on.
        public Audience Without(NotebookSubscription view)
        {
            var left = views.Remove(view);

            return new Audience(left, closed, left.IsEmpty && !views.IsEmpty ? Environment.TickCount64 : aloneSince);
        }
    }
}
