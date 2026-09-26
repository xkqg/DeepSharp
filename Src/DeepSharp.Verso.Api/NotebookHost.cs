// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Diffing;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>One open notebook: Verso's engine with DeepSharp's parts, holding the notebook one file saves.</summary>
/// <remarks>
/// <para>
/// It is opened the way Verso's own editors open a notebook — through the serializer for its format, past the guards
/// that run after reading, with the cells that are only ever shown rendered drawn — and DeepSharp's parts are
/// registered before Verso looks for any beside the application, so an application published as a single file, which
/// has no assemblies beside it, still has them. An extension a notebook asks for is refused: the engine runs what the
/// application carries, and fetches nothing. A host is opened and closed by the <see cref="OpenNotebooks"/> that
/// holds it, never by whoever it was handed to.
/// </para>
/// <para>
/// Everything done to the notebook takes its turn, one at a time and in the order it came, because the engine serves
/// one caller at a time and a change made while another is under way would act on blocks that are going away. A
/// request about a cell a change before it rewrote or took away is refused with <see cref="CellGoneException"/>.
/// </para>
/// <para>
/// Each change makes the next version of the notebook, and every view of it is told, cell by cell — whatever made the
/// change, a click, a clear or a form, for none of which the engine says a word. What a cell shows while it runs is told
/// as it comes. A view never holds the notebook up: one that reads slower than it changes is kept one change behind.
/// </para>
/// </remarks>
public sealed class NotebookHost
{
    private const string CSharp = "csharp";

    // The one thing in this package as wide as the process, because what it guards is: a C# kernel takes over the
    // process's console while it runs (Console.SetOut in Verso's C# kernel), so two C# runs anywhere in the process —
    // in two notebooks, or under two holders of notebooks — print into each other (measured). Each run takes its turn.
    private static readonly Lane CSharpRuns = new();

    // How long what a run shows is gathered before it is published, as Verso's browser editor gathers a burst of output
    // before it draws (32 ms in its ServerNotebookService), so a cell that shows a thousand things is not a thousand versions.
    private static readonly TimeSpan Gathering = TimeSpan.FromMilliseconds(32);

    private readonly Lane _turns = new();
    private Run? _running;

    // Who views the notebook, since when nobody has, and whether it is closed: one value, swapped whole, so no view begins
    // as the notebook closes.
    private Audience _audience = Audience.Opened();

    // How many things asked of the notebook are not done yet, the one under way among them.
    private int _pending;

    // The notebook as the file it was last saved to holds it, read the way it is read when it opens; what is unsaved is
    // what differs from it.
    private NotebookModel _saved;

    // The notebook as of its last version. Only whoever holds the notebook's turn publishes the next, so versions follow
    // one another without anything else to keep them in order.
    private StrongBox<NotebookVersion> _published;

    // The run under way, as the engine last said one began; the end of every turn clears it.
    private StrongBox<HostedRun>? _executing;

    // Set when the engine says a cell began, ended or showed something; a run's turn waits on it to publish what it shows.
    private TaskCompletionSource _said = new(TaskCreationOptions.RunContinuationsAsynchronously);

    // Whether a turn that publishes what the engine said outside any turn is queued already.
    private int _telling;

    private NotebookHost(string filePath, ExtensionHost extensions, Scaffold scaffold, NotebookModel saved)
    {
        FilePath = filePath;
        Extensions = extensions;
        Scaffold = scaffold;
        _saved = saved;
        _published = new(new NotebookVersion(0, [.. scaffold.Cells.Select(cell => cell.Hosted([]))], null));
        scaffold.OnCellExecuting += Began;
        scaffold.OnCellExecuted += Said;
        scaffold.OnCellOutputUpdated += Said;
    }

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string FilePath { get; private set; }

    /// <summary>The notebook as of its last version: every cell in order, and the run under way.</summary>
    public NotebookVersion Current => Volatile.Read(ref _published).Value;

    /// <summary>The notebook's cells as they stand, in order.</summary>
    public IReadOnlyList<HostedCell> Cells => Current.Cells;

    /// <summary>The engine's extensions: Verso's own and DeepSharp's.</summary>
    internal ExtensionHost Extensions { get; }

    /// <summary>The notebook as Verso's engine holds and runs it.</summary>
    internal Scaffold Scaffold { get; }

    /// <summary>Sets a cell's text, as typing it does; nothing runs.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="source">Its new text.</param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    public Task<HostedCell> EditAsync(Guid cell, string source)
    {
        ArgumentNullException.ThrowIfNull(source);

        return TurnAsync(() =>
        {
            Standing(cell);
            Scaffold.UpdateCellSource(cell, source);

            return Task.FromResult(EndTurn().Cells.First(each => each.Id == cell));
        });
    }

    /// <summary>Runs a cell, as pressing its run button does.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The cell as it stands after the run, or after <see cref="Stop"/> ended it.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <remarks>
    /// A C# run takes its turn among every C# run in the process, because a C# kernel takes over the process's console
    /// while it runs, and two at once would print into each other.
    /// </remarks>
    public Task<HostedCell> RunAsync(Guid cell) => TurnAsync(async () =>
    {
        var running = Standing(cell);

        await (string.Equals(running.Language, CSharp, StringComparison.OrdinalIgnoreCase)
            ? CSharpRuns.TakeTurnAsync(() => RunUntilStoppedAsync(() => Scaffold.ExecuteCellAsync(running.Id), running.Language))
            : RunUntilStoppedAsync(() => Scaffold.ExecuteCellAsync(running.Id), running.Language));

        return EndTurn().Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// Stops the run under way — a cell's, or a toolbar button's — if there is one: the only way Verso's engine stops a
    /// run that does not end is a fresh kernel, so the run's kernel is restarted, and what that kernel held — the
    /// notebook's variables, the pipeline handed to C# cells among them — is gone. A run that never ends goes on in the
    /// background until the application does.
    /// </summary>
    public void Stop() => Volatile.Read(ref _running)?.Stop();

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

    /// <summary>Hands a click on a control a cell drew to the part the control names.</summary>
    /// <param name="gesture">The click.</param>
    /// <returns>Whether it changed the cells, and what the part answered, which the cell then shows.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">No part of that name answers a click.</exception>
    public Task<GestureResult> GestureAsync(HostedGesture gesture) => TurnAsync(async () =>
    {
        Standing(gesture.Cell);

        var part = Extensions.GetInteractionHandler(gesture.ExtensionId)
            ?? throw new InvalidOperationException($"No part named '{gesture.ExtensionId}' answers a click.");
        var context = new CellInteractionContext
        {
            Region = CellRegion.Output,
            InteractionType = gesture.Action,
            Payload = gesture.Payload,
            CellId = gesture.Cell,
            ExtensionId = gesture.ExtensionId,
            CancellationToken = CancellationToken.None,
            Variables = Scaffold.Variables,
            Notebook = Scaffold.NotebookOps,
            NotebookModel = Scaffold.Notebook,
        };
        var answer = await part.OnCellInteractionAsync(context);

        if (answer is not null)
        {
            // As Verso's browser editor shows one: the answer is what the cell shows now, while the cell still stands.
            foreach (var shown in Scaffold.Cells.Where(each => each.Id == gesture.Cell))
            {
                shown.Outputs.Clear();
                shown.Outputs.Add(new CellOutput("text/html", answer));
            }
        }

        return new GestureResult(context.StateChanged, answer);
    });

    /// <summary>Every toolbar button the engine has — Verso's own and DeepSharp's — each saying whether it can be pressed now.</summary>
    /// <returns>The buttons, by place and then in their order.</returns>
    public Task<IReadOnlyList<HostedToolbarAction>> ToolbarAsync() => TurnAsync(async () =>
    {
        var context = new ToolbarContext(Scaffold, []);
        var buttons = new List<HostedToolbarAction>();

        foreach (var action in Extensions.GetToolbarActions())
        {
            buttons.Add(new HostedToolbarAction(
                action.ActionId,
                action.DisplayName,
                action.Icon,
                action.IconOnly,
                action.IsPrimary,
                action.ConfirmationPrompt,
                Enum.Parse<ToolbarPlace>(action.Placement.ToString()),
                action.Order,
                await action.IsEnabledAsync(context)));
        }

        return (IReadOnlyList<HostedToolbarAction>)[.. buttons.OrderBy(button => button.Place).ThenBy(button => button.Order)];
    });

    /// <summary>Presses a toolbar button.</summary>
    /// <param name="id">The button.</param>
    /// <param name="cells">The cells it is pressed for, for a button on a cell's toolbar.</param>
    /// <returns>The file it handed over, for whoever pressed it; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">The engine has no button of that name.</exception>
    /// <remarks>
    /// A button may run cells, C# among them, so a press takes its turn among the process's C# runs as a C# run does. A file is
    /// never written beside the notebook: where it is saved is for whoever pressed the button to say.
    /// </remarks>
    public Task<HostedFile?> RunToolbarAsync(string id, params Guid[] cells) => TurnAsync(() => CSharpRuns.TakeTurnAsync(async () =>
    {
        var action = Extensions.GetToolbarActions().FirstOrDefault(each => each.ActionId == id)
            ?? throw new InvalidOperationException($"The notebook has no toolbar button '{id}'.");
        var context = new ToolbarContext(Scaffold, cells);

        // A button that runs the cells can meet one that never ends, and only the notebook's own kernel runs away.
        await RunUntilStoppedAsync(() => action.ExecuteAsync(context), Scaffold.DefaultKernelId);

        return context.Handed;
    }));

    /// <summary>A cell's properties panel: a section from every part that has one for the cell, in their order.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The sections.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    public Task<IReadOnlyList<HostedSection>> PropertiesAsync(Guid cell) => TurnAsync(async () =>
    {
        var shown = Standing(cell);
        var context = new RenderContext(Scaffold, shown);
        var sections = new List<HostedSection>();

        foreach (var part in Extensions.GetPropertyProviders().Where(each => each.AppliesTo(shown, context)).OrderBy(each => each.Order))
        {
            var section = await part.GetPropertiesSectionAsync(shown, context);

            sections.Add(new HostedSection(
                part.ExtensionId,
                section.Title,
                section.Description,
                [.. section.Fields.Select(field => new HostedField(
                    field.Name,
                    field.DisplayName,
                    Enum.Parse<FieldKind>(field.FieldType.ToString()),
                    field.CurrentValue,
                    field.Description,
                    [.. (field.Options ?? []).Select(option => new HostedOption(option.Value, option.DisplayName))],
                    field.IsReadOnly))]));
        }

        return (IReadOnlyList<HostedSection>)sections;
    });

    /// <summary>Changes a field of a cell's properties panel, through the part its section came from.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="part">The part the field's section came from.</param>
    /// <param name="field">The field.</param>
    /// <param name="value">What it is set to.</param>
    /// <returns>When the part has made the change.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">No part of that name has a properties section.</exception>
    public Task SetPropertyAsync(Guid cell, string part, string field, object? value) => TurnAsync(async () =>
    {
        var changed = Standing(cell);
        var provider = Extensions.GetPropertyProviders().FirstOrDefault(each => each.ExtensionId == part)
            ?? throw new InvalidOperationException($"No part named '{part}' has a properties section.");

        await provider.OnPropertyChangedAsync(changed, field, value, new RenderContext(Scaffold, changed));

        return true;
    });

    /// <summary>
    /// Saves the notebook to its file the way Verso's own editors save it: through the serializer for its format, which
    /// leaves out what a block shows, past the guards that run before writing, and written whole under a name of its own
    /// before it takes the file's place, so nothing ever meets half a notebook.
    /// </summary>
    /// <returns>When it is saved.</returns>
    public Task SaveAsync() => TurnAsync(async () =>
    {
        await SaveToAsync(FilePath);

        return true;
    });

    /// <summary>Saves the notebook under another name, which is its file from then on.</summary>
    /// <param name="path">The new file, as a full path.</param>
    /// <returns>When it is saved there.</returns>
    /// <remarks>Only its holder names it, since the holder knows which files are open; <see cref="OpenNotebooks.SaveAsAsync"/>.</remarks>
    internal Task SaveAsAsync(string path) => TurnAsync(async () =>
    {
        await SaveToAsync(path);

        // What the notebook names after itself — the columns saved beside it, an exported pipeline — follows the file.
        Scaffold.SetFilePath(path);
        FilePath = path;

        return true;
    });

    /// <summary>Registers DeepSharp's parts with an engine, before it looks for any beside the application.</summary>
    /// <param name="extensions">The engine.</param>
    /// <returns>When they are registered.</returns>
    /// <remarks>
    /// First, because Verso's own look beside the application passes over a part already registered, while one
    /// registered after it would be refused as a second.
    /// </remarks>
    internal static async Task RegisterAsync(ExtensionHost extensions)
    {
        foreach (var part in Parts())
        {
            await extensions.LoadExtensionAsync(part);
        }
    }

    /// <summary>Every part DeepSharp's notebook has, made anew for one engine.</summary>
    /// <returns>The parts.</returns>
    internal static IExtension[] Parts() =>
    [
        new StepCellType(),
        new StepKernel(),
        new StepRenderer(),
        new StepForm(),
        new RunPipelineAction(),
        new ExportPipelineAction(),
        new TakeOverAction(),
        new JupyterGuard(),
    ];

    /// <summary>Opens the notebook a file holds on an engine, and closes the engine when it cannot be opened.</summary>
    /// <param name="filePath">The notebook's file, as a full path.</param>
    /// <param name="extensions">The engine to open it on; closed when the notebook cannot be opened.</param>
    /// <param name="cancellationToken">Stops the open.</param>
    /// <returns>The host.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows reads the file.</exception>
    internal static async Task<NotebookHost> OpenAsync(string filePath, ExtensionHost extensions, CancellationToken cancellationToken)
    {
        Scaffold? scaffold = null;

        try
        {
            var content = await File.ReadAllTextAsync(filePath, cancellationToken);

            extensions.ConsentHandler = static (_, _) => Task.FromResult(false);
            await RegisterAsync(extensions);
            await extensions.LoadBuiltInExtensionsAsync();

            var serializer = extensions.GetSerializers().FirstOrDefault(each => each.CanImport(filePath))
                ?? throw new NotSupportedException($"No format Verso knows reads '{Path.GetFileName(filePath)}'.");
            var notebook = await ReadAsync(extensions, serializer, content, filePath);
            var saved = await ReadAsync(extensions, serializer, content, filePath);

            // A file can repeat a cell's id — cells copied by hand, or by a tool — and nothing tells such cells apart by it,
            // so each repeat is given an id of its own, as Jupyter's own reader repairs repeated cell ids.
            var ids = new HashSet<Guid>();

            foreach (var cell in notebook.Cells)
            {
                if (!ids.Add(cell.Id))
                {
                    cell.Id = Guid.NewGuid();
                }
            }

            scaffold = new Scaffold(notebook, extensions, filePath);
            scaffold.InitializeSubsystems();
            await scaffold.RenderTransientCellsAsync(cancellationToken);

            return new NotebookHost(filePath, extensions, scaffold, saved);
        }
        catch
        {
            if (scaffold is not null)
            {
                await scaffold.DisposeAsync();
            }

            await extensions.DisposeAsync();

            throw;
        }
    }

    /// <summary>Ends a view: it is told nothing more.</summary>
    /// <param name="view">The view.</param>
    internal void Leave(NotebookSubscription view) => ImmutableInterlocked.Update(ref _audience, audience => audience.Without(view));

    /// <summary>
    /// Closes the notebook, when no view has shown it for the grace, nothing runs or waits, and nothing in it differs from
    /// the file it was last saved to; otherwise leaves it open.
    /// </summary>
    /// <param name="grace">How long no view must have shown it.</param>
    /// <param name="forget">Lets its holder forget it, before its engine closes.</param>
    /// <returns>Whether it is still open.</returns>
    /// <remarks>
    /// It takes its turn, so nothing runs while it looks. What differs from the file is what Verso's own comparison of two
    /// notebooks finds, told which cells' outputs are never saved: what a block shows never counts.
    /// </remarks>
    internal Task<bool> StaysOpenAsync(TimeSpan grace, Action forget) => _turns.TakeTurnAsync(async () =>
    {
        if (Volatile.Read(ref _audience).Closed)
        {
            return false;
        }

        if (Volatile.Read(ref _pending) > 0 || IsUnsaved())
        {
            return true;
        }

        // One swap decides it, so a view that begins meanwhile keeps the notebook open.
        if (!ImmutableInterlocked.Update(ref _audience, audience => audience.Views.IsEmpty && audience.AloneFor >= grace ? Audience.Gone : audience))
        {
            return true;
        }

        forget();
        await CloseEngineAsync();

        return false;
    });

    /// <summary>
    /// Closes the notebook: every view ends and anything asked from now on is refused; what is under way finishes, and then
    /// the engine closes.
    /// </summary>
    /// <returns>When it is closed.</returns>
    internal async ValueTask CloseAsync()
    {
        foreach (var view in Interlocked.Exchange(ref _audience, Audience.Gone).Views)
        {
            view.End();
        }

        await _turns.TakeTurnAsync(async () =>
        {
            await CloseEngineAsync();

            return true;
        });
    }

    // Reads a notebook the way Verso's own editors read one: through its format's serializer, past the guards that run
    // after reading, and naming the kernel and the layout the editors name for a notebook that names neither.
    private static async Task<NotebookModel> ReadAsync(ExtensionHost extensions, INotebookSerializer serializer, string content, string path)
    {
        var notebook = await serializer.DeserializeAsync(content);

        foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(path, serializer.FormatId)).OrderBy(each => each.Priority))
        {
            notebook = await guard.PostDeserializeAsync(notebook, path);
        }

        notebook.DefaultKernelId ??= CSharp;
        notebook.ActiveLayout ??= LayoutDefaults.Reference;

        return notebook;
    }

    private async Task CloseEngineAsync()
    {
        Scaffold.OnCellExecuting -= Began;
        Scaffold.OnCellExecuted -= Said;
        Scaffold.OnCellOutputUpdated -= Said;
        await Scaffold.DisposeAsync();
        await Extensions.DisposeAsync();
    }

    // Everything done to the notebook: refused once it closes, and otherwise one at a time, in the order it came, each
    // ending with the notebook published as it then stands. What was asked before a close and waited behind it is
    // refused when its turn comes, since the engine it asked of is closed by then.
    private Task<T> TurnAsync<T>(Func<Task<T>> change)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _audience).Closed, this);
        Interlocked.Increment(ref _pending);

        return _turns.TakeTurnAsync(async () =>
        {
            try
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _audience).Closed, this);

                try
                {
                    return await change();
                }
                finally
                {
                    EndTurn();
                }
            }
            finally
            {
                Interlocked.Decrement(ref _pending);
            }
        });
    }

    // What in the notebook differs from the file it was last saved to, as Verso's own comparison of two notebooks finds it.
    private bool IsUnsaved()
    {
        var diff = NotebookDiffEngine.Compute(_saved, Scaffold.Notebook, FilePath, Extensions.GetCellTypes());

        return diff.Summary.Added + diff.Summary.Removed + diff.Summary.Modified + diff.Summary.Moved + diff.MetadataChanges.Count > 0;
    }

    // The end of a turn: nothing runs any more, and the notebook is published as it stands.
    private NotebookVersion EndTurn()
    {
        Volatile.Write(ref _executing, null);

        return Publish();
    }

    // Publishes the notebook as it stands as the next version, unless it is the last one again: as the current version,
    // and to every view, each told the cells that came or changed. A cell caught half written by a run keeps what it
    // showed at the last version; the run's next word about it, or the end of its turn, publishes the rest.
    private NotebookVersion Publish()
    {
        var last = Current;
        var before = last.Cells.ToDictionary(cell => cell.Id);

        HostedCell[] cells = [.. Scaffold.Cells.Select(cell => cell.Hosted(before.TryGetValue(cell.Id, out var was) ? was.Outputs : []))];
        HostedCell[] changed = [.. cells.Where(cell => !before.TryGetValue(cell.Id, out var was) || was != cell)];
        var order = cells.Select(cell => cell.Id).SequenceEqual(last.Cells.Select(cell => cell.Id)) ? null : cells.Select(cell => cell.Id).ToArray();
        var running = Volatile.Read(ref _executing)?.Value;

        if (order is null && changed.Length == 0 && running == last.Running)
        {
            return last;
        }

        var next = new NotebookVersion(last.Version + 1, cells, running);

        Volatile.Write(ref _published, new StrongBox<NotebookVersion>(next));

        foreach (var view in Volatile.Read(ref _audience).Views)
        {
            view.Offer(new NotebookChange(next.Version, order, changed, running));
        }

        return next;
    }

    // The engine says a cell began: it is the run under way, since now.
    private void Began(Guid cell)
    {
        Volatile.Write(ref _executing, new StrongBox<HostedRun>(new HostedRun(cell, DateTimeOffset.UtcNow)));
        Said(cell);
    }

    // The engine says a cell began, ended or showed something. It says so from inside the run, and the run waits for what
    // it calls — a view doing its own work here held a click twice as long (measured) — so nothing is done here but
    // asking for it to be published: a run's turn is woken to publish it, and when no turn to tell it is queued already,
    // one is, for what the engine says outside any turn, such as a cell's background task showing more after its run.
    private void Said(Guid cell)
    {
        Volatile.Read(ref _said).TrySetResult();

        if (Interlocked.Exchange(ref _telling, 1) == 0)
        {
            _ = Task.Run(() => _turns.TakeTurnAsync(() =>
            {
                Volatile.Write(ref _telling, 0);

                return Task.FromResult(Publish());
            }));
        }
    }

    // A fresh wait for the engine's next word.
    private Task Listen()
    {
        var said = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Volatile.Write(ref _said, said);

        return said.Task;
    }

    private CellModel Standing(Guid cell) => Scaffold.GetCell(cell) ?? throw new CellGoneException(cell, Current.Version);

    private async Task SaveToAsync(string path)
    {
        var serializer = Extensions.GetSerializers().FirstOrDefault(each => each.CanImport(path))
            ?? throw new NotSupportedException($"No format Verso knows writes '{Path.GetFileName(path)}'.");

        // What a live output shows now is what is saved, as Verso's own editors ask before they write.
        await Scaffold.RefreshLiveOutputsAsync();

        var notebook = Scaffold.Notebook;

        foreach (var guard in Extensions.GetPostProcessors().Where(each => each.CanProcess(path, serializer.FormatId)).OrderBy(each => each.Priority))
        {
            notebook = await guard.PreSerializeAsync(notebook, path);
        }

        var whole = Path.Join(Path.GetDirectoryName(path), $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        var text = await serializer.SerializeAsync(notebook);

        await File.WriteAllTextAsync(whole, text);
        File.Move(whole, path, overwrite: true);

        // From now on, what is unsaved is what differs from what this file holds.
        _saved = await ReadAsync(Extensions, serializer, text, path);
    }

    // Runs a cell until it ends, or until Stop. A run is not asked to stop, since one that does not end does not listen
    // either (measured: a C# loop that awaits goes on with its token cancelled); a fresh kernel ends the turn, and the
    // run is left behind. Only the run under way can be stopped, so the slot holds one, and a Stop after it ended
    // meets a run nobody waits for any more. While it runs, what the engine says it shows is gathered for a moment and
    // published, so a view sees it before the run ends.
    private async Task<bool> RunUntilStoppedAsync(Func<Task> start, string? kernel)
    {
        var run = new Run();

        Volatile.Write(ref _running, run);

        try
        {
            var said = Listen();
            var ran = start();
            var ended = Task.WhenAny(ran, run.Stopped);

            while (await Task.WhenAny(ended, said) == said)
            {
                said = Listen();
                await Task.WhenAny(ended, Task.Delay(Gathering));
                Publish();
            }

            if (await ended == ran)
            {
                await ran;
            }
            else
            {
                await Scaffold.RestartKernelAsync(kernel);
            }

            return true;
        }
        finally
        {
            Volatile.Write(ref _running, null);
        }
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

    // A run under way, and the way to stop it.
    private sealed class Run
    {
        private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task Stopped => _stopped.Task;

        public void Stop() => _stopped.TrySetResult();
    }
}
