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
/// one caller at a time and a change made while another is under way would act on blocks that are going away. Typing,
/// running or opening the panel of a cell a change before it rewrote or took away is refused with
/// <see cref="CellGoneException"/>; a click from its card is not, since the part it names knows the block it became.
/// </para>
/// <para>
/// Each change makes the next version of the notebook, and every view of it is told, cell by cell — whatever made the
/// change, a click, a clear or a form, for none of which the engine says a word. What a cell shows while it runs is told
/// as it comes. Every version also says what runs — the run under way, and what the engine runs that no run owns, what a
/// stop left behind among it — what became of the kernels, whether the notebook differs from its file, what the layout
/// it is shown in draws of its own, and what the notebook says of itself. A view never holds the notebook up: one that
/// reads slower than it changes is kept one change behind.
/// </para>
/// </remarks>
public sealed class NotebookHost
{
    private const string CSharp = "csharp";

    // Acts on an arrangement named in this space are the host's own, as Verso's editors keep them: never a layout part's.
    private const string HostsOwn = "verso/";

    // The one thing in this package as wide as the process, because what it guards is: a C# kernel takes over the
    // process's console while it runs (Console.SetOut in Verso's C# kernel), so two C# runs anywhere in the process —
    // in two notebooks, or under two holders of notebooks — print into each other (measured). Each run takes its turn.
    private static readonly Lane CSharpRuns = new();

    // How long what a run shows is gathered before it is published, as Verso's browser editor gathers a burst of output
    // before it draws (32 ms in its ServerNotebookService), so a cell that shows a thousand things is not a thousand versions.
    private static readonly TimeSpan Gathering = TimeSpan.FromMilliseconds(32);

    private readonly Lane _turns = new();
    private Run? _running;

    // The run whose work the engine is in, as the flow that started it carries it: the engine says a cell began or ended
    // from inside the execution that ran it, however late that is, so a run left behind that ends while another run of the
    // same cell is under way ends its own record and never the other's.
    private readonly AsyncLocal<Run?> _raising = new();

    // The number the last run was given: each run of the notebook is counted, so a stop names the run it means.
    private long _runs;

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

    // Set when the notebook is to be published, which only the turn under way does: by the engine's word that a cell began,
    // ended or showed something, gathered for a moment first (false), or by an ask to publish at once (true).
    private TaskCompletionSource<bool> _said = new(TaskCreationOptions.RunContinuationsAsynchronously);

    // The version the turn under way publishes next, as whoever asked for it waits for it.
    private readonly NextVersion _next = new();

    // What the engine runs that no run owns — a block a change runs, or what a stop left behind — each from the engine's
    // word it began to its word it ended.
    private readonly Executions _executions = new();

    // What became of the notebook's kernels, as the engine says it.
    private Restarts _restarts = Restarts.None;

    // Whether a turn that publishes what the engine said outside any turn is queued already.
    private int _telling;

    private NotebookHost(
        string filePath, ExtensionHost extensions, Scaffold scaffold, NotebookModel saved, IReadOnlyList<HostedToolbarAction> buttons, bool unsaved, HostedArrangement arrangement)
    {
        FilePath = filePath;
        Extensions = extensions;
        Scaffold = scaffold;
        _blocks = ((IExtensionHostContext)extensions).GetLoadedExtensions().OfType<StepCellType>().First();
        Kinds = KindsOf(extensions);
        _saved = saved;
        Layouts = LayoutsOf(extensions);
        Themes = ThemesOf(extensions);
        _published = new(new NotebookVersion(0, [.. scaffold.Cells.Select(cell => cell.Hosted([]))], null, [], Layout, buttons, Restarts.None.Hosted, unsaved, ThemeId, arrangement, Metadata));
        scaffold.OnCellExecuting += Began;
        scaffold.OnCellExecuted += Ended;
        scaffold.OnCellOutputUpdated += Showed;
        scaffold.OnKernelRestarting += Restarting;
        scaffold.OnKernelRestarted += Restarted;
        scaffold.OnKernelRestartFailed += RestartFailed;
    }

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string FilePath { get; private set; }

    /// <summary>
    /// The notebook as of its last version: every cell in order, what runs, what became of its kernels, and whether it
    /// differs from its file.
    /// </summary>
    public NotebookVersion Current => Volatile.Read(ref _published).Value;

    /// <summary>The notebook's cells as they stand, in order.</summary>
    public IReadOnlyList<HostedCell> Cells => Current.Cells;

    /// <summary>
    /// Every kind a cell can be added as or turned into, as Verso's editors offer them: code in each language the engine
    /// runs, Markdown, and every other kind of cell the engine has — a pipeline block among them.
    /// </summary>
    /// <remarks>
    /// Code in the blocks' own language is left out: such a cell runs as a block does and draws its card, but it is no
    /// block, so the pipeline the notebook declares never counts it.
    /// </remarks>
    public IReadOnlyList<HostedKind> Kinds { get; }

    /// <summary>Every layout the engine can show the notebook in, in the engine's order, as Verso's View panel lists them.</summary>
    public IReadOnlyList<HostedLayoutChoice> Layouts { get; }

    /// <summary>
    /// Every theme the engine can draw the notebook in, in the engine's order, as Verso's View panel lists them, each with the
    /// block of custom properties Verso's surfaces style themselves from.
    /// </summary>
    public IReadOnlyList<HostedTheme> Themes { get; }

    /// <summary>The engine's extensions: Verso's own and DeepSharp's.</summary>
    internal ExtensionHost Extensions { get; }

    /// <summary>The notebook as Verso's engine holds and runs it.</summary>
    internal Scaffold Scaffold { get; }

    // The block type the engine loaded, which keeps the notebook's session: found the way every part of the notebook
    // finds it, among everything the engine loaded.
    // The notebook's block type, found once as the notebook opens: a stop tells it without asking the engine for anything.
    private readonly StepCellType _blocks;

    // The theme the notebook chose, as the engine resolved it; none while it chose none — the engine's own default is drawn
    // by nobody but the engine's parts, and a view draws the notebook in its own look.
    private string? ThemeId => Scaffold.Notebook.PreferredThemeId is null ? null : Scaffold.ThemeEngine?.ActiveTheme?.ThemeId;

    // What the notebook says of itself, as the engine holds it now.
    private HostedMetadata Metadata => new(Scaffold.Title, Scaffold.DefaultKernelId, Scaffold.Notebook.Created, Scaffold.Notebook.Modified, Scaffold.Notebook.FormatVersion);

    // The layout the notebook is shown in, as the engine holds it now.
    private HostedLayout Layout => new(
        Scaffold.NotebookOps.ActiveLayoutId,
        Enum.Parse<LayoutAllows>(Scaffold.LayoutCapabilities.ToString()),
        Scaffold.LayoutManager?.ActiveLayout?.SupportsPropertiesPanel ?? true);

    /// <summary>
    /// Sets a cell's text, as typing it does; nothing runs, and the notebook is told at once, so what was worked out from
    /// a block as it was — a grid, the pipeline handed to C# cells — is taken back before anything else is asked of it.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="source">Its new text.</param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell's text be changed.</exception>
    public Task<HostedCell> EditAsync(Guid cell, string source)
    {
        ArgumentNullException.ThrowIfNull(source);

        return TurnAsync(async () =>
        {
            Standing(cell);
            MayEdit();
            Scaffold.UpdateCellSource(cell, source);

            // The notebook is told at once: what was worked out from the block as it was is taken back.
            await TellAsync();

            return (await PublishedAsync()).Cells.First(each => each.Id == cell);
        });
    }

    /// <summary>Adds a cell of a kind right after another, as the add button between two cells does; it starts empty.</summary>
    /// <param name="after">The cell it follows.</param>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The new cell.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell it follows or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be added
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> InsertAsync(Guid after, HostedKind kind) => TurnAsync(() =>
        AddedAsync(Scaffold.Notebook.Cells.IndexOf(Standing(after)) + 1, kind));

    /// <summary>Adds a cell of a kind at the end, as the add button under the last cell does; it starts empty.</summary>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The new cell.</returns>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be added
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> AddAsync(HostedKind kind) => TurnAsync(() => AddedAsync(Scaffold.Cells.Count, kind));

    /// <summary>Takes a cell away.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>When it is gone.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be taken away.</exception>
    public Task RemoveAsync(Guid cell) => TurnAsync(async () =>
    {
        Standing(cell);
        await Scaffold.NotebookOps.RemoveCellAsync(cell);
        await TellAsync();

        return true;
    });

    /// <summary>Moves a cell up past its neighbour, to stand right before it.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="neighbour">The cell it passes: the one above it, as the page showed them.</param>
    /// <returns>When it is moved.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote either cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be moved.</exception>
    /// <remarks>
    /// Named by the neighbour it passes rather than by a place, since another view may have moved cells meanwhile, and a
    /// place counted there is somewhere else here. A cell named as its own neighbour passes nothing.
    /// </remarks>
    public Task MoveBeforeAsync(Guid cell, Guid neighbour) => MoveAsync(cell, neighbour, after: false);

    /// <summary>Moves a cell down past its neighbour, to stand right after it.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="neighbour">The cell it passes: the one below it, as the page showed them.</param>
    /// <returns>When it is moved.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote either cell or took it away.</exception>
    /// <exception cref="LayoutCapabilityException">The layout the notebook is shown in does not let a cell be moved.</exception>
    /// <remarks>
    /// Named by the neighbour it passes rather than by a place, since another view may have moved cells meanwhile, and a
    /// place counted there is somewhere else here. A cell named as its own neighbour passes nothing.
    /// </remarks>
    public Task MoveAfterAsync(Guid cell, Guid neighbour) => MoveAsync(cell, neighbour, after: true);

    /// <summary>
    /// Turns a cell into another kind in one step, as Verso's editors do: its type and language change, its text stays,
    /// and what it showed is cleared. The same kind again changes nothing.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="kind">
    /// One of <see cref="Kinds"/>; one named with no language is given the language Verso's editors give it — code in the
    /// notebook's default kernel, else C#.
    /// </param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be changed
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> ChangeKindAsync(Guid cell, HostedKind kind) => TurnAsync(async () =>
    {
        var listed = Listed(kind);
        var changed = Standing(cell);

        MayEdit();

        if (!Same(changed.Type, listed.Type) || !Same(changed.Language, listed.Language))
        {
            changed.Type = listed.Type;
            changed.Language = listed.Language;
            changed.Outputs.Clear();
            Scaffold.OutputChannels.CloseForCell(cell, "the cell's outputs were cleared");
            await TellAsync();
        }

        return (await PublishedAsync()).Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// What a cell's kernel offers to write next where the cursor stands, as Verso's editors ask it while a person types.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it, which may be ahead of what was sent.</param>
    /// <param name="position">Where the cursor stands in it, counted in characters.</param>
    /// <returns>
    /// What is offered; nothing for a cell whose text no kernel reads, and nothing when the kernel was started afresh while it
    /// was asked, whether it answered or failed as it was put away.
    /// </returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <remarks>
    /// Answered beside whatever holds the notebook's turn, as Verso's editors ask while a cell runs. The kernel is started
    /// first when it has not been, as Verso's editors start it, rather than in the background, so the first thing offered
    /// may take the time a kernel takes to start.
    /// </remarks>
    public Task<IReadOnlyList<HostedCompletion>> CompletionsAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return ReadAsync(cell, standing => KernelReadAsync(async () =>
        {
            if (await KernelOfAsync(standing) is not { } kernel)
            {
                return (IReadOnlyList<HostedCompletion>)[];
            }

            return [.. (await kernel.GetCompletionsAsync(code, position))
                .Select(offered => new HostedCompletion(offered.DisplayText, offered.InsertText, offered.Kind, offered.Description, offered.SortText))];
        }, nothing: []));
    }

    /// <summary>What a word in a cell's text means, as Verso's editors ask it when the cursor rests on the word.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it.</param>
    /// <param name="position">Where the cursor rests in it, counted in characters.</param>
    /// <returns>
    /// What the word means; nothing where the kernel says nothing, for a cell whose text no kernel reads, or when the kernel
    /// was started afresh while it was asked, whether it answered or failed as it was put away.
    /// </returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <remarks>Answered beside whatever holds the notebook's turn, as Verso's editors ask while a cell runs.</remarks>
    public Task<HostedHover?> HoverAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return ReadAsync(cell, standing => KernelReadAsync(async () =>
        {
            if (await KernelOfAsync(standing) is not { } kernel || await kernel.GetHoverInfoAsync(code, position) is not { } said)
            {
                return (HostedHover?)null;
            }

            return new HostedHover(
                said.Content,
                said.MimeType,
                said.Range is { } range ? new HostedRange(range.StartLine, range.StartColumn, range.EndLine, range.EndColumn) : null);
        }, nothing: null));
    }

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
        var run = RunFor(running.Id, IsCSharp(KernelOf(running)));

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
    /// <exception cref="InvalidOperationException">No part of that name answers a click.</exception>
    /// <remarks>
    /// A page sends a click from the card it shows until it draws again, and a change may have rewritten the card's block
    /// meanwhile; the click is handed on all the same, as Verso's own editors hand every click on, because the part knows
    /// the block it became. What the part answers is shown only by a cell that still stands. A click is a change: it runs
    /// DeepSharp's blocks as its own, and anything else it asks to run is a run of its own, told and stopped as any run
    /// is; a click whose run was stopped ends as a stopped run ends, with nothing to answer.
    /// </remarks>
    public Task<GestureResult> GestureAsync(HostedGesture gesture) => TurnAsync(async () =>
    {
        var part = Extensions.GetInteractionHandler(gesture.ExtensionId)
            ?? throw new InvalidOperationException($"No part named '{gesture.ExtensionId}' answers a click.");
        var change = new ChangePort(this);
        var context = new CellInteractionContext
        {
            Region = CellRegion.Output,
            InteractionType = gesture.Action,
            Payload = gesture.Payload,
            CellId = gesture.Cell,
            ExtensionId = gesture.ExtensionId,
            CancellationToken = CancellationToken.None,
            Variables = Scaffold.Variables,
            Notebook = change,
            NotebookModel = Scaffold.Notebook,
        };
        var answer = await ChangeAsync(change, () => part.OnCellInteractionAsync(context));

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

    /// <summary>Presses a toolbar button.</summary>
    /// <param name="id">The button.</param>
    /// <param name="cells">The cells it is pressed for, for a button on a cell's toolbar.</param>
    /// <returns>The file it handed over, for whoever pressed it; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">The engine has no button of that name.</exception>
    /// <remarks>
    /// A button may run cells, C# among them, so a press takes its turn among the process's C# runs as a C# run does. A
    /// press is one run, and the button acts on the notebook through it, so once the run is stopped the notebook refuses
    /// what the button asks. A file is never written beside the notebook: where it is saved is for whoever pressed the
    /// button to say.
    /// </remarks>
    public Task<HostedFile?> RunToolbarAsync(string id, params Guid[] cells) => TurnAsync(async () =>
    {
        var action = Extensions.GetToolbarActions().FirstOrDefault(each => each.ActionId == id)
            ?? throw new InvalidOperationException($"The notebook has no toolbar button '{id}'.");
        var run = RunFor(null, runsCSharp: true);

        // The button acts on the notebook through its run, so a stop reaches every cell it would still run; one that runs a
        // cell, or code, that never ends is stopped as a cell's run is, and a stop starts afresh only what runs.
        var context = new ToolbarContext(Scaffold, cells, new RunPort(Scaffold, run));

        await RunUntilStoppedAsync(run, () => action.ExecuteAsync(context));

        // A stopped press hands nothing over: what it wrote after the stop is no one's.
        return run.Claimed ? null : context.Handed;
    });

    /// <summary>A cell's properties panel: a section from every part that has one for the cell, in their order.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The sections.</returns>
    /// <exception cref="CellGoneException">The notebook's last version holds no such cell: a change rewrote it or took it away.</exception>
    /// <exception cref="ObjectDisposedException">The notebook was closed.</exception>
    /// <exception cref="InvalidOperationException">The layout the notebook is shown in has no properties panel.</exception>
    /// <remarks>Drawn beside whatever holds the notebook's turn, as Verso's editors draw the panel while a cell runs.</remarks>
    public Task<IReadOnlyList<HostedSection>> PropertiesAsync(Guid cell) => ReadAsync(cell, async shown =>
    {
        HasPanel();

        // Drawing a section is a look, and a look does nothing to the notebook.
        var context = new RenderContext(Scaffold, shown, new ReadPort(Scaffold));
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
    /// <exception cref="InvalidOperationException">
    /// No part of that name has a properties section, or the layout the notebook is shown in has no properties panel.
    /// </exception>
    public Task SetPropertyAsync(Guid cell, string part, string field, object? value) => TurnAsync(async () =>
    {
        var changed = Standing(cell);

        HasPanel();
        var provider = Extensions.GetPropertyProviders().FirstOrDefault(each => each.ExtensionId == part)
            ?? throw new InvalidOperationException($"No part named '{part}' has a properties section.");

        var change = new ChangePort(this);

        return await ChangeAsync(change, async () =>
        {
            await provider.OnPropertyChangedAsync(changed, field, value, new RenderContext(Scaffold, changed, change));

            return true;
        });
    });

    /// <summary>
    /// Hands what a person did to the arrangement the notebook's layout drew — a tile moved, resized or run — to the layout's
    /// own part, as Verso's editors hand it on. An act named in the host's own space is the host's: every version carries the
    /// arrangement as it is drawn, so asking for it again leaves nothing to do. An act the layout has no part for changes
    /// nothing.
    /// </summary>
    /// <param name="interaction">The act.</param>
    /// <returns>The file the part handed over, for whoever acted; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">
    /// The notebook is no longer shown in the layout that drew what was acted on; nothing is done.
    /// </exception>
    /// <remarks>
    /// An act is a change: it runs DeepSharp's blocks as its own, and anything else it asks to run — a tile's C# cell — is a
    /// run of its own, told and stopped as any run is, so a tile moved or resized runs nothing and waits for no C# run. A
    /// layout may add, take away or move cells through the notebook's operations, as Verso's notebook layout does, so the
    /// notebook is told its cells changed after every act.
    /// </remarks>
    public Task<HostedFile?> InteractAsync(HostedLayoutInteraction interaction) => TurnAsync(async () =>
    {
        var layout = Scaffold.LayoutManager?.ActiveLayout;

        if (layout is null || !Names(layout.LayoutId, interaction.Layout))
        {
            throw new InvalidOperationException($"The notebook is no longer shown in the layout '{interaction.Layout}'.");
        }

        var extension = (layout as IExtension)?.ExtensionId ?? string.Empty;

        if (interaction.Action.StartsWith(HostsOwn, StringComparison.Ordinal) || !Extensions.TryGetLayoutInteractionHandler(extension, layout.LayoutId, out var part))
        {
            return null;
        }

        var change = new ChangePort(this);
        var context = new ToolbarContext(Scaffold, [], change);

        await ChangeAsync(change, async () =>
        {
            // What the part asks to draw again is drawn with the next version, whatever it asks for.
            await part.OnLayoutInteractionAsync(new LayoutInteractionContext
            {
                ExtensionId = extension,
                LayoutId = layout.LayoutId,
                InteractionType = interaction.Action,
                Payload = interaction.Payload,
                TargetId = interaction.Target,
                Verso = context,
            });

            return true;
        });

        await TellAsync();

        // A stopped act hands nothing over, as a stopped press hands nothing over.
        return change.Stopped ? null : context.Handed;
    });

    /// <summary>
    /// Titles the notebook, as Verso's Metadata panel does: the title names what it is exported as. The same title again
    /// changes nothing.
    /// </summary>
    /// <param name="title">The title; nothing for none.</param>
    /// <returns>When it is titled.</returns>
    public Task RetitleAsync(string? title) => TurnAsync(() =>
    {
        Scaffold.Title = title;

        return Task.FromResult(true);
    });

    /// <summary>
    /// Shows the notebook in another layout, as Verso's View panel switches it: through the engine's own operations, which
    /// name the layout in the notebook, so it is saved as the notebook's choice.
    /// </summary>
    /// <param name="id">One of <see cref="Layouts"/>.</param>
    /// <returns>When it is shown in it.</returns>
    /// <exception cref="InvalidOperationException">The engine has no layout of that id; nothing changes.</exception>
    public Task SwitchLayoutAsync(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        return TurnAsync(() =>
        {
            var layout = Layouts.FirstOrDefault(each => Names(each.Id, id));

            if (layout.Id is null)
            {
                throw new InvalidOperationException($"The notebook has no layout '{id}' to be shown in.");
            }

            Scaffold.NotebookOps.SetActiveLayout(layout.Id);

            return Task.FromResult(true);
        });
    }

    /// <summary>
    /// Draws the notebook in a theme, as Verso's View panel switches it: an explicit choice, which the notebook names from
    /// then on and saves as its own. Only such a choice names one; until then the notebook is drawn in a view's own look.
    /// </summary>
    /// <param name="id">One of <see cref="Themes"/>.</param>
    /// <returns>When it is drawn in it.</returns>
    /// <exception cref="InvalidOperationException">The engine has no theme of that id; nothing changes.</exception>
    public Task SwitchThemeAsync(string id)
    {
        ArgumentNullException.ThrowIfNull(id);

        return TurnAsync(() =>
        {
            var theme = Themes.FirstOrDefault(each => Names(each.Id, id));

            if (theme.Id is null)
            {
                throw new InvalidOperationException($"The notebook has no theme '{id}' to be drawn in.");
            }

            Scaffold.NotebookOps.SetActiveTheme(theme.Id);

            return Task.FromResult(true);
        });
    }

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
        new FormatGuard(),
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
            EnsureDefaults(extensions, scaffold);
            await RestoreAsync(scaffold);
            await scaffold.RenderTransientCellsAsync(cancellationToken);

            // Drawn first, since drawing places what the layout had not placed yet; what differs from the file as it opens —
            // a cell's id repaired, a tile placed afresh — is unsaved from the first version.
            var arrangement = await ArrangementAsync(scaffold);

            await FlushAsync(scaffold);

            return new NotebookHost(
                filePath, extensions, scaffold, saved, await ButtonsAsync(extensions, scaffold), Differs(saved, scaffold, filePath, extensions), arrangement);
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
    /// Closes the notebook, when no view has shown it for the grace, nothing runs or waits — nor runs on, left behind by a
    /// stop — and nothing in it differs from the file it was last saved to; otherwise leaves it open.
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

        // A comparison caught by a cell still showing more keeps the notebook open: what it shows is not in the file yet.
        if (Volatile.Read(ref _pending) > 0 || Executing.Count > 0 || await UnsavedAsync(takeBack: true, caught: true))
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
    /// Shuts the notebook, the first half of closing it, at once: every view ends, anything asked from now on is refused,
    /// and the run under way is stopped as <see cref="Stop"/> stops it — one that waits never runs, one that runs is left
    /// behind — so no close waits for a run.
    /// </summary>
    internal void Shut()
    {
        foreach (var view in Interlocked.Exchange(ref _audience, Audience.Gone).Views)
        {
            view.End();
        }

        Volatile.Read(ref _running)?.Stop();
    }

    /// <summary>
    /// Closes the notebook: it is shut, so every view ends, anything asked from now on is refused and the run under way is
    /// stopped; a change under way finishes, and then the engine closes.
    /// </summary>
    /// <returns>When it is closed.</returns>
    internal async ValueTask CloseAsync()
    {
        Shut();

        await _turns.TakeTurnAsync(async () =>
        {
            await CloseEngineAsync();

            return true;
        });
    }

    // Reads a notebook the way Verso's own editors read one: through its format's serializer, and past the guards that run
    // after reading. Nothing the engine falls back on is written into it, as Verso's browser editor writes nothing: a file
    // that names no kernel or layout is saved naming none.
    private static async Task<NotebookModel> ReadAsync(ExtensionHost extensions, INotebookSerializer serializer, string content, string path)
    {
        var notebook = await serializer.DeserializeAsync(content);

        foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(path, serializer.FormatId)).OrderBy(each => each.Priority))
        {
            notebook = await guard.PostDeserializeAsync(notebook, path);
        }

        return notebook;
    }

    // What the engine falls back on when the notebook names none, or names one it does not have — the notebook's own
    // layout, a light theme — set on the engine and never written into the notebook, as Verso's browser editor sets them.
    private static void EnsureDefaults(ExtensionHost extensions, Scaffold scaffold)
    {
        if (scaffold.LayoutManager is { ActiveLayout: null } layouts)
        {
            layouts.TryActivate(LayoutDefaults.LayoutId);
        }

        if (scaffold.ThemeEngine is { ActiveTheme: null } themes && extensions.GetThemes().FirstOrDefault(theme => theme.ThemeKind == ThemeKind.Light) is { } light)
        {
            themes.SetActiveTheme(light.ThemeId);
        }
    }

    // What the file holds for the notebook's layouts and for the parts' settings, handed back to them as the notebook opens,
    // as Verso's editors hand them back: before anything draws the notebook, which would otherwise arrange the layouts
    // afresh, and before any save takes them back, which would otherwise take the parts' defaults in their place.
    private static async Task RestoreAsync(Scaffold scaffold)
    {
        if (scaffold.LayoutManager is { } layouts)
        {
            // Restoring an arrangement is a look at the notebook, and a look does nothing to it.
            await layouts.RestoreMetadataAsync(scaffold.Notebook, new ToolbarContext(scaffold, [], new ReadPort(scaffold)));
        }

        if (scaffold.SettingsManager is { } settings)
        {
            await settings.RestoreSettingsAsync(scaffold.Notebook);
        }
    }

    // What the layouts and the parts' settings hold now, taken back into the notebook — before every save and every look
    // at what is unsaved, as Verso's editors take them — since each keeps what a person changed in it until then.
    private static async Task FlushAsync(Scaffold scaffold)
    {
        if (scaffold.LayoutManager is { } layouts)
        {
            await layouts.SaveMetadataAsync(scaffold.Notebook);
        }

        if (scaffold.SettingsManager is { } settings)
        {
            await settings.SaveSettingsAsync(scaffold.Notebook);
        }
    }

    private async Task CloseEngineAsync()
    {
        Scaffold.OnCellExecuting -= Began;
        Scaffold.OnCellExecuted -= Ended;
        Scaffold.OnCellOutputUpdated -= Showed;
        Scaffold.OnKernelRestarting -= Restarting;
        Scaffold.OnKernelRestarted -= Restarted;
        Scaffold.OnKernelRestartFailed -= RestartFailed;
        await Scaffold.DisposeAsync();
        await Extensions.DisposeAsync();
    }

    // Everything done to the notebook: refused once it closes, and otherwise one at a time, in the order it came, each
    // telling every view what it does while it does it, and ending with the notebook published as it then stands. A close
    // stops the run under way and waits only for a change under way: whatever else was asked before it and still waits its
    // turn is refused when that turn comes, since a close discards what has not begun, as it discards what is not saved.
    private Task<T> TurnAsync<T>(Func<Task<T>> change)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _audience).Closed, this);
        Interlocked.Increment(ref _pending);

        return _turns.TakeTurnAsync(async () =>
        {
            try
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _audience).Closed, this);

                return await PublishingAsync(change);
            }
            finally
            {
                Interlocked.Decrement(ref _pending);
            }
        });
    }

    // A read of a cell, answered beside whatever holds the notebook's turn, as Verso's editors read a cell's panel and ask its
    // kernel while a cell runs: refused once the notebook is closed; answered for a cell the notebook's last version holds,
    // so one a change under way added is not read before any view is told it, and one it took away is gone; and counted
    // among what is asked of the notebook, so it stays open while the read reads.
    private async Task<T> ReadAsync<T>(Guid cell, Func<CellModel, Task<T>> read)
    {
        Interlocked.Increment(ref _pending);

        try
        {
            // Looked at once counted, so a close that did not count it has marked the notebook closed by now.
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _audience).Closed, this);

            var current = Current;

            if (!current.Cells.Any(each => each.Id == cell) || Scaffold.GetCell(cell) is not { } standing)
            {
                throw new CellGoneException(cell, current.Version);
            }

            return await read(standing);
        }
        finally
        {
            Interlocked.Decrement(ref _pending);
        }
    }

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

    // Whether the notebook differs from the file it was last saved to, as Verso's own comparison of two notebooks finds it:
    // with what the layouts and the parts' settings hold taken back into it first when nothing else acts on the notebook,
    // since a turn's work may be writing into it meanwhile. A comparison caught by a run writing what a cell shows — which
    // Verso's engine catches the same way when it reads such a list — answers what it is told to fall back on.
    private async Task<bool> UnsavedAsync(bool takeBack, bool caught)
    {
        if (takeBack)
        {
            await FlushAsync(Scaffold);
        }

        try
        {
            return Differs(_saved, Scaffold, FilePath, Extensions);
        }
        catch (Exception written) when (written is InvalidOperationException or ArgumentException)
        {
            return caught;
        }
    }

    // What in a notebook differs from the file it was last saved to, as Verso's own comparison of two notebooks finds it,
    // told which cells' outputs are never saved.
    private static bool Differs(NotebookModel saved, Scaffold scaffold, string path, ExtensionHost extensions)
    {
        var diff = NotebookDiffEngine.Compute(saved, scaffold.Notebook, path, extensions.GetCellTypes());

        return diff.Summary.Added + diff.Summary.Removed + diff.Summary.Modified + diff.Summary.Moved + diff.MetadataChanges.Count > 0;
    }

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
    // written by a run keeps what it showed at the last version, and so does the word on what is unsaved; the run's next
    // word about it, or the end of its turn, publishes the rest.
    private async Task<NotebookVersion> NextAsync(bool whole)
    {
        var buttons = await ButtonsAsync(Extensions, Scaffold);
        var last = Current;

        // Drawn before what is unsaved is taken, since drawing places what the layout had not placed yet.
        var arrangement = await ArrangementAsync(Scaffold);
        var unsaved = await UnsavedAsync(takeBack: whole, caught: last.Unsaved);
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
        var layout = Layout;
        var theme = ThemeId;

        var pressable = buttons.SequenceEqual(last.Buttons) ? null : buttons;
        HostedArrangement? arranged = arrangement == last.Arrangement ? null : arrangement;
        var metadata = Metadata;
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

    // The engine says a cell began: it is what the run whose work the engine is in runs now. A cell the engine begins
    // outside every run — a block a change runs — is no run's: every view is told it runs, with nothing to stop. Either
    // way the kernels answer again, so a start afresh that failed before is past.
    private void Began(Guid cell)
    {
        ImmutableInterlocked.Update(ref _restarts, restarts => restarts.Answered());

        // The engine found the cell a moment ago, and nothing else changes the notebook while a run holds its turn.
        if (_raising.Value is { } run)
        {
            run.Began(cell, KernelOf(Scaffold.Cells.First(each => each.Id == cell)));
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

    // Every toolbar button the engine has — Verso's own and DeepSharp's — each saying whether it can be pressed now, by place
    // and then in their order. A button of the notebook as a whole is asked once, with no cell chosen; one on a cell's
    // toolbar or in its menu is asked for every cell, as Verso's editors ask it for the cell it is drawn on, so a page draws
    // it pressable where it is — running a cell wherever the layout lets cells run, clearing one once it shows something.
    // Asking is a look, which does nothing to the notebook.
    private static async Task<IReadOnlyList<HostedToolbarAction>> ButtonsAsync(ExtensionHost extensions, Scaffold scaffold)
    {
        var look = new ReadPort(scaffold);
        var buttons = new List<HostedToolbarAction>();

        foreach (var action in extensions.GetToolbarActions())
        {
            var place = Enum.Parse<ToolbarPlace>(action.Placement.ToString());
            var cells = new List<Guid>();
            string? fault = null;

            if (place is ToolbarPlace.CellToolbar or ToolbarPlace.ContextMenu)
            {
                foreach (var cell in scaffold.Cells)
                {
                    var forCell = await AnswerAsync(action, new ToolbarContext(scaffold, [cell.Id], look));

                    if (forCell.Pressable)
                    {
                        cells.Add(cell.Id);
                    }

                    fault ??= forCell.Fault;
                }
            }

            var whole = await AnswerAsync(action, new ToolbarContext(scaffold, [], look));

            buttons.Add(new HostedToolbarAction(
                action.ActionId,
                action.DisplayName,
                action.Icon,
                action.IconOnly,
                action.IsPrimary,
                action.ConfirmationPrompt,
                place,
                action.Order,
                whole.Pressable,
                cells,
                whole.Fault ?? fault));
        }

        return [.. buttons.OrderBy(button => button.Place).ThenBy(button => button.Order)];
    }

    // What the layout the notebook is shown in draws of its own, as Verso's editors ask it to draw: only a layout that draws
    // an arrangement in the page itself rather than the notebook's list, or in a frame of its own — and never the engine's
    // own notebook layout, the list of the cells, whose arrangement says nothing a version does not. Drawing is a look, and
    // a look does nothing to the notebook. A part that fails to draw holds up no version, and says why instead.
    private static async Task<HostedArrangement> ArrangementAsync(Scaffold scaffold)
    {
        if (scaffold.LayoutManager?.ActiveLayout is not { RequiresCustomRenderer: true, RendererIsolation: LayoutRendererIsolation.Inline } layout
            || (Names(layout.LayoutId, LayoutDefaults.LayoutId) && layout is IExtension { ExtensionId: var extension } && Names(extension, LayoutDefaults.ExtensionId)))
        {
            return default;
        }

        try
        {
            var drawn = await layout.RenderLayoutAsync(scaffold.Notebook.Cells, new ToolbarContext(scaffold, [], new ReadPort(scaffold)));

            return Names(drawn.MimeType, "text/html") ? new HostedArrangement(drawn.Content, null) : default;
        }
        catch (Exception failed)
        {
            return new HostedArrangement(null, failed.Message);
        }
    }

    // Whether a button can be pressed, as its part says. A part that fails to say is not pressed: every version asks every
    // button, and one part that fails holds up no version, so what failed is told with the button instead.
    private static async Task<Answer> AnswerAsync(IToolbarAction action, ToolbarContext context)
    {
        try
        {
            return new Answer(await action.IsEnabledAsync(context), null);
        }
        catch (Exception failed)
        {
            return new Answer(false, failed.Message);
        }
    }

    // A fresh wait for the next word that the notebook is to be published: whether to publish it at once.
    private Task<bool> Listen()
    {
        var said = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        Volatile.Write(ref _said, said);

        return said.Task;
    }

    private CellModel Standing(Guid cell) => Scaffold.GetCell(cell) ?? throw new CellGoneException(cell, Current.Version);

    // The kinds Verso's editors offer, with each language folded in: code in every language the engine runs but the
    // blocks' own, then Markdown, then every other kind of cell in the order the engine has them. The languages are the
    // engine's kernels, which is what its list of registered languages holds in a notebook opened here.
    private static HostedKind[] KindsOf(ExtensionHost extensions) =>
    [
        .. extensions.GetKernels()
            .Where(kernel => !Same(kernel.LanguageId, StepKernel.Language))
            .Select(kernel => new HostedKind("code", kernel.LanguageId, kernel.DisplayName, Editable: true, Rendered: false)),
        .. extensions.GetCellTypes()
            .Where(type => !Same(type.CellTypeId, "code"))
            .OrderBy(type => Same(type.CellTypeId, "markdown") ? 0 : 1)
            .Select(type => new HostedKind(type.CellTypeId, type.Kernel?.LanguageId, type.DisplayName, type.IsEditable, Rendered(extensions, type))),
    ];

    // Whether a kind is shown rendered once it has run, as Verso's editors decide it: by the renderer the engine has for its
    // type, else by the one its cell type brings.
    private static bool Rendered(ExtensionHost extensions, ICellType type) =>
        (extensions.GetRenderers().FirstOrDefault(renderer => Names(renderer.CellTypeId, type.CellTypeId)) ?? type.Renderer)?.CollapsesInputOnExecute ?? false;

    private static bool Same(string? one, string? other) => string.Equals(one, other, StringComparison.OrdinalIgnoreCase);

    // The layouts the engine has, as Verso's View panel lists them.
    private static HostedLayoutChoice[] LayoutsOf(ExtensionHost extensions) =>
    [
        .. extensions.GetLayouts().Select(layout =>
            new HostedLayoutChoice(layout.LayoutId, layout.DisplayName, Enum.Parse<LayoutAllows>(layout.Capabilities.ToString()))),
    ];

    // The themes the engine has, as Verso's View panel lists them, each with the block Verso's engine writes for it.
    private static HostedTheme[] ThemesOf(ExtensionHost extensions) =>
    [
        .. extensions.GetThemes().Select(theme =>
            new HostedTheme(theme.ThemeId, theme.DisplayName, Enum.Parse<ThemeTone>(theme.ThemeKind.ToString()), ThemeCss.BuildRootBlock(theme))),
    ];

    // A kind the notebook lists, as it lists it; one it does not list is refused. A kind named with no language is given
    // the one Verso's editors give it: its cell type's kernel's; none for a type a renderer draws; else the notebook's
    // default kernel, else C#.
    private HostedKind Listed(HostedKind kind)
    {
        var language = kind.Language ?? LanguageOf(kind.Type);

        foreach (var each in Kinds.Where(each => Same(each.Type, kind.Type) && Same(each.Language, language)))
        {
            return each;
        }

        throw new InvalidOperationException($"The notebook has no kind of cell '{kind.Type}' in '{kind.Language}' to add or turn a cell into.");
    }

    // The language Verso's editors give a cell of a type named with no language.
    private string? LanguageOf(string type)
    {
        if (Extensions.GetCellTypes().FirstOrDefault(each => Names(each.CellTypeId, type)) is { } cellType)
        {
            return cellType.Kernel?.LanguageId;
        }

        return Extensions.GetRenderers().Any(renderer => Names(renderer.CellTypeId, type)) ? null : Scaffold.DefaultKernelId ?? CSharp;
    }

    // A cell of a listed kind added at a place, empty, through the port the notebook's layout guards; the notebook is told.
    private async Task<HostedCell> AddedAsync(int at, HostedKind kind)
    {
        var listed = Listed(kind);
        var added = Guid.Parse(await Scaffold.NotebookOps.InsertCellAsync(at, listed.Type, listed.Language));

        await TellAsync();

        return (await PublishedAsync()).Cells.First(each => each.Id == added);
    }

    // A cell moved to stand right before or right after its neighbour, wherever either stands now.
    private Task MoveAsync(Guid cell, Guid neighbour, bool after) => TurnAsync(async () =>
    {
        var cells = Scaffold.Notebook.Cells;
        var from = cells.IndexOf(Standing(cell));
        var to = cells.IndexOf(Standing(neighbour));

        if (from != to)
        {
            await Scaffold.NotebookOps.MoveCellAsync(cell, to - (from < to ? 1 : 0) + (after ? 1 : 0));
            await TellAsync();
        }

        return true;
    });

    // A layout with no properties panel refuses the panel and its fields, as Verso's editors offer the panel only in a
    // layout that has one; a form's field rewrites a block, which such a layout does not let a person do.
    private void HasPanel()
    {
        if (!Layout.HasPropertiesPanel)
        {
            throw new InvalidOperationException("The layout the notebook is shown in has no properties panel.");
        }
    }

    // A layout that does not let a cell's text or kind be changed refuses it, as the engine's own port refuses what its
    // layout does not allow.
    private void MayEdit()
    {
        if (!Scaffold.LayoutCapabilities.HasFlag(LayoutCapabilities.CellEdit))
        {
            throw new LayoutCapabilityException(LayoutCapabilities.CellEdit);
        }
    }

    // The kernel a cell's text is written for, started first as Verso's editors start it before they ask it anything;
    // none for a cell whose language no kernel the engine has reads.
    private async Task<ILanguageKernel?> KernelOfAsync(CellModel cell)
    {
        if (cell.Language is not { } language || Scaffold.GetKernel(language) is not { } kernel)
        {
            return null;
        }

        await Scaffold.WarmUpKernelAsync(language);

        return kernel;
    }

    // Tells the notebook its cells changed, so what was worked out from the blocks as they were is taken back.
    private Task TellAsync() => _blocks.BlocksChangedAsync(Scaffold.Notebook, Scaffold.Variables, new ChangePort(this));

    // What a part does as a change: a change whose run was stopped ends as a stopped run ends, with nothing to answer, and a
    // change is over only once every run it asked for has ended or was stopped, whether or not it waited for them.
    private static async Task<T?> ChangeAsync<T>(ChangePort change, Func<Task<T>> act)
    {
        var answer = default(T);

        try
        {
            answer = await act();
        }
        catch (OperationCanceledException) when (change.Stopped)
        {
            // A change whose run was stopped ends as that run ends.
        }
        finally
        {
            await change.SettledAsync();
        }

        // A stopped change writes nothing, what it answers included.
        return change.Stopped ? default : answer;
    }

    /// <summary>A run, numbered for the notebook, taking the C# turn when it runs C#.</summary>
    /// <param name="cell">The cell it runs; none when it runs several, or code in no cell.</param>
    /// <param name="runsCSharp">Whether it runs C#.</param>
    /// <returns>The run.</returns>
    /// <remarks>Its stop tells the notebook in the step that stops it, so what it asks for from then on writes nothing.</remarks>
    internal Run RunFor(Guid? cell, bool runsCSharp) => new(Interlocked.Increment(ref _runs), cell, takesTheCSharpTurn: runsCSharp, Said, _blocks.StoppedAsync);

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
    internal bool RunsCSharp(Guid cell) => Scaffold.GetCell(cell) is { } found && IsCSharp(KernelOf(found));

    /// <summary>Whether code in a language — the notebook's default kernel when it names none — runs in the C# kernel.</summary>
    /// <param name="language">The language.</param>
    /// <returns>Whether it does.</returns>
    internal bool RunsCSharp(string? language) => IsCSharp(language ?? Scaffold.DefaultKernelId);

    private static bool IsCSharp(string? kernel) => string.Equals(kernel, CSharp, StringComparison.OrdinalIgnoreCase);

    private async Task SaveToAsync(string path)
    {
        var serializer = Extensions.GetSerializers().FirstOrDefault(each => each.CanImport(path))
            ?? throw new NotSupportedException($"No format Verso knows writes '{Path.GetFileName(path)}'.");

        // What a live output shows now is what is saved, as Verso's own editors ask before they write; what the layouts and
        // the parts' settings hold now is taken back into the notebook; and the notebook is stamped with the time it is saved.
        await Scaffold.RefreshLiveOutputsAsync();
        await FlushAsync(Scaffold);
        Scaffold.Notebook.Modified = DateTimeOffset.UtcNow;

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

    // Which kernel runs a cell, in the order the engine asks when it runs one: a cell type it has answers first — with
    // its own kernel, or with none when it only draws; a cell of no such type runs in the kernel its language names; a
    // type a renderer claims is drawn and runs none; a cell that names no language runs in the notebook's default kernel.
    // None is a cell no kernel runs, so a run takes the C# turn, and a stop starts a kernel afresh, only when a kernel
    // really runs the cell.
    private string? KernelOf(CellModel cell)
    {
        if (Extensions.GetCellTypes().FirstOrDefault(type => Names(type.CellTypeId, cell.Type)) is { } type)
        {
            return type.Kernel?.LanguageId;
        }

        if (!string.IsNullOrEmpty(cell.Language) && KernelNamed(cell.Language) is { } named)
        {
            return named.LanguageId;
        }

        if (Extensions.GetRenderers().Any(renderer => Names(renderer.CellTypeId, cell.Type)))
        {
            return null;
        }

        // A language no kernel reads runs in none: the engine takes it for the default kernel's name, and finds no kernel.
        return cell.Language is null ? Scaffold.DefaultKernelId : null;
    }

    // The kernel the engine finds for a language: one it was given, one a part brings, or one a cell type brings.
    private ILanguageKernel? KernelNamed(string language) =>
        Scaffold.GetKernel(language)
        ?? Extensions.GetCellTypes().Select(type => type.Kernel).FirstOrDefault(kernel => kernel is not null && Names(kernel.LanguageId, language));

    // Whether two of the engine's names name the same thing, as the engine compares them.
    private static bool Names(string one, string other) => string.Equals(one, other, StringComparison.OrdinalIgnoreCase);

    // What a button's part answered when asked whether it can be pressed: whether it can, or why it could not say.
    private readonly record struct Answer(bool Pressable, string? Fault);

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
