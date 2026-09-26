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
        Kinds = KindsOf(extensions);
        _saved = saved;
        _published = new(new NotebookVersion(0, [.. scaffold.Cells.Select(cell => cell.Hosted([]))], null, Layout));
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

    /// <summary>
    /// Every kind a cell can be added as or turned into, as Verso's editors offer them: code in each language the engine
    /// runs, Markdown, and every other kind of cell the engine has — a pipeline block among them.
    /// </summary>
    /// <remarks>
    /// Code in the blocks' own language is left out: such a cell runs as a block does and draws its card, but it is no
    /// block, so the pipeline the notebook declares never counts it.
    /// </remarks>
    public IReadOnlyList<HostedKind> Kinds { get; }

    /// <summary>The engine's extensions: Verso's own and DeepSharp's.</summary>
    internal ExtensionHost Extensions { get; }

    /// <summary>The notebook as Verso's engine holds and runs it.</summary>
    internal Scaffold Scaffold { get; }

    // The block type the engine loaded, which keeps the notebook's session: found the way every part of the notebook
    // finds it, among everything the engine loaded.
    private StepCellType Blocks => ((IExtensionHostContext)Extensions).GetLoadedExtensions().OfType<StepCellType>().First();

    // The layout the notebook is shown in, as the engine holds it now.
    private HostedLayout Layout => new(Scaffold.NotebookOps.ActiveLayoutId, Enum.Parse<LayoutAllows>(Scaffold.LayoutCapabilities.ToString()));

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

            return EndTurn().Cells.First(each => each.Id == cell);
        });
    }

    /// <summary>Adds a cell of a kind right after another, as the add button between two cells does; it starts empty.</summary>
    /// <param name="after">The cell it follows.</param>
    /// <param name="kind">One of <see cref="Kinds"/>.</param>
    /// <returns>The new cell.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell it follows or took it away.</exception>
    /// <exception cref="InvalidOperationException">
    /// The notebook lists no such kind, or the layout it is shown in does not let a cell be added
    /// (<see cref="LayoutCapabilityException"/>).
    /// </exception>
    public Task<HostedCell> InsertAsync(Guid after, HostedKind kind) => TurnAsync(() =>
        AddedAsync(Scaffold.Notebook.Cells.IndexOf(Standing(after)) + 1, kind));

    /// <summary>Adds a cell of a kind at the end, as the add button under the last cell does; it starts empty.</summary>
    /// <param name="kind">One of <see cref="Kinds"/>.</param>
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
    /// <param name="kind">One of <see cref="Kinds"/>.</param>
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

        return EndTurn().Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// What a cell's kernel offers to write next where the cursor stands, as Verso's editors ask it while a person types.
    /// </summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it, which may be ahead of what was sent.</param>
    /// <param name="position">Where the cursor stands in it, counted in characters.</param>
    /// <returns>What is offered; nothing for a cell whose text no kernel reads.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <remarks>
    /// The kernel is started first when it has not been, as Verso's editors start it, rather than in the background, so
    /// the first thing offered may take the time a kernel takes to start.
    /// </remarks>
    public Task<IReadOnlyList<HostedCompletion>> CompletionsAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return TurnAsync(async () =>
        {
            if (await KernelOfAsync(Standing(cell)) is not { } kernel)
            {
                return (IReadOnlyList<HostedCompletion>)[];
            }

            return [.. (await kernel.GetCompletionsAsync(code, position))
                .Select(offered => new HostedCompletion(offered.DisplayText, offered.InsertText, offered.Kind, offered.Description, offered.SortText))];
        });
    }

    /// <summary>What a word in a cell's text means, as Verso's editors ask it when the cursor rests on the word.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="code">Its text as the person has it.</param>
    /// <param name="position">Where the cursor rests in it, counted in characters.</param>
    /// <returns>What the word means; nothing where the kernel says nothing, or for a cell whose text no kernel reads.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    public Task<HostedHover?> HoverAsync(Guid cell, string code, int position)
    {
        ArgumentNullException.ThrowIfNull(code);

        return TurnAsync(async () =>
        {
            if (await KernelOfAsync(Standing(cell)) is not { } kernel || await kernel.GetHoverInfoAsync(code, position) is not { } said)
            {
                return (HostedHover?)null;
            }

            return new HostedHover(
                said.Content,
                said.MimeType,
                said.Range is { } range ? new HostedRange(range.StartLine, range.StartColumn, range.EndLine, range.EndColumn) : null);
        });
    }

    /// <summary>Runs a cell, as pressing its run button does.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The cell as it stands after the run, or after <see cref="Stop"/> ended it.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <remarks>
    /// The cell runs in its language's kernel, else the notebook's default kernel, as the engine decides. A C# run takes
    /// its turn among every C# run in the process, because a C# kernel takes over the process's console while it runs,
    /// and two at once would print into each other; while it waits for that turn it is the run under way all the same,
    /// told to every view and stopped like any other.
    /// </remarks>
    public Task<HostedCell> RunAsync(Guid cell) => TurnAsync(async () =>
    {
        var running = Standing(cell);
        var kernel = KernelOf(running);

        await RunUntilStoppedAsync(
            new Run(Interlocked.Increment(ref _runs), running.Id, takesTheCSharpTurn: string.Equals(kernel, CSharp, StringComparison.OrdinalIgnoreCase)),
            () => Scaffold.ExecuteCellAsync(running.Id),
            kernel);

        return EndTurn().Cells.First(each => each.Id == cell);
    });

    /// <summary>
    /// Stops a run — a cell's, or a toolbar button's — whether it still waits for another notebook's C# run or runs. One
    /// that waits never runs: its wait ends, and when the C# turn it waited for comes, nothing starts. One that runs is
    /// stopped the only way Verso's engine stops a run that does not end, with a fresh kernel — the kernel of the cell
    /// that runs — so what that kernel held, the notebook's variables and the pipeline handed to C# cells among them, is
    /// gone. The notebook is told first, so what the run left behind asks for from then on writes nothing, and the
    /// notebook takes its next change at once; a run that never ends goes on in the background until the application does.
    /// </summary>
    /// <param name="run">The run, by its number — so a stop sent again after that run ended stops no other.</param>
    /// <returns>Whether it stopped that run; nothing is stopped when no run of that number is under way.</returns>
    public bool Stop(long run) => Volatile.Read(ref _running) is { } under && under.Number == run && under.Stop();

    /// <summary>
    /// The run under way as it stands now — the cell that runs or waits, since when, and the number a stop names — or
    /// nothing; every view is told it with each version.
    /// </summary>
    public HostedRun? Running => Volatile.Read(ref _running) is { Waits: true } waiting
        ? new HostedRun(waiting.Number, waiting.Cell, waiting.Asked, Waits: true)
        : Volatile.Read(ref _executing)?.Value;

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
    /// the block it became. What the part answers is shown only by a cell that still stands.
    /// </remarks>
    public Task<GestureResult> GestureAsync(HostedGesture gesture) => TurnAsync(async () =>
    {
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
    /// <remarks>
    /// A button of the notebook as a whole is asked once, with no cell chosen. One on a cell's toolbar or in its menu is
    /// asked for every cell, as Verso's editors ask it for the cell it is drawn on, so a page can draw it pressable where
    /// it is — running a cell wherever the layout lets cells run, clearing one once it shows something.
    /// </remarks>
    public Task<IReadOnlyList<HostedToolbarAction>> ToolbarAsync() => TurnAsync(async () =>
    {
        var context = new ToolbarContext(Scaffold, []);
        var buttons = new List<HostedToolbarAction>();

        foreach (var action in Extensions.GetToolbarActions())
        {
            var place = Enum.Parse<ToolbarPlace>(action.Placement.ToString());
            var cells = new List<Guid>();

            if (place is ToolbarPlace.CellToolbar or ToolbarPlace.ContextMenu)
            {
                foreach (var cell in Scaffold.Cells)
                {
                    if (await action.IsEnabledAsync(new ToolbarContext(Scaffold, [cell.Id])))
                    {
                        cells.Add(cell.Id);
                    }
                }
            }

            buttons.Add(new HostedToolbarAction(
                action.ActionId,
                action.DisplayName,
                action.Icon,
                action.IconOnly,
                action.IsPrimary,
                action.ConfirmationPrompt,
                place,
                action.Order,
                await action.IsEnabledAsync(context),
                cells));
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
    public Task<HostedFile?> RunToolbarAsync(string id, params Guid[] cells) => TurnAsync(async () =>
    {
        var action = Extensions.GetToolbarActions().FirstOrDefault(each => each.ActionId == id)
            ?? throw new InvalidOperationException($"The notebook has no toolbar button '{id}'.");
        var context = new ToolbarContext(Scaffold, cells);

        // A button that runs the cells can meet one that never ends; a stop starts afresh the kernel of the cell it runs.
        await RunUntilStoppedAsync(new Run(Interlocked.Increment(ref _runs), cell: null, takesTheCSharpTurn: true), () => action.ExecuteAsync(context), Scaffold.DefaultKernelId);

        return context.Handed;
    });

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
    // ending with the notebook published as it then stands. A close waits only for what is under way: whatever else was
    // asked before it and still waits its turn is refused when that turn comes, since a close discards what has not
    // begun, as it discards what is not saved.
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
        var running = Running;
        var layout = Layout;

        if (order is null && changed.Length == 0 && running == last.Running && layout == last.Layout)
        {
            return last;
        }

        var next = new NotebookVersion(last.Version + 1, cells, running, layout);

        Volatile.Write(ref _published, new StrongBox<NotebookVersion>(next));

        foreach (var view in Volatile.Read(ref _audience).Views)
        {
            view.Offer(new NotebookChange(next.Version, order, changed, running, layout));
        }

        return next;
    }

    // The engine says a cell began: it is the run under way, since now.
    private void Began(Guid cell)
    {
        Volatile.Write(ref _executing, new StrongBox<HostedRun>(new HostedRun(Volatile.Read(ref _running)?.Number ?? 0, cell, DateTimeOffset.UtcNow, Waits: false)));
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

    // The kinds Verso's editors offer, with each language folded in: code in every language the engine runs but the
    // blocks' own, then Markdown, then every other kind of cell in the order the engine has them. The languages are the
    // engine's kernels, which is what its list of registered languages holds in a notebook opened here.
    private static HostedKind[] KindsOf(ExtensionHost extensions) =>
    [
        .. extensions.GetKernels()
            .Where(kernel => !Same(kernel.LanguageId, StepKernel.Language))
            .Select(kernel => new HostedKind("code", kernel.LanguageId, kernel.DisplayName, Editable: true)),
        .. extensions.GetCellTypes()
            .Where(type => !Same(type.CellTypeId, "code"))
            .OrderBy(type => Same(type.CellTypeId, "markdown") ? 0 : 1)
            .Select(type => new HostedKind(type.CellTypeId, type.Kernel?.LanguageId, type.DisplayName, type.IsEditable)),
    ];

    private static bool Same(string? one, string? other) => string.Equals(one, other, StringComparison.OrdinalIgnoreCase);

    // A kind the notebook lists, as it lists it; one it does not list is refused.
    private HostedKind Listed(HostedKind kind)
    {
        foreach (var each in Kinds.Where(each => Same(each.Type, kind.Type) && Same(each.Language, kind.Language)))
        {
            return each;
        }

        throw new InvalidOperationException($"The notebook has no kind of cell '{kind.Type}' in '{kind.Language}' to add or turn a cell into.");
    }

    // A cell of a listed kind added at a place, empty, through the port the notebook's layout guards; the notebook is told.
    private async Task<HostedCell> AddedAsync(int at, HostedKind kind)
    {
        var listed = Listed(kind);
        var added = Guid.Parse(await Scaffold.NotebookOps.InsertCellAsync(at, listed.Type, listed.Language));

        await TellAsync();

        return EndTurn().Cells.First(each => each.Id == added);
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
    private Task TellAsync() => Blocks.BlocksChangedAsync(Scaffold.Notebook, Scaffold.Variables, Scaffold.NotebookOps);

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

    // Runs until it ends, or until Stop. The run is under way from here, so a Stop finds it even while it waits: a C#
    // run first waits its turn among the process's C# runs, and when that turn is not free every view is told the run
    // waits. A Stop then ends the wait at once, and the turn, when it comes, starts nothing. Only the run under way can
    // be stopped, so the slot holds one, and a Stop after it ended meets a run nobody waits for any more.
    private async Task<bool> RunUntilStoppedAsync(Run run, Func<Task> start, string? kernel)
    {
        Volatile.Write(ref _running, run);

        try
        {
            if (!run.Waits)
            {
                return await GoAsync(run, start, kernel);
            }

            var turned = CSharpRuns.TakeTurnAsync(() => run.Starts() ? GoAsync(run, start, kernel) : Task.FromResult(false));

            // The turn was free when it began at once; otherwise the run waits, and every view is told.
            if (run.Waits)
            {
                Publish();
            }

            await Task.WhenAny(turned, run.Stopped);

            return !run.StoppedBeforeItRan && await turned;
        }
        finally
        {
            Volatile.Write(ref _running, null);
        }
    }

    // A run under way, until it ends or a stop ends it. It is not asked to stop, since one that does not end does not
    // listen either (measured: a C# loop that awaits goes on with its token cancelled); a fresh kernel — the kernel of
    // the cell that runs — ends the turn, and the run is left behind. While it runs, what the engine says it shows is
    // gathered for a moment and published, so a view sees it before the run ends.
    private async Task<bool> GoAsync(Run run, Func<Task> start, string? kernel)
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
            // The notebook is told first, so what the run left behind asks for from now on writes nothing, and the
            // notebook takes its next change at once.
            Blocks.Stopped();
            await Scaffold.RestartKernelAsync(KernelNow() ?? kernel);
        }

        return true;
    }

    // Which kernel runs a cell: its language, else the notebook's default kernel — the engine's own rule, so a run takes
    // the C# turn, and a stop starts a kernel afresh, by the kernel that really runs the cell.
    private string? KernelOf(CellModel cell) => cell.Language ?? Scaffold.DefaultKernelId;

    // The kernel of the cell the engine runs now, if one has begun and still stands.
    private string? KernelNow() =>
        Volatile.Read(ref _executing)?.Value.Cell is { } cell && Scaffold.Cells.FirstOrDefault(each => each.Id == cell) is { } running
            ? KernelOf(running)
            : null;

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

    // A run, from the moment it is asked: its number, the cell it runs, and whether it still waits for the C# turn, runs,
    // or was stopped before it ran. Each change of that is one exchange, so a stop and the C# turn that arrive together
    // agree on which came first.
    private sealed class Run(long number, Guid? cell, bool takesTheCSharpTurn)
    {
        private const int Waiting = 0;
        private const int Going = 1;
        private const int StoppedFirst = 2;

        private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _state = takesTheCSharpTurn ? Waiting : Going;

        public long Number => number;

        public Guid? Cell => cell;

        public DateTimeOffset Asked { get; } = DateTimeOffset.UtcNow;

        public Task Stopped => _stopped.Task;

        // Whether it still waits for the C# turn.
        public bool Waits => Volatile.Read(ref _state) == Waiting;

        // Whether a stop came before it ran: then it never runs.
        public bool StoppedBeforeItRan => Volatile.Read(ref _state) == StoppedFirst;

        // Its C# turn came: it runs, unless a stop came first.
        public bool Starts() => Interlocked.CompareExchange(ref _state, Going, Waiting) == Waiting;

        // Stops it: a run that waits never runs; one that runs is ended by whoever runs it. Whether this stop was the first.
        public bool Stop()
        {
            Interlocked.CompareExchange(ref _state, StoppedFirst, Waiting);

            return _stopped.TrySetResult();
        }
    }
}
