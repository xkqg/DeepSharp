// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
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
public sealed partial class NotebookHost : IRunListener
{
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

    // The notebook's file: what it last held, whether the notebook differs from it, and writing it.
    private readonly NotebookFile _file;

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

    private NotebookHost(string filePath, ExtensionHost extensions, Scaffold scaffold, Opening opening)
    {
        Extensions = extensions;
        Scaffold = scaffold;
        _blocks = ((IExtensionHostContext)extensions).GetLoadedExtensions().OfType<StepCellType>().First();
        Kinds = extensions.KindsOf();
        _file = new NotebookFile(filePath, opening.Saved, extensions, scaffold);
        Layouts = extensions.LayoutsOf();
        Themes = extensions.ThemesOf();
        _published = new(new NotebookVersion(
            0, [.. scaffold.Cells.Select(cell => cell.Hosted([]))], null, [], scaffold.Layout, opening.Buttons, Restarts.None.Hosted, opening.Unsaved,
            scaffold.ThemeId, opening.Arrangement, scaffold.SaysOfItself));
        scaffold.OnCellExecuting += Began;
        scaffold.OnCellExecuted += Ended;
        scaffold.OnCellOutputUpdated += Showed;
        scaffold.OnKernelRestarting += Restarting;
        scaffold.OnKernelRestarted += Restarted;
        scaffold.OnKernelRestartFailed += RestartFailed;
    }

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string FilePath => _file.Path;

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
}
