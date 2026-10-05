// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// A block of a pipeline: one step, written as the step's own JSON.
/// </summary>
/// <remarks>
/// The notebook is the declaration, block by block: the blocks in the order they stand are the steps in the order
/// they run, each block's text is exactly what the step writes itself as, and saving the notebook saves the steps.
/// What a block shows is never saved — it is worked out from the data each time it is asked for, and the data is
/// somebody's own. Edited as text in the editor, or field by field in Verso's properties panel; both write the
/// same text.
/// </remarks>
[VersoExtension]
public sealed class StepCellType : NotebookExtension, ICellType
{
    /// <summary>The cell type's id as an extension.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.cell";

    /// <summary>The type a block's cell is saved under.</summary>
    public const string StepType = "deepsharp.step";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline step";

    /// <inheritdoc />
    public override string Description => "A block of a DeepSharp pipeline: one step, written as its own JSON.";

    /// <inheritdoc />
    public string CellTypeId => StepType;

    /// <inheritdoc />
    public string DisplayName => "Pipeline step";

    /// <inheritdoc />
    /// <remarks>Three boxes joined in order: the steps of a pipeline.</remarks>
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M2 3h8v5H2zM14 9.5h8v5h-8zM2 16h8v5H2zM10 5h6v4.5h-1.5V6.5H10zM16 14.5V19h-6v-1.5h4.5v-3z\"/></svg>";

    /// <inheritdoc />
    /// <remarks>
    /// The kernel Verso runs this type's blocks through, which reaches the session this type keeps directly. The
    /// kernel Verso loads as a part of its own serves the editor, and runs the blocks only while this type is
    /// switched off.
    /// </remarks>
    public ILanguageKernel Kernel => _kernel ??= new StepKernel(this);

    private StepKernel? _kernel;

    /// <summary>What the notebook's gestures leave for its blocks, and what they show: the notebook's session.</summary>
    /// <remarks>
    /// Kept here because the block type is the one object every part reaches: the kernel it carries directly, every
    /// other part through the host that loaded it, this type switched off included. It lives as long as this type
    /// and is shared with nothing else; nothing reaches for it of its own accord.
    /// </remarks>
    internal NotebookSession Session { get; } = new();

    /// <inheritdoc />
    public ICellRenderer Renderer { get; } = new StepRenderer();

    /// <inheritdoc />
    public bool IsEditable => true;

    /// <inheritdoc />
    public bool PersistsOutputs => false;

    /// <inheritdoc />
    /// <remarks>
    /// The first step of a course, waiting for what only its person knows: the step that reads a file, its path still to say —
    /// the block a new notebook starts with when <c>DeepSharp.Verso.Api</c> makes one. Neither Verso's engine nor its editors
    /// ask a cell type for its default text, so a block added to a notebook that is there already starts empty in them;
    /// <c>DeepSharp.Verso.Api</c> asks <see cref="StartingText"/> instead, for the step that belongs where the block goes.
    /// </remarks>
    public string GetDefaultContent() => StartingText([], 0);

    /// <summary>The text a block added to a notebook starts with: the step of the course that belongs where the block goes.</summary>
    /// <param name="cells">The notebook's cells, as they stand before the block is added.</param>
    /// <param name="at">The place the block goes: how many cells stand above it.</param>
    /// <returns>
    /// The step of the course the blocks follow that belongs at that place, as the skeleton a block holds, waiting for what
    /// only its person knows; nothing at all when the blocks have gone their own way, already say the whole course, or no step
    /// belongs there.
    /// </returns>
    /// <remarks>
    /// Neither Verso's engine nor its editors ask a cell type what a block starts with, so a host that adds blocks itself —
    /// as <c>DeepSharp.Verso.Api</c> does — asks here. Which of the two courses the blocks follow is the one they say most of,
    /// and the table's when they say as much of both. A block put between two others is the step that belongs between them, so
    /// it never lands out of the order the course teaches.
    /// </remarks>
    public string StartingText(IReadOnlyList<CellModel> cells, int at)
    {
        ArgumentNullException.ThrowIfNull(cells);
        ArgumentOutOfRangeException.ThrowIfNegative(at);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(at, cells.Count);

        var catalog = NotebookVerbs.Catalog();

        return CourseProgress.FollowedBy(cells, catalog) is { } course && CourseProgress.NextAt(course, cells, at, catalog) is { } next
            ? next.AsBlockText(catalog)
            : string.Empty;
    }

    /// <summary>
    /// Tells the notebook its cells changed where no gesture saw it: a block inserted, taken away or moved, a cell turned
    /// into another kind, or text typed into one. What was worked out from the blocks as they were is taken back — a grid
    /// they no longer make is cleared, and the pipeline handed to C# cells is taken back unless they still make it.
    /// </summary>
    /// <param name="notebook">The notebook, its cells as they stand now.</param>
    /// <param name="variables">The notebook's variables, as a host hands them to a part.</param>
    /// <param name="operations">What can be done to the notebook: a grid is cleared through it.</param>
    /// <returns>A task that ends when the notebook has caught up.</returns>
    /// <remarks>
    /// A host that changes cells itself calls it after each such change, as <c>DeepSharp.Verso.Api</c> does. Verso's own
    /// editors call nothing, and the notebook catches up at their next gesture instead. It waits its turn among the
    /// notebook's changes, so it never lands in the middle of one.
    /// </remarks>
    public Task BlocksChangedAsync(NotebookModel notebook, IVariableStore variables, INotebookOperations operations)
    {
        ArgumentNullException.ThrowIfNull(notebook);
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(operations);

        return Session.OneAtATimeAsync(CancellationToken.None, turn => Session.LetThroughAsync(turn, async () =>
        {
            var now = NotebookPipeline.Of(notebook.Cells);

            foreach (var stale in Session.StaleIn(now, except: null))
            {
                await operations.ClearOutputAsync(stale);
            }

            Session.CaughtUp(now, variables, except: null);
        }));
    }

    /// <summary>
    /// Tells the notebook the run under way was stopped: what that run asked for writes nothing more — no grid, and nothing
    /// handed to C# cells — and the notebook takes its next change at once, while the run goes on by itself.
    /// </summary>
    /// <returns>
    /// A task that ends once every write the notebook let through before the stop has landed; at once when none is on its
    /// way. The notebook is told before the method returns: the task is only what the host waits for next.
    /// </returns>
    /// <remarks>
    /// A host that stops a run marks the run first and calls this next, as <c>DeepSharp.Verso.Api</c> does. It starts a
    /// kernel afresh only after the task ended, since a fresh kernel takes away what the kernels hold and a write still on
    /// its way would land after it; and it lets the next change write only after that.
    /// </remarks>
    public Task StoppedAsync() => Session.Stopped();
}
