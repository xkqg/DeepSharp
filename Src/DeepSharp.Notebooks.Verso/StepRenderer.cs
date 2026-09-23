// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// How a block is drawn, and the part the gestures made on a block reach.
/// </summary>
/// <remarks>
/// A block is edited as text and runs through its kernel, so drawing it is showing what it holds and what its run
/// wrote, unchanged. The gestures on a block reach this part by its id — the id the block's buttons carry, written
/// from the one constant here — and this part sees the whole notebook, which the block's own kernel does not. So a
/// gesture assembles the pipeline from every block as the blocks stand now, leaves the block's kernel a request, and
/// runs the block: what the kernel writes while it runs reaches the screen, where the answer to a gesture does not.
/// </remarks>
[VersoExtension]
public sealed class StepRenderer : NotebookExtension, ICellRenderer, ICellInteractionHandler
{
    /// <summary>The renderer's id, which a block's buttons name so that a click reaches this part.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.renderer";

    /// <summary>The gesture that shows the data at a block.</summary>
    internal const string Show = "deepsharp.show";

    /// <summary>The gesture that shows another page of the data at a block; it carries the page, counting from nought.</summary>
    internal const string Page = "deepsharp.page";

    /// <summary>The gesture that excludes a column from the pipeline; it carries the column's name.</summary>
    internal const string Drop = "deepsharp.drop";

    /// <summary>The gesture that marks a declared column a category; it carries the column's name.</summary>
    internal const string Category = "deepsharp.category";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline step view";

    /// <inheritdoc />
    public override string Description => "Shows what a block of a pipeline holds, and takes the gestures made on it.";

    /// <inheritdoc />
    public string CellTypeId => StepCellType.StepType;

    /// <inheritdoc />
    public string DisplayName => "Pipeline step";

    /// <inheritdoc />
    /// <remarks>The step stays in view after a run: it is what a person edits.</remarks>
    public bool CollapsesInputOnExecute => false;

    /// <inheritdoc />
    public CellVisibilityHint DefaultVisibility => CellVisibilityHint.Content;

    /// <inheritdoc />
    public Task<RenderResult> RenderInputAsync(string source, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(source);

        return Task.FromResult(new RenderResult("text/plain", source));
    }

    /// <inheritdoc />
    public Task<RenderResult> RenderOutputAsync(CellOutput output, ICellRenderContext context)
    {
        ArgumentNullException.ThrowIfNull(output);

        return Task.FromResult(new RenderResult(output.MimeType, output.Content));
    }

    /// <inheritdoc />
    /// <remarks>A step is written as JSON, and an editor colours it as such.</remarks>
    public string GetEditorLanguage() => "json";

    /// <inheritdoc />
    /// <remarks>
    /// Every gesture carries the state it asks for, whole, so the same gesture twice does the same thing once. A
    /// gesture answers nothing: what it shows, its block's run writes, and a host that is answered replaces the block's
    /// outputs with the answer. Only a gesture this part cannot act on at all answers, saying why.
    /// </remarks>
    public async Task<string?> OnCellInteractionAsync(CellInteractionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        if (LoadedSession is not { } session)
        {
            return NotLoadedBeside;
        }

        if (context.NotebookModel is not { } notebook || context.Notebook is not { } operations || context.Variables is not { } variables)
        {
            return "This gesture came without the notebook it was made in, so there is nothing for it to act on.";
        }

        var gesture = new Gesture(
            session, notebook, operations, variables, context.CellId,
            LayoutLets(operations, LayoutCapabilities.CellInsert | LayoutCapabilities.CellDelete));

        return await session.OneAtATimeAsync(() => HandleAsync(gesture, context));
    }

    private static async Task<string?> HandleAsync(Gesture gesture, CellInteractionContext context)
    {
        switch (context.InteractionType)
        {
            case Show:
                return await ShowAsync(gesture, page: 0, NotebookPipeline.Of(gesture.Notebook.Cells));

            case Page:
                return await ShowAsync(gesture, PageOf(context.Payload), NotebookPipeline.Of(gesture.Notebook.Cells));

            case Drop or Category when Change(gesture, context.InteractionType, context.Payload) is { } change:
                context.StateChanged = await CommitAsync(gesture, change);

                return null;

            default:
                return null;
        }
    }

    // What a grid gesture changes in the declaration, or nothing when the notebook already is what it asks for:
    // the column does not reach the end, the column is a category, or the block shows no grid to have made it on.
    private static Edit? Change(Gesture gesture, string interaction, string column)
    {
        var assembled = NotebookPipeline.Of(gesture.Notebook.Cells);
        var position = assembled.PositionOf(gesture.Cell);
        var declaration = assembled.Readable;

        if (position < 0 || position >= declaration.Steps.Count || declaration.Steps.OfType<DeclareStep>().FirstOrDefault() is not { } declare)
        {
            return null;
        }

        var declareAt = declaration.Steps.ToList().IndexOf(declare);
        var declared = declare.Columns.FirstOrDefault(each => each.Name == column);

        if (interaction == Category)
        {
            return declared is { Kind: not ColumnKind.Category }
                ? new Edit(assembled, declareAt, new DeclareStep(
                    [.. declare.Columns.Select(each => each == declared ? each with { Kind = ColumnKind.Category } : each)], declare.Remainder), Inserted: false)
                : null;
        }

        // Left out by the schema, or taken away by a step below: the column is excluded already.
        if (!declaration.ColumnsBefore(declaration.Steps.Count).Allows(column))
        {
            return null;
        }

        var readers = Enumerable.Range(0, declaration.Steps.Count)
            .Where(at => declaration.Steps[at].ColumnsRead.Any(read => read.Column == column))
            .ToArray();

        // A column nobody reads, that the schema names and nothing else keeps, is taken out of the schema: a
        // column nobody names is not there.
        if (readers.Length == 0 && declared is not null && declare.Remainder == Remainder.Drop)
        {
            return new Edit(assembled, declareAt, new DeclareStep([.. declare.Columns.Where(each => each != declared)], declare.Remainder), Inserted: false);
        }

        // Otherwise it is dropped after the last step that reads it, or after the step that made it; a drop that
        // already stands there takes one more name.
        var after = Math.Max(readers.DefaultIfEmpty(-1).Max(), MadeAt(declaration, column));

        return after + 1 < declaration.Steps.Count && declaration.Steps[after + 1] is DropColumnsStep drop
            ? new Edit(assembled, after + 1, new DropColumnsStep([.. drop.Columns, column]), Inserted: false)
            : new Edit(assembled, after + 1, new DropColumnsStep([column]), Inserted: true);
    }

    // Where a column comes to be: the first step after which it is there.
    private static int MadeAt(PipelineDeclaration declaration, string column) =>
        Enumerable.Range(0, declaration.Steps.Count).First(at => declaration.ColumnsBefore(at + 1).Allows(column));

    // Makes the change, once, unless it would break a step or the layout cannot hold it; either way the block shows
    // what came of it. A change makes stale every view worked out from the steps it changed, and every grid whose
    // offers it changed, and those are cleared rather than left showing what is no longer so.
    private static async Task<bool> CommitAsync(Gesture gesture, Edit edit)
    {
        List<IPipelineStep> steps = [.. edit.Assembled.Readable.Steps];

        if (edit.Inserted)
        {
            steps.Insert(edit.At, edit.Step);
        }
        else
        {
            steps[edit.At] = edit.Step;
        }

        IReadOnlyList<string> notMade = !gesture.MayAddAndRemove
            ? ["the layout this notebook is shown in cannot add or remove a block; show it in the notebook layout to change the pipeline from the grid."]
            : [.. PipelineDeclaration.FaultsIn(steps).Select(fault => fault.ToString())];

        if (notMade.Count > 0)
        {
            gesture.Session.Request(
                gesture.Cell, edit.Assembled.RequestFor(gesture.Cell, page: 0, run: false) with { NotMade = notMade });
            await gesture.Operations.ExecuteCellAsync(gesture.Cell);

            return false;
        }

        var written = await WriteAsync(gesture, edit);
        var now = NotebookPipeline.Of(gesture.Notebook.Cells);
        var shown = gesture with { Cell = written.Gesture };

        foreach (var cell in gesture.Session.ForgetStale(now, except: shown.Cell))
        {
            // A block deleted since it was shown has nothing left to clear.
            if (gesture.Notebook.Cells.Any(each => each.Id == cell))
            {
                await gesture.Operations.ClearOutputAsync(cell);
            }
        }

        // The block written is one a front end has never seen, and its run is what makes a front end read the
        // notebook again; when it is the block the gesture was made on, showing the result runs it.
        if (written.Block != shown.Cell)
        {
            await gesture.Operations.ExecuteCellAsync(written.Block);
        }

        await ShowAsync(shown, page: 0, now);

        return true;
    }

    // Writes the change as a block no front end has seen: a new block after the step it follows, or a new block in
    // the place of the one rewritten. Verso tells a front end nothing of a block whose text a part changed, and the
    // next keystroke there would put the old text back.
    private static async Task<Written> WriteAsync(Gesture gesture, Edit edit)
    {
        var blocks = edit.Assembled.Blocks;
        var text = edit.Step.AsBlockText();

        if (edit.Inserted)
        {
            var after = gesture.Notebook.Cells.FindIndex(cell => cell.Id == blocks[edit.At - 1].Cell) + 1;
            var inserted = await InsertAsync(gesture, after, text);

            return new Written(inserted.Id, gesture.Cell);
        }

        var old = gesture.Notebook.Cells.Single(cell => cell.Id == blocks[edit.At].Cell);
        var replacing = await InsertAsync(gesture, gesture.Notebook.Cells.IndexOf(old), text);

        foreach (var (key, value) in old.Metadata)
        {
            replacing.Metadata[key] = value;
        }

        await gesture.Operations.RemoveCellAsync(old.Id);
        gesture.Session.Hidden(old.Id);
        gesture.Session.Accepted(old.Id);

        return new Written(replacing.Id, old.Id == gesture.Cell ? replacing.Id : gesture.Cell);
    }

    private static async Task<CellModel> InsertAsync(Gesture gesture, int at, string text)
    {
        var id = Guid.Parse(await gesture.Operations.InsertCellAsync(at, StepCellType.StepType, StepKernel.Language));
        var cell = gesture.Notebook.Cells.Single(each => each.Id == id);

        cell.Source = text;

        return cell;
    }

    private static async Task<string?> ShowAsync(Gesture gesture, int page, NotebookPipeline assembled)
    {
        if (assembled.PositionOf(gesture.Cell) < 0)
        {
            return "This cell is not a block of the pipeline.";
        }

        gesture.Session.Publish(assembled);

        gesture.Session.HandOver(gesture.Variables, assembled);
        gesture.Session.Request(gesture.Cell, assembled.RequestFor(gesture.Cell, page, run: false));

        await gesture.Operations.ExecuteCellAsync(gesture.Cell);

        return null;
    }

    private static int PageOf(string? payload) =>
        int.TryParse(payload, NumberStyles.None, CultureInfo.InvariantCulture, out var page) ? page : 0;

    /// <summary>A gesture, with the notebook it was made in.</summary>
    /// <param name="Session">The notebook's session, which the block's kernel reads.</param>
    /// <param name="Notebook">The notebook, every cell of it.</param>
    /// <param name="Operations">What can be done to the notebook: running a cell, clearing one.</param>
    /// <param name="Variables">The values the notebook's cells share.</param>
    /// <param name="Cell">The block the gesture was made on.</param>
    /// <param name="MayAddAndRemove">Whether the layout the notebook is shown in lets a part add and remove blocks.</param>
    private readonly record struct Gesture(
        NotebookSession Session, NotebookModel Notebook, INotebookOperations Operations, IVariableStore Variables, Guid Cell,
        bool MayAddAndRemove);

    /// <summary>One change a grid gesture makes to the declaration.</summary>
    /// <param name="Assembled">The pipeline the blocks made when the gesture was made.</param>
    /// <param name="At">The block the change is written into, or the place a new block goes.</param>
    /// <param name="Step">The step that block holds after the change.</param>
    /// <param name="Inserted">Whether the step is a new block rather than a block rewritten.</param>
    private sealed record Edit(NotebookPipeline Assembled, int At, IPipelineStep Step, bool Inserted);

    /// <summary>What a commit wrote: the new block, and the block the gesture now stands on.</summary>
    /// <param name="Block">The block written.</param>
    /// <param name="Gesture">The block the gesture was made on, or the new block in its place when it was the one rewritten.</param>
    private readonly record struct Written(Guid Block, Guid Gesture);
}
