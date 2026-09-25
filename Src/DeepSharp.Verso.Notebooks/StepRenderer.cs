// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

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
        var assembled = NotebookPipeline.Of(gesture.Notebook.Cells);

        switch (context.InteractionType)
        {
            case Show:
                return await StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Show, page: 0);

            case Page:
                return await StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Page, PageOf(context.Payload));

            case Drop or Category when Change(assembled, gesture.Cell, context.InteractionType, context.Payload) is { } steps:
                context.StateChanged = await StepCommit.CommitAsync(gesture, assembled, steps);

                return null;

            default:
                return null;
        }
    }

    // The steps a grid gesture makes of the declaration, or nothing when the notebook already is what it asks for: the
    // column does not reach the end, the column is a category, or the block shows no grid to have made it on.
    private static IReadOnlyList<IPipelineStep>? Change(NotebookPipeline assembled, Guid cell, string interaction, string column)
    {
        var position = assembled.PositionOf(cell);
        var declaration = assembled.Readable;

        if (position < 0 || position >= declaration.Steps.Count || declaration.Steps.OfType<DeclareStep>().FirstOrDefault() is not { } declare)
        {
            return null;
        }

        List<IPipelineStep> steps = [.. declaration.Steps];
        var declareAt = steps.IndexOf(declare);
        var declared = declare.Columns.FirstOrDefault(each => each.Name == column);

        if (interaction == Category)
        {
            if (declared is not { Kind: not ColumnKind.Category })
            {
                return null;
            }

            steps[declareAt] = new DeclareStep(
                [.. declare.Columns.Select(each => each == declared ? each with { Kind = ColumnKind.Category } : each)], declare.Remainder);

            return steps;
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
            steps[declareAt] = new DeclareStep([.. declare.Columns.Where(each => each != declared)], declare.Remainder);

            return steps;
        }

        // Otherwise it is dropped after the last step that reads it, or after the step that made it; a drop that
        // already stands there takes one more name.
        var after = Math.Max(readers.DefaultIfEmpty(-1).Max(), MadeAt(declaration, column)) + 1;

        if (after < steps.Count && steps[after] is DropColumnsStep drop)
        {
            steps[after] = new DropColumnsStep([.. drop.Columns, column]);
        }
        else
        {
            steps.Insert(after, new DropColumnsStep([column]));
        }

        return steps;
    }

    // Where a column comes to be: the first step after which it is there.
    private static int MadeAt(PipelineDeclaration declaration, string column) =>
        Enumerable.Range(0, declaration.Steps.Count).First(at => declaration.ColumnsBefore(at + 1).Allows(column));

    private static int PageOf(string? payload) =>
        int.TryParse(payload, NumberStyles.None, CultureInfo.InvariantCulture, out var page) ? page : 0;
}
