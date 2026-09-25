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
/// wrote, unchanged. The gestures on a block reach this part by its id — the id the block's buttons and boxes carry,
/// written from the one constant here — and this part sees the whole notebook, which the block's own kernel does not. So a
/// gesture assembles the pipeline from every block as the blocks stand now, leaves the block's kernel a request, and
/// runs the block: what the kernel writes while it runs reaches the screen, where the answer to a gesture does not.
/// </remarks>
[VersoExtension]
public sealed class StepRenderer : NotebookExtension, ICellRenderer, ICellInteractionHandler
{
    /// <summary>The renderer's id, which a block's buttons and boxes name so that a click reaches this part.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.renderer";

    /// <summary>The gesture that shows the data at a block.</summary>
    internal const string Show = "deepsharp.show";

    /// <summary>The gesture that shows another page of the data at a block; it carries the page, counting from nought.</summary>
    internal const string Page = "deepsharp.page";

    /// <summary>
    /// The gesture a grid's box for whether a column is in sends, with the column: ticked, the column is taken in;
    /// unticked, it is left out.
    /// </summary>
    internal const string Include = "deepsharp.include";

    /// <summary>
    /// The gesture a grid's box for whether a column the schema takes is a category sends, with the column: ticked, it
    /// becomes one; unticked, it goes back to the kind it was.
    /// </summary>
    internal const string Category = "deepsharp.category";

    /// <summary>
    /// The gesture a take-over's list sends, with the saved columns it listed and the key of the blocks it listed them
    /// for: ticked, what it listed is made.
    /// </summary>
    internal const string Apply = "deepsharp.apply";

    /// <summary>The key a box carries its column under.</summary>
    internal const string ColumnKey = "column";

    /// <summary>The key a take-over's list carries the saved columns under, as the file held them when it was listed.</summary>
    internal const string PresetKey = "preset";

    /// <summary>The key a take-over's list carries the key of the blocks it was listed for under.</summary>
    internal const string DrawnKey = "drawn";

    // Said at a block when the list of a take-over is applied to blocks that are no longer the ones it listed.
    private const string ListedForOtherBlocks = "the blocks changed since the list was shown — take over again";

    // Said at a block when a column the schema does not name is ticked before anything read the source in this session.
    private const string SourceNotRead =
        "The source is not read in this session, so where a column it has stands among its columns is not known yet: "
        + "its data is shown here now, and ticking the column again takes it in where it stands.";

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

            default:
                if (ControlAction.Read(context.InteractionType) is not { } action || StateOf(context.Payload) is not { } ticked)
                {
                    return null;
                }

                if (action.Gesture == Apply)
                {
                    return await AppliedAsync(gesture, assembled, context, action, ticked);
                }

                return action.Text(ColumnKey) is { } column ? await TickedAsync(gesture, assembled, context, action.Gesture, column, ticked) : null;
        }
    }

    // A take-over's list, sent with the saved columns it listed and the key of the blocks it listed them for. Ticked,
    // what it listed is made, in one commit; asked again — the click's echo — the blocks hold it already, and nothing
    // happens. The blocks must still make the whole pipeline the list was worked out for, or the change is not made and
    // the block says why.
    private static async Task<string?> AppliedAsync(
        Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action, bool ticked)
    {
        if (!ticked || action.Text(PresetKey) is not { } saved || action.Text(DrawnKey) is not { } drawn)
        {
            return null;
        }

        PresetTakeOver takenOver;

        try
        {
            takenOver = PipelinePreset.FromJson(saved, NotebookVerbs.Catalog()).TakeOver(assembled.Readable, header: null);
        }
        catch (PipelineFileException unreadable)
        {
            await StepCommit.NotMadeAsync(gesture, assembled, [.. unreadable.Faults.Select(fault => fault.ToString())]);

            return null;
        }

        if (takenOver.Faults.Count == 0 && takenOver.Steps.SequenceEqual(assembled.Readable.Steps))
        {
            return null;
        }

        IReadOnlyList<string> notMade = !assembled.Whole
            ? [$"the blocks do not make a pipeline yet: {string.Join("; ", assembled.Stopping)}"]
            : drawn != NotebookSession.KeyOf(assembled.Readable)
                ? [ListedForOtherBlocks]
                : [.. takenOver.Faults.Select(fault => fault.ToString())];

        if (notMade.Count > 0)
        {
            await StepCommit.NotMadeAsync(gesture, assembled, notMade);

            return null;
        }

        context.StateChanged = await StepCommit.CommitAsync(gesture, assembled, takenOver.Steps);

        return null;
    }

    // A grid's box, sent with the state it is in: the steps the column rules make of that state, written when they
    // differ from the steps there are. A box asks for a state, so a change the rules refuse is said at the block, and
    // one asked for again changes nothing.
    private static async Task<string?> TickedAsync(
        Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, string name, string column, bool ticked)
    {
        var declaration = assembled.Readable;
        var position = assembled.PositionOf(gesture.Cell);

        // Only a block of the declaration shows a grid to have sent it.
        if (position < 0 || position >= declaration.Steps.Count)
        {
            return null;
        }

        IReadOnlyList<IPipelineStep>? steps;

        try
        {
            steps = name switch
            {
                Include when ticked => await TakenInAsync(gesture, assembled, column),
                Include => declaration.Excluding(column),
                Category => Kinded(declaration, column, ticked),
                _ => null,
            };
        }
        catch (DeclarationException refused)
        {
            // What the schema refuses of itself is said at the block, as a rule the change would break is.
            await StepCommit.NotMadeAsync(gesture, assembled, [.. refused.Faults.Select(fault => fault.ToString())]);

            return null;
        }

        if (steps is not null)
        {
            context.StateChanged = await StepCommit.CommitAsync(gesture, assembled, steps);
        }

        return null;
    }

    // A column taken in. One the schema does not name comes in as text where the source has it, which only the source's
    // own header says: the rows this session read from it. Before anything read them the block shows the source, which
    // reads them, and says so; and a column the source does not have was not ticked on its grid.
    private static async Task<IReadOnlyList<IPipelineStep>?> TakenInAsync(Gesture gesture, NotebookPipeline assembled, string column)
    {
        var declaration = assembled.Readable;
        var header = declaration.Steps[0] is ReadCsvStep read ? gesture.Session.Sources.ColumnNamesFor(read) : null;

        if (declaration.ChoicesFor([column]).Rows[0].Standing != ColumnStanding.NotDeclared)
        {
            return declaration.Including(column, ColumnKind.Text, header ?? []);
        }

        if (header is null)
        {
            await StepCommit.NotMadeAsync(gesture, assembled, [SourceNotRead]);

            return null;
        }

        return header.Contains(column, StringComparer.Ordinal) ? declaration.Including(column, ColumnKind.Text, header) : null;
    }

    // A column the schema takes made a category, or given back the kind it was; nothing for a column it does not take,
    // nor for a category that does not say what it was.
    private static IReadOnlyList<IPipelineStep>? Kinded(PipelineDeclaration declaration, string column, bool ticked) =>
        declaration.Steps.OfType<DeclareStep>().FirstOrDefault()?.Taking.FirstOrDefault(each => each.Name == column) switch
        {
            null => null,
            _ when ticked => declaration.WithKind(column, ColumnKind.Category),
            { Kind: ColumnKind.Category, Was: { } was } => declaration.WithKind(column, was),
            _ => null,
        };

    // A box sends whether it is ticked; anything else says nothing about the state it asks for.
    private static bool? StateOf(string? payload) => payload switch
    {
        "true" => true,
        "false" => false,
        _ => null,
    };

    private static int PageOf(string? payload) =>
        int.TryParse(payload, NumberStyles.None, CultureInfo.InvariantCulture, out var page) ? page : 0;
}
