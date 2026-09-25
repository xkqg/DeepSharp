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

    /// <summary>The gesture that lists every column of the source at the schema's block.</summary>
    internal const string Columns = "deepsharp.columns";

    /// <summary>
    /// The gesture a row's box on the list sends, with its column, the kind the row shows, and what the list was drawn
    /// from: ticked, the column is taken in; unticked, it is left out.
    /// </summary>
    internal const string ListInclude = "deepsharp.list.include";

    /// <summary>
    /// The gesture a row's kind select on the list sends, with its column and what the list was drawn from; the router
    /// sends the value it is at: a kind, or none.
    /// </summary>
    internal const string ListKind = "deepsharp.list.kind";

    /// <summary>
    /// The gesture the list's type select sends, with what the list was drawn from; the router sends the kind of output
    /// picked. A pick: it commits nothing, and the list is drawn again making that kind.
    /// </summary>
    internal const string ListType = "deepsharp.list.type";

    /// <summary>
    /// The gesture a row's output box sends, with its column, the kind of output it makes, the kind a tick takes the
    /// column in with, and what the list was drawn from: ticked, the column is put into the output; unticked, taken out.
    /// </summary>
    internal const string ListOutput = "deepsharp.list.output";

    /// <summary>The gesture the list's box that takes the output away sends, with what the list was drawn from.</summary>
    internal const string ListRemoveOutput = "deepsharp.list.removeOutput";

    /// <summary>
    /// The gesture a select setting one of the output's own values sends, with the value's key, the kind of output the
    /// list makes, and what the list was drawn from; the router sends the value it is at.
    /// </summary>
    internal const string ListParameter = "deepsharp.list.parameter";

    /// <summary>The key a select setting one of the output's own values carries that value's key under.</summary>
    internal const string ParameterKey = "key";

    /// <summary>The key a list's control carries the kind of output the list makes under.</summary>
    internal const string TypeKey = "type";

    /// <summary>The key a box carries its column under.</summary>
    internal const string ColumnKey = "column";

    /// <summary>The key a take-over's list carries the saved columns under, as the file held them when it was listed.</summary>
    internal const string PresetKey = "preset";

    /// <summary>The key a take-over's list, or the list of the source's columns, carries the key of the blocks it was drawn for under.</summary>
    internal const string DrawnKey = "drawn";

    /// <summary>The key a row of the list carries the kind it shows under: the one a tick takes the column in with.</summary>
    internal const string KindKey = "kind";

    /// <summary>The key a row of the list carries the fingerprint of the source's bytes it was drawn from under.</summary>
    internal const string SourceKey = "source";

    // Said at the list when a row is sent in a session that has not read the source the list was drawn from.
    private const string ListSourceNotRead = "Choose the columns again — the source is not read in this session.";

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

            case Columns:
                return await StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Show, page: 0, ListPicks.None);

            default:
                if (ControlAction.Read(context.InteractionType) is not { } action)
                {
                    return null;
                }

                // A select sends the value it is at; a box, whether it is ticked.
                if (action.Gesture == ListKind)
                {
                    return await KindPickedAsync(gesture, assembled, context, action);
                }

                if (action.Gesture == ListType)
                {
                    return await TypePickedAsync(gesture, assembled, context);
                }

                if (action.Gesture == ListParameter)
                {
                    return await ParameterPickedAsync(gesture, assembled, context, action);
                }

                if (StateOf(context.Payload) is not { } ticked)
                {
                    return null;
                }

                return action.Gesture switch
                {
                    Apply => await AppliedAsync(gesture, assembled, context, action, ticked),
                    ListInclude => await ListedAsync(gesture, assembled, context, action, ticked),
                    ListOutput => await AnsweredAsync(gesture, assembled, context, action, ticked),
                    ListRemoveOutput => await RemovedAsync(gesture, assembled, context, action, ticked),
                    _ => action.Text(ColumnKey) is { } column ? await TickedAsync(gesture, assembled, context, action.Gesture, column, ticked) : null,
                };
        }
    }

    // The list's type select: a pick. It commits nothing; the list is drawn again with the kind picked in its boxes.
    private static async Task<string?> TypePickedAsync(Gesture gesture, NotebookPipeline assembled, CellInteractionContext context) =>
        assembled.SchemaBlock is { } schema && !string.IsNullOrEmpty(context.Payload)
            ? await StepCommit.ShowAsync(gesture with { Cell = schema }, assembled, ViewTrigger.Show, page: 0, new ListPicks(context.Payload))
            : null;

    // A row's output box. The output is changed only while the blocks make one pipeline; a box from a list drawn before
    // the blocks or the source's bytes changed draws the list again and changes nothing; otherwise the column is put into
    // the output of the kind the box makes, or taken out, as the output box's one rule says.
    private static async Task<string?> AnsweredAsync(
        Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action, bool ticked)
    {
        if (await RowOfAsync(gesture, assembled, action) is not { } row)
        {
            return null;
        }

        var picks = PicksOf(action);

        if (!assembled.Whole)
        {
            await StepCommit.NotMadeAsync(row.List, assembled, [NotWhole(assembled)], picks);

            return null;
        }

        if (!DrawnOver(action, assembled, row.Source))
        {
            await StepCommit.ShowAsync(row.List, assembled, ViewTrigger.Show, page: 0, picks);

            return null;
        }

        var change = OutputBox.Change(
            NotebookVerbs.Catalog(), assembled.Readable, action.Text(TypeKey) ?? string.Empty, row.Column,
            KindNamed(action.Text(KindKey)) ?? ColumnKind.Text, row.Source.Rows.ColumnNames, ticked);

        if (change.Steps is not { } steps)
        {
            await StepCommit.NotMadeAsync(row.List, assembled, change.NotMade, picks);

            return null;
        }

        context.StateChanged = await StepCommit.CommitAsync(row.List, assembled, steps, picks);

        return null;
    }

    // The list's box that takes the output away. Ticked with no output there — the click's echo — it changes nothing;
    // otherwise it needs blocks that make one pipeline and a list that still says what holds.
    private static async Task<string?> RemovedAsync(
        Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action, bool ticked)
    {
        if (!ticked || assembled.SchemaBlock is not { } schema || assembled.Readable.Output is null)
        {
            return null;
        }

        var list = gesture with { Cell = schema };
        var picks = PicksOf(action);

        if (!assembled.Whole)
        {
            await StepCommit.NotMadeAsync(list, assembled, [NotWhole(assembled)], picks);

            return null;
        }

        if (assembled.Readable.Steps[0] is not ReadCsvStep read || gesture.Session.Sources.KeptFor(read) is not { } source || !DrawnOver(action, assembled, source))
        {
            await StepCommit.ShowAsync(list, assembled, ViewTrigger.Show, page: 0, picks);

            return null;
        }

        context.StateChanged = await StepCommit.CommitAsync(list, assembled, assembled.Readable.WithoutOutput(), picks);

        return null;
    }

    // What a list's control was drawn with: the kind of output its boxes make.
    private static ListPicks PicksOf(ControlAction action) => new(action.Text(TypeKey));

    // Why a change to the output is not made while the blocks make no pipeline, naming the block that stops them.
    private static string NotWhole(NotebookPipeline assembled) => $"the blocks do not make a pipeline yet: {string.Join("; ", assembled.Stopping)}";

    // A row's box on the list of the source's columns. A box from a list drawn before the blocks or the source's bytes
    // changed draws the list again and changes nothing; otherwise ticked takes the column in with the kind its row showed,
    // and unticked leaves it out.
    private static async Task<string?> ListedAsync(
        Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action, bool ticked)
    {
        if (await RowOfAsync(gesture, assembled, action) is not { } row)
        {
            return null;
        }

        if (!DrawnOver(action, assembled, row.Source))
        {
            await StepCommit.ShowAsync(row.List, assembled, ViewTrigger.Show, page: 0, PicksOf(action));

            return null;
        }

        IReadOnlyList<IPipelineStep> steps;

        try
        {
            steps = ticked
                ? assembled.Readable.Including(row.Column, KindNamed(action.Text(KindKey)) ?? ColumnKind.Text, row.Source.Rows.ColumnNames)
                : assembled.Readable.Excluding(row.Column);
        }
        catch (DeclarationException refused)
        {
            await StepCommit.NotMadeAsync(row.List, assembled, [.. refused.Faults.Select(fault => fault.ToString())], PicksOf(action));

            return null;
        }

        context.StateChanged = await StepCommit.CommitAsync(row.List, assembled, steps, PicksOf(action));

        return null;
    }

    // A row's kind select on the list. Verso's router sends its value on every key and every change, with no end to a
    // walk, so each value is one pick from the state the list was drawn in, replacing what the walk committed before:
    // a send is fresh when the list still says what holds, and goes on from the select's own last change when nothing
    // else changed since; anything else is stale, draws the list again and changes nothing, an echo included. The select
    // commits the value it ends on, and a category walked away from and back to still remembers the kind it was.
    private static async Task<string?> KindPickedAsync(Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action)
    {
        if (await RowOfAsync(gesture, assembled, action) is not { } row)
        {
            return null;
        }

        var now = assembled.Readable;
        var drawn = DrawnOver(action, assembled, row.Source)
            ? now.Steps
            : gesture.Session.ContinuationOf(context.InteractionType, NotebookSession.KeyOf(now), row.Source.Fingerprint);

        if (drawn is null)
        {
            await StepCommit.ShowAsync(row.List, assembled, ViewTrigger.Show, page: 0, PicksOf(action));

            return null;
        }

        // Asked for what already holds — the value it is at, sent again — it changes nothing.
        if (Picked(new PipelineDeclaration(drawn), row.Column, context.Payload, row.Source.Rows.ColumnNames) is not { } steps || steps.SequenceEqual(now.Steps))
        {
            return null;
        }

        if (await StepCommit.CommitAsync(row.List, assembled, steps, PicksOf(action)))
        {
            gesture.Session.SelectCommitted(context.InteractionType, drawn, NotebookSession.KeyOf(new PipelineDeclaration(steps)), row.Source.Fingerprint);
            context.StateChanged = true;
        }

        return null;
    }

    // A select setting one of the output's own values, under the rule a row's kind select keeps: a send is fresh while its
    // list still says what holds, goes on from the select's own last change when nothing else changed since, and is
    // stale otherwise, drawing the list again and changing nothing, an echo included. The value is read into the output
    // as its block's form reads it, from the state the list was drawn in, and the output placed as the column rules place
    // one. It needs blocks that make one pipeline.
    private static async Task<string?> ParameterPickedAsync(Gesture gesture, NotebookPipeline assembled, CellInteractionContext context, ControlAction action)
    {
        if (await ListOfAsync(gesture, assembled) is not { } list)
        {
            return null;
        }

        var picks = PicksOf(action);

        if (!assembled.Whole)
        {
            await StepCommit.NotMadeAsync(list.On, assembled, [NotWhole(assembled)], picks);

            return null;
        }

        var now = assembled.Readable;
        var drawn = DrawnOver(action, assembled, list.Source)
            ? now.Steps
            : gesture.Session.ContinuationOf(context.InteractionType, NotebookSession.KeyOf(now), list.Source.Fingerprint);

        if (drawn is null)
        {
            await StepCommit.ShowAsync(list.On, assembled, ViewTrigger.Show, page: 0, picks);

            return null;
        }

        var pick = OutputParameters.Picked(
            NotebookVerbs.Catalog(), new PipelineDeclaration(drawn), action.Text(ParameterKey) ?? string.Empty, context.Payload, list.Source.Rows.ColumnNames);

        if (pick.Steps is not { } steps)
        {
            if (pick.NotMade.Count > 0)
            {
                await StepCommit.NotMadeAsync(list.On, assembled, pick.NotMade, picks);
            }

            return null;
        }

        // Asked for what already holds — the value it is at, sent again — it changes nothing.
        if (steps.SequenceEqual(now.Steps))
        {
            return null;
        }

        if (await StepCommit.CommitAsync(list.On, assembled, steps, picks))
        {
            gesture.Session.SelectCommitted(context.InteractionType, drawn, NotebookSession.KeyOf(new PipelineDeclaration(steps)), list.Source.Fingerprint);
            context.StateChanged = true;
        }

        return null;
    }

    // The one pick a select's value makes from the state its list was drawn in: for a column the schema names, that kind;
    // for one it does not, taking it in with that kind; for none, the state as it was drawn. Nothing for a value that
    // names no kind.
    private static IReadOnlyList<IPipelineStep>? Picked(PipelineDeclaration drawn, string column, string? value, IReadOnlyList<string> header)
    {
        if (string.IsNullOrEmpty(value))
        {
            return drawn.Steps;
        }

        if (KindNamed(value) is not { } kind)
        {
            return null;
        }

        return drawn.Steps.OfType<DeclareStep>().FirstOrDefault()?.Columns.Any(each => each.Name == column) == true
            ? drawn.WithKind(column, kind)
            : drawn.Including(column, kind, header);
    }

    // The row a list's control is about: nothing when the control names no column.
    private static async Task<ListRow?> RowOfAsync(Gesture gesture, NotebookPipeline assembled, ControlAction action) =>
        action.Text(ColumnKey) is { } column && await ListOfAsync(gesture, assembled) is { } list ? new ListRow(list.On, column, list.Source) : null;

    // The list a control is on: the schema's block the blocks hold now — found by what the control carries, whichever
    // block the send names, since a change writes that block anew — with the rows this session read the source as.
    // Nothing when the blocks hold no schema; the list drawn again, saying why, in a session that has not read the source.
    private static async Task<ListOn?> ListOfAsync(Gesture gesture, NotebookPipeline assembled)
    {
        if (assembled.SchemaBlock is not { } schema)
        {
            return null;
        }

        var list = gesture with { Cell = schema };

        if (assembled.Readable.Steps[0] is not ReadCsvStep read || gesture.Session.Sources.KeptFor(read) is not { } source)
        {
            await StepCommit.NotMadeAsync(list, assembled, [ListSourceNotRead], ListPicks.None);

            return null;
        }

        return new ListOn(list, source);
    }

    // Whether a list's control was drawn over the blocks and the source's bytes as they are now.
    private static bool DrawnOver(ControlAction action, NotebookPipeline assembled, SourceRows source) =>
        action.Text(DrawnKey) == NotebookSession.KeyOf(assembled.Readable) && action.Text(SourceKey) == source.Fingerprint;

    // A kind by its word; nothing for a word that names none.
    private static ColumnKind? KindNamed(string? word) =>
        Enum.TryParse<ColumnKind>(word, ignoreCase: true, out var kind) && Enum.IsDefined(kind) ? kind : null;

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
            ? [NotWhole(assembled)]
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
        var header = declaration.Steps[0] is ReadCsvStep read ? gesture.Session.Sources.KeptFor(read)?.Rows.ColumnNames : null;

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

    /// <summary>The row a list's control is about.</summary>
    /// <param name="List">The gesture, at the list's block.</param>
    /// <param name="Column">The row's column.</param>
    /// <param name="Source">The rows this session read the source as, and their fingerprint.</param>
    private readonly record struct ListRow(Gesture List, string Column, SourceRows Source);

    /// <summary>The list a control is on.</summary>
    /// <param name="On">The gesture, at the list's block.</param>
    /// <param name="Source">The rows this session read the source as, and their fingerprint.</param>
    private readonly record struct ListOn(Gesture On, SourceRows Source);
}
