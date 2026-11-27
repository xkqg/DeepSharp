// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
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
    /// The gesture an alert's box under a profile sends, with its column, what answering it does and the value it says:
    /// ticked, the alert is answered — the column left out, or the value said to stand for a gap.
    /// </summary>
    internal const string Answer = "deepsharp.answer";

    /// <summary>What answering an alert does, as an alert's box carries it: its word, in lower case.</summary>
    internal const string AnswerKey = "answer";

    /// <summary>The value an alert's box says stands for a gap.</summary>
    internal const string ValueKey = "value";

    /// <summary>The format a box takes a column of moments in with, when it is not ISO 8601's.</summary>
    internal const string FormatKey = "format";

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

    /// <summary>
    /// The gesture the list's select for ticking one column or a range sends, with what the list was drawn with; the
    /// router sends <c>one</c> or <c>range</c>. A pick: it commits nothing.
    /// </summary>
    internal const string ListRange = "deepsharp.list.range";

    /// <summary>
    /// The gesture a range's kind select sends, with what the range is of and what the list was drawn with; the router
    /// sends the kind. A pick: it commits nothing.
    /// </summary>
    internal const string ListRangeKind = "deepsharp.list.rangeKind";

    /// <summary>The key a list's control carries whether its ticks are ranges under.</summary>
    internal const string RangeKey = "range";

    /// <summary>The key a list's control carries where a range of columns to take in starts under.</summary>
    internal const string IncludeFromKey = "includeFrom";

    /// <summary>The key a list's control carries where a range of the output's columns starts under.</summary>
    internal const string OutputFromKey = "outputFrom";

    /// <summary>The key a list's control carries the kind a range takes its columns in with under.</summary>
    internal const string IncludeKindKey = "includeKind";

    /// <summary>The key a list's control carries the kind a range of the output's columns takes them in with under.</summary>
    internal const string OutputKindKey = "outputKind";

    /// <summary>The key a range's kind select carries what the range is of under: <c>include</c> or <c>output</c>.</summary>
    internal const string ForKey = "for";

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

        var mayAddAndRemove = LayoutLets(operations, LayoutCapabilities.CellInsert | LayoutCapabilities.CellDelete);

        return await session.OneAtATimeAsync(context.CancellationToken, turn =>
            HandleAsync(new Gesture(session, notebook, operations, variables, context.CellId, mayAddAndRemove, turn), context));
    }

    private static async Task<string?> HandleAsync(Gesture gesture, CellInteractionContext context)
    {
        // A gesture that can change the blocks waits for what a front end may still be sending, and reads the blocks only
        // then: a change written over a block before a held keystroke lands would lose the keystroke. A view or a pick
        // changes nothing, and does not wait.
        if (ControlAction.Read(context.InteractionType) is { Gesture: not (ListType or ListRange or ListRangeKind) })
        {
            await gesture.Session.SettleAsync();
        }

        var assembled = NotebookPipeline.Of(gesture.Notebook.Cells);

        // What changed since the last gesture without one seeing it — a block added, taken away, moved or typed into in
        // the editor — is caught up with first: a grid the blocks no longer make is cleared.
        await gesture.Session.LetThroughAsync(gesture.Turn, async () =>
        {
            foreach (var stale in gesture.Session.StaleIn(assembled, except: gesture.Cell))
            {
                await gesture.Operations.ClearOutputAsync(stale);
            }

            gesture.Session.CaughtUp(assembled, gesture.Variables, except: gesture.Cell);
        });

        switch (context.InteractionType)
        {
            case Show:
                return await StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Show);

            case Page:
                return await StepCommit.PageAsync(gesture, assembled, PageOf(context.Payload));

            case Columns:
                return await StepCommit.ShowAsync(gesture, assembled, ViewTrigger.Show, ListPicks.None);

            default:
                if (ControlAction.Read(context.InteractionType) is not { } action)
                {
                    return null;
                }

                var sent = new Sent(gesture, assembled, context, action);

                // A select sends the value it is at; a box, whether it is ticked.
                if (action.Gesture == ListKind)
                {
                    return await sent.KindPickedAsync();
                }

                if (action.Gesture == ListType)
                {
                    return await sent.PickedAsync(sent.Picks with { Type = context.Payload });
                }

                if (action.Gesture == ListRange)
                {
                    return await sent.PickedAsync(Ranging(sent.Picks, context.Payload == "range"));
                }

                if (action.Gesture == ListRangeKind)
                {
                    return await sent.PickedAsync(action.Text(ForKey) == "output"
                        ? sent.Picks with { OutputKind = context.Payload.AsKind() }
                        : sent.Picks with { IncludeKind = context.Payload.AsKind() });
                }

                if (action.Gesture == ListParameter)
                {
                    return await sent.ParameterPickedAsync();
                }

                if (StateOf(context.Payload) is not { } ticked)
                {
                    return null;
                }

                return action.Gesture switch
                {
                    Apply => await sent.AppliedAsync(ticked),
                    ListInclude => await sent.ListedAsync(ticked),
                    ListOutput => await sent.AnsweredAsync(ticked),
                    ListRemoveOutput => await sent.RemovedAsync(ticked),
                    Answer => await sent.AnsweredAlertAsync(ticked),
                    _ => action.Text(ColumnKey) is { } column ? await sent.TickedAsync(column, ticked) : null,
                };
        }
    }

    // Ticking one column or a range: leaving ranges forgets where one started; staying in them keeps it.
    private static ListPicks Ranging(ListPicks picks, bool range) =>
        range ? picks with { Range = true } : picks with { Range = false, IncludeFrom = null, OutputFrom = null };

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
