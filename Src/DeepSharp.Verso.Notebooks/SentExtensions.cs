// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using static DeepSharp.Verso.Notebooks.StepRenderer;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What each control on a block or a list does when it sends, each asked of the thing that was sent.
/// </summary>
/// <remarks>
/// Every gesture here is a state a control asks for, so the same gesture twice does the same thing once. Each reads the
/// blocks as they stood when it was sent, changes them only through the one commit, and says at a block why it did not.
/// The router that picks one by the gesture's name stays with <see cref="StepRenderer"/>.
/// </remarks>
internal static class SentExtensions
{
    // Said at a block when the list of a take-over is applied to blocks that are no longer the ones it listed.
    private const string ListedForOtherBlocks = "the blocks changed since the list was shown — take over again";

    // Said at a block when a column the schema does not name is ticked before anything read the source in this session.
    private const string SourceNotRead =
        "The source is not read in this session, so where a column it has stands among its columns is not known yet: "
        + "its data is shown here now, and ticking the column again takes it in where it stands.";

    extension(Sent sent)
    {
        // A pick on the list — the kind of output, ticking one column or a range, a range's kind. It commits nothing; the list
        // is drawn again with it, keeping the others.
        internal async Task<string?> PickedAsync(ListPicks picks) =>
            sent.Assembled.SchemaBlock is { } schema && !string.IsNullOrEmpty(sent.Context.Payload)
                ? await StepCommit.ShowAsync(sent.Gesture with { Cell = schema }, sent.Assembled, ViewTrigger.Show, picks)
                : null;

        // A row's output box. The output is changed only while the blocks make one pipeline; a box from a list drawn before
        // the blocks or the source's bytes changed draws the list again and changes nothing; otherwise the column is put into
        // the output of the kind the box makes, or taken out, as the output box's one rule says. In a range of an output of
        // many, the first tick says where the range starts and changes nothing, and the second puts every column between in,
        // with the range's kind; a range refused keeps its start, so another end can be ticked.
        internal async Task<string?> AnsweredAsync(bool ticked)
        {
            var (_, assembled, context, action) = sent;

            if (await sent.RowAsync() is not { } row)
            {
                return null;
            }

            var picks = sent.Picks;

            if (!await sent.WholeOrRefuseAsync(row.List, picks) || !await sent.ListOrRefreshAsync(row.List, row.Source))
            {
                return null;
            }

            var catalog = NotebookVerbs.Catalog();
            var verb = action.Text(TypeKey) ?? string.Empty;
            var header = row.Source.Rows.ColumnNames;
            var kinds = ticked && picks.Range ? OutputBox.RangeKinds(catalog, verb) : null;

            if (kinds is not null && picks.OutputFrom is null)
            {
                await StepCommit.ShowAsync(row.List, assembled, ViewTrigger.Show, picks with { OutputFrom = row.Column });

                return null;
            }

            var boxes = new OutputBox(catalog, assembled.Readable, verb, header);
            var change = kinds is not null
                ? boxes.Ranged(header.Between(picks.OutputFrom!, row.Column), picks.OutputKind ?? kinds[0])
                : boxes.Change(row.Column, action.Text(KindKey).AsKind() ?? ColumnKind.Text, ticked);

            if (change.Steps is not { } steps)
            {
                await StepCommit.NotMadeAsync(row.List, assembled, change.NotMade, picks);

                return null;
            }

            context.StateChanged = await StepCommit.CommitAsync(row.List, assembled, steps, picks with { OutputFrom = null });

            return null;
        }

        // The list's box that takes the output away. Ticked with no output there — the click's echo — it changes nothing;
        // otherwise it needs a session that read the source, blocks that make one pipeline, and a list that still says what
        // holds.
        internal async Task<string?> RemovedAsync(bool ticked)
        {
            var assembled = sent.Assembled;

            if (!ticked || assembled.Readable.Output is null || await sent.ListAsync() is not { } list)
            {
                return null;
            }

            if (!await sent.WholeOrRefuseAsync(list.On, sent.Picks) || !await sent.ListOrRefreshAsync(list.On, list.Source))
            {
                return null;
            }

            sent.Context.StateChanged = await StepCommit.CommitAsync(list.On, assembled, assembled.Readable.WithoutOutput(), sent.Picks);

            return null;
        }

        // A row's box on the list of the source's columns. A box from a list drawn before the blocks or the source's bytes
        // changed draws the list again and changes nothing; otherwise ticked takes the column in with the kind its row showed,
        // and unticked leaves it out. In a range, the first tick says where it starts and changes nothing, and the second
        // takes every column between in, with the range's one kind; unticking leaves one column out.
        internal async Task<string?> ListedAsync(bool ticked)
        {
            var (_, assembled, context, action) = sent;

            if (await sent.RowAsync() is not { } row)
            {
                return null;
            }

            var picks = sent.Picks;

            if (!await sent.ListOrRefreshAsync(row.List, row.Source))
            {
                return null;
            }

            if (ticked && picks.Range && picks.IncludeFrom is null)
            {
                await StepCommit.ShowAsync(row.List, assembled, ViewTrigger.Show, picks with { IncludeFrom = row.Column });

                return null;
            }

            var header = row.Source.Rows.ColumnNames;
            IReadOnlyList<IPipelineStep> steps;

            try
            {
                steps = !ticked ? assembled.Readable.Excluding(row.Column)
                    : picks.Range ? assembled.Readable.Including(header.Between(picks.IncludeFrom!, row.Column), picks.IncludeKind ?? ColumnKind.Text, header)
                    : TakenIn.CarriedBy(action).Into(assembled.Readable, row.Column, header);
            }
            catch (DeclarationException refused)
            {
                await StepCommit.NotMadeAsync(row.List, assembled, [.. refused.Faults.Select(fault => fault.ToString())], picks);

                return null;
            }

            context.StateChanged = await StepCommit.CommitAsync(row.List, assembled, steps, picks with { IncludeFrom = null });

            return null;
        }

        // A row's kind select on the list. Verso's router sends its value on every key and every change, with no end to a
        // walk, so each value is one pick from the state the list was drawn in, replacing what the walk committed before:
        // a send is fresh when the list still says what holds, and goes on from the select's own last change when nothing
        // else changed since; anything else is stale, draws the list again and changes nothing, an echo included. The select
        // commits the value it ends on, and a category walked away from and back to still remembers the kind it was.
        internal async Task<string?> KindPickedAsync()
        {
            var (gesture, assembled, context, _) = sent;

            if (await sent.RowAsync() is not { } row)
            {
                return null;
            }

            var now = assembled.Readable;
            var drawn = sent.DrawnOver(row.Source)
                ? now.Steps
                : gesture.Session.ContinuationOf(context.InteractionType, NotebookSession.KeyOf(now), row.Source.Fingerprint);

            if (drawn is null)
            {
                await sent.RefreshAsync(row.List);

                return null;
            }

            // Asked for what already holds — the value it is at, sent again — it changes nothing.
            if (Picked(new PipelineDeclaration(drawn), row.Column, context.Payload, row.Source.Rows.ColumnNames) is not { } steps || steps.SequenceEqual(now.Steps))
            {
                return null;
            }

            if (await StepCommit.CommitAsync(row.List, assembled, steps, sent.Picks))
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
        internal async Task<string?> ParameterPickedAsync()
        {
            var (gesture, assembled, context, action) = sent;

            if (await sent.ListAsync() is not { } list)
            {
                return null;
            }

            var picks = sent.Picks;

            if (!await sent.WholeOrRefuseAsync(list.On, picks))
            {
                return null;
            }

            var now = assembled.Readable;
            var drawn = sent.DrawnOver(list.Source)
                ? now.Steps
                : gesture.Session.ContinuationOf(context.InteractionType, NotebookSession.KeyOf(now), list.Source.Fingerprint);

            if (drawn is null)
            {
                await sent.RefreshAsync(list.On);

                return null;
            }

            var pick = new OutputParameters(NotebookVerbs.Catalog(), new PipelineDeclaration(drawn), list.Source.Rows.ColumnNames)
                .Picked(action.Text(ParameterKey) ?? string.Empty, context.Payload);

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

        // A take-over's list, sent with the saved columns it listed and the key of the blocks it listed them for. Ticked,
        // what it listed is made, in one commit; asked again — the click's echo — the blocks hold it already, and nothing
        // happens. The blocks must still make the whole pipeline the list was worked out for, or the change is not made and
        // the block says why.
        internal async Task<string?> AppliedAsync(bool ticked)
        {
            var (gesture, assembled, context, action) = sent;

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

            if (!await sent.WholeOrRefuseAsync(gesture, picks: null))
            {
                return null;
            }

            IReadOnlyList<string> notMade = drawn != NotebookSession.KeyOf(assembled.Readable)
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
        internal async Task<string?> TickedAsync(string column, bool ticked)
        {
            var (gesture, assembled, context, action) = sent;
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
                steps = action.Gesture switch
                {
                    Include when ticked => await sent.TakenInAsync(column, TakenIn.CarriedBy(action)),
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

        // A column taken in. One the schema does not name comes in as its box says — by the rule the list's box keeps too —
        // where the source has it, which only the source's own header says: the rows this session read from it. Before
        // anything read them the block shows the source, which reads them, and says so; a column the source does not have
        // was not ticked on its grid, nor was a column of rows handed in, which no grid shows.
        internal async Task<IReadOnlyList<IPipelineStep>?> TakenInAsync(string column, TakenIn taken)
        {
            var declaration = sent.Assembled.Readable;

            // One the schema names, or a drop leaves out, is taken back where it stands already.
            if (declaration.ChoicesFor([column]).Rows[0].Standing != ColumnStanding.NotDeclared)
            {
                return declaration.Including(column, ColumnKind.Text, []);
            }

            if (SourceCache.FilesOf(declaration) is null)
            {
                return null;
            }

            if (sent.Gesture.Session.Sources.KeptFor(declaration)?.Rows.ColumnNames is not { } header)
            {
                await StepCommit.NotMadeAsync(sent.Gesture, sent.Assembled, [SourceNotRead]);

                return null;
            }

            return header.Contains(column, StringComparer.Ordinal) ? taken.Into(declaration, column, header) : null;
        }

        // An alert's box under a profile: ticked, its answer is given through the column rules — the column left out, or the
        // value said to stand for a gap on the schema; unticked, nothing, since the box only ever asks for its answer. What the
        // rules refuse is said at the block. Given twice, an answer changes nothing the second time.
        internal async Task<string?> AnsweredAlertAsync(bool ticked)
        {
            var (gesture, assembled, context, action) = sent;
            var position = assembled.PositionOf(gesture.Cell);

            if (!ticked || position < 0 || position >= assembled.Readable.Steps.Count
                || action.Text(ColumnKey) is not { } column || AnswerOf(action.Text(AnswerKey)) is not { } answering)
            {
                return null;
            }

            IReadOnlyList<IPipelineStep> steps;

            try
            {
                steps = new AlertAnswer(answering, column, Value: action.Text(ValueKey)).AppliedTo(assembled.Readable);
            }
            catch (DeclarationException refused)
            {
                await StepCommit.NotMadeAsync(gesture, assembled, [.. refused.Faults.Select(fault => fault.ToString())]);

                return null;
            }

            context.StateChanged = await StepCommit.CommitAsync(gesture, assembled, steps);

            return null;
        }
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

        if (value.AsKind() is not { } kind)
        {
            return null;
        }

        // A list is drawn only over blocks that hold a schema.
        return drawn.Schema?.Columns.Any(each => each.Name == column) == true
            ? drawn.WithKind(column, kind)
            : drawn.Including(column, kind, header);
    }

    // What an alert's box says answering it does: leaving its column out, or saying a value stands for a gap there.
    private static AlertAction? AnswerOf(string? word) =>
        Enum.GetValues<AlertAction>().Where(action => action != AlertAction.Step && string.Equals(action.ToString(), word, StringComparison.OrdinalIgnoreCase))
            .Cast<AlertAction?>().FirstOrDefault();

    // A column the schema takes made a category, or given back the kind it was; nothing for a column it does not take,
    // nor for a category that does not say what it was.
    private static IReadOnlyList<IPipelineStep>? Kinded(PipelineDeclaration declaration, string column, bool ticked) =>
        declaration.Schema?.Taking.FirstOrDefault(each => each.Name == column) switch
        {
            null => null,
            _ when ticked => declaration.WithKind(column, ColumnKind.Category),
            { Kind: ColumnKind.Category, Was: { } was } => declaration.WithKind(column, was),
            _ => null,
        };
}
