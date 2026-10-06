// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The language a block is written in: one step of a pipeline, as its own JSON.
/// </summary>
/// <remarks>
/// Running a block reads its text through the catalog a pipeline file is read with and shows the block's card:
/// the stage it belongs to and what its verb does, or every fault at its line and column. Everything a block shows
/// is written while the block runs rather than handed back at the end, so the front end is told at once — which is
/// also the only way a run started by a gesture reaches the screen. An editor is offered the verbs, the keys a step
/// takes and the words a key takes, all read from the same descriptions the reader checks a file against, and for a
/// key naming columns the columns the notebook's pipeline knew at the last gesture.
/// <para>
/// There are two of these kernels, one for each thing Verso asks of them. The block type carries one, and Verso runs
/// the blocks through it. Verso also loads this kernel as a part of its own and asks that one for completions, hover
/// and diagnostics — and runs the blocks through it only while the block type is switched off. Neither keeps
/// anything between calls: the notebook's session lives on the block type, and both reach it, the one through the
/// block type that made it, the other through the host that loaded it.
/// </para>
/// </remarks>
[VersoExtension]
public sealed class StepKernel : NotebookExtension, ILanguageKernel
{
    /// <summary>The kernel's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.kernel";

    /// <summary>The language a block is written in.</summary>
    public const string Language = "pdd";

    /// <summary>
    /// The name the notebook's pipeline is handed to C# cells under, as text: the declaration its blocks make, and
    /// what the fit learned once the whole pipeline has run.
    /// </summary>
    /// <remarks>
    /// A C# cell reads it with <c>Variables.TryGet&lt;string&gt;</c>, because it is often not there: not until "Show the
    /// data here" or the toolbar's run reads the blocks, and not again after Verso's Run All, a change made in a block's
    /// form, a block run by hand with other text, blocks that make no whole pipeline, or a stopped run that started a
    /// kernel afresh — whenever the blocks may no longer make what it held. A block added, taken away, moved, turned into another kind or typed into
    /// takes it back too, unless the blocks still make it, once the notebook hears of it: at the next gesture, or at once
    /// from a host that changes cells itself. Either of those two reads the blocks and hands it over again. The text reads back
    /// through a catalog of the verbs this package brings,
    /// <c>StepCatalog.BuiltIn().WithIndicators().WithParquet().WithExcel().WithJson().WithNetworks().WithML()</c>, and a relative source path in it
    /// is read from the folder handed over beside it, under <see cref="Folder"/>.
    /// <para>
    /// Not a name a C# variable can have, on purpose. Verso declares a variable for every value a cell can name,
    /// once, and a value handed over under such a name would be read as it was the first time, forever; a key no
    /// variable can have is read afresh every time it is asked for. Text, because the notebook's types and a C# cell's
    /// are loaded apart and are not the same types even when their names are.
    /// </para>
    /// </remarks>
    public const string HandOver = "deepsharp.pipeline";

    /// <summary>
    /// The name the notebook's folder is handed to C# cells under, as text: where a relative path in the handed-over
    /// pipeline is read from, by the rule the notebook itself reads it by — <c>SourceFolder.Of</c> the folder. Absent
    /// for a notebook never saved, which reads from the working directory, <c>SourceFolder.WorkingDirectory</c>.
    /// </summary>
    public const string Folder = "deepsharp.folder";

    /// <summary>
    /// The name a C# cell hands a trained model's predictions back to the notebook under, as text: what
    /// <c>Measures.PredictionsToJson</c> writes of the measures the pipeline's report took of the model.
    /// </summary>
    /// <remarks>
    /// A cell that trains writes <c>Variables.Set("deepsharp.predictions", trained.Measures!.PredictionsToJson())</c>, and
    /// "Show the data here" at the report's block measures what it holds by the report, on the notebook's own run of its
    /// blocks, and draws it under the grid as the report says it is shown. Text, and a name no C# variable can have, for the
    /// reasons the pipeline is handed over so: the cell's types are not the notebook's, and a cell's own variables are copied
    /// into the notebook's under their names.
    /// </remarks>
    public const string HandedBack = "deepsharp.predictions";

    // The block type this kernel was made by, when it was; otherwise the one Verso loaded beside it.
    private readonly StepCellType? _owner;

    /// <summary>A kernel as Verso makes it, finding the notebook's session through the host that loads it.</summary>
    public StepKernel()
    {
    }

    /// <summary>The kernel the block type carries, which keeps the block type's session.</summary>
    /// <param name="owner">The block type.</param>
    internal StepKernel(StepCellType owner) => _owner = owner;

    /// <summary>What the notebook's gestures leave for its blocks, and what they show.</summary>
    /// <remarks>
    /// Kept by the block type Verso loaded, the one object every part reaches — Verso may run a block through this
    /// kernel or through the one the block type carries, and a gesture must reach whichever runs it.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Neither made by the block type nor loaded by Verso: there is no notebook.</exception>
    internal NotebookSession Session => _owner?.Session ?? RequiredSession;

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline steps";

    /// <inheritdoc />
    public override string Description => "Reads every block of a pipeline as one step, and shows what the step is and does.";

    /// <inheritdoc />
    public string LanguageId => Language;

    /// <inheritdoc />
    public string DisplayName => "Pipeline step";

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } = [".pdd"];

    /// <inheritdoc />
    public Task InitializeAsync() => Task.CompletedTask;

    /// <inheritdoc />
    /// <remarks>
    /// A block run because somebody ran it shows its card. A block run by a gesture shows its card and what the
    /// gesture asked for — the data there, or why there is none — and the request is taken once, so running the
    /// block again afterwards shows the card alone. Every write is let through whole, once it is worked out, under the
    /// ticket of the block's run and, for what a gesture asked, that gesture's turn: a block whose run was stopped shows
    /// what it wrote before the stop — its card, when it got that far — and nothing after it, and hands nothing over; one
    /// stopped before its kernel began shows nothing, since the engine clears a block as it begins it. A block run by hand
    /// whose step is not the one the last gesture read withdraws the pipeline handed to C# cells, which no longer is the
    /// one the blocks make.
    /// </remarks>
    public async Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context)
    {
        ArgumentNullException.ThrowIfNull(code);
        ArgumentNullException.ThrowIfNull(context);

        var catalog = NotebookVerbs.Catalog();
        var session = Session;
        var ticket = session.Enter(context.CancellationToken);
        var request = session.Take(context.CellId, ticket);
        var cell = context.CellId;
        var variables = context.Variables;
        var folder = context.NotebookMetadata.FolderPath();

        session.LetThrough(ticket, () => NotebookSession.HandOverFolder(variables, folder));

        IPipelineStep step;

        try
        {
            step = catalog.ReadStep(code);
        }
        catch (PipelineFileException refused)
        {
            await session.LetThroughAsync(ticket, async () =>
            {
                session.Hidden(cell);
                NotebookSession.Withdraw(variables);
                await context.WriteOutputAsync(StepCard.Refused(refused.Faults));
            });

            return [];
        }

        await session.LetThroughAsync(ticket, () => context.WriteOutputAsync(StepCard.Of(step, catalog.Describe(step.Verb).Purpose)));

        if (request is { } asked)
        {
            await ShowAsync(new BlockRun(session, ticket, context), asked);
        }
        else
        {
            // Run by hand with other text than the last gesture read: the pipeline handed to C# cells is no longer
            // the one the blocks make, and a C# cell sees none until a gesture reads them again.
            var other = session.Assembled?.Blocks.FirstOrDefault(block => block.Cell == cell).Step?.Equals(step) != true;

            session.LetThrough(ticket, () =>
            {
                session.Hidden(cell);

                if (other)
                {
                    NotebookSession.Withdraw(variables);
                }
            });
        }

        return [];
    }

    // The data at the block, over the rows the notebook's folder holds; and, when the whole pipeline was asked
    // for, what it learned, handed over beside the declaration. A view is worked out once for the steps and the
    // bytes it comes from and shown again from memory while both stay, so another page, or the same view again,
    // runs no step. What stops the rows on the way — a file that is not there, a column the rows lack, a value
    // that is not what its column declares — is said at the block, in the words of what stopped them. Each write is
    // one whole write, let through for the gesture's turn and the block's run.
    private static async Task ShowAsync(BlockRun block, ViewRequest request)
    {
        var (session, ticket, context) = block;
        var cell = context.CellId;

        // A card asked for in place of the data — a take-over's list — reads no rows and changes nothing.
        if (request.Card is { } card)
        {
            await block.LetThroughAsync(request, async () =>
            {
                session.Hidden(cell);
                await context.WriteOutputAsync(card);
            });

            return;
        }

        if (request.NotMade.Count > 0)
        {
            await block.LetThroughAsync(request, () => context.WriteOutputAsync(StepCard.NotMade(request.NotMade)));
        }

        if (request.Declaration is not { } declaration)
        {
            await block.LetThroughAsync(request, async () =>
            {
                session.Hidden(cell);
                await context.WriteOutputAsync(StepCard.NoData(request.Faults));
            });

            return;
        }

        if (request.List is { } picks)
        {
            await ListAsync(block, request, declaration, picks);

            return;
        }

        // What the blocks decided is saved before the rows are read: a decision stands whatever the rows meet.
        if (request.SavesTheColumns && context.NotebookMetadata.ColumnsFilePath() is { } path)
        {
            var columns = new ColumnsFile(path);
            var saving = columns.Saving(declaration);

            await block.LetThroughAsync(request, async () =>
            {
                if (columns.Write(saving) is { } notSaved)
                {
                    await context.WriteOutputAsync(notSaved);
                }
            });
        }

        var key = declaration.ViewKeyAt(request.Position);
        PipelineView view;
        KindProposal proposal;
        WholeRun run;

        try
        {
            // A relative path is read from the folder the notebook is saved in, by the rule a pipeline file is read
            // by; a notebook that was never saved has no folder, and reads from the working directory.
            var folder = context.NotebookMetadata.SourceFolder();
            var source = session.Sources.RowsFor(declaration, folder);
            proposal = source.Proposal;
            var pipeline = new Pipeline(declaration, source.Rows, folder);

            // One run of the whole pipeline at most, for the fit handed over and a report's measures alike.
            run = new WholeRun(session, pipeline, source.Fingerprint);
            view = session.ViewFor(key, source.Fingerprint)
                ?? session.Keep(key, source.Fingerprint, pipeline.ViewAt(request.Position + 1));

            HandOverOnceRead(request, ticket, run, context.Variables);
        }
        catch (Exception refused) when (refused is IOException or UnauthorizedAccessException or FormatException or InvalidOperationException)
        {
            // The rows are gone or other than a fit learned from: nothing learned from them is handed on. A run stopped
            // since it was asked for says nothing of what it met.
            await block.LetThroughAsync(request, async () =>
            {
                session.Hidden(cell);

                if (session.Assembled is { } assembled)
                {
                    session.HandOver(context.Variables, assembled, SourceBytes.Unreadable);
                }

                await context.WriteOutputAsync(StepCard.RowsRefused(refused.Message));
            });

            return;
        }

        // The grid, what the block measured under it, and the record that the block shows it are one write: a run
        // stopped since it was asked for writes none of it, and a stop while it is written waits for all of it.
        // A column taken in from the grid is taken in by the rule the list keeps: as the file saved beside the notebook
        // declares it, else as its cells propose. A report's block measures under its grid what a C# cell handed back,
        // on this run of the blocks; a refusal of what was handed back is no refusal of the rows, so it is measured here.
        var stored = context.NotebookMetadata.ColumnsFilePath() is { } beside ? new ColumnsFile(beside).Stored().Preset : null;
        var grid = DataGrid.Of(view, request.Page, declaration, column => TakenIn.Of(column, stored, proposal));
        var measured = view.Evidence.TryGetValue(request.Position, out var evidence) ? evidence.Accept(new EvidenceView())
            : declaration.Steps[request.Position] is INamesTheMeasures report ? new ReportView(run).Of(report, context.Variables)
            : null;

        await block.LetThroughAsync(request, async () =>
        {
            await context.WriteOutputAsync(grid.Output);

            if (measured is { } drawn)
            {
                await context.WriteOutputAsync(drawn);
            }

            session.Showing(cell, key, grid.Header);
        });
    }

    // Every column of the source as one row, from the rows as the source reads them — no step runs. The list marks new
    // the columns the saved file never showed, then says in the file that it showed them; a change made from the list
    // saves the decisions with the header it showed. A saved file that cannot be read is said to be so and never written.
    // What the list shows and what it saves are worked out first and written as one write.
    private static async Task ListAsync(BlockRun block, ViewRequest request, PipelineDeclaration declaration, ListPicks picks)
    {
        var (session, _, context) = block;
        var cell = context.CellId;
        SourceRows source;

        try
        {
            source = session.Sources.RowsFor(declaration, context.NotebookMetadata.SourceFolder());
        }
        catch (Exception refused) when (refused is IOException or UnauthorizedAccessException or FormatException or InvalidOperationException)
        {
            // A list stopped since it was asked for says nothing of what it met.
            await block.LetThroughAsync(request, async () =>
            {
                session.Hidden(cell);
                await context.WriteOutputAsync(StepCard.RowsRefused(refused.Message));
            });

            return;
        }

        var header = source.Rows.ColumnNames;
        var file = context.NotebookMetadata.ColumnsFilePath() is { } path ? new ColumnsFile(path) : (ColumnsFile?)null;
        var stored = file?.Stored() ?? default;
        var saving = stored.Unreadable is not null || file is not { } columns ? default
            : request.SavesTheColumns ? columns.Saving(declaration, header)
            : request.Trigger == ViewTrigger.Show ? ColumnsSave.OfSource(stored, header)
            : default;
        var drawn = NotebookSession.KeyOf(declaration);
        var list = new ColumnList(NotebookVerbs.Catalog(), declaration, source, stored.Preset).Drawn(picks, request.Whole);

        // A list stopped since it was asked for is neither drawn nor saved.
        await block.LetThroughAsync(request, async () =>
        {
            if (stored.Unreadable is { } unreadable)
            {
                await context.WriteOutputAsync(unreadable);
            }
            else if (file?.Write(saving) is { } notSaved)
            {
                await context.WriteOutputAsync(notSaved);
            }

            await context.WriteOutputAsync(list);
            session.Listing(cell, drawn);
        });
    }

    // What the notebook hands to C# cells once the rows were read. A run of the whole pipeline hands over what it
    // learned — and a run of these steps over these bytes runs once: asked again, it hands the same over again. A
    // view knows the bytes the gesture did not, and keeps what a run learned only while it learned it from them. Each is
    // one write, let through for the gesture's turn and the block's run.
    private static void HandOverOnceRead(ViewRequest request, NotebookTurn ticket, WholeRun run, IVariableStore variables)
    {
        var session = run.Session;

        // A run already stopped fits nothing it would only throw away; whether anything is handed over is decided where it
        // is let through.
        if (session.StoppedSince(request.Turn) || ticket.Mark.IsCancellationRequested)
        {
            return;
        }

        if (!request.RunsTheWholePipeline)
        {
            session.LetThrough(request.Turn, ticket.Mark, () =>
            {
                if (session.Assembled is { } assembled)
                {
                    session.HandOver(variables, assembled, SourceBytes.Of(run.Fingerprint));
                }
            });

            return;
        }

        if (!session.HandOverFitAgain(variables, run, request.Turn, ticket.Mark))
        {
            session.HandOverFit(variables, run, request.Turn, ticket.Mark);
        }
    }

    /// <summary>A block's run, for what a gesture asked of it: the notebook's session, the run's ticket, and what the engine hands the block.</summary>
    /// <param name="Session">The notebook's session.</param>
    /// <param name="Ticket">The ticket of the block's run: a stop marks it.</param>
    /// <param name="Context">What the engine hands the block: its cell, its notebook, and where it writes.</param>
    private readonly record struct BlockRun(NotebookSession Session, NotebookTurn Ticket, IExecutionContext Context)
    {
        /// <summary>Lets one whole write through for the gesture's turn and the block's run, unless either was stopped since.</summary>
        /// <param name="request">What the gesture asked, under its turn.</param>
        /// <param name="write">The write, whole.</param>
        /// <returns>A task that ends once the write was let through and written, or refused.</returns>
        public Task<bool> LetThroughAsync(ViewRequest request, Func<Task> write) => Session.LetThroughAsync(request.Turn, Ticket.Mark, write);
    }

    /// <inheritdoc />
    /// <remarks>
    /// A key naming columns is offered the columns the notebook's pipeline knew at the last gesture, at any of its
    /// blocks: this kernel is not told which block is being written, and never assembles the notebook itself.
    /// </remarks>
    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition)
    {
        ArgumentNullException.ThrowIfNull(code);

        // A kernel with no notebook knows no columns: it offers what the descriptions say and nothing more.
        var scope = (_owner?.Session ?? LoadedSession)?.Assembled?.ColumnsKnown ?? [];

        return Task.FromResult(Completions(StepText.Of(code), cursorPosition, NotebookVerbs.Catalog(), scope));
    }

    /// <inheritdoc />
    /// <remarks>The same faults a run shows, with lines and columns counted from one.</remarks>
    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code)
    {
        ArgumentNullException.ThrowIfNull(code);

        try
        {
            NotebookVerbs.Catalog().ReadStep(code);

            return Task.FromResult<IReadOnlyList<Diagnostic>>([]);
        }
        catch (PipelineFileException refused)
        {
            return Task.FromResult<IReadOnlyList<Diagnostic>>(
                [.. refused.Faults.Select(fault => new Diagnostic(
                    DiagnosticSeverity.Error, fault.Message, fault.Line, fault.Column, fault.Line, fault.Column + 1))]);
        }
    }

    /// <inheritdoc />
    /// <remarks>Over a key, what the parameter means; over the verb, what the step does; elsewhere, nothing.</remarks>
    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition)
    {
        ArgumentNullException.ThrowIfNull(code);

        var text = StepText.Of(code);
        var catalog = NotebookVerbs.Catalog();

        return Task.FromResult(text.QuotedAt(cursorPosition) switch
        {
            { Depth: 1, IsKey: false } verb when verb.Key == StepCatalog.StepKey && catalog.Knows(verb.Text) => new HoverInfo(catalog.Describe(verb.Text).Purpose),
            { Depth: 1, IsKey: true } key when Parameter(text, key.Text, catalog) is { } parameter => new HoverInfo(parameter.Description),
            _ => null,
        });
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private static IReadOnlyList<Completion> Completions(StepText text, int cursor, StepCatalog catalog, IReadOnlyList<KnownColumn> scope)
    {
        var place = text.PlaceOf(cursor);

        return place.Spot switch
        {
            Spot.InKey or Spot.BeforeKey => Keys(text, place, catalog),
            Spot.InValue or Spot.BeforeValue when place.Key == StepCatalog.StepKey => Verbs(place, catalog),
            Spot.InValue or Spot.BeforeValue or Spot.InItem =>
                Values(text, place, catalog, new StepWords(scope, place.Key is { } key ? text.ItemsOf(key, cursor) : [])),
            _ => [],
        };
    }

    // The keys the step takes that the block does not hold yet.
    private static IReadOnlyList<Completion> Keys(StepText text, CursorPlace place, StepCatalog catalog)
    {
        if (text.Verb is not { } verb || !catalog.Knows(verb))
        {
            return [];
        }

        var written = text.Keys;
        var offered = new List<Completion>();

        foreach (var parameter in catalog.Describe(verb).Parameters)
        {
            foreach (var key in parameter.Keys.Where(key => !written.Contains(key) && key.StartsWith(place.Typed, StringComparison.Ordinal)))
            {
                offered.Add(new Completion(key, place.Spot == Spot.InKey ? key : $"\"{key}\": ", "Property", parameter.Description));
            }
        }

        return offered;
    }

    private static IReadOnlyList<Completion> Verbs(CursorPlace place, StepCatalog catalog) =>
        [.. catalog.Descriptions
            .Where(description => description.Verb.StartsWith(place.Typed, StringComparison.Ordinal))
            .Select(description => new Completion(
                description.Verb,
                place.Spot == Spot.InValue ? description.Verb : $"\"{description.Verb}\"",
                "Function",
                description.Purpose))];

    // The words the key under the cursor takes, read from its parameter's kind.
    private static IReadOnlyList<Completion> Values(StepText text, CursorPlace place, StepCatalog catalog, StepWords taken)
    {
        if (Parameter(text, place.Key, catalog) is not { } parameter)
        {
            return [];
        }

        var words = parameter.Accept(taken);

        return [.. (place.Spot == Spot.BeforeValue ? words.Outside : words.Inside)
            .Where(word => word.StartsWith(place.Typed, StringComparison.Ordinal))
            .Select(word => new Completion(word, word, "EnumMember", parameter.Description))];
    }

    private static StepParameter? Parameter(StepText text, string? key, StepCatalog catalog) =>
        text.Verb is { } verb && catalog.Knows(verb)
            ? catalog.Describe(verb).Parameters.FirstOrDefault(parameter => key is not null && parameter.Keys.Contains(key))
            : null;
}
