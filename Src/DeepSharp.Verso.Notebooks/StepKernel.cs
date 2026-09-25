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
    /// form, a block run by hand with other text, or blocks that make no whole pipeline — whenever the blocks may no
    /// longer make what it held. Either of those two reads the blocks and hands it over again. The text reads back
    /// through a catalog of the packages' own verbs, <c>StepCatalog.BuiltIn().WithIndicators()</c>, and a relative
    /// source path in it is read from the folder handed over beside it, under <see cref="Folder"/>.
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
    /// block again afterwards shows the card alone. A block run by hand whose step is not the one the last gesture
    /// read withdraws the pipeline handed to C# cells, which no longer is the one the blocks make.
    /// </remarks>
    public async Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context)
    {
        ArgumentNullException.ThrowIfNull(code);
        ArgumentNullException.ThrowIfNull(context);

        var catalog = NotebookVerbs.Catalog();
        var session = Session;
        var request = session.Take(context.CellId);

        NotebookSession.HandOverFolder(context.Variables, context.NotebookMetadata.FolderPath());

        IPipelineStep step;

        try
        {
            step = catalog.ReadStep(code);
        }
        catch (PipelineFileException refused)
        {
            session.Hidden(context.CellId);
            NotebookSession.Withdraw(context.Variables);
            await context.WriteOutputAsync(StepCard.Refused(refused.Faults));

            return [];
        }

        await context.WriteOutputAsync(StepCard.Of(step, catalog.Describe(step.Verb).Purpose));

        if (request is { } asked)
        {
            await ShowAsync(session, asked, context);
        }
        else
        {
            session.Hidden(context.CellId);

            // Run by hand with other text than the last gesture read: the pipeline handed to C# cells is no longer
            // the one the blocks make, and a C# cell sees none until a gesture reads them again.
            if (session.Assembled?.Blocks.FirstOrDefault(block => block.Cell == context.CellId).Step?.Equals(step) != true)
            {
                NotebookSession.Withdraw(context.Variables);
            }
        }

        return [];
    }

    // The data at the block, over the rows the notebook's folder holds; and, when the whole pipeline was asked
    // for, what it learned, handed over beside the declaration. A view is worked out once for the steps and the
    // bytes it comes from and shown again from memory while both stay, so another page, or the same view again,
    // runs no step. What stops the rows on the way — a file that is not there, a column the rows lack, a value
    // that is not what its column declares — is said at the block, in the words of what stopped them.
    private static async Task ShowAsync(NotebookSession session, ViewRequest request, IExecutionContext context)
    {
        if (request.NotMade.Count > 0)
        {
            await context.WriteOutputAsync(StepCard.NotMade(request.NotMade));
        }

        if (request.Declaration is not { } declaration)
        {
            session.Hidden(context.CellId);
            await context.WriteOutputAsync(StepCard.NoData(request.Faults));

            return;
        }

        var key = declaration.ViewKeyAt(request.Position);
        PipelineView view;

        try
        {
            // A relative path is read from the folder the notebook is saved in, by the rule a pipeline file is read
            // by; a notebook that was never saved has no folder, and reads from the working directory.
            var folder = context.NotebookMetadata.SourceFolder();
            var source = session.Sources.RowsFor(declaration, folder);
            var pipeline = new Pipeline(declaration, source.Rows, folder);

            view = session.ViewFor(key, source.Fingerprint)
                ?? session.Keep(key, source.Fingerprint, pipeline.ViewAt(request.Position + 1));

            HandOverOnceRead(session, request, pipeline, source.Fingerprint, context.Variables);
        }
        catch (Exception refused) when (refused is IOException or UnauthorizedAccessException or FormatException or InvalidOperationException)
        {
            // The rows are gone or other than a fit learned from: nothing learned from them is handed on.
            session.Hidden(context.CellId);

            if (session.Assembled is { } assembled)
            {
                session.HandOver(context.Variables, assembled, SourceBytes.Unreadable);
            }

            await context.WriteOutputAsync(StepCard.RowsRefused(refused.Message));

            return;
        }

        var grid = DataGrid.Of(view, request.Page, declaration);

        await context.WriteOutputAsync(grid.Output);

        // A block that declares evidence shows what it measured under the rows it measured it on.
        if (view.Evidence.TryGetValue(request.Position, out var evidence))
        {
            await context.WriteOutputAsync(evidence.Accept(new EvidenceView()));
        }

        session.Showing(context.CellId, key, grid.Header);
    }

    // What the notebook hands to C# cells once the rows were read. A run of the whole pipeline hands over what it
    // learned — and a run of these steps over these bytes runs once: asked again, it hands the same over again. A
    // view knows the bytes the gesture did not, and keeps what a run learned only while it learned it from them.
    private static void HandOverOnceRead(
        NotebookSession session, ViewRequest request, Pipeline pipeline, string fingerprint, IVariableStore variables)
    {
        if (!request.RunsTheWholePipeline)
        {
            if (session.Assembled is { } assembled)
            {
                session.HandOver(variables, assembled, SourceBytes.Of(fingerprint));
            }

            return;
        }

        if (!session.HandOverFitAgain(variables, pipeline.Declaration, fingerprint))
        {
            session.HandOverFit(variables, pipeline.Run(), fingerprint);
        }
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
            _ => (HoverInfo?)null,
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
