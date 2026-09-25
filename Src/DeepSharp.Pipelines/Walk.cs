// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// The one walk over a declaration: every step, in the order it was written, doing the one thing it does.
/// </summary>
/// <remarks>
/// A run, a replay and the grid under one block of a notebook are this walk. There used to be two: the run
/// lifted every feature above every dropped row, the replay dropped nothing, and the same rows came out as
/// fifty-six from one and sixty from the other. The only difference left is the mode — whether the steps
/// that learn are fitted here or replay what was fitted before.
/// <para>
/// The walk hands each step to itself and never asks what it is: each capability says what doing it means
/// in terms of what the walk holds — the rows, the table, the parts, what was learned. A new capability is a
/// new interface that does that, and nothing here changes.
/// </para>
/// </remarks>
/// <param name="declaration">The steps, in the order they were written.</param>
/// <param name="mode">Fitting on the training rows, or replaying what was fitted.</param>
/// <param name="folder">Where a relative path a source holds is read from.</param>
internal sealed class Walk(PipelineDeclaration declaration, WalkMode mode, SourceFolder folder)
{
    private readonly List<Witnessed> _evidence = [];
    private IRowSource? _rows;
    private Table? _table;
    private Table? _divided;
    private Part[]? _parts;
    private Table? _captured;
    private int _captureAfter = -1;
    private int _at;
    private int _through;

    /// <summary>The table as it stands at the step being walked.</summary>
    /// <remarks>
    /// The declaration holds every step that works on columns after the step that names them, so there is
    /// always a table by the time one asks.
    /// </remarks>
    public Table Table => _table!;

    /// <summary>Walks every step over the given rows, or over the rows the declaration opens itself.</summary>
    /// <param name="rows">Rows handed in, or nothing to open the declared source.</param>
    /// <returns>The table as the last step left it, and where every row landed.</returns>
    /// <exception cref="InvalidOperationException">The declaration cannot be walked over these rows.</exception>
    public Walked Through(IRowSource? rows) => Walking(rows, declaration.Steps.Count);

    /// <summary>The rows as they stand after some of the steps, joined to the split wherever it is declared.</summary>
    /// <param name="rows">Rows handed in, or nothing to open the declared source.</param>
    /// <param name="steps">How many steps from the start; at least one past the step that declares the columns.</param>
    /// <returns>The rows there, and where each stands.</returns>
    /// <remarks>
    /// The walk goes on past this place to the split, when the split is below it, so every row here knows the
    /// part it will land in — or that it is dropped before it gets there.
    /// </remarks>
    public PipelineView Viewed(IRowSource? rows, int steps)
    {
        _captureAfter = steps;
        var walked = Walking(rows, declaration.WalkedFor(steps));

        return new PipelineView(_captured!, Standings.Of(_captured!, _divided, _parts), Measured, walked.Evidence);
    }

    /// <summary>Walks to the split, when there is one, so rows read elsewhere can be told where they stand.</summary>
    /// <param name="rows">Rows handed in, or nothing to open the declared source.</param>
    /// <param name="read">The rows to place, read from the same source.</param>
    /// <param name="steps">How many steps from the start the rows are placed after, all of them above the schema.</param>
    /// <returns>Those rows, and where each stands.</returns>
    public PipelineView Placed(IRowSource? rows, Table read, int steps)
    {
        var evidence = declaration.SplitAt >= 0 ? Walking(rows, declaration.WalkedFor(steps)).Evidence : NothingProduced;

        return new PipelineView(read, Standings.Of(read, _divided, _parts), Measured, evidence);
    }

    /// <summary>Walks as far as the step that turns the rows into columns, and stops there.</summary>
    /// <param name="rows">Rows handed in, or nothing to open the declared source.</param>
    /// <returns>The rows as they were read into the declared columns.</returns>
    /// <exception cref="InvalidOperationException">The declaration names no columns, or no source.</exception>
    public Table Bound(IRowSource? rows) => Walking(rows, declaration.ColumnsAt + 1).Table;

    /// <summary>Takes the rows the declaration's source opens, unless rows were handed in to take their place.</summary>
    /// <param name="open">How the source opens its rows, reading a relative path from the pipeline's folder.</param>
    /// <remarks>Rows handed in take the place of the declared source, which is how serving works.</remarks>
    public void Open(Func<SourceFolder, IRowSource> open) => _rows ??= open(folder);

    /// <summary>Reads the rows into the declared columns.</summary>
    /// <param name="bind">How the step reads rows into columns.</param>
    /// <exception cref="InvalidOperationException">There are no rows: no source, and none handed in.</exception>
    public void Bind(Func<IRowSource, Table> bind)
    {
        var table = bind(mode.Prepare(declaration, _rows ?? throw new InvalidOperationException(
            "This pipeline never says where its rows come from, so there is nothing to prepare.")));

        _table = table;

        // Before any step runs, the columns the rows turned out to have are followed down the steps this walk
        // goes through: a column the schema allowed to be absent, and the rows lack, is refused at every one of
        // them that reads it. A run goes through every step; a view through those it is worked out from, so a
        // step below it is refused at its own view and at the run, and two views with one key show one thing.
        var followed = ColumnFlow.Follow(declaration.Steps, ColumnState.Of(table), _at + 1, _through, declared: true);

        if (followed.Faults.Count > 0)
        {
            throw new DeclarationException(followed.Faults);
        }

        mode.Bound(declaration, table);
    }

    /// <summary>Divides the rows, when this walk divides them, and writes down what the division saw.</summary>
    /// <param name="assign">How the step assigns every row a part.</param>
    /// <param name="describe">What only this kind of split adds to what every split writes down.</param>
    public void Divide(Func<Table, Part[]> assign, Action<Table, IReadOnlyList<Part>, FittedStepValues> describe)
    {
        if (!mode.Divides)
        {
            return;
        }

        var table = Table;
        var parts = assign(table);

        _divided = table;

        mode.ValuesFor(_at, () =>
        {
            var seen = new FittedStepValues();

            foreach (var part in new[] { Part.Train, Part.Validation, Part.Test, Part.Predict })
            {
                seen.Learned($"rows.{part.ToString().ToLowerInvariant()}", parts.Count(each => each == part));
            }

            seen.Learned("digest", table.Digest());
            describe(table, parts, seen);

            return seen;
        });

        _parts = parts;
    }

    /// <summary>Applies what a step learned from the training rows: learned here, or learned before and replayed.</summary>
    /// <param name="fit">How the step learns from the training rows.</param>
    /// <param name="apply">How the step applies what it learned.</param>
    public void Learn(Func<Table, IReadOnlyList<Part>, FittedStepValues> fit, Action<Table, FittedStepValues> apply)
    {
        var table = Table;

        // The declaration holds every step that learns below the split, so a walk that fits has divided the
        // rows by the time one is reached; a walk that replays never calls the fit.
        apply(table, mode.ValuesFor(_at, () => fit(table, _parts!)));
    }

    /// <summary>Keeps the rows a mask says to keep.</summary>
    /// <param name="keep">One answer per row: whether it stays.</param>
    public void Keep(IReadOnlyList<bool> keep) => _table = Table.Keep(keep);

    /// <summary>The rows as a step that drops rows judges them: every column, less the answers this walk awaits.</summary>
    /// <remarks>
    /// A served row is the question, so its answers arrive as gaps, and a gap there is no reason to drop the row or
    /// to count a warm-up. A fit awaits nothing and judges every column: a training row without its answer cannot
    /// be learned from.
    /// </remarks>
    public Table Judged => mode.Awaited(declaration) is { Count: > 0 } awaited ? Table.Without(awaited) : Table;

    /// <summary>Takes note of the rows here, for evidence produced once the walk knows where every row lands.</summary>
    /// <param name="produce">How the step produces its evidence from the rows and their standings.</param>
    /// <remarks>
    /// When fitting only: evidence is what a run shows, and a replay shows nothing. The rows are copied as they
    /// stand, since later steps change the table, and measured after the walk has reached the split.
    /// </remarks>
    public void Evidence(Func<PipelineView, Evidence> produce)
    {
        if (mode.Divides)
        {
            _evidence.Add(new Witnessed(_at, Table.Snapshot(), produce));
        }
    }

    /// <summary>Puts the rows in an order, in every mode: a replay hands rows back in the declared order too.</summary>
    /// <param name="order">The rows as they are now, in the order they are to stand.</param>
    public void Reorder(IReadOnlyList<int> order) => _table = Table.Ordered(order);

    // A column a step reads can be gone by the time it is reached though the declaration allowed it: a fill that
    // saw too many gaps and left only its marker, a category the training rows never held. The refusal says
    // which step, which column, and why it may be gone, rather than that some table lacks some key.
    private void ThrowIfAColumnItReadsIsGone(IPipelineStep step)
    {
        if (_table is not { } table)
        {
            return;
        }

        foreach (var read in step.ColumnsRead)
        {
            if (!table.Has(read.Column))
            {
                var why = declaration.ColumnsBefore(_at).WhyItMayBeGone(read.Column)
                          ?? "no step above left it here";

                throw new InvalidOperationException(
                    $"Step {_at + 1}, '{step.Verb}', reads '{read.Column}', which is not here: {why}.");
            }
        }
    }

    // What a view measures: the training rows of a pipeline that divides its rows, the rest when it does not.
    private Standing Measured => declaration.SplitAt >= 0 ? Standing.Train : Standing.Undivided;

    // What a view carries where no step on its way produced any evidence.
    private static readonly IReadOnlyDictionary<int, Evidence> NothingProduced = new Dictionary<int, Evidence>();

    private Walked Walking(IRowSource? rows, int through)
    {
        if (declaration.ColumnsAt < 0)
        {
            throw new InvalidOperationException(
                "This pipeline never says which columns take part. Declare them, and the rest is dropped.");
        }

        _rows = rows;
        _through = through;

        for (_at = 0; _at < through; _at++)
        {
            ThrowIfAColumnItReadsIsGone(declaration.Steps[_at]);

            // An output that names the answer acts on nothing; every other step does exactly one thing.
            if (declaration.Steps[_at] is IActsInAWalk acting)
            {
                acting.ActOn(this);
            }

            if (_at == _captureAfter - 1)
            {
                _captured = Table.Snapshot();
            }
        }

        var evidence = _evidence.ToDictionary(
            witnessed => witnessed.At,
            witnessed => witnessed.Produce(new PipelineView(witnessed.Rows, Standings.Of(witnessed.Rows, _divided, _parts), Measured, NothingProduced)));

        // A pipeline that learns nothing needs no split, and rows nothing divided are not training rows.
        return new Walked(Table, _parts ?? [.. Enumerable.Repeat(Part.Undivided, Table.RowCount)], evidence);
    }
}

/// <summary>What a walk leaves behind: the table, whose rows know where they were read, the part each landed in, and its evidence.</summary>
/// <param name="Table">The table as the last step left it.</param>
/// <param name="Parts">Which part each row belongs to.</param>
/// <param name="Evidence">What each step that produces evidence produced, by its place.</param>
internal readonly record struct Walked(Table Table, Part[] Parts, IReadOnlyDictionary<int, Evidence> Evidence);

/// <summary>The rows a step producing evidence saw, and how it produces its evidence from them.</summary>
/// <param name="At">The step's place.</param>
/// <param name="Rows">The rows as they stood there.</param>
/// <param name="Produce">How the step produces its evidence.</param>
internal readonly record struct Witnessed(int At, Table Rows, Func<PipelineView, Evidence> Produce);

/// <summary>
/// What a walk does with the steps that learn and the step that divides.
/// </summary>
internal abstract class WalkMode
{
    /// <summary>The rows as the walk will read them; the same rows, unless the mode has a reason to add to them.</summary>
    /// <param name="declaration">The declaration being walked.</param>
    /// <param name="source">The rows.</param>
    /// <returns>The rows to bind.</returns>
    public virtual IRowSource Prepare(PipelineDeclaration declaration, IRowSource source) => source;

    /// <summary>Told once the rows have been read into columns, before anything changes them.</summary>
    /// <param name="declaration">The declaration being walked.</param>
    /// <param name="table">The rows as they were read.</param>
    public virtual void Bound(PipelineDeclaration declaration, Table table)
    {
    }

    /// <summary>Whether the rows are divided at the split, which fitting does and replaying does not.</summary>
    public abstract bool Divides { get; }

    /// <summary>The answers this walk awaits rather than reads: none when fitting, every one the output names when replaying.</summary>
    /// <param name="declaration">The declaration being walked.</param>
    /// <returns>The answer columns, in the output's order.</returns>
    public virtual IReadOnlyList<string> Awaited(PipelineDeclaration declaration) => [];

    /// <summary>What a step that learns applies, at its place in the walk.</summary>
    /// <param name="at">The step's place in the declaration.</param>
    /// <param name="fit">How the step learns, when this walk fits.</param>
    /// <returns>The values the step applies.</returns>
    public abstract FittedStepValues ValuesFor(int at, Func<FittedStepValues> fit);
}

/// <summary>
/// Fitting: the rows are divided at the split, and every step that learns does so from the training rows alone.
/// </summary>
internal sealed class FitOnTheTrainingRows : WalkMode
{
    private readonly Dictionary<int, FittedStepValues> _fitted = [];

    /// <summary>What each step learned, by its place in the declaration.</summary>
    public IReadOnlyDictionary<int, FittedStepValues> Fitted => _fitted;

    /// <summary>The answer as it was read, when there is one to come back to: a number the pipeline may change.</summary>
    public double?[]? AnswerAsRead { get; private set; }

    /// <inheritdoc />
    public override void Bound(PipelineDeclaration declaration, Table table)
    {
        // The way back is walked for an output of one answer, and only for an answer that is a number to begin
        // with: an answer of words is predicted as a category and there is no arithmetic to come back through.
        var target = declaration.Output?.Answers is [var only] ? only : null;

        AnswerAsRead = target is not null && table.Has(target)
                       && table[target].Kind is not (ColumnKind.Text or ColumnKind.Category)
            ? table.NumbersOf(target)
            : null;
    }

    /// <inheritdoc />
    public override bool Divides => true;

    /// <inheritdoc />
    public override FittedStepValues ValuesFor(int at, Func<FittedStepValues> fit)
    {
        var learned = fit();

        _fitted[at] = learned;

        return learned;
    }
}

/// <summary>
/// Replaying: nothing is divided and nothing learns; every step applies what the training rows taught it.
/// </summary>
/// <param name="fitted">What each step learned, by its place in the declaration.</param>
internal sealed class ReplayWhatWasFitted(IReadOnlyDictionary<int, FittedStepValues> fitted) : WalkMode
{
    /// <inheritdoc />
    /// <remarks>
    /// A row served to a model has no answer yet — that is why it is asked. Every answer the output names that the
    /// rows lack arrives as a gap in every row, which is what it is.
    /// </remarks>
    public override IRowSource Prepare(PipelineDeclaration declaration, IRowSource source)
    {
        string[] lacking = [.. Awaited(declaration).Where(answer => !source.ColumnNames.Contains(answer, StringComparer.Ordinal))];

        return lacking.Length == 0 ? source : new RowsAwaitingAnAnswer(source, lacking);
    }

    /// <inheritdoc />
    public override IReadOnlyList<string> Awaited(PipelineDeclaration declaration) => declaration.Output?.Answers ?? [];

    /// <inheritdoc />
    public override bool Divides => false;

    /// <inheritdoc />
    /// <remarks>A saved pipeline is checked for one fit per step that learns when it is loaded, so every one is here.</remarks>
    public override FittedStepValues ValuesFor(int at, Func<FittedStepValues> fit) => fitted[at];
}

/// <summary>
/// Rows handed in without the columns a model predicts, read as if they were there and empty.
/// </summary>
/// <param name="rows">The rows as they were handed in.</param>
/// <param name="answers">The columns they lack.</param>
internal sealed class RowsAwaitingAnAnswer(IRowSource rows, IReadOnlyList<string> answers) : IRowSource
{
    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; } = [.. rows.ColumnNames, .. answers];

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows =>
        rows.Rows.Select(row => (IReadOnlyList<string?>)[.. row, .. answers.Select(_ => (string?)null)]);
}
