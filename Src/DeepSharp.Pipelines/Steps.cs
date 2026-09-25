// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from a comma-separated file.
/// </summary>
/// <remarks>
/// Declaring where the data comes from is not the same act as going to get it: nothing is opened until the
/// pipeline runs, so a declaration can be written, saved and checked on a machine that has no data on it.
/// </remarks>
public sealed record ReadCsvStep : IPipelineStep<ReadCsvStep>, IOpensRows, IDescribesColumns
{
    private static readonly FilePathParameter PathKey = new(
        "path", "Where the comma-separated file will be, when the pipeline runs.", "data.csv");

    /// <summary>Declares that the rows come from the file at this path.</summary>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadCsvStep(string path) => Path = PathKey.Require(path);

    /// <summary>Where the file will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.csv";

    /// <inheritdoc />
    public static string Purpose => "Reads the rows from a comma-separated file.";

    /// <inheritdoc />
    public static StepParameters<ReadCsvStep> Parameters { get; } =
        new StepParameters<ReadCsvStep>().With(PathKey, step => step.Path);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>A relative path is read from the pipeline's folder; the path stays as it was written.</remarks>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return new CsvRowSource(folder.Resolve(Path));
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static ReadCsvStep ReadFrom(JsonElement element) => new(PathKey.Read(element));
}

/// <summary>
/// The rows are handed in rather than opened.
/// </summary>
/// <remarks>
/// A source that lives outside the file: a table already in memory, a reader over a query, anything a
/// package turns into rows. The declaration says so and says what it was, and the rows themselves are
/// handed to the pipeline when it runs — which is also exactly how serving works, so the same declaration
/// covers both without a second shape.
/// </remarks>
public sealed record ReadRowsStep : IPipelineStep<ReadRowsStep>, IOpensRows, IDescribesColumns
{
    private static readonly TextParameter DescriptionKey = new(
        "description", "What the rows are, for whoever reads the file later.", "rows handed in");

    /// <summary>Declares that the rows are handed in.</summary>
    /// <param name="description">What the rows are, for whoever reads the file later.</param>
    /// <exception cref="ArgumentException">The description is empty.</exception>
    public ReadRowsStep(string description) => Description = DescriptionKey.Require(description);

    /// <summary>What the rows are, for whoever reads the file later.</summary>
    public string Description { get; }

    /// <inheritdoc />
    public static string Name => "read.rows";

    /// <inheritdoc />
    public static string Purpose => "Takes rows that are handed in rather than opened: a table already in memory, a reader over a query.";

    /// <inheritdoc />
    public static StepParameters<ReadRowsStep> Parameters { get; } =
        new StepParameters<ReadRowsStep>().With(DescriptionKey, step => step.Description);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public IRowSource Open(SourceFolder folder) =>
        throw new InvalidOperationException(
            $"This pipeline reads rows that are handed in ({Description}). "
            + "Hand them to Run or Prepare, or to Replay when the pipeline is already fitted.");

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ReadRowsStep ReadFrom(JsonElement element) => new(DescriptionKey.Read(element));
}

/// <summary>
/// Split the rows into training, validation and test by where they sit in time.
/// </summary>
/// <remarks>
/// This is the line in the chain. Above it nothing may learn from the data; below it the operations that do
/// become available, and each of them is fitted on the training rows alone.
/// <para>
/// A moment is never divided: every row of one moment lands in the part its first row does, so what happened
/// at one time is on one side of the line or the other. Rows are placed in the order of their time and then
/// of their keys, so the same rows are divided the same way whatever order they arrive in.
/// </para>
/// <para>
/// A gap keeps the last moments of every part apart — before each line and at the end — for an answer read
/// from later rows: without one, the last training rows learn their answers from the rows a model is measured
/// on. The rows in the gap are fitted on by nothing and handed to nothing.
/// </para>
/// </remarks>
public sealed record SplitByTimeStep : ISplitStep, IDividesInTime, IPipelineStep<SplitByTimeStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column that says when a row happened.", "when", ColumnKinds.Ordered);

    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter GapKey = new(
        "gap", "How many of the last moments of every part are kept apart, fitted on by nothing and handed to nothing: at least as many as the rows an answer reads ahead. Left out, none.",
        0, atLeast: 0, leftOut: 0);

    /// <summary>Declares a split in time, by shares that together make a whole.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training, validation, test and predicting.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share: nothing, or more than everything.</exception>
    public SplitByTimeStep(string column, SplitShares shares)
        : this(column, shares, 0)
    {
    }

    /// <summary>Declares a split in time that keeps the last moments of every part apart.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training, validation, test and predicting.</param>
    /// <param name="gap">How many of the last moments of every part are kept apart; none for no gap.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share, or the gap is below nothing.</exception>
    public SplitByTimeStep(string column, SplitShares shares, int gap)
    {
        Column = ColumnKey.Require(column);
        Shares = SharesKey.Require(shares);
        Gap = GapKey.Require(gap);
    }

    /// <inheritdoc />
    public static StepParameters<SplitByTimeStep> Parameters { get; } = new StepParameters<SplitByTimeStep>()
        .With(ColumnKey, step => step.Column)
        .With(SharesKey, step => step.Shares)
        .With(GapKey, step => step.Gap);

    /// <summary>The column that says when a row happened.</summary>
    public string Column { get; }

    /// <summary>How much goes to training, validation, test and predicting.</summary>
    public SplitShares Shares { get; }

    /// <summary>How many of the last moments of every part are kept apart.</summary>
    public int Gap { get; }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A row has no time, or its time is not a number.</exception>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        // A row with no time cannot be placed in a split by time, and guessing where it belongs is how a row
        // from next year ends up in the training data.
        var when = table.RowOrderBy(Column);
        var order = Enumerable.Range(0, table.RowCount).ToArray();

        Array.Sort(order, (one, other) =>
        {
            var compared = when(one, other);

            return compared != 0 ? compared : table.Identities[one].Key.CompareTo(table.Identities[other].Key);
        });

        var parts = Shares.Over(table.RowCount).Placed(order);

        parts.KeptTogether(order, (one, other) => when(one, other) == 0);
        parts.Gapped(order, Gap, (one, other) => when(one, other) == 0);

        // A gap wider than a part takes the whole of it, and a model with nothing to learn from, or nothing to be
        // measured on, is a run that should stop rather than succeed.
        ThrowIfTheGapTookAll(parts, Part.Train, "learn from");
        ThrowIfTheGapTookAll(parts, Part.Test, "measure on");

        return parts;
    }

    private void ThrowIfTheGapTookAll(Part[] parts, Part part, string purpose)
    {
        if (Gap > 0 && !parts.Contains(part))
        {
            throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"A gap of {Gap} moments takes every row of {part.ToString().ToLowerInvariant()}, and leaves nothing to {purpose}. Keep fewer moments apart, or divide more rows."));
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Where each part starts and ends, in the column it was divided by: a model trained on one stretch of
    /// time and measured on the next can say which stretches they were.
    /// </remarks>
    public void Describe(Table table, IReadOnlyList<Part> parts, FittedStepValues seen)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(seen);

        var column = table[Column];
        var when = Comparer<int>.Create(table.RowOrderBy(Column));

        foreach (var part in new[] { Part.Train, Part.Validation, Part.Test, Part.Predict })
        {
            var rows = Enumerable.Range(0, table.RowCount).Where(row => parts[row] == part).Order(when).ToArray();

            if (rows.Length > 0)
            {
                seen.Learned($"moments.{part.ToString().ToLowerInvariant()}", [Moment(column, rows[0]), Moment(column, rows[^1])]);
            }
        }
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    // A moment is written the one way that reads the same everywhere; a number as its shortest exact form.
    private static string Moment(IColumn column, int row) =>
        column is Column<DateTime> moments
            ? moments[row]!.Value.ToString("o", CultureInfo.InvariantCulture)
            : column.TextAt(row)!;

    /// <inheritdoc />
    public static string Name => "split.byTime";

    /// <inheritdoc />
    public static string Purpose => "Divides the rows by when they happened: the earliest to learn from, the latest to be measured on.";

    /// <inheritdoc />
    /// <remarks>The second version keeps a moment whole and places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public string Verb => Name;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is of the wrong kind.</exception>
    public static SplitByTimeStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), SharesKey.Read(element), GapKey.Read(element));
}

/// <summary>
/// Fill the gaps in a column, the named way.
/// </summary>
/// <remarks>
/// Missing is not the same as not-a-number: a value is missing when it was never there, which is data,
/// while a not-a-number is arithmetic that produced no number, which is a fault further upstream. They get
/// different verbs because they deserve different answers.
/// <para>
/// One verb in two forms, because the two ways of filling are different things: one puts a value learned from
/// the training rows into every gap, the other carries the value before a gap forward and so reads the rows in
/// their order. <see cref="Of"/> picks the form from the strategy, and a file does the same.
/// </para>
/// </remarks>
public abstract record FillMissingStep : IFittedStep, IPipelineStep<FillMissingStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column with gaps in it.", "column", ColumnKinds.Fillable);

    private static readonly FillStrategyParameter WithKey = new(
        "with",
        "What goes in the gaps, learned from the training rows: mean, median, zero, previous, constant, or refuse.",
        With.Median,
        ["mean", "median", "zero", "previous", "constant", "refuse"],
        "filling a gap");

    private static readonly ShareParameter RefuseAboveKey = new(
        "refuseAbove",
        "The share of the training rows that may be gaps and still be filled. Above it the column is not filled, "
        + "and the column that says where the gaps were speaks for it. Left out, every share is filled.");

    private protected FillMissingStep(string column, FillStrategy strategy, double? refuseAbove)
    {
        Column = ColumnKey.Require(column);
        Strategy = WithKey.Require(strategy);
        RefuseAbove = RefuseAboveKey.Require(refuseAbove);
    }

    /// <summary>Declares that the gaps in a column are filled the named way.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">What to put in them — <see cref="With"/> has the names.</param>
    /// <param name="refuseAbove">The share of the training rows that may be gaps and still be filled; nothing for no limit.</param>
    /// <returns>The step, in the form the strategy takes.</returns>
    /// <exception cref="ArgumentException">The column has no name, or the strategy is not one of the names.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The share is not one.</exception>
    public static FillMissingStep Of(string column, FillStrategy strategy, double? refuseAbove = null) =>
        strategy.Name == With.Previous.Name
            ? new FillMissingByPreviousStep(column, refuseAbove)
            : new FillMissingByValueStep(column, strategy, refuseAbove);

    /// <inheritdoc />
    public static StepParameters<FillMissingStep> Parameters { get; } = new StepParameters<FillMissingStep>()
        .With(ColumnKey, step => step.Column)
        .With(WithKey, step => step.Strategy)
        .With(RefuseAboveKey, step => step.RefuseAbove);

    /// <summary>The column with gaps in it.</summary>
    public string Column { get; }

    /// <summary>What goes in them.</summary>
    public FillStrategy Strategy { get; }

    /// <summary>The share of the training rows that may be gaps and still be filled, or nothing for no limit.</summary>
    /// <remarks>
    /// Past some point a filled column is invention: a value made up for three quarters of the rows, after
    /// which the column saying where the gaps were carries everything the original had. So the decision is
    /// made once, in the open, when the pipeline is fitted — measured on the training rows alone, written down,
    /// and replayed unchanged.
    /// </remarks>
    public double? RefuseAbove { get; }

    /// <summary>The column written beside a filled one, saying where the gaps were.</summary>
    /// <remarks>
    /// Always, and not behind a flag. Filling destroys the difference between "absent" and "the value
    /// happened to be that" permanently, so the difference is written down before it goes.
    /// </remarks>
    public string MarkerColumn => $"{Column}_was_missing";

    /// <inheritdoc />
    /// <remarks>
    /// With a share it will not fill above, whether the column is still there is decided when the pipeline is
    /// fitted, so it is known but not sure from here on.
    /// </remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        var marked = before.With(MarkerColumn, ColumnKind.Number);

        return RefuseAbove is { } share
            ? marked.MaybeGone(
                Column,
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"fill.missing replaced it by '{MarkerColumn}', because more than {share} of its training rows were gaps"))
            : marked;
    }

    /// <inheritdoc />
    public static string Name => "fill.missing";

    /// <inheritdoc />
    public static string Purpose => "Fills the gaps in a column the named way, with a value learned from the training rows, and marks where they were.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var learned = new FittedStepValues();
        var training = table.TrainingValues(Column, parts);

        // The gaps among the training rows: what the fit saw, like every number in its half of the file.
        learned.Learned("gaps", training.Gaps);

        if (RefuseAbove is { } ceiling)
        {
            var share = training.Rows == 0 ? 0 : training.Gaps / (double)training.Rows;

            learned.Learned("share", share);
            learned.Learned("filled", share <= ceiling ? 1 : 0);

            if (share > ceiling)
            {
                return learned;
            }
        }

        Learn(table, parts, learned);

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var column = table[Column];
        var marker = new Column<double>(
            MarkerColumn, ColumnKind.Number,
            Enumerable.Range(0, table.RowCount).Select(row => (double?)(column.IsMissing(row) ? 1 : 0)));

        // Decided when the pipeline was fitted: too many of the training rows were gaps, so the column is not
        // filled at all, and the marker speaks for it.
        if (fitted.Numbers.TryGetValue("filled", out var filled) && filled == 0)
        {
            table.Remove(Column);
            table.Put(marker);

            return;
        }

        switch (column)
        {
            case Column<double> numbers:
                Fill(numbers, fitted, value => value);
                break;

            case Column<long> whole:
                Fill(whole, fitted, value => (long)Math.Round(value, MidpointRounding.AwayFromZero));
                break;

            default:
                throw new InvalidOperationException(
                    $"'{Column}' holds {column.Kind.ToString().ToLowerInvariant()}, and a gap in it is not filled with a number.");
        }

        table.Put(marker);
    }

    /// <summary>Reads this step back out of a file, in the form its strategy takes.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static FillMissingStep ReadFrom(JsonElement element) =>
        Of(ColumnKey.Read(element), WithKey.Read(element), RefuseAboveKey.Read(element));

    /// <summary>What this form learns from the training rows, beside the count of gaps every form writes down.</summary>
    /// <param name="table">The data.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <param name="learned">Where it writes what it learned.</param>
    private protected abstract void Learn(Table table, IReadOnlyList<Part> parts, FittedStepValues learned);

    /// <summary>Fills every gap in the column, the way this form does.</summary>
    /// <typeparam name="T">What the column holds.</typeparam>
    /// <param name="column">The column, changed in place.</param>
    /// <param name="fitted">What the fit learned.</param>
    /// <param name="asValue">How a learned number becomes a value of the column.</param>
    private protected abstract void Fill<T>(Column<T> column, FittedStepValues fitted, Func<double, T> asValue)
        where T : struct;
}

/// <summary>
/// Fills every gap in a column with one value: learned from the training rows, or said outright.
/// </summary>
/// <remarks>
/// The mean or the median of the training rows, nought, a constant, or a refusal for a column that is not
/// supposed to have gaps at all. Every number here comes from the training rows and from nowhere else: fit on
/// all of them and the validation rows have quietly taught the model about themselves, and nothing goes red.
/// </remarks>
public sealed record FillMissingByValueStep : FillMissingStep
{
    /// <summary>Declares that the gaps in a column are filled with one value.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">Which value: mean, median, zero, a constant, or refuse.</param>
    /// <exception cref="ArgumentException">
    /// The column has no name, or the strategy is not one of the names — or it is the one that carries the
    /// value before a gap forward, which is the other form of this verb.
    /// </exception>
    /// <param name="refuseAbove">The share of the training rows that may be gaps and still be filled; nothing for no limit.</param>
    public FillMissingByValueStep(string column, FillStrategy strategy, double? refuseAbove = null)
        : base(column, strategy, refuseAbove)
    {
        if (strategy.Name == With.Previous.Name)
        {
            throw new ArgumentException(
                "Carrying the value before a gap forward reads the rows in their order; that is FillMissingByPreviousStep.",
                nameof(strategy));
        }
    }

    /// <inheritdoc />
    private protected override void Learn(Table table, IReadOnlyList<Part> parts, FittedStepValues learned)
    {
        var training = table.TrainingValues(Column, parts);

        switch (Strategy.Name)
        {
            case "mean":
                learned.Learned("value", training.Learnable("a fill value").Mean);
                break;

            case "median":
                learned.Learned("value", training.Learnable("a fill value").Median);
                break;

            case "zero":
                learned.Learned("value", 0);
                break;

            case "constant":
                learned.Learned("value", Strategy.Value!.Value);
                break;

            default:
                if (learned.Number("gaps") > 0)
                {
                    throw new InvalidOperationException(
                        $"'{Column}' has {learned.Number("gaps"):0} gaps in its training rows, and this pipeline says there should be none.");
                }

                break;
        }
    }

    /// <inheritdoc />
    private protected override void Fill<T>(Column<T> column, FittedStepValues fitted, Func<double, T> asValue)
    {
        for (var row = 0; row < column.Count; row++)
        {
            if (!column.IsMissing(row))
            {
                continue;
            }

            // A row served a year from now can hold the gap the training rows never did; the refusal holds then too.
            column[row] = Strategy.Name == With.Refuse.Name
                ? throw new InvalidOperationException(
                    $"Row {row + 1} of '{Column}' is a gap, and this pipeline says there should be none.")
                : asValue(fitted.Number("value"));
        }
    }
}

/// <summary>
/// Fills every gap in a column with the value before it.
/// </summary>
/// <remarks>
/// It learns nothing, and the value it uses depends on the row above rather than on the training set — which
/// is why it reads the rows in their order and needs that order declared above it. A gap with nothing before
/// it to carry forward is refused rather than guessed.
/// </remarks>
public sealed record FillMissingByPreviousStep : FillMissingStep, IReadsRowOrder
{
    /// <summary>Declares that the gaps in a column are filled with the value before them.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <param name="refuseAbove">The share of the training rows that may be gaps and still be filled; nothing for no limit.</param>
    public FillMissingByPreviousStep(string column, double? refuseAbove = null)
        : base(column, With.Previous, refuseAbove)
    {
    }

    /// <inheritdoc />
    private protected override void Learn(Table table, IReadOnlyList<Part> parts, FittedStepValues learned)
    {
    }

    /// <inheritdoc />
    private protected override void Fill<T>(Column<T> column, FittedStepValues fitted, Func<double, T> asValue)
    {
        T? previous = null;

        for (var row = 0; row < column.Count; row++)
        {
            if (!column.IsMissing(row))
            {
                previous = column[row];
                continue;
            }

            column[row] = previous ?? throw new InvalidOperationException(
                $"Row {row + 1} of '{Column}' is a gap with nothing before it to carry forward.");
        }
    }
}
