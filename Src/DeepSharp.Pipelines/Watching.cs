// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Keeps, as its fit sees them, the training rows of every column it watches — each value once with how many training rows hold
/// it, and how many hold none — so the rows a served model meets can be measured against the rows it learned from long after the
/// rows themselves are gone.
/// </summary>
/// <remarks>
/// A pipeline read from its file keeps what its steps learned and not the rows they learned it from, so the rows it is served have
/// nothing to be measured against unless a step keeps them. This one does, below the split and from the training rows alone, as
/// counts: both distances a column is measured by need no more, and the counts measure to the bit what the rows would have. It
/// learns nothing a replay reads and changes no column, so what it keeps is said as seen, and a pipeline that declares it serves
/// rows exactly as one that does not. Left unnamed, it watches every column where it stands but the id, the answer and the columns
/// the rows are ordered or divided in time by, which every window holds otherwise than the training rows by construction.
/// </remarks>
public sealed record WatchStep : IPipelineStep<WatchStep>, IFittedStep, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to keep the training rows of; left out, every column where it stands but the id, the answer and the columns the rows are ordered or divided in time by.", ["column"], ColumnKinds.Any)
    {
        Optional = true,
    };

    /// <summary>Keeps the training rows of these columns, or of every column it watches when none is named.</summary>
    /// <param name="warnAbove">The distance above which a window of served rows is flagged: above nought, at most one.</param>
    /// <param name="columns">The columns to keep; none, for every column but the id, the answer and the order.</param>
    /// <exception cref="ArgumentOutOfRangeException">The bound is not above nought and at most one.</exception>
    /// <exception cref="ArgumentException">A column has no name, or one is named twice.</exception>
    public WatchStep(double warnAbove, IEnumerable<string>? columns = null)
    {
        WarnAbove = warnAbove > 0 && warnAbove <= 1
            ? LikenessStep.WarnAboveKey.Require(warnAbove)
            : throw new ArgumentOutOfRangeException(nameof(warnAbove), warnAbove, "A window is flagged above a distance that is above nought and at most one.");
        Columns = ColumnsKey.Require([.. columns ?? []]);
    }

    /// <summary>The distance above which a window of served rows is flagged.</summary>
    public double WarnAbove { get; }

    /// <summary>The columns kept; none, for every column but the id, the answer and the order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "watch.reference";

    /// <inheritdoc />
    public static string Purpose => "Keeps the training rows of the columns it watches, as counts, so rows served later can be measured against them.";

    /// <inheritdoc />
    public static int Since => 11;

    /// <inheritdoc />
    public static StepParameters<WatchStep> Parameters { get; } = new StepParameters<WatchStep>()
        .With(LikenessStep.WarnAboveKey, step => step.WarnAbove)
        .With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>The name under which the fit lists the columns it kept, in the order it kept them.</summary>
    internal const string KeptColumns = "columns";

    /// <inheritdoc />
    public bool Equals(WatchStep? other) => other is not null && WarnAbove.Equals(other.WarnAbove) && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(WarnAbove);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    /// <remarks>
    /// Over a table alone there is no answer and no order to leave out, so every column but the id is kept unless the step names
    /// them; in a walk, the declaration says which those are.
    /// </remarks>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts) => Kept(table, parts, Watched(table, new ViewedDeclaration([], [], Standing.Train)));

    /// <inheritdoc />
    /// <remarks>Nothing: the rows go on exactly as they came.</remarks>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);
    }

    void IActsInAWalk.ActOn(Walk walk)
    {
        var declared = ViewedDeclaration.Of(walk.Declaration);

        walk.Learn((table, parts) => Kept(table, parts, Watched(table, declared)), ApplyTo);
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static WatchStep ReadFrom(JsonElement element) => new(LikenessStep.WarnAboveKey.Read(element), ColumnsKey.Read(element));

    // The columns watched: those named, or every column but the id, the answers and the columns the rows are ordered by.
    private IReadOnlyList<string> Watched(Table table, ViewedDeclaration declared) =>
        Columns.Count > 0
            ? Columns
            : [.. table.Columns.Where(column => column.Kind != ColumnKind.Media).Select(column => column.Name)
                .Where(name => name != table.Id && !declared.Answers.Contains(name, StringComparer.Ordinal) && !declared.OrderedBy.Contains(name, StringComparer.Ordinal))];

    // Each watched column of the training rows, counted, and said as seen.
    private static FittedStepValues Kept(Table table, IReadOnlyList<Part> parts, IReadOnlyList<string> watched)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        int[] training = [.. Enumerable.Range(0, table.RowCount).Where(row => parts[row] == Part.Train)];
        var kept = new FittedStepValues();

        kept.Saw(KeptColumns, watched);

        foreach (var column in watched)
        {
            switch (Tally.Of(table, column, training))
            {
                case NumbersTallied numbers:
                    kept.Saw($"{column}.values", numbers.Values);
                    kept.Saw($"{column}.counts", [.. numbers.Counts.Select(count => (double)count)]);
                    kept.Saw($"{column}.gaps", numbers.Gaps);
                    break;
                case WordsTallied words:
                    kept.Saw($"{column}.words", words.Words);
                    kept.Saw($"{column}.counts", [.. words.Counts.Select(count => (double)count)]);
                    kept.Saw($"{column}.gaps", words.Gaps);
                    break;
            }
        }

        return kept;
    }

    /// <summary>The training rows of one kept column, as the fit counted them.</summary>
    /// <param name="fitted">What the fit kept.</param>
    /// <param name="column">The column.</param>
    /// <returns>The count.</returns>
    internal static Tally KeptOf(FittedStepValues fitted, string column)
    {
        int[] counts = [.. fitted.Curve($"{column}.counts").Select(count => (int)count)];
        var gaps = (int)fitted.Number($"{column}.gaps");

        return fitted.Lists.ContainsKey($"{column}.words") || fitted.Curves.ContainsKey($"{column}.words")
            ? new WordsTallied(fitted.List($"{column}.words"), counts, gaps)
            : new NumbersTallied(fitted.Curve($"{column}.values"), counts, gaps);
    }
}

/// <summary>How far one watched column of a window of served rows is from the training rows the pipeline kept.</summary>
/// <param name="Column">The column.</param>
/// <param name="Distance">The distance its kind takes: Kolmogorov–Smirnov for numbers and moments, total variation for words.</param>
/// <param name="Apart">The distance, from nought to one; not a number when the window or the training rows hold no value.</param>
/// <param name="Above">Whether it is above the bound the watch was declared with.</param>
public readonly record struct WindowColumn(string Column, Distance Distance, double Apart, bool Above)
{
    /// <summary>How many training rows held a value of the column.</summary>
    public int TrainingRows { get; init; }

    /// <summary>How many training rows held none.</summary>
    public int TrainingGaps { get; init; }

    /// <summary>How many rows of the window held a value of the column.</summary>
    public int WindowRows { get; init; }

    /// <summary>How many rows of the window held none.</summary>
    public int WindowGaps { get; init; }
}

/// <summary>A window of rows a served model met, measured column by column against the training rows the pipeline kept.</summary>
/// <param name="WarnAbove">The bound above which a distance is flagged, as the watch was declared.</param>
/// <param name="Rows">How many rows the window held, as the pipeline replayed them to the watch.</param>
/// <param name="Columns">Every watched column, in the order the watch kept them.</param>
/// <remarks>
/// A distance over a few rows is noise as much as anything — one row is as far from the training rows as its place among them says,
/// a half for a row at their middle — which is why every column says how many rows it was worked out from beside the flag, as the
/// likeness of the parts does; how many rows make a window is the caller's to say, by the rows it hands in.
/// </remarks>
public sealed record WindowLikeness(double WarnAbove, int Rows, IReadOnlyList<WindowColumn> Columns)
{
    /// <summary>The earliest moment the window holds in the first column its rows are ordered or divided in time by; nothing without one.</summary>
    public DateTime? From { get; init; }

    /// <summary>The latest such moment; nothing without one.</summary>
    public DateTime? To { get; init; }
}

/// <summary>The rows a served model meets, measured against the training rows a pipeline kept, one window at a time.</summary>
public static class Watching
{
    extension(PreparedData prepared)
    {
        /// <summary>
        /// Measures a window of rows a served model met against the training rows the pipeline kept: every column the watch
        /// keeps, by the distance its kind takes, flagged above its bound.
        /// </summary>
        /// <param name="window">The rows, as they arrived — a day of them, an hour, or whatever the question is asked over.</param>
        /// <returns>The window, column by column, with how many rows each distance was worked out from and when the window was.</returns>
        /// <exception cref="InvalidOperationException">
        /// The pipeline keeps none of the rows it was fitted on — it declares no <c>watch.reference</c> — or keeps them at more than
        /// one place; or a row is refused as the replay to the watch refuses it.
        /// </exception>
        /// <remarks>
        /// The rows are replayed with what the training rows taught every step above the watch and stopped there, so a category is
        /// still a word, and one the training rows never held is measured rather than refused by an encoder below. Nothing is
        /// learned from them: the reference is the one the pipeline was fitted on, whichever window is measured and however often.
        /// The library keeps no window between calls; the caller groups the rows the way its question asks, and the windows it
        /// measured are drawn one beside another by <c>DeepSharp.Charts</c>.
        /// </remarks>
        public WindowLikeness Watched(IRowSource window)
        {
            ArgumentNullException.ThrowIfNull(prepared);
            ArgumentNullException.ThrowIfNull(window);

            var steps = prepared.Course.Steps;
            int[] watches = [.. Enumerable.Range(0, steps.Count).Where(at => steps[at] is WatchStep)];

            if (watches is not [var at])
            {
                throw new InvalidOperationException(watches.Length == 0
                    ? "This pipeline keeps none of the rows it was fitted on, so the rows it is served have nothing to be measured against. "
                      + "Declare watch.reference below the split, fit the pipeline again and train the model behind that fit."
                    : string.Create(CultureInfo.InvariantCulture, $"This pipeline keeps the rows it was fitted on at {watches.Length} places, and a window is measured against one: keep one watch.reference."));
            }

            var watch = (WatchStep)steps[at];
            var kept = prepared.Fitted[at];
            var table = new Walk(prepared.Course, new ReplayWhatWasFitted(prepared.Fitted), SourceFolder.WorkingDirectory).Viewed(window, at + 1).Table;
            int[] rows = [.. Enumerable.Range(0, table.RowCount)];

            WindowColumn Measured(string column)
            {
                var trained = WatchStep.KeptOf(kept, column);
                var served = Tally.Of(table, column, rows);
                var apart = trained.ApartFrom(served);

                return new WindowColumn(column, trained.Distance, apart, apart > watch.WarnAbove)
                {
                    TrainingRows = trained.Held,
                    TrainingGaps = trained.Gaps,
                    WindowRows = served.Held,
                    WindowGaps = served.Gaps,
                };
            }

            var moments = ViewedDeclaration.Of(prepared.Declaration).OrderedBy
                .Where(column => table.Has(column) && table[column] is Column<DateTime>)
                .Select(column => (Column<DateTime>)table[column])
                .FirstOrDefault() is { } ordered
                ? rows.Select(row => ordered[row]).OfType<DateTime>().ToArray()
                : [];

            return new WindowLikeness(watch.WarnAbove, table.RowCount, [.. kept.List(WatchStep.KeptColumns).Select(Measured)])
            {
                From = moments.Length > 0 ? moments.Min() : null,
                To = moments.Length > 0 ? moments.Max() : null,
            };
        }
    }
}
