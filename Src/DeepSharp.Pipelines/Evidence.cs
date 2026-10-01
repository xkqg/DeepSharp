// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How a piece of evidence is shown: drawn, or as a grid of numbers.</summary>
public enum Shown
{
    /// <summary>Drawn: a correlation as a coloured grid, a profile as bars, a model's measures as charts.</summary>
    Drawn,

    /// <summary>As the numbers themselves.</summary>
    Numbers,
}

/// <summary>
/// A step that produces evidence about the rows where it stands, and changes nothing.
/// </summary>
/// <remarks>
/// Declared with the pipeline, before the numbers exist, so every run produces the same evidence and the
/// report cannot shrink to whatever happened to look good. It is measured when the pipeline is fitted, on the
/// rows the split trains on — the split below it as much as one above — and not again when the pipeline is
/// replayed. What it produces is output, kept with the run and never written into the pipeline's file.
/// </remarks>
public interface IProducesEvidence : IActsInAWalk
{
    /// <summary>The evidence about the rows as they stand here.</summary>
    /// <param name="view">The rows here, and where each stands.</param>
    /// <returns>The evidence.</returns>
    Evidence Produce(PipelineView view);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Evidence(Produce);
}

/// <summary>
/// Something a run shows as proof: a profile of the columns, the rows a correlation is drawn from.
/// </summary>
public abstract class Evidence
{
    private protected Evidence()
    {
    }

    /// <summary>Hands this evidence to whatever shows it, as the kind it is.</summary>
    /// <typeparam name="TResult">What the visitor builds.</typeparam>
    /// <param name="visitor">The visitor.</param>
    /// <returns>What it built.</returns>
    public abstract TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor);
}

/// <summary>Something built from each kind of evidence: a picture, a table of numbers.</summary>
/// <typeparam name="TResult">What it builds.</typeparam>
public interface IEvidenceVisitor<out TResult>
{
    /// <summary>A profile of the columns.</summary>
    /// <param name="profile">The profile.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(DataProfile profile);

    /// <summary>The rows a correlation is drawn from.</summary>
    /// <param name="correlation">The rows, and how many were kept.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(CorrelationInput correlation);

    /// <summary>The measures a trained model's predictions were held to, part by part, each beside predicting the training rows' average.</summary>
    /// <param name="measures">The measures.</param>
    /// <returns>What the visitor builds for them.</returns>
    TResult Visit(Measures measures);
}

/// <summary>One column, as a profile measures it on the measured rows.</summary>
/// <param name="Name">The column's name.</param>
/// <param name="Kind">What it holds.</param>
/// <param name="Rows">How many rows were measured.</param>
/// <param name="Gaps">How many of them are gaps.</param>
/// <param name="NotFinite">How many hold a value that is not a finite number.</param>
/// <param name="Distinct">How many different values there are, a gap not counted.</param>
/// <param name="Constant">Whether every value there is is the same one.</param>
/// <param name="Min">The smallest number, for a column of numbers with any.</param>
/// <param name="Max">The largest number, for a column of numbers with any.</param>
/// <param name="Mean">The average, for a column of numbers with any.</param>
/// <param name="Median">The middle value, for a column of numbers with any.</param>
public readonly record struct ColumnProfile(
    string Name,
    ColumnKind Kind,
    int Rows,
    int Gaps,
    int NotFinite,
    int Distinct,
    bool Constant,
    double? Min,
    double? Max,
    double? Mean,
    double? Median);

/// <summary>What answering an alert does.</summary>
public enum AlertAction
{
    /// <summary>A step is written for the column, one the column rules keep for its kind: its verb says which.</summary>
    Step,

    /// <summary>
    /// The column is left out, as the column rules leave one out: excluded in the schema when it is declared and no step
    /// names it — a category the step encoding every category turns into columns among them — and dropped after the last
    /// step that reads it otherwise.
    /// </summary>
    LeaveOut,

    /// <summary>The schema says a value stands for a gap in the column, so every cell holding it is read as one.</summary>
    SayMissing,
}

/// <summary>How an alert is answered.</summary>
/// <param name="Action">What answering it does.</param>
/// <param name="Column">The column it is done to.</param>
/// <param name="Verb">For a step, its verb; nothing otherwise.</param>
/// <param name="Value">For a value said to stand for a gap, the value; nothing otherwise.</param>
public readonly record struct AlertAnswer(AlertAction Action, string Column, string? Verb = null, string? Value = null)
{
    /// <summary>A step written for a column.</summary>
    /// <param name="verb">The step's verb.</param>
    /// <param name="column">The column.</param>
    /// <returns>The answer.</returns>
    public static AlertAnswer Step(string verb, string column) => new(AlertAction.Step, column, verb);

    /// <summary>A column left out.</summary>
    /// <param name="column">The column.</param>
    /// <returns>The answer.</returns>
    public static AlertAnswer LeaveOut(string column) => new(AlertAction.LeaveOut, column);

    /// <summary>A value the schema says stands for a gap in a column.</summary>
    /// <param name="column">The column.</param>
    /// <param name="value">The value.</param>
    /// <returns>The answer.</returns>
    public static AlertAnswer SayMissing(string column, string value) => new(AlertAction.SayMissing, column, Value: value);

    /// <summary>The steps with this answer given, when giving it is a change to the columns.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <returns>
    /// The steps with the column left out, as the column rules leave one out, or with the value said on the schema; the
    /// steps as they are for a step to be written, which a person places where it belongs.
    /// </returns>
    /// <exception cref="DeclarationException">
    /// The rules refuse the answer: the column is the last one the schema takes, or the schema does not name it.
    /// </exception>
    public IReadOnlyList<IPipelineStep> AppliedTo(PipelineDeclaration declaration)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        return Action switch
        {
            AlertAction.LeaveOut => declaration.Excluding(Column),
            AlertAction.SayMissing => declaration.WithMissing(Column, Value),
            _ => declaration.Steps,
        };
    }
}

/// <summary>Something a profile found, the columns it is about, and how it is answered.</summary>
/// <param name="Columns">
/// The columns it is about: the one it was found in, then, for a column that says again what another says or hands a
/// model the answer, that other column.
/// </param>
/// <param name="Says">What it found, in words, with what it counted.</param>
/// <param name="Answer">How it is answered.</param>
public readonly record struct ProfileAlert(IReadOnlyList<string> Columns, string Says, AlertAnswer Answer)
{
    /// <summary>The column it was found in.</summary>
    public string Column => Columns[0];

    /// <inheritdoc />
    public bool Equals(ProfileAlert other) => Columns.SequenceEqual(other.Columns) && Says == other.Says && Answer == other.Answer;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Column, Columns.Count, Says, Answer);
}

/// <summary>
/// A profile of the columns where it stands: what they hold, what is wrong with them or should not be there, and how
/// each is answered.
/// </summary>
public sealed class DataProfile : Evidence
{
    internal DataProfile(PipelineView view, IReadOnlyList<ColumnProfile> columns, IReadOnlyList<ProfileAlert> alerts, DuplicateRows duplicates)
    {
        Over = view.Measured;
        Rows = view.MeasuredRows().Length;
        Columns = columns;
        Alerts = alerts;
        Duplicates = duplicates;
    }

    /// <summary>The rows it measured: the training rows, or the undivided ones in a pipeline with no split.</summary>
    public Standing Over { get; }

    /// <summary>How many rows it measured.</summary>
    public int Rows { get; }

    /// <summary>Each column it profiled.</summary>
    public IReadOnlyList<ColumnProfile> Columns { get; }

    /// <summary>What it found, each with how it is answered: a step, the column left out, or a value said to stand for a gap.</summary>
    public IReadOnlyList<ProfileAlert> Alerts { get; }

    /// <summary>The rows that are there more than once, among all the rows where it stands.</summary>
    public DuplicateRows Duplicates { get; }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>
/// The rows a correlation is drawn from: the measured rows with a finite number in every column named.
/// </summary>
/// <remarks>
/// The correlation itself is worked out by whatever draws it. What the pipeline says is which rows it may be
/// worked out from, how many of the measured rows that is, and the rule that left the others out — because a
/// correlation over the rows that happened to be complete is a different number from one over all of them.
/// </remarks>
public sealed class CorrelationInput : Evidence
{
    internal CorrelationInput(PipelineView view, IReadOnlyList<string> columns, IReadOnlyList<double[]> rows, Shown shown)
    {
        Over = view.Measured;
        Columns = columns;
        Rows = rows;
        Total = view.MeasuredRows().Length;
        Shown = shown;
    }

    /// <summary>Which rows it was drawn from: the training rows, or the undivided ones when nothing divides them.</summary>
    public Standing Over { get; }

    /// <summary>The columns, in the order every row lists them.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The complete rows, each with one number per column.</summary>
    public IReadOnlyList<double[]> Rows { get; }

    /// <summary>How many rows were kept.</summary>
    public int Kept => Rows.Count;

    /// <summary>How many rows were measured, kept or not.</summary>
    public int Total { get; }

    /// <summary>The rule that decided which rows were kept.</summary>
    public string Policy => "complete rows";

    /// <summary>How it is shown.</summary>
    public Shown Shown { get; }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>
/// Profiles the columns where it stands, on the rows the split trains on.
/// </summary>
/// <remarks>
/// For each column: how many rows, gaps and values that are not numbers; how many different values; for
/// numbers, the smallest, the largest, the average and the middle. And what is wrong, each thing with how it is
/// answered: a gap by a step the column rules keep for the column's kind — <c>fill.missing</c> among numbers, the
/// encoder among words, which makes a gap no category and marks it, <c>drop.gaps</c> otherwise; a value that is not
/// a number by <c>fill.nan</c>; words by an encoder. A column is left out when it never changes, when it only names
/// its row — words no two rows share, or a running number, unless the rows are put in order or divided by it — when
/// it hands a model the answer, going with it value for value, or when it says again what a column before it says.
/// A number far from every other, held by more than one row and written as files write that nothing is known — 0,
/// −1, nines — is answered by the schema saying it stands for a gap. Two columns go with each other value for value
/// only where their values repeat, each held by two rows or more on average: columns that never repeat a value would
/// pair up by chance. An extreme is not something that should not be there: it is measured, and a step that clips
/// or refuses one is declared.
/// </remarks>
public sealed record ProfileStep : IPipelineStep<ProfileStep>, IProducesEvidence, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to profile; left out, every column where the step stands.", ["column"], ColumnKinds.Any)
    {
        Optional = true,
    };

    /// <summary>Declares a profile of these columns, or of every column where it stands.</summary>
    /// <param name="columns">The columns to profile; none, for every column.</param>
    /// <exception cref="ArgumentException">A column has no name, or one is named twice.</exception>
    public ProfileStep(IEnumerable<string>? columns = null) =>
        Columns = ColumnsKey.Require([.. columns ?? []]);

    /// <summary>The columns to profile; empty for every column where the step stands.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "evidence.profile";

    /// <inheritdoc />
    public static string Purpose => "Profiles the columns where it stands, on the rows the split trains on, and names what is wrong or should not be there, with how each is answered.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<ProfileStep> Parameters { get; } =
        new StepParameters<ProfileStep>().With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(ProfileStep? other) => other is not null && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public Evidence Produce(PipelineView view)
    {
        ArgumentNullException.ThrowIfNull(view);

        var table = view.Table;
        var measured = view.MeasuredRows();
        var names = Columns.Count > 0 ? Columns : [.. table.Columns.Select(column => column.Name)];
        var profiles = names.Select(name => Profiled(view, table[name], measured)).ToArray();
        ProfileAlert[] alerts = [.. profiles.SelectMany(profile => Alerts(view, profile)), .. Pairs(view, names, measured)];

        return new DataProfile(view, profiles, alerts, Duplicates(view));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ProfileStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));

    private static ColumnProfile Profiled(PipelineView view, IColumn column, int[] measured)
    {
        var gaps = measured.Count(column.IsMissing);
        var distinct = measured.Where(row => !column.IsMissing(row)).Select(column.TextAt).Distinct(StringComparer.Ordinal).Count();

        if (column.Kind is ColumnKind.Text or ColumnKind.Category or ColumnKind.Timestamp)
        {
            return new ColumnProfile(column.Name, column.Kind, measured.Length, gaps, 0, distinct, distinct == 1, null, null, null, null);
        }

        var values = view.MeasuredValues(column.Name);
        var any = values.Finite.Count > 0;

        return new ColumnProfile(
            column.Name, column.Kind, measured.Length, gaps, values.NotFinite, distinct, distinct == 1,
            any ? values.Finite[0] : null,
            any ? values.Finite[^1] : null,
            any ? values.Mean : null,
            any ? values.Median : null);
    }

    private static IEnumerable<ProfileAlert> Alerts(PipelineView view, ColumnProfile column)
    {
        if (column.Gaps > 0)
        {
            yield return One(column, $"{column.Gaps} of {column.Rows} rows are gaps.", AlertAnswer.Step(GapStep(column.Kind), column.Name));
        }

        if (column.NotFinite > 0)
        {
            yield return One(
                column, $"{column.NotFinite} of {column.Rows} rows hold a value that is not a finite number.", AlertAnswer.Step("fill.nan", column.Name));
        }

        if (column.Constant)
        {
            yield return One(column, "Every row holds the same value, so it tells a model nothing.", AlertAnswer.LeaveOut(column.Name));
        }

        if (column.Kind == ColumnKind.Category)
        {
            yield return One(
                column, "It holds groups as words, which a model cannot take as they are.", AlertAnswer.Step("encode.categories", column.Name));
        }

        if (column.Kind == ColumnKind.Text)
        {
            yield return One(column, "It holds words, which a model cannot take as they are.", AlertAnswer.Step("encode", column.Name));
        }

        if (NamesItsRow(view, column))
        {
            yield return One(
                column,
                $"Every one of the {column.Rows - column.Gaps} values is a row's own, as an identifier's is: it tells a model which row it is, not what the row holds.",
                AlertAnswer.LeaveOut(column.Name));
        }

        foreach (var nothingKnown in NothingKnown(view, column))
        {
            yield return nothingKnown;
        }
    }

    private static ProfileAlert One(ColumnProfile column, string says, AlertAnswer answer) => new([column.Name], says, answer);

    // The step that answers a gap, by the rule each step keeps for the kinds it takes: a fill where it can fill, the
    // encoder among words, which makes a gap no category and marks it, and dropping the rows otherwise.
    private static string GapStep(ColumnKind kind) =>
        ColumnKinds.Fillable.Contains(kind) ? "fill.missing"
        : kind == ColumnKind.Category ? "encode.categories"
        : kind == ColumnKind.Text ? "encode"
        : "drop.gaps";

    // Words no two rows share, or a running number, in a column the rows are neither put in order nor divided by.
    private static bool NamesItsRow(PipelineView view, ColumnProfile column)
    {
        var values = column.Rows - column.Gaps;

        return values >= 2 && column.Distinct == values && !view.OrderedBy.Contains(column.Name, StringComparer.Ordinal) && column.Kind switch
        {
            ColumnKind.Text => true,
            ColumnKind.Integer => column.Max.GetValueOrDefault() - column.Min.GetValueOrDefault() + 1 == values,
            _ => false,
        };
    }

    // How far from every other a number is, as a multiple of the step the values usually take from one to the next.
    private const double Far = 10;

    // The smallest and the largest number, each where more than one row holds it, it is written as files write that
    // nothing is known, and it lies far from every other value.
    private static IEnumerable<ProfileAlert> NothingKnown(PipelineView view, ColumnProfile column)
    {
        if (column.Kind is not (ColumnKind.Number or ColumnKind.Integer))
        {
            yield break;
        }

        var held = view.MeasuredValues(column.Name).Finite.GroupBy(value => value).Select(group => new Held(group.Key, group.Count())).ToArray();

        if (held.Length < 3)
        {
            yield break;
        }

        double[] steps = [.. held.Skip(1).Select((value, at) => value.Value - held[at].Value).Order()];
        var usual = (steps[(steps.Length - 1) / 2] + steps[steps.Length / 2]) / 2;

        End[] ends = [new(held[0], held[1], "smallest"), new(held[^1], held[^2], "largest")];

        foreach (var end in ends.Where(end => end.At.Rows >= 2 && WritesNothingKnown(end.At.Value) && Math.Abs(end.At.Value - end.Next.Value) > Far * usual))
        {
            var written = end.At.Value.ToString("R", CultureInfo.InvariantCulture);

            yield return One(
                column,
                $"{end.At.Rows} of {column.Rows} rows hold {written}, the {end.Word} value and far from every other: if {written} is how the file "
                + "writes that nothing is known, the schema can say so, and those rows are read as gaps.",
                AlertAnswer.SayMissing(column.Name, written));
        }
    }

    // 0, -1, and a run of nines either way: the values files write where nothing is known.
    private static bool WritesNothingKnown(double value) =>
        value is 0 or -1 || (double.IsInteger(value) && Math.Abs(value) >= 9 && Math.Abs(value).ToString("R", CultureInfo.InvariantCulture).All(digit => digit == '9'));

    // Every column that hands a model the answer, value for value, and every one that says again what a column before it
    // says; each once, against the answer or the first column it repeats.
    private static IEnumerable<ProfileAlert> Pairs(PipelineView view, IReadOnlyList<string> names, int[] measured)
    {
        var table = view.Table;
        string[] answers = [.. view.Answers.Where(table.Has)];
        var coded = names.Concat(answers).Distinct(StringComparer.Ordinal)
            .Select(name => Coded.Of(table[name], measured))
            .Where(column => column.Repeats)
            .ToDictionary(column => column.Name, StringComparer.Ordinal);

        foreach (var name in names.Where(name => coded.ContainsKey(name) && !answers.Contains(name, StringComparer.Ordinal)))
        {
            var column = coded[name];

            if (answers.Select(coded.GetValueOrDefault).FirstOrDefault(answer => answer is not null && column.GoesWith(answer)) is { } restated)
            {
                yield return new ProfileAlert(
                    [name, restated.Name],
                    $"Each of its values goes with one value of {restated.Name}, the answer, and each answer with one of its values, on all "
                    + $"{measured.Length} rows — {column.Pairs(restated)} — so it hands a model the answer.",
                    AlertAnswer.LeaveOut(name));

                continue;
            }

            var before = names.TakeWhile(other => other != name).Where(other => !answers.Contains(other, StringComparer.Ordinal));

            if (before.Select(coded.GetValueOrDefault).FirstOrDefault(other => other is not null && column.GoesWith(other)) is { } repeated)
            {
                yield return new ProfileAlert(
                    [name, repeated.Name],
                    $"Each of its values goes with one value of {repeated.Name} and each of those with one of its, on all {measured.Length} rows — "
                    + $"{column.Pairs(repeated)} — so it says again what {repeated.Name} says.",
                    AlertAnswer.LeaveOut(name));
            }
        }
    }

    // A number, and how many of the measured rows hold it.
    private readonly record struct Held(double Value, int Rows);

    // One end of a column's numbers: the value there, the one next to it, and the word for which end it is.
    private readonly record struct End(Held At, Held Next, string Word);

    // The part of each standing: the one of the same name, and none for a row dropped before the split reaches it.
    private static readonly Part[] PartOf =
        [.. Enum.GetValues<Standing>().Select(standing => Enum.TryParse<Part>(standing.ToString(), out var part) ? part : Part.Undivided)];

    // Among every row where it stands; across parts only where there are parts to be across.
    private static DuplicateRows Duplicates(PipelineView view) =>
        view.Measured == Standing.Undivided
            ? view.Table.Duplicates()
            : view.Table.Duplicates([.. view.Standings.Select(standing => PartOf[(int)standing])]);

    /// <summary>A column's values on the measured rows as numbers standing for them, a gap one of its own.</summary>
    private sealed class Coded
    {
        // How many measured values of a column one of its values stands for, at least, on average, before the column is
        // paired with another: columns that never repeat a value pair up by chance.
        private const int RowsPerValue = 2;

        private Coded(string name, int[] codes, IReadOnlyList<string?> values)
        {
            Name = name;
            Codes = codes;
            Values = values;
        }

        public string Name { get; }

        // Where each measured row's value stands among the values.
        public int[] Codes { get; }

        // The values, in the order they were first met; a gap is nothing.
        public IReadOnlyList<string?> Values { get; }

        public bool Repeats => Values.Count >= 2 && Values.Count * RowsPerValue <= Codes.Length;

        public static Coded Of(IColumn column, int[] measured)
        {
            var places = new Dictionary<string, int>(StringComparer.Ordinal);
            var values = new List<string?>();
            var gap = -1;
            var codes = new int[measured.Length];

            for (var at = 0; at < measured.Length; at++)
            {
                if (column.TextAt(measured[at]) is not { } text)
                {
                    if (gap < 0)
                    {
                        gap = values.Count;
                        values.Add(null);
                    }

                    codes[at] = gap;

                    continue;
                }

                if (!places.TryGetValue(text, out var place))
                {
                    places[text] = place = values.Count;
                    values.Add(text);
                }

                codes[at] = place;
            }

            return new(column.Name, codes, values);
        }

        // Whether each value goes with one value of the other column, and each of those with one of these.
        public bool GoesWith(Coded other)
        {
            var mine = Enumerable.Repeat(-1, Values.Count).ToArray();
            var theirs = Enumerable.Repeat(-1, other.Values.Count).ToArray();

            for (var at = 0; at < Codes.Length; at++)
            {
                var one = Codes[at];
                var two = other.Codes[at];

                if (mine[one] < 0 && theirs[two] < 0)
                {
                    mine[one] = two;
                    theirs[two] = one;
                }
                else if (mine[one] != two || theirs[two] != one)
                {
                    return false;
                }
            }

            return true;
        }

        // The values and the ones they go with, as a person reads them, a few at most.
        public string Pairs(Coded other)
        {
            const int Shown = 5;

            var pairs = Enumerable.Range(0, Values.Count)
                .Select(code => new Pair(Values[code], other.Values[other.Codes[Array.IndexOf(Codes, code)]], Codes.Count(each => each == code)))
                .OrderBy(pair => pair.Value ?? string.Empty, StringComparer.Ordinal)
                .ToArray();
            var shown = string.Join(", ", pairs.Take(Shown).Select(pair => $"{Said(pair.Value)} with {Said(pair.With)} in {pair.Rows}"));

            return pairs.Length > Shown ? $"{shown}, and {pairs.Length - Shown} more" : shown;
        }

        private static string Said(string? value) => value ?? "a gap";

        // One value, the value of the other column it goes with, and how many rows hold the two.
        private readonly record struct Pair(string? Value, string? With, int Rows);
    }
}

/// <summary>
/// Sets out the rows a correlation between columns is drawn from, on the rows the split trains on.
/// </summary>
/// <remarks>
/// Only the rows with a finite number in every column named, and it says how many of the measured rows that is
/// and by which rule: a correlation over the rows that happened to be complete is a different number from one
/// over all of them, and a picture that does not say so is a picture of something else.
/// </remarks>
public sealed record CorrelationStep : IPipelineStep<CorrelationStep>, IProducesEvidence, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to correlate with one another: two or more, each holding numbers.", ["left", "right"], ColumnKinds.Numbers);

    private static readonly OneOfParameter<Shown> ShownKey = new(
        "shown", "How it is shown: drawn as a coloured grid, or as the numbers themselves.", Shown.Drawn);

    /// <summary>Declares the rows a correlation between these columns is drawn from.</summary>
    /// <param name="columns">The columns: two or more, each holding numbers.</param>
    /// <param name="shown">How it is shown.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    public CorrelationStep(IEnumerable<string> columns, Shown shown = Shown.Drawn)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        Shown = ShownKey.Require(shown);

        // A rule on the number of columns, so it lives with the step.
        if (Columns.Count < 2)
        {
            throw new ArgumentException("A correlation is between two columns or more.", nameof(columns));
        }
    }

    /// <summary>The columns.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How it is shown.</summary>
    public Shown Shown { get; }

    /// <inheritdoc />
    public static string Name => "evidence.correlation";

    /// <inheritdoc />
    public static string Purpose => "Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<CorrelationStep> Parameters { get; } = new StepParameters<CorrelationStep>()
        .With(ColumnsKey, step => step.Columns)
        .With(ShownKey, step => step.Shown);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(CorrelationStep? other) => other is not null && Shown == other.Shown && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Shown);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public Evidence Produce(PipelineView view)
    {
        ArgumentNullException.ThrowIfNull(view);

        var table = view.Table;
        var values = Columns.Select(table.NumbersOf).ToArray();
        var measured = view.MeasuredRows();

        var complete = measured
            .Where(row => values.All(column => column[row] is { } value && double.IsFinite(value)))
            .Select(row => values.Select(column => column[row]!.Value).ToArray())
            .ToArray();

        return new CorrelationInput(view, Columns, complete, Shown);
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CorrelationStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element), ShownKey.Read(element));
}
