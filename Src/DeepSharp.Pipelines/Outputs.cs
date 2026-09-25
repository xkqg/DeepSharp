// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that names what a model is asked to predict: the output of the pipeline.
/// </summary>
/// <remarks>
/// A pipeline has one output, and the kind of output decides everything about the answer: which columns hold it,
/// what the handover hands over, what a row served later waits for. The kinds are the verbs that implement this
/// — one column, several, a column a number of rows ahead — and a new kind is a new verb, registered like any
/// other.
/// <para>
/// Naming the answer is not doing something to the data, so an output that only names its answers acts on
/// nothing, and it is the one kind of step the run is allowed to leave alone.
/// </para>
/// </remarks>
public interface INamesTheAnswer : IPipelineStep
{
    /// <summary>The columns that hold the answer, in their order, as they stand after this step.</summary>
    IReadOnlyList<string> Answers { get; }

    /// <summary>Why a row's answers cannot be handed over as this output's answers, when they cannot.</summary>
    /// <param name="answers">One row's answers, in the order <see cref="Answers"/> names them, each a finite number.</param>
    /// <returns>The reason, or nothing when the row may be handed over.</returns>
    /// <remarks>
    /// Nothing, for an output that takes any number as its answer. A kind of output that promises more — a
    /// distribution that sums to one, labels that are nought or one — says so here, and the handover asks it of
    /// every row it hands over: a model trained towards an answer its output could not have meant learns the
    /// wrong thing, and says nothing about it.
    /// </remarks>
    string? Refusal(IReadOnlyList<double> answers) => null;

    /// <summary>Whether this output makes its answer from the rows, rather than naming columns the rows bring.</summary>
    /// <remarks>
    /// An answer the rows bring is what a served row lacks and awaits; an answer made from later rows is one a
    /// served row cannot have, and awaits nothing from the rows handed in. Said by <see cref="IMakesTheAnswer"/>, and
    /// by nothing else.
    /// </remarks>
    bool MakesItsAnswer => false;
}

/// <summary>
/// An output that makes its answer from the rows, where the rows do not bring it.
/// </summary>
/// <remarks>
/// The price five days on is in the rows, five rows later; the answer is made from them where the output stands,
/// when the pipeline is fitted. A served row has no later rows, so there it is a gap: it is what is being asked.
/// </remarks>
public interface IMakesTheAnswer : IActsInAWalk, INamesTheAnswer
{
    /// <summary>Makes the answer columns from the rows as they stand.</summary>
    /// <param name="table">The data, changed in place.</param>
    void MakeAnswers(Table table);

    /// <inheritdoc />
    bool INamesTheAnswer.MakesItsAnswer => true;

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Answer(MakeAnswers, Answers);
}

/// <summary>What an answer read from later rows is.</summary>
public enum AheadAs
{
    /// <summary>The value itself, as it stands those rows later.</summary>
    Value,

    /// <summary>The return on the row's own value by then: the later value divided by the row's, less one.</summary>
    Return,
}

/// <summary>
/// Names the column a model is being asked to predict.
/// </summary>
/// <remarks>
/// The answer is not a feature, so it is taken out of what a model is shown and handed over separately. A
/// pipeline that left it among the inputs would produce a model that scores perfectly and knows nothing —
/// and the same mistake wears a quieter costume when a column merely restates the answer, which is what
/// declaring the columns is for.
/// </remarks>
public sealed record TargetStep : IPipelineStep<TargetStep>, INamesTheAnswer, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column a model is asked to predict, handed over apart from the numbers it is shown.", "answer", ColumnKinds.Any);

    /// <summary>Declares which column holds the answer.</summary>
    /// <param name="column">The column being predicted.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public TargetStep(string column) => Column = ColumnKey.Require(column);

    /// <summary>The column being predicted.</summary>
    public string Column { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => [Column];

    /// <inheritdoc />
    public static string Name => "target";

    /// <inheritdoc />
    public static string Purpose => "Names the column a model is asked to predict, which is handed over apart from the numbers it is shown.";

    /// <inheritdoc />
    public static StepParameters<TargetStep> Parameters { get; } =
        new StepParameters<TargetStep>().With(ColumnKey, step => step.Column);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static TargetStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element));
}

/// <summary>
/// Names several columns that together hold one answer: how a whole is divided among them.
/// </summary>
/// <remarks>
/// A flock weighed in bands of fifty grams is one answer of seventy numbers, the share of its birds in each band, and
/// the order of the bands is part of it. Every row's shares are at least nought and sum to one, which the handover
/// asks of each row it hands over: a row that does not was counted rather than divided, and dividing each row by its
/// sum above this — <c>normalise.row</c> with L1 — makes it one.
/// <para>
/// Named with the column that says how many there were, the shares come back as how many fell in each part: each
/// share times that column, as its row was read. A served row cannot divide its own counts back, since the counts
/// are what it asks for, but it does know how many birds its flock has. The run tries the way back on every row it
/// was fitted on, so bands that do not add up to their flock are refused there.
/// </para>
/// </remarks>
public sealed record DistributionStep : IPipelineStep<DistributionStep>, INamesTheAnswer, IUndoesItself, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns the answer is divided among, in their order: at least two.", ["share1", "share2"], ColumnKinds.Numbers);

    private static readonly ColumnParameter ScaleByKey = new(
        "scaleBy",
        "The column saying how many the shares are shares of, so predictions come back as how many fell in each; left out, they come back as shares.",
        "count",
        ColumnKinds.Numbers,
        optional: true);

    /// <summary>Declares the columns an answer is divided among.</summary>
    /// <param name="columns">The columns, in their order.</param>
    /// <param name="scaleBy">The column saying how many the shares are shares of, or nothing to have shares come back as shares.</param>
    /// <exception cref="ArgumentException">
    /// There are fewer than two columns, one has no name, or one is named twice; or the column the shares are shares of
    /// is one of them.
    /// </exception>
    public DistributionStep(IEnumerable<string> columns, string? scaleBy = null)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);

        if (Columns.Count < 2)
        {
            throw new ArgumentException(
                $"'{ColumnsKey.Key}' names at least two columns: an answer held in one column is a target.", nameof(columns));
        }

        ScaleBy = ScaleByKey.Require(scaleBy ?? string.Empty) is { Length: > 0 } named ? named : null;

        if (ScaleBy is not null && Columns.Contains(ScaleBy, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                $"'{ScaleByKey.Key}' says how many the shares are shares of, and '{ScaleBy}' is one of the shares.", nameof(scaleBy));
        }
    }

    /// <summary>The columns the answer is divided among, in their order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The column saying how many the shares are shares of, when there is one.</summary>
    public string? ScaleBy { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => Columns;

    /// <inheritdoc />
    /// <remarks>The first of the columns; a way back asks <see cref="Undoes"/> of each of them.</remarks>
    public string Produces => Columns[0];

    /// <inheritdoc />
    public static string Name => "target.distribution";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the columns a model is asked to predict as one answer: how a whole is divided among them, in their order.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<DistributionStep> Parameters { get; } = new StepParameters<DistributionStep>()
        .With(ColumnsKey, step => step.Columns)
        .With(ScaleByKey, step => step.ScaleBy ?? string.Empty);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public string? Refusal(IReadOnlyList<double> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);

        if (answers.FirstOrDefault(share => share < 0) is var below and < 0)
        {
            return string.Create(CultureInfo.InvariantCulture, $"a share of {below} is below nought, and a share of a whole never is.");
        }

        var sum = answers.Sum();

        return Math.Abs(sum - 1) <= 1e-9 * answers.Count
            ? null
            : string.Create(
                CultureInfo.InvariantCulture,
                $"its shares sum to {sum}, not to one. Divide each row by its sum above this: normalise.row with L1.");
    }

    /// <inheritdoc />
    /// <remarks>Every one of its columns, when it names what the shares are shares of; none otherwise.</remarks>
    public bool Undoes(string column) => ScaleBy is not null && Columns.Contains(column, StringComparer.Ordinal);

    /// <inheritdoc />
    /// <remarks>How many a share stands for depends on its row, so this refuses: hand the rows over with the predictions.</remarks>
    public double Undo(double value, FittedStepValues? fitted) =>
        ScaleBy is null
            ? value
            : throw new InvalidOperationException(
                $"A share comes back as how many it stands for only with the row it belongs to, which says how many '{ScaleBy}' there were. "
                + "Hand the rows over with the predictions.");

    /// <inheritdoc />
    public double Undo(double value, FittedStepValues? fitted, RowAsRead row) =>
        ScaleBy is null
            ? value
            : value * (row[ScaleBy] ?? throw new InvalidOperationException(
                $"Row {row.ReadAt + 1} has no '{ScaleBy}', so its shares cannot come back as how many fell in each."));

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(DistributionStep? other) =>
        other is not null && ScaleBy == other.ScaleBy && Columns.SequenceEqual(other.Columns, StringComparer.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(ScaleBy);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DistributionStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element), ScaleByKey.Read(element));
}

/// <summary>
/// Names several columns that together hold one answer: labels, each nought or one on every row.
/// </summary>
/// <remarks>
/// Which of several things a row is, or which of several things hold for it: a row holds one label when its things
/// exclude each other, and any number of them when they do not, and the output says which it means. The handover
/// refuses a row with a label that is neither nought nor one, or with more or fewer ones than the output says a row
/// holds. Labels are in their own units, so they come back as they were handed over.
/// </remarks>
public sealed record LabelsStep : IPipelineStep<LabelsStep>, INamesTheAnswer, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns holding the labels, each nought or one on every row: at least two.", ["label1", "label2"], ColumnKinds.Numbers);

    private static readonly WholeNumberParameter OnesKey = new(
        "ones",
        "How many of the columns hold a one on every row: one when a row is exactly one of its things; left out, any number.",
        0,
        atLeast: 0,
        leftOut: 0);

    /// <summary>Declares the columns holding an answer of labels.</summary>
    /// <param name="columns">The columns, in their order.</param>
    /// <param name="ones">How many of them hold a one on every row, or nought for any number.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The number of ones is below nought, or above the number of columns: a row cannot hold more ones than it has labels.
    /// </exception>
    public LabelsStep(IEnumerable<string> columns, int ones = 0)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);

        if (Columns.Count < 2)
        {
            throw new ArgumentException(
                $"'{ColumnsKey.Key}' names at least two columns: an answer held in one column is a target.", nameof(columns));
        }

        Ones = OnesKey.Require(ones);

        if (Ones > Columns.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ones), ones,
                string.Create(CultureInfo.InvariantCulture, $"'{OnesKey.Key}' says a row holds {Ones} ones, and there are only {Columns.Count} labels."));
        }
    }

    /// <summary>The columns holding the labels, in their order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How many of the columns hold a one on every row; nought for any number.</summary>
    public int Ones { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => Columns;

    /// <inheritdoc />
    public static string Name => "target.labels";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the columns a model is asked to predict as one answer of labels, each nought or one on every row.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<LabelsStep> Parameters { get; } = new StepParameters<LabelsStep>()
        .With(ColumnsKey, step => step.Columns)
        .With(OnesKey, step => step.Ones);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public string? Refusal(IReadOnlyList<double> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);

        if (answers.FirstOrDefault(label => label is not (0 or 1), 0) is var odd and not 0)
        {
            return string.Create(CultureInfo.InvariantCulture, $"a label of {odd} is neither nought nor one.");
        }

        var held = answers.Count(label => label == 1);

        return Ones > 0 && held != Ones
            ? string.Create(CultureInfo.InvariantCulture, $"it holds {held} ones, and a row holds {Ones} of these labels.")
            : null;
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(LabelsStep? other) =>
        other is not null && Ones == other.Ones && Columns.SequenceEqual(other.Columns, StringComparer.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Ones);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static LabelsStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element), OnesKey.Read(element));
}

/// <summary>
/// Names an answer read from a column a number of rows later, in the declared order: the value then, or the return
/// on the row's own value by then.
/// </summary>
/// <remarks>
/// The answer is made where the output stands, into a column of its own named after the column and how far ahead,
/// and the last rows, with nothing that far after them, have none. It stands below a split in time whose gap is at
/// least as wide as how far it reads, the rows ordered by the column the split divides by alone.
/// <para>
/// It comes back as a value of the column it was read from: a value by the steps above it that changed that column,
/// a return by the row's own value as it was read — which is why a return stands above every step that changes the
/// column it is made from.
/// </para>
/// </remarks>
public sealed record AheadStep : IPipelineStep<AheadStep>, IMakesTheAnswer, IReadsRowsAhead, IUndoesItself, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column the answer is read from, rows later.", "close", ColumnKinds.Numbers);

    private static readonly WholeNumberParameter AheadKey = new(
        "ahead", "How many rows later the answer is read, in the declared order: at least one.", 1, atLeast: 1);

    private static readonly OneOfParameter<AheadAs> AsKey = new(
        "as", "What the answer is: the value itself then, or the return on the row's own value by then.", AheadAs.Value);

    /// <summary>Declares an answer read from a column rows later.</summary>
    /// <param name="column">The column the answer is read from.</param>
    /// <param name="ahead">How many rows later, at least one.</param>
    /// <param name="as">The value itself, or the return on the row's own value.</param>
    /// <exception cref="ArgumentException">The column has no name, or the form is not one of the names.</exception>
    /// <exception cref="ArgumentOutOfRangeException">It reads less than one row ahead.</exception>
    public AheadStep(string column, int ahead, AheadAs @as = AheadAs.Value)
    {
        Column = ColumnKey.Require(column);
        Ahead = AheadKey.Require(ahead);
        As = AsKey.Require(@as);
    }

    /// <summary>The column the answer is read from.</summary>
    public string Column { get; }

    /// <inheritdoc />
    public int Ahead { get; }

    /// <summary>Whether the answer is the value itself or the return on the row's own value.</summary>
    public AheadAs As { get; }

    /// <summary>
    /// Whether the answer is made from the column as it was read — a return is — so it stands above every step that
    /// changes that column: said once, for the rule that holds it and for whatever places the step.
    /// </summary>
    internal bool IsMadeFromItsColumnAsRead => As == AheadAs.Return;

    /// <summary>The column the answer is made into: the column's name and how far ahead.</summary>
    public string Answer => string.Create(CultureInfo.InvariantCulture, $"{Column}.ahead{Ahead}");

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => [Answer];

    /// <inheritdoc />
    public string Produces => Answer;

    /// <inheritdoc />
    public static string Name => "target.ahead";

    /// <inheritdoc />
    public static string Purpose =>
        "Names an answer read from a column rows later in the declared order: the value then, or the return on the row's own value by then.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<AheadStep> Parameters { get; } = new StepParameters<AheadStep>()
        .With(ColumnKey, step => step.Column)
        .With(AheadKey, step => step.Ahead)
        .With(AsKey, step => step.As);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void MakeAnswers(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var values = table.NumbersOf(Column);
        var answers = new double?[values.Length];

        for (var row = 0; row + Ahead < values.Length; row++)
        {
            if (values[row + Ahead] is not { } later)
            {
                continue;
            }

            // A return on nothing, or on a gap, is not a number: that row has no answer.
            answers[row] = As == AheadAs.Value
                ? later
                : values[row] is { } now && now != 0 ? (later / now) - 1 : null;
        }

        table.Put(new Column<double>(Answer, ColumnKind.Number, answers));
    }

    /// <inheritdoc />
    /// <remarks>The column the answer was read from: whatever changed that column above is undone next.</remarks>
    public string From(string column) => Column;

    /// <inheritdoc />
    /// <remarks>A value is itself; a return comes back as a value only by the row's own value, so it refuses here.</remarks>
    public double Undo(double value, FittedStepValues? fitted) =>
        As == AheadAs.Value
            ? value
            : throw new InvalidOperationException(
                $"A return comes back as a value of '{Column}' only with the row it was made on, whose '{Column}' it is a return on. "
                + "Hand the rows over with the predictions.");

    /// <inheritdoc />
    public double Undo(double value, FittedStepValues? fitted, RowAsRead row) =>
        As == AheadAs.Value
            ? value
            : (row[Column] ?? throw new InvalidOperationException(
                $"Row {row.ReadAt + 1} has no '{Column}', so a return on it cannot come back as a value.")) * (1 + value);

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Answer, ColumnKind.Number);
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static AheadStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), AheadKey.Read(element), AsKey.Read(element));
}
