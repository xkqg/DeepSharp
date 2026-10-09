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
/// nothing, and the run leaves it alone — as it leaves a report, which names what those answers are measured by.
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

    /// <summary>Whether this output's answers can be classes, each nought or one, as the measures that count classes read them.</summary>
    /// <remarks>
    /// They can, unless the output says they are amounts: a share of a whole or a return is not a class, and a report
    /// that counts classes of one is refused where it is written. Where they can be, a row whose answer is neither nought
    /// nor one is refused when it is measured.
    /// </remarks>
    bool AnswersCanBeClasses => true;

    /// <summary>How many of this output's answers are one on every row, when they are classes; nought for any number.</summary>
    /// <remarks>
    /// Which classes a prediction names follows from it. One on every row, a row is exactly one of its things and the
    /// class predicted is the answer predicted likeliest, the first of equals; more, the answers predicted likeliest;
    /// any number, each answer is predicted by itself, at a half or more — a column of noughts and ones among them.
    /// </remarks>
    int Ones => 0;

    /// <summary>Whether this output's answers are shares of a whole in an order that means something, as bands of weight are.</summary>
    /// <remarks>
    /// They are not, unless the output says so: the shares of a flock among its farms have no first and last that a distance
    /// between them could follow. Where they are, a measure or a loss may say how far a
    /// prediction's shares lie from the answer's along that order — the earth mover's distance, the ranked probability score —
    /// and one that does is refused where it is written against any other output.
    /// </remarks>
    bool IsOrdered => false;
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
/// <para>
/// A flock that arrives with fewer birds than were planned does not add up to the planned flock, and the planned flock is
/// what a served row knows. Named with a remainder, the output makes one more answer, after the bands: what is left of the
/// whole once the bands are counted. It makes every answer from the counts as they were read — each band's count, and what
/// is left, over how many there were — so the bands hold counts where it stands, not shares, and nothing above it changes
/// them or the whole; the shares then sum to one, and come back as birds that add up to the planned flock. A flock that
/// arrives with more birds than were planned leaves less than nothing, which is no share of anything, and is refused,
/// naming its row. What is left stands outside the bands' order: a measure or a loss that follows the order compares the
/// bands alone.
/// </para>
/// </remarks>
public sealed record DistributionStep : IPipelineStep<DistributionStep>, IMakesTheAnswer, IUndoesItself, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey =
        ColumnsParameter.OfOneAnswer("The columns the answer is divided among, in their order: at least two.", ["share1", "share2"]);

    private static readonly ColumnParameter ScaleByKey = new(
        "scaleBy",
        "The column saying how many the shares are shares of, so predictions come back as how many fell in each; left out, they come back as shares.",
        ColumnKinds.Numbers);

    private static readonly TrueOrFalseParameter OrderedKey = new(
        "ordered",
        "Whether the columns are in an order that means something, as bands of weight are, so a prediction is measured by how far its shares lie from the answer's along it; left out, they are not.",
        false)
    {
        LeftOut = false,
    };

    private static readonly NewColumnParameter RemainderKey = new(
        "remainder",
        "The column made for what is left of the whole once the columns are counted, when fewer arrive than the column saying how many there were; left out, the columns hold the whole.");

    /// <summary>Declares the columns an answer is divided among.</summary>
    /// <param name="columns">The columns, in their order.</param>
    /// <param name="scaleBy">The column saying how many the shares are shares of, or nothing to have shares come back as shares.</param>
    /// <exception cref="ArgumentException">
    /// There are fewer than two columns, one has no name, or one is named twice; or the column the shares are shares of
    /// is one of them.
    /// </exception>
    public DistributionStep(IEnumerable<string> columns, string? scaleBy = null)
        : this(columns, scaleBy, ordered: false)
    {
    }

    /// <summary>Declares the columns an answer is divided among, whether they are in an order, and what is left of the whole.</summary>
    /// <param name="columns">The columns, in their order.</param>
    /// <param name="scaleBy">The column saying how many the shares are shares of, or nothing to have shares come back as shares.</param>
    /// <param name="ordered">Whether the columns are in an order that means something, as bands of weight are.</param>
    /// <param name="remainder">
    /// The column made for what is left of the whole once the columns are counted, or nothing when the columns hold the
    /// whole; it needs the column saying how many there were.
    /// </param>
    /// <exception cref="ArgumentException">
    /// There are fewer than two columns, one has no name, or one is named twice; the column the shares are shares of is one
    /// of them; or a remainder is named without it, or under the name of one of the columns or of it.
    /// </exception>
    public DistributionStep(IEnumerable<string> columns, string? scaleBy, bool ordered, string? remainder = null)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        ScaleBy = ScaleByKey.Require(scaleBy ?? string.Empty) is { Length: > 0 } named ? named : null;
        Ordered = OrderedKey.Require(ordered);
        Remainder = RemainderKey.Require(remainder);

        if (ScaleBy is not null && Columns.Contains(ScaleBy, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                $"'{ScaleByKey.Key}' says how many the shares are shares of, and '{ScaleBy}' is one of the shares.", nameof(scaleBy));
        }

        if (Remainder is null)
        {
            return;
        }

        if (ScaleBy is null)
        {
            throw new ArgumentException(
                $"'{RemainderKey.Key}' is what is left of how many there were, and nothing says how many: name that column with '{ScaleByKey.Key}'.",
                nameof(remainder));
        }

        if (Columns.Contains(Remainder, StringComparer.Ordinal) || Remainder == ScaleBy)
        {
            throw new ArgumentException(
                $"'{RemainderKey.Key}' is a column of its own, made for what is left, and '{Remainder}' is one of the shares or '{ScaleByKey.Key}' itself.",
                nameof(remainder));
        }
    }

    /// <summary>The columns the answer is divided among, in their order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The column saying how many the shares are shares of, when there is one.</summary>
    public string? ScaleBy { get; }

    /// <summary>Whether the columns are in an order that means something, as bands of weight are.</summary>
    public bool Ordered { get; }

    /// <summary>The column made for what is left of the whole once the columns are counted, when there is one.</summary>
    public string? Remainder { get; }

    /// <inheritdoc />
    /// <remarks>The columns in their order, and after them what is left of the whole, when the output makes it.</remarks>
    public IReadOnlyList<string> Answers => Remainder is null ? Columns : [.. Columns, Remainder];

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
        .With(ScaleByKey, step => step.ScaleBy ?? string.Empty)
        .With(OrderedKey, step => step.Ordered)
        .With(RemainderKey, step => step.Remainder);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public bool IsOrdered => Ordered;

    /// <inheritdoc />
    /// <remarks>Only with a remainder, which no row brings: the bands are the rows' own, and a served row awaits them.</remarks>
    public bool MakesItsAnswer => Remainder is not null;

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
    /// <remarks>Never: a share of a whole is an amount, measured by how far it is from the share there was.</remarks>
    public bool AnswersCanBeClasses => false;

    /// <inheritdoc />
    /// <remarks>
    /// Nothing, without a remainder: the bands are the rows' own. With one, each band's share of the whole and what is left of
    /// it: the counts over how many there were, row by row. A row with a
    /// gap in a band or in the whole keeps its gaps, which the handover refuses; a row whose whole is nought, or whose bands
    /// hold more than it, is refused here.
    /// </remarks>
    /// <exception cref="InvalidOperationException">A row's bands hold more than its whole, or its whole is nought or less.</exception>
    public void MakeAnswers(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        // Without a remainder the bands are the rows' own shares, and there is nothing to make.
        if (Remainder is null)
        {
            return;
        }

        var whole = table.NumbersOf(ScaleBy!);
        var counts = Columns.Select(table.NumbersOf).ToArray();
        var shares = Columns.Select(_ => new double?[table.RowCount]).ToArray();
        var left = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (whole[row] is not { } total || counts.Any(band => band[row] is null))
            {
                continue;
            }

            var counted = counts.Sum(band => band[row]!.Value);

            if (total <= 0)
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {table.Identities[row].ReadAt + 1} says '{ScaleBy}' is {total}, and a whole of nought has no shares for '{Remainder}' to be what is left of."));
            }

            // A count is counted, so a whole a millionth short of its bands is the rounding of how it was written down.
            if (counted - total > 1e-9 * total)
            {
                throw new InvalidOperationException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {table.Identities[row].ReadAt + 1} holds {counted} in its bands, more than the {total} '{ScaleBy}' says there were: what is left would be below nothing, which is no share of anything."));
            }

            for (var band = 0; band < counts.Length; band++)
            {
                shares[band][row] = counts[band][row]!.Value / total;
            }

            left[row] = Math.Max(0, total - counted) / total;
        }

        for (var band = 0; band < Columns.Count; band++)
        {
            table.Put(new Column<double>(Columns[band], ColumnKind.Number, shares[band]));
        }

        table.Put(new Column<double>(Remainder!, ColumnKind.Number, left));
    }

    /// <inheritdoc />
    /// <remarks>Only an output that makes what is left of its whole acts on the rows; one that names the bands as they are leaves them alone.</remarks>
    void IActsInAWalk.ActOn(Walk walk)
    {
        if (Remainder is not null)
        {
            walk.Answer(MakeAnswers, Answers);
        }
    }

    /// <inheritdoc />
    /// <remarks>Every one of its answers, when it names what the shares are shares of; none otherwise.</remarks>
    public bool Undoes(string column) => ScaleBy is not null && Answers.Contains(column, StringComparer.Ordinal);

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
    /// <remarks>With a remainder, every band holds a share of a number where it stands, and what is left is one more column.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Remainder is null ? before : Answers.Aggregate(before, (state, answer) => state.With(answer, ColumnKind.Number));
    }

    /// <inheritdoc />
    public bool Equals(DistributionStep? other) =>
        other is not null
        && ScaleBy == other.ScaleBy
        && Ordered == other.Ordered
        && Remainder == other.Remainder
        && Columns.SequenceEqual(other.Columns, StringComparer.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(ScaleBy);
        hash.Add(Ordered);
        hash.Add(Remainder);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static DistributionStep ReadFrom(JsonElement element) =>
        new(ColumnsKey.Read(element), ScaleByKey.Read(element), OrderedKey.Read(element), RemainderKey.Read(element));
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
    private static readonly ColumnsParameter ColumnsKey =
        ColumnsParameter.OfOneAnswer("The columns holding the labels, each nought or one on every row: at least two.", ["label1", "label2"]);

    private static readonly WholeNumberParameter OnesKey = new(
        "ones",
        "How many of the columns hold a one on every row: one when a row is exactly one of its things; left out, any number.",
        0)
    {
        AtLeast = 0,
        LeftOut = 0,
    };

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
/// Names several columns that together hold one answer of free numbers, each any finite number.
/// </summary>
/// <remarks>
/// A flock's mean weight and its spread, predicted together: numbers that are neither shares of a whole nor classes, so
/// nothing is asked of a row beyond what the handover asks of every answer — no gap, and a finite number. They are amounts,
/// so a report that counts classes of them is refused where it is written, and they come back in their own units, through
/// whatever the steps above them did to each.
/// </remarks>
public sealed record NumbersStep : IPipelineStep<NumbersStep>, INamesTheAnswer, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey =
        ColumnsParameter.OfOneAnswer("The columns holding the numbers of the answer, in their order: at least two.", ["number1", "number2"]);

    /// <summary>Declares the columns holding an answer of free numbers.</summary>
    /// <param name="columns">The columns, in their order.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    public NumbersStep(IEnumerable<string> columns)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
    }

    /// <summary>The columns holding the numbers, in their order.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Answers => Columns;

    /// <inheritdoc />
    public static string Name => "target.numbers";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the columns a model is asked to predict as one answer of free numbers, each any finite number.";

    /// <inheritdoc />
    /// <remarks>New in the eighth version of the file.</remarks>
    public static int Since => 8;

    /// <inheritdoc />
    public static StepParameters<NumbersStep> Parameters { get; } = new StepParameters<NumbersStep>()
        .With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>Never: a free number is an amount, measured by how far it is from the number there was.</remarks>
    public bool AnswersCanBeClasses => false;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(NumbersStep? other) => other is not null && Columns.SequenceEqual(other.Columns, StringComparer.Ordinal);

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

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NumbersStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));
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
        "ahead", "How many rows later the answer is read, in the declared order: at least one.", 1)
    {
        AtLeast = 1,
    };

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
    /// <remarks>A value read ahead can be one, when the column holds noughts and ones; a return is an amount, and never.</remarks>
    public bool AnswersCanBeClasses => As == AheadAs.Value;

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

/// <summary>Which of an output's answers the rows bring, and which of them stand in its order.</summary>
internal static class AnswerExtensions
{
    extension(INamesTheAnswer output)
    {
        /// <summary>The answers the rows bring, which a served row lacks and awaits.</summary>
        /// <returns>
        /// Every answer of an output that names its columns; none of one that makes its whole answer; a distribution's bands,
        /// without what is left of its whole, which it makes.
        /// </returns>
        public IReadOnlyList<string> Brought() => output switch
        {
            DistributionStep distribution => distribution.Columns,
            { MakesItsAnswer: true } => [],
            _ => output.Answers,
        };

        /// <summary>How many of the answers, from the first, stand in the output's order.</summary>
        /// <returns>A distribution's bands, without what is left of its whole; every answer of any other output.</returns>
        public int InTheOrder() => output is DistributionStep distribution ? distribution.Columns.Count : output.Answers.Count;

        /// <summary>Whether this output's answers can be measured by the measures of a family.</summary>
        /// <param name="family">The family.</param>
        /// <returns>
        /// Amounts, always; classes, where its answers can be classes; shares, where they are a distribution's or in an order;
        /// shares in an order, where they are in one.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">The value is none of the families.</exception>
        /// <remarks>The one rule for it: where a report is written, and wherever a refusal says what the answers are measured by.</remarks>
        public bool Takes(MetricFamily family) => family switch
        {
            MetricFamily.Amounts => true,
            MetricFamily.Classes => output.AnswersCanBeClasses,
            MetricFamily.Shares => output.IsOrdered || output is DistributionStep,
            MetricFamily.OrderedShares => output.IsOrdered,
            _ => throw new ArgumentOutOfRangeException(nameof(family), family, "This is none of the families of measures."),
        };

        /// <summary>What this output's answers are measured by as shares, as a refusal adds it to the measures of amounts.</summary>
        /// <returns>", or as shares with kl", say; nothing, for answers that are no shares.</returns>
        public string SharesMeasured()
        {
            string[] words =
            [
                .. Enum.GetValues<Metric>()
                    .Where(metric => metric.ComparesShares() && output.Takes(metric.Family()))
                    .Select(metric => metric.Word()),
            ];

            return words switch
            {
                [] => string.Empty,
                [var only] => $", or as shares with {only}",
                _ => $", or as shares in their order with {string.Join(", ", words[..^1])} or {words[^1]}",
            };
        }
    }
}
