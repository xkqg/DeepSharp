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
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
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
    /// <exception cref="ArgumentOutOfRangeException">The number of ones is below nought.</exception>
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
