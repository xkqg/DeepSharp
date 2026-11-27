// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

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
        "What goes in the gaps: the mean or the median of the training rows, the value before the gap, nought, a constant, or refuse.",
        With.Median,
        ["mean", "median", "zero", "previous", "constant", "refuse"])
    {
        What = "filling a gap",
    };

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
    public string MarkerColumn => Column.Marked;

    /// <inheritdoc />
    /// <remarks>
    /// With a share it will not fill above, whether the column is still there is decided when the pipeline is
    /// fitted, so it is known but not sure from here on.
    /// </remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        var marked = before.Filled(Column, Strategy).With(MarkerColumn, ColumnKind.Number, Form.Unit);

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
    public static string Purpose => "Fills the gaps in a column the named way, below the split, and marks where they were. A value no row decided is settle.gaps, above it.";

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
        learned.Saw("gaps", training.Gaps);

        if (RefuseAbove is { } ceiling)
        {
            var share = training.Rows == 0 ? 0 : training.Gaps / (double)training.Rows;

            learned.Saw("share", share);
            learned.Decided("filled", share <= ceiling ? 1 : 0);

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

        table.Marks(Column);

        // Decided when the pipeline was fitted: too many of the training rows were gaps, so the column is not
        // filled at all, and the marker speaks for it.
        if (fitted.Choice("filled") is 0)
        {
            table.Remove(Column);

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
                    $"'{Column}' holds {column.Kind.Word()}, and a gap in it is not filled with a number.");
        }
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
