// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Deals with a value that is not a number a model can use: a not-a-number, or an infinity.
/// </summary>
/// <remarks>
/// Missing and not-a-number are different things and get different verbs. A value is missing when it was
/// never there, which is data; a value is not a number when arithmetic produced no usable number, which is a
/// fault further upstream — a division by nothing, a product too large to hold. So this refuses by default:
/// quietly replacing it would carry somebody else's broken division into a model and call it a measurement.
/// </remarks>
public sealed record FillNaNStep : IFittedStep, IPipelineStep<FillNaNStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to watch for values that are not numbers a model can use.", "column", ColumnKinds.Fractions);

    // Not previous: the row above a not-a-number says nothing about what it should have been.
    private static readonly FillStrategyParameter WithKey = new(
        "with",
        "What happens to a value that is not a number: refuse, which is the default and usually the right answer, or mean, median, zero or constant.",
        With.Refuse,
        ["refuse", "mean", "median", "zero", "constant"],
        "dealing with a value that is not a number");

    /// <summary>Declares what happens to a value in this column that is not a number.</summary>
    /// <param name="column">The column to watch.</param>
    /// <param name="strategy">What to do — refusing is the default and usually the right answer.</param>
    /// <exception cref="ArgumentException">The column has no name, or the strategy is not one of the names.</exception>
    public FillNaNStep(string column, FillStrategy strategy = default)
    {
        Column = ColumnKey.Require(column);
        Strategy = WithKey.Require(string.IsNullOrWhiteSpace(strategy.Name) ? With.Refuse : strategy);
    }

    /// <inheritdoc />
    public static StepParameters<FillNaNStep> Parameters { get; } = new StepParameters<FillNaNStep>()
        .With(ColumnKey, step => step.Column)
        .With(WithKey, step => step.Strategy);

    /// <summary>The column being watched.</summary>
    public string Column { get; }

    /// <summary>What happens to a value that is not a number.</summary>
    public FillStrategy Strategy { get; }

    /// <inheritdoc />
    public static string Name => "fill.nan";

    /// <inheritdoc />
    public static string Purpose => "Deals with a value that is not a number a model can use; refusing it is the default.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var training = table.TrainingValues(Column, parts);
        var learned = new FittedStepValues();

        // "Not a number" in the sense a model cares about: a not-a-number and an infinity alike. Both are
        // arithmetic that produced no usable value, and a mean with an infinity in it is an infinity. Counted
        // among the training rows, like every number in the fitted half.
        learned.Learned("notNumbers", training.NotFinite);

        // This is the step that deals with those values, so it learns from the finite ones rather than refusing.
        switch (Strategy.Name)
        {
            case "mean":
                learned.Learned("value", training.Measurable("a fill value").Mean);
                break;

            case "median":
                learned.Learned("value", training.Measurable("a fill value").Median);
                break;

            case "zero":
                learned.Learned("value", 0);
                break;

            case "constant":
                learned.Learned("value", Strategy.Value!.Value);
                break;

            default:
                break;
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        if (table[Column] is not Column<double> numbers)
        {
            throw new InvalidOperationException(
                $"'{Column}' holds {table[Column].Kind.ToString().ToLowerInvariant()}, "
                + "which cannot hold a value that is not a number.");
        }

        for (var row = 0; row < numbers.Count; row++)
        {
            if (numbers[row] is not { } value || double.IsFinite(value))
            {
                continue;
            }

            numbers[row] = Strategy.Name == "refuse"
                ? throw new InvalidOperationException(
                    $"Row {row + 1} of '{Column}' is not a number a model can use "
                    + $"({value.ToString(CultureInfo.InvariantCulture)}). Something upstream produced it, and this "
                    + "pipeline will not carry it into a model as if it were a measurement.")
                : fitted.Number("value");
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static FillNaNStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), WithKey.Read(element));
}
