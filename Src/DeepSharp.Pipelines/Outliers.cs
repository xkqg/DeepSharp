// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How the bounds an extreme value is held to are worked out.</summary>
public enum Bounds
{
    /// <summary>By quantile: hold the lowest and highest few per cent at the edge of the rest.</summary>
    Quantile,

    /// <summary>By spread: so many standard deviations either side of the middle.</summary>
    Sigma,

    /// <summary>By the middle half: so many interquartile ranges beyond the quartiles.</summary>
    Iqr,
}

/// <summary>What happens to a value outside the bounds.</summary>
public enum Outlier
{
    /// <summary>Hold it at the bound. The row survives and its other columns still count.</summary>
    Clip,

    /// <summary>Make it a gap, and let the step that fills gaps decide what goes there.</summary>
    Blank,

    /// <summary>Stop, and say which row and how far out it was.</summary>
    Refuse,
}

/// <summary>
/// The bounds of one line: how they are worked out, and the columns they hold for.
/// </summary>
/// <remarks>
/// The ways of finding the bounds are the methods, as the kinds of a scaling are, so a line reads as the sentence it is
/// and the set stays open-closed: a way added to <see cref="Bounds"/> adds a method here and changes no call site. Each
/// hands back one step a column, which is what the declaration, the file and a notebook's blocks have always held — the
/// line is a door, not a new shape. Every column is held at its bound. What happens to a value outside the bounds is a
/// fact about one column — a refusal belongs on the column that must never see one — and stays on
/// <see cref="FittingBuilder.ClipOutliers(string, Bounds, double, Outlier)"/>.
/// <para>
/// A distance left out is the one the step starts from, so it is said in one place. A quantile is the exception: its
/// distance is a share of the column, the step refuses that starting multiple there, and a line that offered a quantile
/// without a share could only fail.
/// </para>
/// </remarks>
public sealed class BoundsLine : IDeclaresSteps
{
    private readonly List<IPipelineStep> _steps = [];

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>Columns held to the middle half, widened by the usual distance beyond the quartiles.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next way can be written after it.</returns>
    /// <remarks>The robust middle: the middle half is what an extreme value cannot move.</remarks>
    public BoundsLine Iqr(params string[] columns) => Add(columns, Bounds.Iqr);

    /// <summary>Columns held to the middle half, widened by a distance you choose beyond the quartiles.</summary>
    /// <param name="at">How far beyond the quartiles the bounds sit, as a multiple of the width of the middle half.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next way can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The distance is not a finite number above nothing.</exception>
    public BoundsLine Iqr(double at, params string[] columns) => Add(columns, Bounds.Iqr, at);

    /// <summary>Columns held to so many standard deviations either side of their mean, at the usual distance.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next way can be written after it.</returns>
    /// <remarks>
    /// Assumes the column is roughly symmetric, and is dragged around by exactly the values it is meant to catch.
    /// </remarks>
    public BoundsLine Sigma(params string[] columns) => Add(columns, Bounds.Sigma);

    /// <summary>Columns held to a number of standard deviations you choose either side of their mean.</summary>
    /// <param name="at">How many standard deviations either side of the middle the bounds sit.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next way can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The distance is not a finite number above nothing.</exception>
    public BoundsLine Sigma(double at, params string[] columns) => Add(columns, Bounds.Sigma, at);

    /// <summary>Columns whose lowest and highest few per cent are held at the edge of the rest.</summary>
    /// <param name="at">
    /// The share set aside at each end, less than half of the column; nothing for the smallest and largest value the
    /// training rows hold.
    /// </param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next way can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The share is not a finite number, at least nothing and less than half of the column.
    /// </exception>
    /// <remarks>Honest about tails of any shape. It is the one way with no line that leaves the share unsaid.</remarks>
    public BoundsLine Quantile(double at, params string[] columns) => Add(columns, Bounds.Quantile, at);

    private BoundsLine Add(string[] columns, Bounds bounds, double? at = null)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            // Left out, the distance is the one the step starts from, which keeps it said in one place.
            _steps.Add(at is { } distance
                ? new ClipOutliersStep(column, bounds, distance)
                : new ClipOutliersStep(column, bounds));
        }

        return this;
    }
}

/// <summary>
/// Holds the extreme values of a column to bounds learned from the training rows.
/// </summary>
/// <remarks>
/// An extreme value is not automatically a mistake, and this does not pretend otherwise: it is a declared
/// decision about how much of the tail a model is allowed to see. The bounds are learned on the training
/// rows alone, like everything else below the split, so a validation row that lies outside them is held at
/// an edge the fit never saw it push.
/// <para>
/// The three ways of finding the bounds answer different questions. A quantile asks which few per cent to
/// set aside and is honest about tails of any shape. A spread of so many standard deviations assumes the
/// column is roughly symmetric and is dragged around by exactly the values it is meant to catch. The
/// interquartile range is the robust middle: it uses the middle half, which extremes cannot move.
/// </para>
/// <para>
/// A quantile bound at nothing is the smallest and largest value the training rows hold, so a refusal there is a range
/// check learned on the training rows: a later row beyond anything the fit saw stops the run and says how far out it was.
/// </para>
/// </remarks>
public sealed record ClipOutliersStep : IFittedStep, IPipelineStep<ClipOutliersStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column whose extremes are held.", "column", ColumnKinds.Numbers);

    private static readonly OneOfParameter<Bounds> BoundsKey = new(
        "bounds", "How the bounds are worked out: by quantile, by spread, or by the middle half.", Bounds.Iqr);

    private static readonly NumberParameter AtKey = new(
        "at",
        "How far out the bounds sit: a share for a quantile, nothing for the training extremes; a multiple of the spread or the middle half otherwise.",
        1.5)
    {
        AtLeast = 0,
    };

    private static readonly OneOfParameter<Outlier> OutlierKey = new(
        "outlier", "What happens to a value outside the bounds: held at the edge, made a gap, or refused.", Outlier.Clip);

    /// <summary>Declares that the extremes of a column are held to bounds.</summary>
    /// <param name="column">The column to hold.</param>
    /// <param name="bounds">How the bounds are worked out.</param>
    /// <param name="at">
    /// How far out the bounds sit: a share for a quantile, and nothing for the smallest and largest value the training rows
    /// hold; a multiple otherwise.
    /// </param>
    /// <param name="outlier">What happens to a value outside them.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The distance is not one this kind of bound can use.</exception>
    public ClipOutliersStep(string column, Bounds bounds = Bounds.Iqr, double at = 1.5,
                            Outlier outlier = Outlier.Clip)
    {
        Column = ColumnKey.Require(column);
        At = AtKey.Require(at);

        // Rules between two parameters, so they live with the step that has both.
        if (bounds == Bounds.Quantile && at >= 0.5)
        {
            throw new ArgumentOutOfRangeException(
                nameof(at), at, "A quantile bound sets aside less than half the column at each end.");
        }

        if (bounds != Bounds.Quantile && at == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(at), at,
                "Bounds at nothing are the training extremes, which a quantile bound says: by spread or by the middle half they would close on the middle itself.");
        }

        Bounds = BoundsKey.Require(bounds);
        Outlier = OutlierKey.Require(outlier);
    }

    /// <inheritdoc />
    public static StepParameters<ClipOutliersStep> Parameters { get; } = new StepParameters<ClipOutliersStep>()
        .With(ColumnKey, step => step.Column)
        .With(BoundsKey, step => step.Bounds)
        .With(AtKey, step => step.At)
        .With(OutlierKey, step => step.Outlier);

    /// <summary>The column being held.</summary>
    public string Column { get; }

    /// <summary>How the bounds are worked out.</summary>
    public Bounds Bounds { get; }

    /// <summary>How far out the bounds sit.</summary>
    public double At { get; }

    /// <summary>What happens to a value outside them.</summary>
    public Outlier Outlier { get; }

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Column, ColumnKind.Number);
    }

    /// <inheritdoc />
    public static string Name => "outliers.clip";

    /// <inheritdoc />
    public static string Purpose => "Holds the extreme values of a column to bounds learned from the training rows.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var training = table.TrainingValues(Column, parts).Learnable("bounds");
        var learned = new FittedStepValues();

        switch (Bounds)
        {
            case Bounds.Quantile:
                learned.Learned("lower", training.Quantile(At));
                learned.Learned("upper", training.Quantile(1 - At));
                break;

            case Bounds.Sigma:
                learned.Learned("lower", training.Mean - (At * training.StandardDeviation));
                learned.Learned("upper", training.Mean + (At * training.StandardDeviation));
                break;

            default:
                var low = training.Quantile(0.25);
                var high = training.Quantile(0.75);
                var middle = high - low;
                learned.Learned("lower", low - (At * middle));
                learned.Learned("upper", high + (At * middle));
                break;
        }

        // How many of the training rows lie outside: what the fit saw, like everything else in its half.
        learned.Learned(
            "outside",
            training.Finite.Count(value => value < learned.Number("lower") || value > learned.Number("upper")));

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var lower = fitted.Number("lower");
        var upper = fitted.Number("upper");
        var values = table.NumbersOf(Column);
        var held = new double?[values.Length];

        for (var row = 0; row < values.Length; row++)
        {
            if (values[row] is not { } value)
            {
                continue;
            }

            held[row] = value >= lower && value <= upper
                ? value
                : Outlier switch
                {
                    Outlier.Clip => Math.Clamp(value, lower, upper),
                    Outlier.Blank => null,
                    _ => throw new InvalidOperationException(
                        string.Create(
                            CultureInfo.InvariantCulture,
                            $"Row {row + 1} of '{Column}' is {value:0.####}, outside the {lower:0.####} to "
                            + $"{upper:0.####} this pipeline was fitted on.")),
                };
        }

        table.Put(new Column<double>(Column, ColumnKind.Number, held));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ClipOutliersStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), BoundsKey.Read(element), AtKey.Read(element), OutlierKey.Read(element));
}
