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
/// </remarks>
public sealed record ClipOutliersStep : IFittedStep, ILearnsFromData, IPipelineStep<ClipOutliersStep>
{
    /// <summary>Declares that the extremes of a column are held to bounds.</summary>
    /// <param name="column">The column to hold.</param>
    /// <param name="bounds">How the bounds are worked out.</param>
    /// <param name="at">How far out the bounds sit: a share for a quantile, a multiple otherwise.</param>
    /// <param name="outlier">What happens to a value outside them.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The distance is not one this kind of bound can use.</exception>
    public ClipOutliersStep(string column, Bounds bounds = Bounds.Iqr, double at = 1.5,
                            Outlier outlier = Outlier.Clip)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        if (!(at > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(at), at, "A bound sits some distance out, not none.");
        }

        if (bounds == Bounds.Quantile && at >= 0.5)
        {
            throw new ArgumentOutOfRangeException(
                nameof(at), at, "A quantile bound sets aside less than half the column at each end.");
        }

        Column = column;
        Bounds = bounds;
        At = at;
        Outlier = outlier;
    }

    /// <summary>The column being held.</summary>
    public string Column { get; }

    /// <summary>How the bounds are worked out.</summary>
    public Bounds Bounds { get; }

    /// <summary>How far out the bounds sit.</summary>
    public double At { get; }

    /// <summary>What happens to a value outside them.</summary>
    public Outlier Outlier { get; }

    /// <inheritdoc />
    public static string Name => "outliers.clip";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var values = Numbers.Of(table, Column);
        var training = Enumerable.Range(0, values.Length)
            .Where(row => parts[row] == Part.Train && values[row] is not null)
            .Select(row => values[row]!.Value)
            .Order()
            .ToArray();

        if (training.Length == 0)
        {
            throw new InvalidOperationException(
                $"Every training row of '{Column}' is a gap, so there are no bounds to learn.");
        }

        var learned = new FittedStepValues();

        switch (Bounds)
        {
            case Bounds.Quantile:
                learned.Learned("lower", Quantile(training, At));
                learned.Learned("upper", Quantile(training, 1 - At));
                break;

            case Bounds.Sigma:
                var mean = training.Average();
                var spread = Math.Sqrt(training.Average(value => (value - mean) * (value - mean)));
                learned.Learned("lower", mean - (At * spread));
                learned.Learned("upper", mean + (At * spread));
                break;

            default:
                var low = Quantile(training, 0.25);
                var high = Quantile(training, 0.75);
                var middle = high - low;
                learned.Learned("lower", low - (At * middle));
                learned.Learned("upper", high + (At * middle));
                break;
        }

        learned.Learned(
            "outside",
            values.Count(value => value is { } number
                                  && (number < learned.Number("lower") || number > learned.Number("upper"))));

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var lower = fitted.Number("lower");
        var upper = fitted.Number("upper");
        var values = Numbers.Of(table, Column);
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

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("bounds", Bounds.ToString().ToLowerInvariant());
        writer.WriteNumber("at", At);
        writer.WriteString("outlier", Outlier.ToString().ToLowerInvariant());
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ClipOutliersStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredEnum<Bounds>("bounds"),
            element.RequiredNumber("at"),
            element.RequiredEnum<Outlier>("outlier"));

    internal static double Quantile(double[] sorted, double at)
    {
        var place = at * (sorted.Length - 1);
        var below = (int)Math.Floor(place);
        var above = Math.Min(below + 1, sorted.Length - 1);

        return sorted[below] + ((sorted[above] - sorted[below]) * (place - below));
    }
}
