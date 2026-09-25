// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What the training rows of one column hold, as everything that learns reads it.
/// </summary>
/// <remarks>
/// One set of values and one way of measuring them, for every step that learns. There used to be four, one
/// per step, each with its own idea of a median and of a value that is not a number: the same column gave a
/// median of 5.5 to one step and 6 to another, and a min-max scale learned its middle from a not-a-number.
/// <para>
/// A gap is not a value, and neither is a not-a-number or an infinity; both are counted and neither is
/// measured. A step that learns refuses to learn while a training value is not a finite number — the step
/// for that is <c>fill.nan</c>, above it — because quietly leaving it out would hide a fault upstream.
/// </para>
/// </remarks>
public readonly record struct TrainingValues
{
    private readonly double[] _finite;

    internal TrainingValues(string column, double[] finite, int gaps, int notFinite)
    {
        Column = column;
        _finite = finite;
        Gaps = gaps;
        NotFinite = notFinite;
    }

    /// <summary>The column the values are from.</summary>
    public string Column { get; }

    /// <summary>The finite values of the training rows, smallest first.</summary>
    public IReadOnlyList<double> Finite => _finite;

    /// <summary>How many training rows are gaps.</summary>
    public int Gaps { get; }

    /// <summary>How many training rows hold a not-a-number or an infinity.</summary>
    public int NotFinite { get; }

    /// <summary>How many training rows there are.</summary>
    public int Rows => _finite.Length + Gaps + NotFinite;

    /// <summary>The average of the finite values.</summary>
    /// <exception cref="InvalidOperationException">There are none.</exception>
    public double Mean => Measured().Average();

    /// <summary>The middle of the finite values; the average of the two in the middle when their number is even.</summary>
    /// <exception cref="InvalidOperationException">There are none.</exception>
    public double Median => Quantile(0.5);

    /// <summary>How far the finite values spread around their average: the square root of their mean squared distance from it.</summary>
    /// <exception cref="InvalidOperationException">There are none.</exception>
    public double StandardDeviation
    {
        get
        {
            var mean = Mean;

            return Math.Sqrt(_finite.Average(value => (value - mean) * (value - mean)));
        }
    }

    /// <summary>The value a share of the finite values lies below, read off between the two it falls between.</summary>
    /// <param name="at">The share, from nought to one.</param>
    /// <returns>The value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The share is not one.</exception>
    /// <exception cref="InvalidOperationException">There are no finite values.</exception>
    public double Quantile(double at)
    {
        if (!(at >= 0 && at <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(at), at, "A quantile is a share, from nought to one.");
        }

        var sorted = Measured();
        var place = at * (sorted.Length - 1);
        var below = (int)Math.Floor(place);
        var above = Math.Min(below + 1, sorted.Length - 1);

        return sorted[below] + ((sorted[above] - sorted[below]) * (place - below));
    }

    /// <summary>These values, for a step that learns from them: every one a finite number, and at least one.</summary>
    /// <param name="what">What the step learns, for the message: "a scale", "a fill value".</param>
    /// <returns>These values.</returns>
    /// <exception cref="InvalidOperationException">A training value is not a finite number, or every training row is a gap.</exception>
    public TrainingValues Learnable(string what)
    {
        if (NotFinite > 0)
        {
            throw new InvalidOperationException(
                $"{NotFinite} of the training rows of '{Column}' hold a value that is not a finite number, and {what} "
                + $"is not learned from those. Say what becomes of them with fill.nan for '{Column}', above this step.");
        }

        return Measurable(what);
    }

    /// <summary>These values, for a step that measures the finite ones and deals with the rest itself: at least one.</summary>
    /// <param name="what">What the step learns, for the message.</param>
    /// <returns>These values.</returns>
    /// <exception cref="InvalidOperationException">No training row holds a finite number.</exception>
    public TrainingValues Measurable(string what) =>
        _finite.Length > 0
            ? this
            : throw new InvalidOperationException(
                $"Every training row of '{Column}' is a gap or not a number, so there is nothing to learn {what} from.");

    private double[] Measured() =>
        _finite.Length > 0
            ? _finite
            : throw new InvalidOperationException($"No training row of '{Column}' holds a finite number to measure.");
}

/// <summary>
/// Reading a table's columns as numbers, and its training rows as what a fit learns from.
/// </summary>
/// <remarks>
/// Public because a package that adds a verb needs exactly this and would otherwise write its own, and two
/// readings of "what is a number here" is one too many: a boolean counts as one and nought, a gap stays a gap,
/// and words are refused by name.
/// </remarks>
public static class TableExtensions
{
    /// <summary>The column's values as numbers, with a gap where a cell is a gap.</summary>
    /// <param name="table">The table to look in.</param>
    /// <param name="column">The column's name.</param>
    /// <returns>One value per row.</returns>
    /// <exception cref="InvalidOperationException">The column holds something that is not a number.</exception>
    public static double?[] NumbersOf(this Table table, string column)
    {
        ArgumentNullException.ThrowIfNull(table);

        return table[column] switch
        {
            Column<double> numbers => [.. Enumerable.Range(0, numbers.Count).Select(row => numbers[row])],
            Column<long> whole => [.. Enumerable.Range(0, whole.Count).Select(row => (double?)whole[row])],
            Column<bool> flags =>
                [.. Enumerable.Range(0, flags.Count).Select(row => flags[row] is { } flag ? flag ? 1 : 0 : (double?)null)],
            var other => throw new InvalidOperationException(
                $"'{column}' holds {other.Kind.ToString().ToLowerInvariant()}, and this step works on numbers."),
        };
    }

    /// <summary>What the training rows of a column hold: its finite values, its gaps and its values that are not numbers.</summary>
    /// <param name="table">The table.</param>
    /// <param name="column">The column's name.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <returns>The training values, the one set every fit reads.</returns>
    /// <exception cref="ArgumentException">The parts are not one per row.</exception>
    /// <exception cref="InvalidOperationException">The column holds something that is not a number.</exception>
    public static TrainingValues TrainingValues(this Table table, string column, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        if (parts.Count != table.RowCount)
        {
            throw new ArgumentException(
                $"There are {parts.Count} parts for {table.RowCount} rows, and a row belongs to exactly one part.", nameof(parts));
        }

        return table.ValuesOf(column, row => parts[row] == Part.Train);
    }

    /// <summary>What some rows of a column hold: the training rows of a fit, the measured rows of a view.</summary>
    /// <param name="table">The table.</param>
    /// <param name="column">The column's name.</param>
    /// <param name="measured">Which rows count.</param>
    /// <returns>Their finite values, their gaps and their values that are not numbers.</returns>
    internal static TrainingValues ValuesOf(this Table table, string column, Func<int, bool> measured)
    {
        var values = table.NumbersOf(column);
        var finite = new List<double>();
        var gaps = 0;
        var notFinite = 0;

        for (var row = 0; row < values.Length; row++)
        {
            if (!measured(row))
            {
                continue;
            }

            switch (values[row])
            {
                case null:
                    gaps++;
                    break;

                case { } value when !double.IsFinite(value):
                    notFinite++;
                    break;

                case { } value:
                    finite.Add(value);
                    break;
            }
        }

        finite.Sort();

        return new TrainingValues(column, [.. finite], gaps, notFinite);
    }

    /// <summary>
    /// What some rows of a column hold as words: the categories an encoder learns from the training rows, the categories
    /// the measured rows of a view hold.
    /// </summary>
    /// <param name="table">The table.</param>
    /// <param name="column">The column's name.</param>
    /// <param name="measured">Which rows count.</param>
    /// <returns>Every value those rows hold, once, in ordinal order; a gap is none.</returns>
    internal static IReadOnlyList<string> CategoriesOf(this Table table, string column, Func<int, bool> measured)
    {
        var values = table[column];

        return
        [
            .. Enumerable.Range(0, table.RowCount)
                .Where(row => measured(row) && !values.IsMissing(row))
                .Select(row => values.TextAt(row)!)
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal),
        ];
    }
}
