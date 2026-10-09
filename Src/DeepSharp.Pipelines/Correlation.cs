// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>Which correlation coefficient a pair of columns is held to.</summary>
public enum Coefficient
{
    /// <summary>Pearson's: how well the two lie on a straight line.</summary>
    Pearson,

    /// <summary>
    /// Spearman's: how well the two keep the same order, which is Pearson's coefficient of their ranks. Values that are equal
    /// share the average of the places they take, so a curve that only ever rises is a perfect one.
    /// </summary>
    Spearman,
}

/// <summary>
/// The coefficients of one kind between every pair of columns: a square of numbers, the same read from either side.
/// </summary>
/// <remarks>
/// A pair has <see cref="double.NaN"/> where no coefficient is defined: a column that never changes has no direction to share,
/// and fewer than two rows correlate nothing. It is not nought — nought is a measurement, and a chart or a profile that
/// read it would take a number nobody worked out for one somebody did.
/// </remarks>
public sealed class CorrelationCoefficients
{
    private readonly double[,] _values;

    internal CorrelationCoefficients(IReadOnlyList<string> columns, double[,] values)
    {
        Columns = columns;
        _values = values;
    }

    /// <summary>The columns, in the order the rows and the columns of the square stand in.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How many columns there are, which is how many rows and columns the square has.</summary>
    public int Size => Columns.Count;

    /// <summary>The coefficient between two columns, by their places.</summary>
    /// <param name="row">The place of one column.</param>
    /// <param name="column">The place of the other.</param>
    /// <returns>The coefficient, from minus one to one; <see cref="double.NaN"/> where none is defined.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A place is outside the columns.</exception>
    public double this[int row, int column]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(row);
            ArgumentOutOfRangeException.ThrowIfNegative(column);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Size);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Size);

            return _values[row, column];
        }
    }

    /// <summary>The coefficient between two columns, by their names.</summary>
    /// <param name="row">The name of one column.</param>
    /// <param name="column">The name of the other.</param>
    /// <returns>The coefficient, from minus one to one; <see cref="double.NaN"/> where none is defined.</returns>
    /// <exception cref="ArgumentException">A name is none of the columns.</exception>
    public double this[string row, string column] => _values[PlaceOf(row), PlaceOf(column)];

    /// <summary>The square as rows of numbers, a copy that may be changed without changing the coefficients.</summary>
    /// <returns>One array per row, each as long as there are columns.</returns>
    public double[][] ToArray() =>
        [.. Enumerable.Range(0, Size).Select(row => Enumerable.Range(0, Size).Select(column => _values[row, column]).ToArray())];

    private int PlaceOf(string name)
    {
        ArgumentNullException.ThrowIfNull(name);

        for (var place = 0; place < Columns.Count; place++)
        {
            if (string.Equals(Columns[place], name, StringComparison.Ordinal))
            {
                return place;
            }
        }

        throw new ArgumentException($"'{name}' is none of the columns: {string.Join(", ", Columns)}.", nameof(name));
    }
}

/// <summary>
/// The correlation between columns, worked out once from the rows a correlation was kept over: Pearson's and Spearman's
/// coefficients for every pair.
/// </summary>
/// <remarks>
/// The chart, the notebook's table and the profile's rank alert read these numbers and work out none of their own, so one
/// pair of columns has one coefficient wherever it is shown. They are worked out from the rows
/// <see cref="CorrelationInput"/> kept, which are the complete ones: a row that has a gap in any of the columns is in none
/// of the pairs. Spearman's ranks tie as their average; no p-value is given, because the usual one is only trusted for very
/// many rows and a coefficient with a number to hide behind is read as more than it says.
/// </remarks>
public sealed class CorrelationMatrix
{
    private CorrelationMatrix(IReadOnlyList<string> columns, int kept, CorrelationCoefficients pearson, CorrelationCoefficients spearman)
    {
        Columns = columns;
        Kept = kept;
        Pearson = pearson;
        Spearman = spearman;
    }

    /// <summary>The columns, in the order the correlation named them.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How many rows the coefficients were worked out from.</summary>
    public int Kept { get; }

    /// <summary>Pearson's coefficient between every pair of columns.</summary>
    public CorrelationCoefficients Pearson { get; }

    /// <summary>Spearman's coefficient between every pair of columns.</summary>
    public CorrelationCoefficients Spearman { get; }

    /// <summary>The coefficients of one kind.</summary>
    /// <param name="coefficient">Which coefficient.</param>
    /// <returns>Its coefficients between every pair of columns.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The value is none of the coefficients: a number cast to the set.</exception>
    public CorrelationCoefficients Of(Coefficient coefficient) => coefficient switch
    {
        Coefficient.Pearson => Pearson,
        Coefficient.Spearman => Spearman,
        _ => throw new ArgumentOutOfRangeException(nameof(coefficient), coefficient, "A correlation is Pearson's or Spearman's."),
    };

    /// <summary>Works out both coefficients between every pair of these columns.</summary>
    /// <param name="columns">The columns, in the order the rows list them.</param>
    /// <param name="rows">The complete rows, each with one finite number per column.</param>
    /// <returns>The coefficients.</returns>
    internal static CorrelationMatrix From(IReadOnlyList<string> columns, IReadOnlyList<double[]> rows)
    {
        double[][] values = [.. Enumerable.Range(0, columns.Count).Select(column => rows.Select(row => row[column]).ToArray())];
        double[][] ranks = [.. values.Select(column => column.AverageRanks())];

        return new CorrelationMatrix(
            columns,
            rows.Count,
            new CorrelationCoefficients(columns, Square(values)),
            new CorrelationCoefficients(columns, Square(ranks)));
    }

    // Every pair, worked out once above the diagonal and read the same from below it: a column is its own correlation of one
    // when it has one at all.
    private static double[,] Square(double[][] columns)
    {
        var square = new double[columns.Length, columns.Length];

        for (var row = 0; row < columns.Length; row++)
        {
            square[row, row] = columns[row].Varies() ? 1.0 : double.NaN;

            for (var column = row + 1; column < columns.Length; column++)
            {
                square[row, column] = square[column, row] = columns[row].PearsonWith(columns[column]);
            }
        }

        return square;
    }
}

/// <summary>The arithmetic of a coefficient, on the numbers of a column.</summary>
internal static class CoefficientExtensions
{
    extension(double[] values)
    {
        /// <summary>The place each value takes among them, counting from one, equal values sharing the average of the places they span.</summary>
        /// <returns>The ranks, in the order of the values.</returns>
        public double[] AverageRanks()
        {
            var keys = (double[])values.Clone();
            var order = Enumerable.Range(0, values.Length).ToArray();
            var ranks = new double[values.Length];

            Array.Sort(keys, order);

            for (var first = 0; first < keys.Length;)
            {
                var last = first;

                while (last + 1 < keys.Length && keys[last + 1] == keys[first])
                {
                    last++;
                }

                for (var at = first; at <= last; at++)
                {
                    ranks[order[at]] = ((first + last) / 2.0) + 1;
                }

                first = last + 1;
            }

            return ranks;
        }

        /// <summary>The Pearson coefficient between these values and as many others.</summary>
        /// <param name="other">The other values, one for each of these.</param>
        /// <returns>
        /// The coefficient, from minus one to one; <see cref="double.NaN"/> when there are fewer than two pairs, or when either
        /// column never changes.
        /// </returns>
        /// <remarks>
        /// Worked out the way the rest of the pipeline measures a spread: from each value's distance to the average, the
        /// average taken first. Ten tenths do not add up to one, so a column that never changes is found by
        /// <see cref="Varies"/> and not by a spread of nought.
        /// </remarks>
        public double PearsonWith(double[] other)
        {
            if (!values.Varies() || !other.Varies())
            {
                return double.NaN;
            }

            var mine = values.Average();
            var theirs = other.Average();
            double together = 0, myNorm = 0, theirNorm = 0;

            for (var at = 0; at < values.Length; at++)
            {
                var one = values[at] - mine;
                var two = other[at] - theirs;

                together += one * two;
                myNorm += one * one;
                theirNorm += two * two;
            }

            return Math.Clamp(together / (Math.Sqrt(myNorm) * Math.Sqrt(theirNorm)), -1, 1);
        }

        /// <summary>The Spearman coefficient between these values and as many others: Pearson's, of their ranks.</summary>
        /// <param name="other">The other values, one for each of these.</param>
        /// <returns>The coefficient; <see cref="double.NaN"/> where <see cref="PearsonWith"/> has none.</returns>
        public double SpearmanWith(double[] other) => values.AverageRanks().PearsonWith(other.AverageRanks());

        /// <summary>Whether there are at least two values and they are not all the one.</summary>
        /// <returns>
        /// <see langword="true"/> when the values have a spread to correlate. That a column never changes is told by what it
        /// holds, not by a spread left over from rounding its average.
        /// </returns>
        public bool Varies() => values.Length >= 2 && values.Any(value => value != values[0]);
    }
}
