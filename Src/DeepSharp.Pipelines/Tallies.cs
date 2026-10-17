// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What one column holds over some rows, counted: each value once, with how many rows hold it, and how many rows hold none.
/// </summary>
/// <remarks>
/// Both distances a column is measured by need no more than the counts, so a set of rows counted measures exactly what the rows
/// would have: the counted Kolmogorov–Smirnov distance and the counted total variation are, to the bit, the ones worked out from the
/// values one by one. That is what lets a reference of the training rows be kept as counts and the rows themselves let go.
/// </remarks>
internal abstract record Tally
{
    /// <summary>How many rows hold a value.</summary>
    public abstract int Held { get; }

    /// <summary>How many rows hold none: a gap, or a number that is not finite.</summary>
    public abstract int Gaps { get; }

    /// <summary>The distance this kind of column is measured by.</summary>
    public abstract Distance Distance { get; }

    /// <summary>How far apart this and another count of the same kind of column are.</summary>
    /// <param name="other">The other count.</param>
    /// <returns>The distance, from nought to one; not a number when either holds no value.</returns>
    /// <exception cref="ArgumentException">The other counts another kind of column.</exception>
    public abstract double ApartFrom(Tally other);

    /// <summary>
    /// What a column holds over some of a table's rows, counted by the one rule every comparison of columns takes: words as words,
    /// a moment as the days it stands from 1970, any other column as its numbers; a gap, and a number that is not finite, as a row
    /// that holds none.
    /// </summary>
    /// <param name="table">The rows.</param>
    /// <param name="column">The column.</param>
    /// <param name="rows">Which rows, by their place in the table.</param>
    /// <returns>The count.</returns>
    public static Tally Of(Table table, string column, IReadOnlyList<int> rows)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(rows);

        if (table[column] is TextColumn words)
        {
            return WordsTallied.Of(rows.Where(row => !words.IsMissing(row)).Select(row => words.TextAt(row)!), rows.Count(words.IsMissing));
        }

        IReadOnlyList<double?> numbers = table[column] is Column<DateTime> moments
            ? [.. Enumerable.Range(0, table.RowCount).Select(row => moments[row] is { } moment ? (moment - DateTime.UnixEpoch).TotalDays : (double?)null)]
            : table.NumbersOf(column);
        var values = rows.Select(row => numbers[row]);
        double[] held = [.. values.Where(value => value is { } number && double.IsFinite(number)).Select(value => value!.Value)];

        return NumbersTallied.Of(held, rows.Count - held.Length);
    }

    // A column's values counted, every count one or more.
    private protected static void ThrowIfNotOnePerValue(int values, IReadOnlyList<int> counts, int gaps)
    {
        if (values != counts.Count || counts.Any(count => count < 1) || gaps < 0)
        {
            throw new ArgumentException("A count is one number of rows for each value, at least one each, beside a number of rows with none.");
        }
    }
}

/// <summary>A column of numbers counted: its distinct values in ascending order, and how many rows hold each.</summary>
internal sealed record NumbersTallied : Tally
{
    /// <summary>A column of numbers counted.</summary>
    /// <param name="values">The distinct values, ascending.</param>
    /// <param name="counts">How many rows hold each, in the same order.</param>
    /// <param name="gaps">How many rows hold no value.</param>
    /// <exception cref="ArgumentException">There is not one count, of at least one, for each value.</exception>
    public NumbersTallied(IReadOnlyList<double> values, IReadOnlyList<int> counts, int gaps)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(counts);
        ThrowIfNotOnePerValue(values.Count, counts, gaps);

        Values = values;
        Counts = counts;
        Gaps = gaps;
        Held = counts.Sum();
    }

    /// <summary>The distinct values, ascending.</summary>
    public IReadOnlyList<double> Values { get; }

    /// <summary>How many rows hold each value.</summary>
    public IReadOnlyList<int> Counts { get; }

    /// <inheritdoc />
    public override int Held { get; }

    /// <inheritdoc />
    public override int Gaps { get; }

    /// <inheritdoc />
    public override Distance Distance => Distance.KolmogorovSmirnov;

    /// <summary>Counts the values a column holds over some rows.</summary>
    /// <param name="values">The values, each a finite number, in any order.</param>
    /// <param name="gaps">How many rows hold no value.</param>
    /// <returns>The count.</returns>
    public static NumbersTallied Of(IEnumerable<double> values, int gaps)
    {
        ArgumentNullException.ThrowIfNull(values);

        var grouped = values.Order().GroupBy(value => value).ToArray();

        return new([.. grouped.Select(group => group.Key)], [.. grouped.Select(group => group.Count())], gaps);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The most the share of one at or below a value differs from the other's, every value either holds taken in order with all of
    /// its copies in both at once, as scipy's <c>ks_2samp</c> takes ties.
    /// </remarks>
    public override double ApartFrom(Tally other)
    {
        if (other is not NumbersTallied numbers)
        {
            throw new ArgumentException("A column of numbers is measured against another column of numbers.", nameof(other));
        }

        if (Held == 0 || numbers.Held == 0)
        {
            return double.NaN;
        }

        int at = 0, elsewhere = 0, below = 0, belowElsewhere = 0;
        var most = 0d;

        while (at < Values.Count || elsewhere < numbers.Values.Count)
        {
            var value = elsewhere >= numbers.Values.Count || (at < Values.Count && Values[at] <= numbers.Values[elsewhere]) ? Values[at] : numbers.Values[elsewhere];

            if (at < Values.Count && Values[at] <= value)
            {
                below += Counts[at++];
            }

            if (elsewhere < numbers.Values.Count && numbers.Values[elsewhere] <= value)
            {
                belowElsewhere += numbers.Counts[elsewhere++];
            }

            most = Math.Max(most, Math.Abs(((double)below / Held) - ((double)belowElsewhere / numbers.Held)));
        }

        return most;
    }
}

/// <summary>A column of words counted: each word once, in the order it first stood, and how many rows hold each.</summary>
internal sealed record WordsTallied : Tally
{
    /// <summary>A column of words counted.</summary>
    /// <param name="words">The distinct words, in the order each first stood.</param>
    /// <param name="counts">How many rows hold each, in the same order.</param>
    /// <param name="gaps">How many rows hold no word.</param>
    /// <exception cref="ArgumentException">There is not one count, of at least one, for each word.</exception>
    public WordsTallied(IReadOnlyList<string> words, IReadOnlyList<int> counts, int gaps)
    {
        ArgumentNullException.ThrowIfNull(words);
        ArgumentNullException.ThrowIfNull(counts);
        ThrowIfNotOnePerValue(words.Count, counts, gaps);

        Words = words;
        Counts = counts;
        Gaps = gaps;
        Held = counts.Sum();
    }

    /// <summary>The distinct words, in the order each first stood.</summary>
    public IReadOnlyList<string> Words { get; }

    /// <summary>How many rows hold each word.</summary>
    public IReadOnlyList<int> Counts { get; }

    /// <inheritdoc />
    public override int Held { get; }

    /// <inheritdoc />
    public override int Gaps { get; }

    /// <inheritdoc />
    public override Distance Distance => Distance.TotalVariation;

    /// <summary>Counts the words a column holds over some rows.</summary>
    /// <param name="words">The words, in the order the rows hold them.</param>
    /// <param name="gaps">How many rows hold no word.</param>
    /// <returns>The count.</returns>
    public static WordsTallied Of(IEnumerable<string> words, int gaps)
    {
        ArgumentNullException.ThrowIfNull(words);

        var grouped = words.GroupBy(word => word, StringComparer.Ordinal).ToArray();

        return new([.. grouped.Select(group => group.Key)], [.. grouped.Select(group => group.Count())], gaps);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Half the sum of how far each word's share in one differs from its share in the other, the words taken in the order they first
    /// stand in this one and then in the other.
    /// </remarks>
    public override double ApartFrom(Tally other)
    {
        if (other is not WordsTallied words)
        {
            throw new ArgumentException("A column of words is measured against another column of words.", nameof(other));
        }

        if (Held == 0 || words.Held == 0)
        {
            return double.NaN;
        }

        var here = Shares(this);
        var there = Shares(words);
        var sum = 0d;

        foreach (var word in Words.Concat(words.Words).Distinct(StringComparer.Ordinal))
        {
            sum += Math.Abs((here.TryGetValue(word, out var share) ? share : 0d) - (there.TryGetValue(word, out var elsewhere) ? elsewhere : 0d));
        }

        return sum / 2;
    }

    // Each word's share of the rows that hold one, worked out as the share of a count over the whole.
    private static Dictionary<string, double> Shares(WordsTallied tally) =>
        tally.Words.Select((word, at) => KeyValuePair.Create(word, (double)tally.Counts[at] / tally.Held)).ToDictionary(StringComparer.Ordinal);
}
