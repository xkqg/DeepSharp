// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

// One row of a key, wherever it is held: its place in the declared order, and the value a history reads.
internal readonly record struct Held(double[] Place, double Value);

// The sum and the count of the rows strictly before a place: all of them, or those on the last places of a window.
internal readonly record struct Counted(double Sum, int Count);

/// <summary>
/// The rows of one key, in their declared order, with the running sum of their values: what came before a place is a
/// search. Where a window reads only the last places, the rows are put in an order that is the same whatever order
/// they were handed in — by place and then by value — and what a window holds is added up row by row, since the
/// difference of two running totals is not the sum of the rows between them once a total has lost the small values a
/// large one swallowed.
/// </summary>
/// <remarks>
/// Shared by <see cref="HistoryStep"/>, whose rows are the training rows alone, and <see cref="KnownHistoryStep"/>,
/// whose rows are the training rows and every other row already known when a later row of the same key arrives — one
/// set of arithmetic for what "the earlier rows of a key" means, once either has assembled which rows those are.
/// </remarks>
internal sealed class Earlier
{
    private readonly double[][] _places;
    private readonly double[] _values;
    private readonly double[] _sums;
    private readonly int[] _starts;

    /// <summary>The rows of one key, ready to be searched.</summary>
    /// <param name="rows">Every row of the key, in no particular order.</param>
    /// <param name="windowed">Whether a window will ever ask for fewer than every row before a place.</param>
    public Earlier(Held[] rows, bool windowed)
    {
        ArgumentNullException.ThrowIfNull(rows);

        Array.Sort(rows, windowed ? (one, other) => Compared(one.Place, other.Place) is var order and not 0 ? order : one.Value.CompareTo(other.Value) : (one, other) => Compared(one.Place, other.Place));

        _places = [.. rows.Select(row => row.Place)];
        _values = [.. rows.Select(row => row.Value)];
        _sums = new double[rows.Length + 1];

        for (var at = 0; at < rows.Length; at++)
        {
            _sums[at + 1] = _sums[at] + rows[at].Value;
        }

        _starts = [.. Enumerable.Range(0, rows.Length).Where(at => at == 0 || Compared(_places[at - 1], _places[at]) != 0)];
    }

    /// <summary>The sum and the count of the rows strictly before a place: all of them, or those on the last places of a window.</summary>
    /// <param name="place">The place a row stands at.</param>
    /// <param name="window">How many of the last places before it are read; nought for every one of them.</param>
    public Counted Before(double[] place, int window)
    {
        ArgumentNullException.ThrowIfNull(place);

        int low = 0, high = _places.Length;

        while (low < high)
        {
            var middle = (low + high) / 2;

            if (Compared(_places[middle], place) < 0)
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }

        if (window == 0)
        {
            return new Counted(_sums[low], low);
        }

        var places = _starts.Count(start => start < low);

        if (places == 0)
        {
            return default;
        }

        var first = _starts[Math.Max(0, places - window)];
        var sum = 0d;

        for (var at = first; at < low; at++)
        {
            sum += _values[at];
        }

        return new Counted(sum, low - first);
    }

    // The greatest place held, by the same ordering Before searches with; nothing when there are no rows at all.
    internal double[]? Latest => _places.Length == 0 ? null : _places[^1];

    internal static int Compared(double[] one, double[] other)
    {
        for (var at = 0; at < one.Length; at++)
        {
            var compared = one[at].CompareTo(other[at]);

            if (compared != 0)
            {
                return compared;
            }
        }

        return 0;
    }
}

// The three columns that say what a history reads and whose rows it reads them from: the value, the key that groups
// rows into one history, and the columns that order them. Held as one so fitting them takes four parameters, not five.
internal readonly record struct EarlierColumns(string Value, string Key, IReadOnlyList<string> OrderBy);

// What HistoryStep and KnownHistoryStep both do at Fit: learn the training rows' prior, and keep their key, value and
// place so a replay can read them back — the training floor every history has, however it is read afterwards.
internal static class EarlierFit
{
    // The m-estimate: the mean of what was counted, pulled towards the prior by a weight above nothing — the prior alone
    // where nothing was counted and there is no weight to pull towards it.
    extension(Counted counted)
    {
        internal double Blended(double shrink, double prior) =>
            (counted.Count + shrink) is 0d ? prior : (counted.Sum + (shrink * prior)) / (counted.Count + shrink);
    }

    extension(Table table)
    {
        internal double[][] PlacesBy(IReadOnlyList<string> orderBy)
        {
            var columns = orderBy.Select(name => table[name]).ToArray();
            var places = new double[table.RowCount][];

            for (var row = 0; row < table.RowCount; row++)
            {
                places[row] = new double[columns.Length];

                for (var at = 0; at < columns.Length; at++)
                {
                    places[row][at] = columns[at] switch
                    {
                        _ when columns[at].IsMissing(row) => throw new InvalidOperationException(
                            $"Row {table.Identities[row].ReadAt + 1} has no '{orderBy[at]}', so nothing says which rows came before it."),
                        Column<DateTime> moments => (moments[row]!.Value - DateTime.UnixEpoch).TotalDays,
                        Column<long> whole => whole[row]!.Value,
                        _ => ((Column<double>)columns[at])[row]!.Value,
                    };
                }
            }

            return places;
        }

        /// <exception cref="InvalidOperationException">No training row holds a number of the column, or a row has a gap where its order is said.</exception>
        internal FittedStepValues FitTheEarlierTrainingRows(IReadOnlyList<Part> parts, EarlierColumns columns)
        {
            var prior = table.TrainingValues(columns.Value, parts).Learnable("the mean of every training row").Mean;
            var values = table.NumbersOf(columns.Value);
            var keys = table[columns.Key];
            var order = table.PlacesBy(columns.OrderBy);
            var kept = Enumerable.Range(0, table.RowCount).Where(row => parts[row] == Part.Train && values[row] is { } x && double.IsFinite(x) && !keys.IsMissing(row)).ToArray();
            var learned = new FittedStepValues();

            learned.Learned("prior", prior);
            learned.Learned("keys", [.. kept.Select(row => keys.TextAt(row)!)]);
            learned.Learned("values", [.. kept.Select(row => values[row]!.Value)]);

            for (var at = 0; at < columns.OrderBy.Count; at++)
            {
                learned.Learned(string.Create(CultureInfo.InvariantCulture, $"order{at + 1}"), [.. kept.Select(row => order[row][at])]);
            }

            return learned;
        }
    }

    extension(FittedStepValues fitted)
    {
        /// <summary>Every training row the fit kept, grouped by key, ready for an <see cref="Earlier"/> of its own.</summary>
        /// <param name="orderBy">How many order columns the fit kept a place for.</param>
        internal Dictionary<string, Held[]> TrainingRowsByKey(int orderBy)
        {
            var learnedKeys = fitted.List("keys");
            var learnedValues = fitted.Curve("values");
            var learnedOrder = Enumerable.Range(1, orderBy).Select(at => fitted.Curve(string.Create(CultureInfo.InvariantCulture, $"order{at}"))).ToArray();

            return Enumerable.Range(0, learnedKeys.Count)
                .GroupBy(row => learnedKeys[row], StringComparer.Ordinal)
                .ToDictionary(group => group.Key, group => group.Select(row => new Held([.. learnedOrder.Select(curve => curve[row])], learnedValues[row])).ToArray(), StringComparer.Ordinal);
        }
    }
}
