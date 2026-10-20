// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How far a history reaches back and how much the mean of every training row weighs beside a key's own rows.</summary>
/// <param name="Shrink">How many rows' worth of weight the mean of every training row has: nothing only with a window.</param>
/// <param name="Window">How many of the last places of the order are read: nought for every earlier row.</param>
/// <remarks>
/// A window counts distinct places of the order, so rows of one date are one place, a window never cuts between them, and it
/// holds more rows than it has places where several stand on one. Without a window there must be a weight, since a key with few
/// rows is pulled towards the mean of every training row or it reads nothing; with one, a weight of nothing is the plain mean of
/// the rows it holds, and a window that holds none reads the mean of every training row.
/// </remarks>
public readonly record struct HistoryReach(double Shrink, int Window);

/// <summary>
/// What the earlier rows of the same kind held, as one number: the mean of a column over the training rows of a row's key that
/// came strictly before it, pulled towards the mean of every training row — the m-estimate a type history or a farm history is.
/// </summary>
/// <remarks>
/// <para>
/// A row reads <c>(sum + shrink · prior) / (count + shrink)</c>, where the sum and the count are of the training rows of its key
/// that come before it in the order its <see cref="OrderBy"/> columns say — the first deciding, each next one between rows the
/// ones before call equal — and the prior is the mean of the column over every training row. A row of the same order as
/// another is not before it, so no row reads its own value, one written beside it, or a later one, and a key no training row
/// holds, or a gap in the key, reads the prior. The weight has no default: how far a key with few rows is trusted is a
/// decision about the data. A window of so many places of the order reads only the last of them before the row, which is a
/// farm's last three flocks, and a window that holds none reads the prior.
/// </para>
/// <para>
/// What it learns is the training rows themselves — each one's key, its place in the order and its value — and the prior, so
/// a row served alone, long afterwards, reads what the same row would read in the set: the training rows that came before it.
/// The column it reads may be the answer: a training row reads only training rows before it, and every other row only
/// training rows, so nothing a model is chosen or measured on reaches a feature. The pipeline's file carries those training
/// rows' values. The number is in the units of the column it reads, and lands in no range until a scale says where.
/// </para>
/// </remarks>
public sealed record HistoryStep : IFittedStep, IPipelineStep<HistoryStep>, IDescribesColumns
{
    private static readonly ColumnParameter KeyKey = new(
        "key", "The column whose value says which rows are of one kind: a farm, a type.", "key", ColumnKinds.Any);

    private static readonly ColumnParameter ValueKey = new(
        "value", "The column of numbers whose mean over the earlier rows of the same kind is read.", "value", ColumnKinds.Numbers);

    private static readonly ColumnsParameter OrderByKey = new(
        "orderBy", "The columns that say which rows came before a row: the first decides, each next one decides between rows the ones before call equal.",
        ["when"], ColumnKinds.Ordered);

    private static readonly NumberParameter ShrinkKey = new(
        "shrink", "How many rows' worth of weight the mean of every training row has beside a key's own rows, above nothing: the larger, the more a key with few rows is pulled towards it.", 3)
    {
        AtLeast = 0,
    };

    private static readonly WholeNumberParameter WindowKey = new(
        "window", "How many of the last places of the order a row reads, rows of one place counting as one; left out, every earlier row. A window of nothing is read with a weight of nothing too.", 3)
    {
        AtLeast = 1,
        LeftOut = 0,
    };

    private static readonly NewColumnParameter IntoKey = new(
        "into", "The name of the column it makes; left out, the value's name, 'history' and the key's name.", "value_history_key", optional: true);

    /// <summary>Declares that each row reads what the earlier training rows of its key held.</summary>
    /// <param name="key">The column whose value says which rows are of one kind.</param>
    /// <param name="value">The column of numbers whose earlier mean is read.</param>
    /// <param name="orderBy">The columns that say which rows came before a row, the one that decides first first.</param>
    /// <param name="shrink">How many rows' worth of weight the training rows' mean has, above nothing.</param>
    /// <exception cref="ArgumentException">A column has no name, the order names none, or names one twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The weight is not above nothing.</exception>
    public HistoryStep(string key, string value, IEnumerable<string> orderBy, double shrink)
        : this(key, value, orderBy, new HistoryReach(shrink, 0))
    {
    }

    /// <summary>Declares that each row reads what the last earlier training rows of its key held.</summary>
    /// <param name="key">The column whose value says which rows are of one kind.</param>
    /// <param name="value">The column of numbers whose earlier mean is read.</param>
    /// <param name="orderBy">The columns that say which rows came before a row, the one that decides first first.</param>
    /// <param name="reach">How far back the rows read and what the training rows' mean weighs.</param>
    /// <exception cref="ArgumentException">A column has no name, the order names none, or names one twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The weight is below nothing or not a number, the window is below nothing, or there is neither a weight above nothing nor a window.</exception>
    public HistoryStep(string key, string value, IEnumerable<string> orderBy, HistoryReach reach)
    {
        ArgumentNullException.ThrowIfNull(orderBy);

        if (!(reach.Shrink >= 0) || reach.Window < 0 || (reach.Shrink == 0 && reach.Window == 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(reach),
                string.Create(CultureInfo.InvariantCulture, $"a weight of {reach.Shrink} and a window of {reach.Window}"),
                "The weight of the training rows' mean is above nothing, unless a window says how many places are read: then it may be nothing.");
        }

        Key = KeyKey.Require(key);
        Value = ValueKey.Require(value);
        OrderBy = OrderByKey.Require([.. orderBy]);
        Shrink = ShrinkKey.Require(reach.Shrink);
        Window = reach.Window;
    }

    /// <inheritdoc />
    public static StepParameters<HistoryStep> Parameters { get; } = new StepParameters<HistoryStep>()
        .With(KeyKey, step => step.Key)
        .With(ValueKey, step => step.Value)
        .With(OrderByKey, step => step.OrderBy)
        .With(ShrinkKey, step => step.Shrink)
        .With(WindowKey, step => step.Window)
        .With(IntoKey, step => step.Written);

    /// <summary>The column whose value says which rows are of one kind.</summary>
    public string Key { get; }

    /// <summary>The column of numbers whose earlier mean is read.</summary>
    public string Value { get; }

    /// <summary>The columns that say which rows came before a row.</summary>
    public IReadOnlyList<string> OrderBy { get; }

    /// <summary>How many rows' worth of weight the training rows' mean has beside a key's own rows.</summary>
    public double Shrink { get; }

    /// <summary>How many of the last places of the order a row reads; nought for every earlier row.</summary>
    public int Window { get; }

    /// <summary>The name of the column it makes, when one was said; nothing, for the name it makes up.</summary>
    public string? Written
    {
        get;
        init => field = IntoKey.Require(value);
    }

    /// <summary>The name of the column it makes: the one said, or the value's name, 'history' and the key's name.</summary>
    public string Into => Written ?? $"{Value}_history_{Key}";

    /// <inheritdoc />
    public static string Name => "history";

    /// <inheritdoc />
    public static string Purpose => "Writes beside each row the mean of a column over the earlier training rows of its key, pulled towards the training rows' mean.";

    /// <inheritdoc />
    public static int Since => 10;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Into, ColumnKind.Number);
    }

    /// <inheritdoc />
    public bool Equals(HistoryStep? other) =>
        other is not null && Key == other.Key && Value == other.Value && OrderBy.SequenceEqual(other.OrderBy) && Shrink.Equals(other.Shrink) && Window == other.Window && Written == other.Written;

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Key);
        hash.Add(Value);

        foreach (var column in OrderBy)
        {
            hash.Add(column);
        }

        hash.Add(Shrink);
        hash.Add(Window);
        hash.Add(Written);

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">No training row holds a number of the column, or a row has a gap where its order is said.</exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var prior = table.TrainingValues(Value, parts).Learnable("the mean of every training row").Mean;
        var values = table.NumbersOf(Value);
        var keys = table[Key];
        var order = Places(table);
        var kept = Enumerable.Range(0, table.RowCount).Where(row => parts[row] == Part.Train && values[row] is { } x && double.IsFinite(x) && !keys.IsMissing(row)).ToArray();
        var learned = new FittedStepValues();

        learned.Learned("prior", prior);
        learned.Learned("keys", [.. kept.Select(row => keys.TextAt(row)!)]);
        learned.Learned("values", [.. kept.Select(row => values[row]!.Value)]);

        for (var at = 0; at < OrderBy.Count; at++)
        {
            learned.Learned(string.Create(CultureInfo.InvariantCulture, $"order{at + 1}"), [.. kept.Select(row => order[row][at])]);
        }

        return learned;
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A row has a gap where its order is said.</exception>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var prior = fitted.Number("prior");
        var learnedKeys = fitted.List("keys");
        var learnedValues = fitted.Curve("values");
        var learnedOrder = Enumerable.Range(1, OrderBy.Count).Select(at => fitted.Curve(string.Create(CultureInfo.InvariantCulture, $"order{at}"))).ToArray();
        var earlier = Enumerable.Range(0, learnedKeys.Count)
            .GroupBy(row => learnedKeys[row], StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => new Earlier([.. group.Select(row => new Held([.. learnedOrder.Select(curve => curve[row])], learnedValues[row]))], Window > 0), StringComparer.Ordinal);
        var keys = table[Key];
        var order = Places(table);
        var read = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            var (sum, count) = !keys.IsMissing(row) && earlier.TryGetValue(keys.TextAt(row)!, out var rows) ? rows.Before(order[row], Window) : default;

            // A window of nothing held, read with no weight, has no mean of its own to read: it reads the prior.
            read[row] = count + Shrink == 0 ? prior : (sum + (Shrink * prior)) / (count + Shrink);
        }

        table.Put(new Column<double>(Into, ColumnKind.Number, read));
    }

    // Each row's place in the order, one number for each column that says it: a moment as days since 1970, a whole number and
    // a number as themselves — the declaration lets no other kind through. A row with a gap there has no place, and is refused
    // naming it.
    private double[][] Places(Table table)
    {
        var columns = OrderBy.Select(name => table[name]).ToArray();
        var places = new double[table.RowCount][];

        for (var row = 0; row < table.RowCount; row++)
        {
            places[row] = new double[columns.Length];

            for (var at = 0; at < columns.Length; at++)
            {
                places[row][at] = columns[at] switch
                {
                    _ when columns[at].IsMissing(row) => throw new InvalidOperationException(
                        $"Row {table.Identities[row].ReadAt + 1} has no '{OrderBy[at]}', so nothing says which rows came before it."),
                    Column<DateTime> moments => (moments[row]!.Value - DateTime.UnixEpoch).TotalDays,
                    Column<long> whole => whole[row]!.Value,
                    _ => ((Column<double>)columns[at])[row]!.Value,
                };
            }
        }

        return places;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static HistoryStep ReadFrom(JsonElement element) =>
        new(KeyKey.Read(element), ValueKey.Read(element), OrderByKey.Read(element), new HistoryReach(ShrinkKey.Read(element), WindowKey.Read(element))) { Written = IntoKey.Read(element) };

    // The sum and the count of the training rows of a key that came before a row.
    private readonly record struct Counted(double Sum, int Count);

    // One training row of a key: its place in the order, and its value.
    private readonly record struct Held(double[] Place, double Value);

    // The training rows of one key, in their order, with the running sum of their values: what came before a place is a search. Where
    // a window reads only the last places, the rows are put in an order that is the same whatever order they were handed in — by place
    // and then by value — and what a window holds is added up row by row, since the difference of two running totals is not the sum of
    // the rows between them once a total has lost the small values a large one swallowed.
    private sealed class Earlier
    {
        private readonly double[][] _places;
        private readonly double[] _values;
        private readonly double[] _sums;
        private readonly int[] _starts;

        public Earlier(Held[] rows, bool windowed)
        {
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

        // The sum and the count of the rows strictly before a place: all of them, or those on the last places of a window.
        public Counted Before(double[] place, int window)
        {
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

        private static int Compared(double[] one, double[] other)
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
}
