// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What the earlier rows of the same kind held, as one number — the training rows' mean pulled towards a key's own
/// rows, as <see cref="HistoryStep"/> reads them — plus every other row of the key already known by the time a
/// later row of it is asked for: a row served with the rows its own producer answered since the pipeline was
/// fitted, in time order, carries them, so a validation or a test row reads them too, the way their outcome is
/// known in production by the time a later row of the same producer arrives.
/// </summary>
/// <remarks>
/// <para>
/// The training rows are the floor every key has, fitted once and never re-read: without them, a key served alone
/// with nothing handed in beside it would have no history at all, which is the one thing a history may never do.
/// Above that floor, a row counts once: the training rows stop at their own last place in the order, and only a row
/// strictly after that place, and strictly before the row being computed, is read from the table it is applied to —
/// so a training row, present in both the fitted file and the table it was fitted from, is never read twice.
/// </para>
/// <para>
/// A gap in the value column is not yet known and is not read; an answer a later step still has to make is a gap in
/// every row it is handed to for serving, so a row served before that answer exists reads nothing from it, and a row
/// served after it reads whatever was handed in.
/// </para>
/// </remarks>
public sealed record KnownHistoryStep : IFittedStep, IPipelineStep<KnownHistoryStep>, IDescribesColumns, IReadsRowOrder, IReadsAnEarlierValue
{
    private static readonly ColumnParameter KeyKey = new(
        "key", "The column whose value says which rows are of one kind: a farm, a type.", "key", ColumnKinds.Any);

    private static readonly ColumnParameter ValueKey = new(
        "value", "The column of numbers whose mean over the earlier rows of the same kind is read.", "value", ColumnKinds.Numbers);

    private static readonly ColumnsParameter OrderByKey = new(
        "orderBy", "The columns that say which rows came before a row: the first is the column the split above divides time by, each next one decides between rows the ones before call equal.",
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
        "into", "The name of the column it makes; left out, the value's name, 'knownHistory' and the key's name.", "value_knownHistory_key", optional: true);

    /// <summary>Declares that each row reads what the earlier rows of its key already held, known at the time it is asked for.</summary>
    /// <param name="key">The column whose value says which rows are of one kind.</param>
    /// <param name="value">The column of numbers whose earlier mean is read.</param>
    /// <param name="orderBy">The columns that say which rows came before a row, the one the split divides time by first.</param>
    /// <param name="reach">How far back the rows read and what the training rows' mean weighs.</param>
    /// <exception cref="ArgumentException">A column has no name, the order names none, or names one twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The weight is below nothing or not a number, the window is below nothing, or there is neither a weight above nothing nor a window.</exception>
    public KnownHistoryStep(string key, string value, IEnumerable<string> orderBy, HistoryReach reach)
    {
        ArgumentNullException.ThrowIfNull(orderBy);

        if (double.IsNaN(reach.Shrink) || reach.Shrink < 0 || reach.Window < 0 || (reach.Shrink is 0d && reach.Window == 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(reach),
                string.Create(System.Globalization.CultureInfo.InvariantCulture, $"a weight of {reach.Shrink} and a window of {reach.Window}"),
                "The weight of the training rows' mean is above nothing, unless a window says how many places are read: then it may be nothing.");
        }

        Key = KeyKey.Require(key);
        Value = ValueKey.Require(value);
        OrderBy = OrderByKey.Require([.. orderBy]);
        Shrink = ShrinkKey.Require(reach.Shrink);
        Window = reach.Window;
    }

    /// <inheritdoc />
    public static StepParameters<KnownHistoryStep> Parameters { get; } = new StepParameters<KnownHistoryStep>()
        .With(KeyKey, step => step.Key)
        .With(ValueKey, step => step.Value)
        .With(OrderByKey, step => step.OrderBy)
        .With(ShrinkKey, step => step.Shrink)
        .With(WindowKey, step => step.Window)
        .With(IntoKey, step => step.Written);

    /// <summary>The column whose value says which rows are of one kind.</summary>
    public string Key { get; }

    /// <inheritdoc />
    public string Value { get; }

    /// <summary>The columns that say which rows came before a row; the first is the one the split above divides time by.</summary>
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

    /// <summary>The name of the column it makes: the one said, or the value's name, 'knownHistory' and the key's name.</summary>
    public string Into => Written ?? $"{Value}_knownHistory_{Key}";

    /// <inheritdoc />
    public static string Name => "history.known";

    /// <inheritdoc />
    public static string Purpose => "Writes beside each row the mean of a column over the earlier rows of its key already known at the time it is asked for, pulled towards the training rows' mean.";

    /// <inheritdoc />
    public static int Since => 16;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Into, ColumnKind.Number);
    }

    /// <inheritdoc />
    public bool Equals(KnownHistoryStep? other) =>
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

        return table.FitTheEarlierTrainingRows(parts, Value, Key, OrderBy);
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A row has a gap where its order is said.</exception>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var prior = fitted.Number("prior");
        var trainingRows = fitted.TrainingRowsByKey(OrderBy.Count);
        var horizon = trainingRows.ToDictionary(group => group.Key, group => group.Value.Select(row => row.Place).Aggregate((latest, place) => Earlier.Compared(place, latest) > 0 ? place : latest), StringComparer.Ordinal);
        var keys = table[Key];
        var values = table.NumbersOf(Value);
        var order = table.PlacesBy(OrderBy);

        var context = Enumerable.Range(0, table.RowCount)
            .Where(row => !keys.IsMissing(row) && values[row] is { } v && double.IsFinite(v))
            .GroupBy(row => keys.TextAt(row)!, StringComparer.Ordinal)
            .ToDictionary(
                group => group.Key,
                group => group.Where(row => !horizon.TryGetValue(group.Key, out var last) || Earlier.Compared(order[row], last) > 0).Select(row => new Held(order[row], values[row]!.Value)).ToArray(),
                StringComparer.Ordinal);

        var earlier = trainingRows.Keys.Union(context.Keys, StringComparer.Ordinal)
            .ToDictionary(
                key => key,
                key => new Earlier([.. trainingRows.GetValueOrDefault(key, []), .. context.GetValueOrDefault(key, [])], Window > 0),
                StringComparer.Ordinal);

        var read = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            var counted = !keys.IsMissing(row) && earlier.TryGetValue(keys.TextAt(row)!, out var rows) ? rows.Before(order[row], Window) : default;

            read[row] = counted.Blended(Shrink, prior);
        }

        table.Put(new Column<double>(Into, ColumnKind.Number, read));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static KnownHistoryStep ReadFrom(JsonElement element) =>
        new(KeyKey.Read(element), ValueKey.Read(element), OrderByKey.Read(element), new HistoryReach(ShrinkKey.Read(element), WindowKey.Read(element))) { Written = IntoKey.Read(element) };
}
