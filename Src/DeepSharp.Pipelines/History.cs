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
/// What the earlier rows of the same kind held, as one number: the mean of a column over the rows of a row's key that came
/// strictly before it, pulled towards the mean of every training row — the m-estimate a type history or a farm history is.
/// </summary>
/// <typeparam name="TSelf">The step itself.</typeparam>
/// <remarks>
/// <para>
/// A row reads <c>(sum + shrink · prior) / (count + shrink)</c>, where the sum and the count are of the rows of its key that come
/// before it in the order its <see cref="OrderBy"/> columns say — the first deciding, each next one between rows the ones before
/// call equal — and the prior is the mean of the column over every training row. A row of the same order as another is not before
/// it, so no row reads its own value, one written beside it, or a later one, and a key no row holds, or a gap in the key, reads
/// the prior.
/// </para>
/// <para>
/// What every such step learns is the training rows themselves — each one's key, its place in the order and its value — and the
/// prior: the floor every key has, fitted once and replayed unchanged, so a row served alone, long afterwards, still reads its
/// key's training rows. A training row reads the floor and nothing else, so nothing a model is chosen or measured on reaches
/// what it learns from. What one step of this kind differs from another in is only which rows besides the floor a row that is
/// not a training row may read.
/// </para>
/// </remarks>
public abstract record EarlierRowsStep<TSelf>
    where TSelf : EarlierRowsStep<TSelf>
{
    private protected static readonly ColumnParameter KeyKey = new(
        "key", "The column whose value says which rows are of one kind: a farm, a type.", "key", ColumnKinds.Any);

    private protected static readonly ColumnParameter ValueKey = new(
        "value", "The column of numbers whose mean over the earlier rows of the same kind is read.", "value", ColumnKinds.Numbers);

    private protected static readonly ColumnsParameter OrderByKey = new(
        "orderBy", "The columns that say which rows came before a row: the first decides, each next one decides between rows the ones before call equal.",
        ["when"], ColumnKinds.Ordered);

    private protected static readonly NumberParameter ShrinkKey = new(
        "shrink", "How many rows' worth of weight the mean of every training row has beside a key's own rows, above nothing: the larger, the more a key with few rows is pulled towards it.", 3)
    {
        AtLeast = 0,
    };

    private protected static readonly WholeNumberParameter WindowKey = new(
        "window", "How many of the last places of the order a row reads, rows of one place counting as one; left out, every earlier row. A window of nothing is read with a weight of nothing too.", 3)
    {
        AtLeast = 1,
        LeftOut = 0,
    };

    /// <summary>Declares the step from what every history is told.</summary>
    /// <param name="key">The column whose value says which rows are of one kind.</param>
    /// <param name="value">The column of numbers whose earlier mean is read.</param>
    /// <param name="orderBy">The columns that say which rows came before a row, the one that decides first first.</param>
    /// <param name="reach">How far back the rows read and what the training rows' mean weighs.</param>
    /// <exception cref="ArgumentException">A column has no name, the order names none, or names one twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The weight is below nothing or not a number, the window is below nothing, or there is neither a weight above nothing nor a window.</exception>
    private protected EarlierRowsStep(string key, string value, IEnumerable<string> orderBy, HistoryReach reach)
    {
        ArgumentNullException.ThrowIfNull(orderBy);

        if (double.IsNaN(reach.Shrink) || reach.Shrink < 0 || reach.Window < 0 || (reach.Shrink is 0d && reach.Window == 0))
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

    /// <summary>The column whose value says which rows are of one kind.</summary>
    public string Key { get; }

    /// <summary>The column of numbers whose earlier mean is read.</summary>
    public string Value { get; }

    /// <summary>The columns that say which rows came before a row, the one that decides first first.</summary>
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

    /// <summary>The name of the column it makes: the one said, or the value's name, what the step is called, and the key's name.</summary>
    public string Into => Written ?? $"{Value}_{Called}_{Key}";

    // The key the name of the column a step makes is written under, with the example its description gives.
    private protected abstract NewColumnParameter IntoKey { get; }

    // What the step is called in the name of the column it makes when none is said.
    private protected abstract string Called { get; }

    /// <summary>The columns a step leaves behind: those it found, and the one it makes.</summary>
    /// <param name="before">The columns before the step.</param>
    /// <returns>The columns after it.</returns>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Into, ColumnKind.Number);
    }

    /// <summary>Learns the training rows of every key, each with its place and its value, and the mean of every training row.</summary>
    /// <param name="table">The data.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <returns>What was learned.</returns>
    /// <exception cref="InvalidOperationException">No training row holds a number of the column, or a row has a gap where its order is said.</exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var learned = table.FitTheEarlierTrainingRows(parts, new EarlierColumns(Value, Key, OrderBy));

        Remember(table, parts, learned);

        return learned;
    }

    /// <summary>Writes beside every row the mean of the rows of its key it may read that came strictly before it.</summary>
    /// <param name="table">The data, changed in place.</param>
    /// <param name="fitted">What the fit learned.</param>
    /// <exception cref="InvalidOperationException">A row has a gap where its order is said.</exception>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var prior = fitted.Number("prior");
        var floorRows = fitted.TrainingRowsByKey(OrderBy.Count);
        var keys = table[Key];
        var order = table.PlacesBy(OrderBy);
        var known = Known(table, fitted, floorRows, order);
        var floor = floorRows.ToDictionary(group => group.Key, group => new Earlier(group.Value, Window > 0), StringComparer.Ordinal);
        var every = known.Context.Count == 0
            ? floor
            : floorRows.Keys.Union(known.Context.Keys, StringComparer.Ordinal).ToDictionary(
                key => key,
                key => new Earlier([.. floorRows.GetValueOrDefault(key, []), .. known.Context.GetValueOrDefault(key, [])], Window > 0),
                StringComparer.Ordinal);
        var read = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            // A training row reads the floor alone; every other row the floor and whatever else it may read.
            var readable = known.IsMember(row) ? floor : every;
            var counted = !keys.IsMissing(row) && readable.TryGetValue(keys.TextAt(row)!, out var rows) ? rows.Before(order[row], Window) : default;

            read[row] = counted.Blended(Shrink, prior);
        }

        table.Put(new Column<double>(Into, ColumnKind.Number, read));
    }

    // What a step keeps of the fit beyond the floor; nothing, unless it reads more than the floor.
    private protected virtual void Remember(Table table, IReadOnlyList<Part> parts, FittedStepValues learned)
    {
    }

    // Which rows are training rows, reading the floor alone, and which rows besides the floor the others may read, by key.
    private protected abstract KnownRows Known(Table table, FittedStepValues fitted, IReadOnlyDictionary<string, Held[]> floor, double[][] order);
}

/// <summary>Which rows read the floor alone, and what the other rows may read besides it.</summary>
/// <param name="IsMember">Whether a row is one of the training rows, and so reads the floor alone.</param>
/// <param name="Context">The rows a row that is not may read besides the floor, by key.</param>
internal readonly record struct KnownRows(Func<int, bool> IsMember, IReadOnlyDictionary<string, Held[]> Context)
{
    /// <summary>The floor alone, for every row.</summary>
    internal static KnownRows FloorAlone { get; } = new(_ => true, new Dictionary<string, Held[]>(StringComparer.Ordinal));
}

/// <summary>
/// What the earlier training rows of the same kind held, as one number: the mean of a column over the training rows of a row's
/// key that came strictly before it, pulled towards the mean of every training row.
/// </summary>
/// <remarks>
/// Every row reads the training rows alone, so the column it reads may be the answer: a training row reads only training rows
/// before it, and every other row only training rows, so nothing a model is chosen or measured on reaches a feature. It reads
/// no row in the order the rows stand in, so it asks for no order and no particular split above it. The pipeline's file
/// carries those training rows' values. The number is in the units of the column it reads, and lands in no range until a scale
/// says where. <see cref="EarlierRowsStep{TSelf}"/> says how the number is made.
/// </remarks>
public sealed record HistoryStep : EarlierRowsStep<HistoryStep>, IFittedStep, IPipelineStep<HistoryStep>, IDescribesColumns, IReadsAnEarlierValue
{
    private static readonly NewColumnParameter IntoParameter = new(
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
        : base(key, value, orderBy, reach)
    {
    }

    /// <inheritdoc />
    public static StepParameters<HistoryStep> Parameters { get; } = new StepParameters<HistoryStep>()
        .With(KeyKey, step => step.Key)
        .With(ValueKey, step => step.Value)
        .With(OrderByKey, step => step.OrderBy)
        .With(ShrinkKey, step => step.Shrink)
        .With(WindowKey, step => step.Window)
        .With(IntoParameter, step => step.Written);

    /// <inheritdoc />
    public static string Name => "history";

    /// <inheritdoc />
    public static string Purpose => "Writes beside each row the mean of a column over the earlier training rows of its key, pulled towards the training rows' mean.";

    /// <inheritdoc />
    public static int Since => 10;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    private protected override NewColumnParameter IntoKey => IntoParameter;

    /// <inheritdoc />
    private protected override string Called => "history";

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static HistoryStep ReadFrom(JsonElement element) =>
        new(KeyKey.Read(element), ValueKey.Read(element), OrderByKey.Read(element), new HistoryReach(ShrinkKey.Read(element), WindowKey.Read(element))) { Written = IntoParameter.Read(element) };

    /// <inheritdoc />
    private protected override KnownRows Known(Table table, FittedStepValues fitted, IReadOnlyDictionary<string, Held[]> floor, double[][] order) => KnownRows.FloorAlone;
}
