// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What the earlier rows of the same kind held, as one number — the training rows' mean pulled towards a key's own rows, as
/// <see cref="HistoryStep"/> reads them — plus every other row of the key already known by the time a later row of it is asked
/// for, whatever part it is in: a row served with the rows its own producer answered since the pipeline was fitted, in time
/// order, carries them, so a validation or a test row reads them too, the way their outcome is known in production by the time
/// a later row of the same producer arrives.
/// </summary>
/// <remarks>
/// <para>
/// The training rows are the floor every key has, fitted once and never re-read: without them, a key served alone with nothing
/// handed in beside it would have no history at all, which is the one thing a history may never do. A training row reads that
/// floor and nothing else, so nothing a model is chosen or measured on reaches what it learns from, under any split. Every
/// other row reads the floor and every row of its key strictly before it in the table it is applied to that is not a training
/// row — a validation row an earlier test row too, since its outcome is known by then. A training row is known again by what
/// it says in the step's own columns — its key, its order, its value and the declared id — which the fit keeps as the
/// <c>members</c>; so a training row, present in both the fitted file and the table it was fitted from, is never read twice, and
/// a fit replayed over its own rows writes the column it wrote.
/// </para>
/// <para>
/// A gap in the value column is not yet known and is not read; an answer a later step still has to make is a gap in every row
/// it is handed to for serving, so a row served before that answer exists reads nothing from it, and a row served after it
/// reads whatever was handed in. A fit that kept no members, as one an earlier release wrote, reads the rows after a key's last
/// training row, as it did then.
/// </para>
/// <para>
/// It reads no row in the order the rows stand in, since it orders by its own columns, so it asks for no order and stands below
/// any split. Its value may not be a column an output made from rows ahead: that column holds a value from later than the row it
/// stands on.
/// </para>
/// </remarks>
public sealed record KnownHistoryStep : EarlierRowsStep<KnownHistoryStep>, IFittedStep, IPipelineStep<KnownHistoryStep>, IDescribesColumns, IReadsAnEarlierValue
{
    private const string Members = "members";

    private static readonly NewColumnParameter IntoParameter = new(
        "into", "The name of the column it makes; left out, the value's name, 'knownHistory' and the key's name.", "value_knownHistory_key", optional: true);

    /// <summary>Declares that each row reads what the earlier rows of its key already held, known at the time it is asked for.</summary>
    /// <param name="key">The column whose value says which rows are of one kind.</param>
    /// <param name="value">The column of numbers whose earlier mean is read.</param>
    /// <param name="orderBy">The columns that say which rows came before a row, the one that decides first first.</param>
    /// <param name="reach">How far back the rows read and what the training rows' mean weighs.</param>
    /// <exception cref="ArgumentException">A column has no name, the order names none, or names one twice.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The weight is below nothing or not a number, the window is below nothing, or there is neither a weight above nothing nor a window.</exception>
    public KnownHistoryStep(string key, string value, IEnumerable<string> orderBy, HistoryReach reach)
        : base(key, value, orderBy, reach)
    {
    }

    /// <inheritdoc />
    public static StepParameters<KnownHistoryStep> Parameters { get; } = new StepParameters<KnownHistoryStep>()
        .With(KeyKey, step => step.Key)
        .With(ValueKey, step => step.Value)
        .With(OrderByKey, step => step.OrderBy)
        .With(ShrinkKey, step => step.Shrink)
        .With(WindowKey, step => step.Window)
        .With(IntoParameter, step => step.Written);

    /// <inheritdoc />
    public static string Name => "history.known";

    /// <inheritdoc />
    public static string Purpose => "Writes beside each row the mean of a column over the earlier rows of its key already known at the time it is asked for, pulled towards the training rows' mean.";

    /// <inheritdoc />
    public static int Since => 16;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    private protected override NewColumnParameter IntoKey => IntoParameter;

    /// <inheritdoc />
    private protected override string Called => "knownHistory";

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static KnownHistoryStep ReadFrom(JsonElement element) =>
        new(KeyKey.Read(element), ValueKey.Read(element), OrderByKey.Read(element), new HistoryReach(ShrinkKey.Read(element), WindowKey.Read(element))) { Written = IntoParameter.Read(element) };

    /// <inheritdoc />
    /// <remarks>Every training row, gaps in the value included, by what it says in the step's own columns.</remarks>
    private protected override void Remember(Table table, IReadOnlyList<Part> parts, FittedStepValues learned) =>
        learned.Learned(Members, [.. Enumerable.Range(0, table.RowCount).Where(parts.IsTraining).Select(row => MemberOf(table, row)).Distinct().Order(StringComparer.Ordinal)]);

    /// <inheritdoc />
    private protected override KnownRows Known(Table table, FittedStepValues fitted, IReadOnlyDictionary<string, Held[]> floor, double[][] order)
    {
        var keys = table[Key];
        var values = table.NumbersOf(Value);

        if (!fitted.Lists.TryGetValue(Members, out var kept))
        {
            return AfterTheFloor(table, floor, order);
        }

        var members = kept.ToHashSet(StringComparer.Ordinal);
        var isMember = Enumerable.Range(0, table.RowCount).Select(row => members.Contains(MemberOf(table, row))).ToArray();
        var context = Enumerable.Range(0, table.RowCount)
            .Where(row => !isMember[row] && !keys.IsMissing(row) && values[row] is { } v && double.IsFinite(v))
            .GroupBy(row => keys.TextAt(row)!, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Select(row => new Held(order[row], values[row]!.Value)).ToArray(), StringComparer.Ordinal);

        return new KnownRows(row => isMember[row], context);
    }

    // A fit an earlier release wrote kept no members: every row read the floor and the rows of its key after the key's last
    // training row, which under the split in time it stood below were the rows known since the fit.
    private KnownRows AfterTheFloor(Table table, IReadOnlyDictionary<string, Held[]> floor, double[][] order)
    {
        var keys = table[Key];
        var values = table.NumbersOf(Value);
        var horizon = floor.ToDictionary(group => group.Key, group => group.Value.Select(row => row.Place).Aggregate((latest, place) => Earlier.Compared(place, latest) > 0 ? place : latest), StringComparer.Ordinal);
        var context = Enumerable.Range(0, table.RowCount)
            .Where(row => !keys.IsMissing(row) && values[row] is { } v && double.IsFinite(v))
            .GroupBy(row => keys.TextAt(row)!, StringComparer.Ordinal)
            .ToDictionary(
                group => group.Key,
                group => group.Where(row => !horizon.TryGetValue(group.Key, out var last) || Earlier.Compared(order[row], last) > 0).Select(row => new Held(order[row], values[row]!.Value)).ToArray(),
                StringComparer.Ordinal);

        return new KnownRows(_ => false, context);
    }

    // A row as the step knows it again: what it says in the step's own columns, and the declared id where one stands.
    private string MemberOf(Table table, int row)
    {
        string[] names = table.IdColumnInForce is { } id ? [Key, .. OrderBy, Value, id.Name] : [Key, .. OrderBy, Value];

        return RowKey.Of(names, [.. names.Select(name => table[name].TextAt(row))]).ToString();
    }
}
