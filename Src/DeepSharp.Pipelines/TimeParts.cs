// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>Which piece of a moment in time becomes a column of its own.</summary>
public enum TimePart
{
    /// <summary>The minute within the hour.</summary>
    Minute,

    /// <summary>The hour of the day.</summary>
    Hour,

    /// <summary>The day of the week, written as its name.</summary>
    DayOfWeek,

    /// <summary>The day of the month.</summary>
    DayOfMonth,

    /// <summary>The month, written as its number.</summary>
    Month,

    /// <summary>The quarter of the year.</summary>
    Quarter,

    /// <summary>The season, by the meteorological months of the northern hemisphere.</summary>
    Season,

    /// <summary>The year.</summary>
    Year,
}

/// <summary>
/// A step that says which of the columns it produces stand for a group rather than for themselves.
/// </summary>
/// <remarks>
/// The schema says it for the columns that came out of the file; this says it for the columns a step made.
/// Either way it is said once, and <see cref="FittingBuilder.EncodeCategories"/> collects from both rather
/// than asking again.
/// </remarks>
public interface IDeclaresCategories : IPipelineStep
{
    /// <summary>The columns this step produces that stand for a group.</summary>
    IEnumerable<string> Categories { get; }
}

/// <summary>
/// Takes a moment in time apart into the pieces people actually reason with.
/// </summary>
/// <remarks>
/// A month is not a quantity. March is not three of anything, and a model told that December is twelve
/// times January has been told something false about the world. So these arrive as categories by default,
/// and <see cref="FittingBuilder.EncodeCategories"/> turns them into the numbers a model can take.
/// <para>
/// Ask for them as numbers when the order is the point rather than the grouping — a year over a long span,
/// say, where each one being its own category would be a column per year and a certain refusal the first
/// time next year arrives. And where a piece of time wraps round, <see cref="PipelineBuilder.Cyclical"/>
/// says that better than either: eleven at night and midnight are neighbours, which no category knows.
/// </para>
/// </remarks>
public sealed record TimePartsStep : IPipelineStep<TimePartsStep>, IAddsColumns, IDeclaresCategories, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column holding the moment in time.", "when", ColumnKinds.Moments);

    private static readonly TrueOrFalseParameter AsCategoriesKey = new(
        "asCategories", "Whether the pieces stand for a group, which they do unless the order is the point.", true);

    private static readonly SeveralOfParameter<TimePart> PartsKey = new(
        "parts", "Which pieces of the moment become columns of their own.", [TimePart.Month]);

    /// <summary>Declares that a moment in time is taken apart.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="parts">Which pieces to take out of it.</param>
    /// <param name="asCategories">Whether the pieces stand for a group; they do, unless you say otherwise.</param>
    /// <exception cref="ArgumentException">The column has no name, or no parts were asked for.</exception>
    public TimePartsStep(string column, IEnumerable<TimePart> parts, bool asCategories = true)
    {
        ArgumentNullException.ThrowIfNull(parts);

        Column = ColumnKey.Require(column);
        Parts = PartsKey.Require([.. parts]);
        AsCategories = asCategories;
    }

    /// <inheritdoc />
    public static StepParameters<TimePartsStep> Parameters { get; } = new StepParameters<TimePartsStep>()
        .With(ColumnKey, step => step.Column)
        .With(AsCategoriesKey, step => step.AsCategories)
        .With(PartsKey, step => step.Parts);

    /// <summary>The column holding the moment.</summary>
    public string Column { get; }

    /// <summary>Which pieces are taken out of it.</summary>
    public IReadOnlyList<TimePart> Parts { get; }

    /// <summary>Whether the pieces stand for a group rather than for themselves.</summary>
    public bool AsCategories { get; }

    /// <inheritdoc />
    public IEnumerable<string> Categories =>
        AsCategories ? Parts.Select(NameOf) : [];

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Parts.Aggregate(before, (state, part) => state.With(NameOf(part), AsCategories ? ColumnKind.Category : ColumnKind.Number));
    }

    /// <inheritdoc />
    public static string Name => "feature.timeParts";

    /// <inheritdoc />
    public static string Purpose => "Takes a moment in time apart into the pieces people reason with: an hour, a weekday, a month.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public bool Equals(TimePartsStep? other) =>
        other is not null
        && Column == other.Column
        && AsCategories == other.AsCategories
        && Parts.SequenceEqual(other.Parts);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Column);
        hash.Add(AsCategories);

        foreach (var part in Parts)
        {
            hash.Add(part);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        if (table[Column] is not Column<DateTime> moments)
        {
            throw new InvalidOperationException(
                $"'{Column}' holds {table[Column].Kind.ToString().ToLowerInvariant()} "
                + "and there is no moment in time to take apart.");
        }

        foreach (var part in Parts)
        {
            var written = Enumerable.Range(0, table.RowCount)
                .Select(row => moments[row] is { } moment ? Piece(part, moment) : null)
                .ToArray();

            table.Put(AsCategories
                ? new TextColumn(NameOf(part), ColumnKind.Category, written)
                : new Column<double>(
                    NameOf(part), ColumnKind.Number,
                    written.Select(value => value is null
                        ? (double?)null
                        : double.Parse(value, CultureInfo.InvariantCulture))));
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or names a piece nobody defined.</exception>
    public static TimePartsStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), PartsKey.Read(element), AsCategoriesKey.Read(element));

    private string NameOf(TimePart part) => $"{Column}_{part.ToString().ToLowerInvariant()}";

    private static string Piece(TimePart part, DateTime moment) => part switch
    {
        TimePart.Minute => moment.Minute.ToString(CultureInfo.InvariantCulture),
        TimePart.Hour => moment.Hour.ToString(CultureInfo.InvariantCulture),
        // The name rather than the number, because a category written as a number invites somebody to
        // average it, and the average of Tuesday and Thursday is not Wednesday.
        TimePart.DayOfWeek => moment.DayOfWeek.ToString().ToLowerInvariant(),
        TimePart.DayOfMonth => moment.Day.ToString(CultureInfo.InvariantCulture),
        TimePart.Month => moment.Month.ToString(CultureInfo.InvariantCulture),
        TimePart.Quarter => (((moment.Month - 1) / 3) + 1).ToString(CultureInfo.InvariantCulture),
        // Meteorological seasons, northern hemisphere: December to February is winter. A convention rather
        // than a fact, so it is written down here where somebody can disagree with it on purpose.
        TimePart.Season => moment.Month switch
        {
            12 or 1 or 2 => "winter",
            3 or 4 or 5 => "spring",
            6 or 7 or 8 => "summer",
            _ => "autumn",
        },
        _ => moment.Year.ToString(CultureInfo.InvariantCulture),
    };
}
