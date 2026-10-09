// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that writes a value as a place on a circle, so that the ends of a cycle meet.
/// </summary>
/// <remarks>
/// A moment on the cycle of its day and a number on a cycle of a length are two questions, each with its own verb, and they
/// answer alike: two values, a sine and a cosine, written in the form the pipeline lands its features in. Saying so in the
/// type is what lets the one rule that governs the form — where the features land is said above every step it governs —
/// hold for both, and for a step another package brings.
/// </remarks>
public interface IPlacesOnACircle : IAddsColumns
{
    /// <summary>How the two values of a place are written down.</summary>
    Form Form { get; }
}

/// <summary>Where a value stands on a circle: how far along it is, and how long the circle is.</summary>
/// <param name="Position">How far along the circle the value stands, counted from its start, in the units of its length.</param>
/// <param name="Length">How long one turn is.</param>
/// <remarks>
/// The turn is worked out in the one order, so a place gives the same sine and cosine whichever step asked for it: twice
/// pi times the position, over the length.
/// </remarks>
internal readonly record struct CirclePlace(double Position, double Length)
{
    /// <summary>The sine of the place.</summary>
    public double Sine => Math.Sin(Turn);

    /// <summary>The cosine of the place.</summary>
    public double Cosine => Math.Cos(Turn);

    private double Turn => 2 * Math.PI * Position / Length;
}

/// <summary>
/// Where a moment stands, and what it is called, on the cycles of the calendar.
/// </summary>
internal static class MomentExtensions
{
    // The seasons by the meteorological months of the northern hemisphere, in the order a year meets them from the start of
    // December: a convention rather than a fact, so it is written down once, here, where somebody can disagree with it on
    // purpose. A piece of a moment and a place on a circle both read it, so they cannot disagree about a month.
    private static readonly string[] SeasonNames = ["winter", "spring", "summer", "autumn"];

    extension(DateTime moment)
    {
        /// <summary>The season the moment falls in, counted from winter (nought): December to February, then each three months.</summary>
        internal int Season => moment.Month is 12 or 1 or 2 ? 0 : ((moment.Month - 3) / 3) + 1;

        /// <summary>The name of the season the moment falls in.</summary>
        internal string SeasonName => SeasonNames[moment.Season];

        /// <summary>Where the moment stands on the cycle of a period.</summary>
        /// <param name="period">The cycle.</param>
        /// <returns>The place on it: how far along, and how long the turn is.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The period is none of the cycles there are: a number cast to the set.</exception>
        /// <remarks>
        /// Every cycle is its own arm, so a cycle added to the set is a cycle nobody can forget to place: a moment on one that
        /// has no arm is refused here instead of being placed on the nearest.
        /// </remarks>
        internal CirclePlace PlaceOn(Period period) => period switch
        {
            Period.HourOfDay => new(moment.Hour + (moment.Minute / 60.0), 24),
            Period.DayOfWeek => new((int)moment.DayOfWeek, 7),
            Period.DayOfMonth => new(moment.Day - 1, 31),
            Period.MonthOfYear => new(moment.Month - 1, 12),
            Period.DayOfYear => new(moment.DayOfYear - 1, DateTime.IsLeapYear(moment.Year) ? 366 : 365),
            Period.Season => new(moment.Season, 4),
            _ => throw new ArgumentOutOfRangeException(nameof(period), period, "There is no such cycle."),
        };
    }
}

/// <summary>
/// Writing places on a circle onto a table, in the form the pipeline lands its features in.
/// </summary>
internal static class CirclePlacesExtensions
{
    extension(Table table)
    {
        /// <summary>Puts the sine and the cosine of places onto the table; a row without a place stays a gap.</summary>
        /// <param name="places">The place of each row, or nothing where the row has none.</param>
        /// <param name="stems">The names of the sine and the cosine.</param>
        /// <param name="form">How the two values are written down.</param>
        internal void PutPlaces(CirclePlace?[] places, string[] stems, Form form)
        {
            var sines = new double?[places.Length];
            var cosines = new double?[places.Length];

            for (var row = 0; row < places.Length; row++)
            {
                sines[row] = places[row]?.Sine;
                cosines[row] = places[row]?.Cosine;
            }

            foreach (var column in form.Written(stems[0], sines).Concat(form.Written(stems[1], cosines)))
            {
                table.Put(column);
            }
        }
    }
}

/// <summary>
/// Writes a number as a place on a circle of a length you give, so that the ends of the cycle meet.
/// </summary>
/// <remarks>
/// An age in days is a place in a week of seven, a phase is a place in a period of its own, and the end of a week is next to
/// its start, which a plain number tells a model is as far off as two values can be. The length is said, never worked out
/// from the rows: a length taken from the largest value would be learned from the rows, and a cycle of seven does not depend on
/// which ages a flock happened to reach. A value beyond one turn, or below nought, is placed by where it falls in the length.
/// A moment in time is the other question, and has its own verb: <see cref="CyclicalStep"/>.
/// </remarks>
public sealed record CycleStep : IPipelineStep<CycleStep>, IPlacesOnACircle, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column holding the number.", "age", ColumnKinds.Quantities);

    private static readonly NumberParameter LengthKey = new(
        "length", "How long one turn is, in the column's own units: 7 for an age in days on a week. Said by you, never worked out from the rows.", 7)
    {
        Above = 0,
    };

    private static readonly OneOfParameter<Form> FormKey = new(
        "form", "How the two values are written down: as they are, shifted between nothing and one, or split into how far up and how far down.", Form.Signed);

    /// <summary>Declares a number written as a place on a circle.</summary>
    /// <param name="column">The column holding the number.</param>
    /// <param name="length">How long one turn is, in the column's own units.</param>
    /// <param name="form">How to write the two values down.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The length is not a number above nought.</exception>
    public CycleStep(string column, double length, Form form = Form.Signed)
    {
        Column = ColumnKey.Require(column);
        Length = LengthKey.Require(length);
        Form = FormKey.Require(form);
    }

    /// <inheritdoc />
    public static StepParameters<CycleStep> Parameters { get; } = new StepParameters<CycleStep>()
        .With(ColumnKey, step => step.Column)
        .With(LengthKey, step => step.Length)
        .With(FormKey, step => step.Form);

    /// <summary>The column holding the number.</summary>
    public string Column { get; }

    /// <summary>How long one turn is, in the column's own units.</summary>
    public double Length { get; }

    /// <inheritdoc />
    public Form Form { get; }

    /// <inheritdoc />
    public static string Name => "feature.cycle";

    /// <inheritdoc />
    public static string Purpose => "Writes a number as a place on a circle of a length you give, so that the ends of the cycle meet.";

    /// <inheritdoc />
    /// <remarks>New in the eighth version of the file.</remarks>
    public static int Since => 8;

    /// <inheritdoc />
    public string Verb => Name;

    // The two values a number becomes, before a form writes each of them down; named by the length, which is part of what
    // they are: a place in a week and a place in a month of one column are not the same two values.
    private string[] Stems
    {
        get
        {
            var stem = string.Create(CultureInfo.InvariantCulture, $"{Column}_cycle{Length}");

            return [$"{stem}_sin", $"{stem}_cos"];
        }
    }

    /// <inheritdoc />
    /// <remarks>Written split by its sign, each value becomes two halves, and each is known to be one.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Stems.Aggregate(before, (state, stem) => Form == Form.SplitSign
            ? Form.Names(stem).Aggregate(state, (halves, half) => halves.WithHalf(half, stem))
            : state.With(stem, ColumnKind.Number, Form));
    }

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var numbers = table.NumbersOf(Column);

        table.PutPlaces(
            [.. numbers.Select(number => number is { } value ? new CirclePlace(value - (Length * Math.Floor(value / Length)), Length) : (CirclePlace?)null)],
            Stems,
            Form);
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CycleStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), LengthKey.Read(element), FormKey.Read(element));
}

/// <summary>
/// The cycles of one line: how long one turn is, and the columns whose numbers are placed on it.
/// </summary>
/// <remarks>
/// A length is a setting of the whole group and a column is not, so the line names the length once and then the columns that
/// stand on it, as <see cref="TimePartLine{TLine}"/> names the pieces once and then their columns:
/// <c>cycle =&gt; cycle.Every(7).Of("age", "day").Every(30).Of("phase")</c>. Each column reaches the declaration as the step of
/// its own the verb for one column writes, so the file, the notebook's blocks and every fit are what they are. How the two
/// values of a place are written down is where the pipeline lands its features; a form of a column's own is said on
/// <see cref="PipelineBuilder.Cycle(string, double, Form)"/>.
/// </remarks>
public sealed class CycleLine : IDeclaresSteps
{
    private readonly List<IPipelineStep> _steps = [];

    private Form _features = NormaliseStep.DefaultFeatures;

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>Where the pipeline declared its features land, which is where a place on a circle is written.</summary>
    Form IDeclaresSteps.Features
    {
        set => _features = value;
    }

    /// <summary>Names how long one turn is, before the columns whose numbers stand on it.</summary>
    /// <param name="length">How long one turn is, in the columns' own units.</param>
    /// <returns>The length, waiting for the columns.</returns>
    public CycleOfLength Every(double length) => new(columns => Add(columns, length));

    private CycleLine Add(string[] columns, double length)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            _steps.Add(new CycleStep(column, length, _features));
        }

        return this;
    }
}

/// <summary>
/// A length of one turn a line has named, waiting for the columns whose numbers stand on it.
/// </summary>
public sealed class CycleOfLength
{
    private readonly Func<string[], CycleLine> _of;

    // What adding columns to the line means is the line's own business, so it is handed over rather than reached into.
    internal CycleOfLength(Func<string[], CycleLine> of) => _of = of;

    /// <summary>Places the numbers of these columns on the cycle.</summary>
    /// <param name="columns">The columns holding a number.</param>
    /// <returns>The line, so another length can be named after it.</returns>
    /// <exception cref="ArgumentNullException">The columns are missing.</exception>
    /// <exception cref="ArgumentException">A column has no name.</exception>
    public CycleLine Of(params string[] columns) => _of(columns);
}
