// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How a signed value is written down.</summary>
public enum Form
{
    /// <summary>One column, between minus one and one. The plain value.</summary>
    Signed,

    /// <summary>One column, between nothing and one. The same feature, shifted; note that zero becomes a half.</summary>
    Unit,

    /// <summary>Two columns, each between nothing and one: how far up, and how far down.</summary>
    SplitSign,
}

/// <summary>What a derived column does with the two it is made from.</summary>
public enum Arithmetic
{
    /// <summary>Added together.</summary>
    Plus,

    /// <summary>The second taken from the first.</summary>
    Minus,

    /// <summary>Multiplied.</summary>
    Times,

    /// <summary>The first divided by the second; a division by nothing leaves a gap.</summary>
    DividedBy,
}

/// <summary>Which cycle a moment in time is placed on.</summary>
public enum Period
{
    /// <summary>The hour of the day, where midnight follows eleven at night.</summary>
    HourOfDay,

    /// <summary>The day of the week, where Monday follows Sunday.</summary>
    DayOfWeek,

    /// <summary>The day of the month.</summary>
    DayOfMonth,

    /// <summary>The month of the year, where January follows December.</summary>
    MonthOfYear,
}

/// <summary>
/// Writing a signed value down in the shape a model is going to read it.
/// </summary>
internal static class FormExtensions
{
    /// <summary>The names of the columns one signed value becomes, in this form.</summary>
    /// <param name="form">How it is written down.</param>
    /// <param name="name">What the value is called.</param>
    /// <returns>One name, or two.</returns>
    internal static IEnumerable<string> Names(this Form form, string name) =>
        form == Form.SplitSign ? [$"{name}_pos", $"{name}_neg"] : [name];

    /// <summary>The columns one signed value becomes, in this form.</summary>
    /// <param name="form">How to write it down.</param>
    /// <param name="name">What the value is called.</param>
    /// <param name="values">The value for each row, between minus one and one.</param>
    /// <returns>One column, or two, named as <see cref="Names"/> says.</returns>
    internal static IEnumerable<IColumn> Written(this Form form, string name, double?[] values)
    {
        var names = form.Names(name).ToArray();

        return form switch
        {
            Form.Signed => [new Column<double>(names[0], ColumnKind.Number, values)],
            Form.Unit => [new Column<double>(names[0], ColumnKind.Number, values.Select(Shifted))],
            _ =>
            [
                new Column<double>(names[0], ColumnKind.Number, values.Select(value => Half(value, up: true))),
                new Column<double>(names[1], ColumnKind.Number, values.Select(value => Half(value, up: false))),
            ],
        };
    }

    private static double? Shifted(double? value) => value is { } number ? (number + 1) / 2 : null;

    private static double? Half(double? value, bool up) =>
        value is { } number ? Math.Max(up ? number : -number, 0) : null;
}

/// <summary>
/// Adds a column worked out from two others.
/// </summary>
/// <remarks>
/// Written as a named operation on two columns rather than as a piece of code, because a piece of code
/// cannot be saved in a file and read back a year later. That is the whole trade: the vocabulary is
/// smaller, and everything in it survives being written down. Nothing here learns from the data — it is
/// arithmetic on one row — so a feature belongs before the split.
/// </remarks>
public sealed record AddFeatureStep : IPipelineStep<AddFeatureStep>, IAddsColumns, IDescribesColumns
{
    private static readonly NewColumnParameter ColumnKey = new(
        "column", "What the new column is called.", "feature");

    private static readonly ColumnParameter LeftKey = new(
        "left", "The column on the left of the arithmetic.", "left", ColumnKinds.Numbers);

    private static readonly OneOfParameter<Arithmetic> ArithmeticKey = new(
        "arithmetic", "What is done with the two columns: added, subtracted, multiplied or divided.", Arithmetic.Minus);

    private static readonly ColumnParameter RightKey = new(
        "right", "The column on the right of the arithmetic.", "right", ColumnKinds.Numbers);

    /// <summary>Declares a column worked out from two others.</summary>
    /// <param name="name">What the new column is called.</param>
    /// <param name="left">The column on the left.</param>
    /// <param name="arithmetic">What to do with them.</param>
    /// <param name="right">The column on the right.</param>
    /// <exception cref="ArgumentException">A name is empty.</exception>
    public AddFeatureStep(string name, string left, Arithmetic arithmetic, string right)
    {
        Column = ColumnKey.Require(name)!;
        Left = LeftKey.Require(left);
        Arithmetic = ArithmeticKey.Require(arithmetic);
        Right = RightKey.Require(right);
    }

    /// <inheritdoc />
    public static StepParameters<AddFeatureStep> Parameters { get; } = new StepParameters<AddFeatureStep>()
        .With(ColumnKey, step => step.Column)
        .With(LeftKey, step => step.Left)
        .With(ArithmeticKey, step => step.Arithmetic)
        .With(RightKey, step => step.Right);

    /// <summary>What the new column is called.</summary>
    public string Column { get; }

    /// <summary>The column on the left.</summary>
    public string Left { get; }

    /// <summary>What is done with the two columns.</summary>
    public Arithmetic Arithmetic { get; }

    /// <summary>The column on the right.</summary>
    public string Right { get; }

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Column, ColumnKind.Number);
    }

    /// <inheritdoc />
    public static string Name => "feature.add";

    /// <inheritdoc />
    public static string Purpose => "Adds a column worked out from two others by plain arithmetic.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var left = table.NumbersOf(Left);
        var right = table.NumbersOf(Right);

        var values = new double?[table.RowCount];

        for (var row = 0; row < values.Length; row++)
        {
            if (left[row] is not { } first || right[row] is not { } second)
            {
                // A gap on either side leaves a gap: inventing a value here would quietly become a
                // measurement, and the step that fills gaps is the one allowed to do that, after the split.
                continue;
            }

            values[row] = Arithmetic switch
            {
                Arithmetic.Plus => first + second,
                Arithmetic.Minus => first - second,
                Arithmetic.Times => first * second,
                _ => second == 0 ? null : first / second,
            };
        }

        table.Put(new Column<double>(Column, ColumnKind.Number, values));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static AddFeatureStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element)!, LeftKey.Read(element), ArithmeticKey.Read(element), RightKey.Read(element));
}

/// <summary>
/// Writes a moment in time as a place on a circle.
/// </summary>
/// <remarks>
/// An hour of the day is not a number between nothing and twenty-three. Eleven at night and midnight are
/// neighbours, and a model handed the plain number is told they are as far apart as two values can be.
/// A sine and a cosine put the wrap where it belongs, and nothing about it is learned from the data.
/// </remarks>
public sealed record CyclicalStep : IPipelineStep<CyclicalStep>, IAddsColumns, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column holding the moment in time.", "when", ColumnKinds.Moments);

    private static readonly OneOfParameter<Period> PeriodKey = new(
        "period", "Which cycle the moment is placed on: the hour of the day, the day of the week or of the month, the month of the year.", Period.MonthOfYear);

    private static readonly OneOfParameter<Form> FormKey = new(
        "form", "How the two values are written down: as they are, shifted between nothing and one, or split into how far up and how far down.", Form.Signed);

    /// <summary>Declares a moment in time written as a place on a circle.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="period">Which cycle to place it on.</param>
    /// <param name="form">How to write the two values down.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public CyclicalStep(string column, Period period, Form form = Form.Signed)
    {
        Column = ColumnKey.Require(column);
        Period = PeriodKey.Require(period);
        Form = FormKey.Require(form);
    }

    /// <inheritdoc />
    public static StepParameters<CyclicalStep> Parameters { get; } = new StepParameters<CyclicalStep>()
        .With(ColumnKey, step => step.Column)
        .With(PeriodKey, step => step.Period)
        .With(FormKey, step => step.Form);

    /// <summary>The column holding the moment.</summary>
    public string Column { get; }

    /// <summary>Which cycle the moment is placed on.</summary>
    public Period Period { get; }

    /// <summary>How the two values are written down.</summary>
    public Form Form { get; }

    /// <inheritdoc />
    /// <remarks>Written split by its sign, each value becomes two halves, and each is known to be one.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Stems.Aggregate(before, (state, stem) => Form == Form.SplitSign
            ? Form.Names(stem).Aggregate(state, (halves, half) => halves.WithHalf(half, stem))
            : state.With(stem, ColumnKind.Number));
    }

    // The two values a moment becomes, before a form writes each of them down.
    private IEnumerable<string> Stems => [$"{Column}_{Period.ToString().ToLowerInvariant()}_sin", $"{Column}_{Period.ToString().ToLowerInvariant()}_cos"];

    /// <inheritdoc />
    public static string Name => "feature.cyclical";

    /// <inheritdoc />
    public static string Purpose => "Writes a moment in time as a place on a circle, so that the ends of a cycle meet.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        if (table[Column] is not Column<DateTime> moments)
        {
            throw new InvalidOperationException(
                $"'{Column}' holds {table[Column].Kind.ToString().ToLowerInvariant()} and a cycle needs a moment in time.");
        }

        var sines = new double?[table.RowCount];
        var cosines = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (moments[row] is not { } moment)
            {
                continue;
            }

            var turn = 2 * Math.PI * Place(moment) / Length;

            sines[row] = Math.Sin(turn);
            cosines[row] = Math.Cos(turn);
        }

        var stems = Stems.ToArray();

        foreach (var column in Form.Written(stems[0], sines).Concat(Form.Written(stems[1], cosines)))
        {
            table.Put(column);
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CyclicalStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), PeriodKey.Read(element), FormKey.Read(element));

    private double Length => Period switch
    {
        Period.HourOfDay => 24,
        Period.DayOfWeek => 7,
        Period.DayOfMonth => 31,
        _ => 12,
    };

    private double Place(DateTime moment) => Period switch
    {
        Period.HourOfDay => moment.Hour + (moment.Minute / 60.0),
        Period.DayOfWeek => (int)moment.DayOfWeek,
        Period.DayOfMonth => moment.Day - 1,
        _ => moment.Month - 1,
    };
}

/// <summary>
/// A step that puts columns onto the table without learning anything from the data.
/// </summary>
public interface IAddsColumns : IActsInAWalk
{
    /// <summary>Works the new columns out and puts them on the table.</summary>
    /// <param name="table">The data, changed in place.</param>
    void AddTo(Table table);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => AddTo(walk.Table);
}
