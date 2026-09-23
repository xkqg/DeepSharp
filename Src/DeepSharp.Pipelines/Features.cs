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
internal static class Forms
{
    /// <summary>The columns one signed value becomes, in this form.</summary>
    /// <param name="form">How to write it down.</param>
    /// <param name="name">What the value is called.</param>
    /// <param name="values">The value for each row, between minus one and one.</param>
    /// <returns>One column, or two.</returns>
    internal static IEnumerable<IColumn> Write(Form form, string name, double?[] values) => form switch
    {
        Form.Signed => [new Column<double>(name, ColumnKind.Number, values)],
        Form.Unit => [new Column<double>(name, ColumnKind.Number, values.Select(Shifted))],
        _ =>
        [
            new Column<double>($"{name}_pos", ColumnKind.Number, values.Select(value => Half(value, up: true))),
            new Column<double>($"{name}_neg", ColumnKind.Number, values.Select(value => Half(value, up: false))),
        ],
    };

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
public sealed record AddFeatureStep : IPipelineStep<AddFeatureStep>, IAddsColumns
{
    /// <summary>Declares a column worked out from two others.</summary>
    /// <param name="name">What the new column is called.</param>
    /// <param name="left">The column on the left.</param>
    /// <param name="arithmetic">What to do with them.</param>
    /// <param name="right">The column on the right.</param>
    /// <exception cref="ArgumentException">A name is empty.</exception>
    public AddFeatureStep(string name, string left, Arithmetic arithmetic, string right)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(left);
        ArgumentException.ThrowIfNullOrWhiteSpace(right);

        Column = name;
        Left = left;
        Arithmetic = arithmetic;
        Right = right;
    }

    /// <summary>What the new column is called.</summary>
    public string Column { get; }

    /// <summary>The column on the left.</summary>
    public string Left { get; }

    /// <summary>What is done with the two columns.</summary>
    public Arithmetic Arithmetic { get; }

    /// <summary>The column on the right.</summary>
    public string Right { get; }

    /// <inheritdoc />
    public static string Name => "feature.add";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var left = Numbers.Of(table, Left);
        var right = Numbers.Of(table, Right);

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

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("left", Left);
        writer.WriteString("arithmetic", Arithmetic.ToString().ToLowerInvariant());
        writer.WriteString("right", Right);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static AddFeatureStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredString("left"),
            element.RequiredEnum<Arithmetic>("arithmetic"),
            element.RequiredString("right"));
}

/// <summary>
/// Writes a moment in time as a place on a circle.
/// </summary>
/// <remarks>
/// An hour of the day is not a number between nothing and twenty-three. Eleven at night and midnight are
/// neighbours, and a model handed the plain number is told they are as far apart as two values can be.
/// A sine and a cosine put the wrap where it belongs, and nothing about it is learned from the data.
/// </remarks>
public sealed record CyclicalStep : IPipelineStep<CyclicalStep>, IAddsColumns
{
    /// <summary>Declares a moment in time written as a place on a circle.</summary>
    /// <param name="column">The column holding the moment.</param>
    /// <param name="period">Which cycle to place it on.</param>
    /// <param name="form">How to write the two values down.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public CyclicalStep(string column, Period period, Form form = Form.Signed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        Column = column;
        Period = period;
        Form = form;
    }

    /// <summary>The column holding the moment.</summary>
    public string Column { get; }

    /// <summary>Which cycle the moment is placed on.</summary>
    public Period Period { get; }

    /// <summary>How the two values are written down.</summary>
    public Form Form { get; }

    /// <inheritdoc />
    public static string Name => "feature.cyclical";

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

        var stem = $"{Column}_{Period.ToString().ToLowerInvariant()}";

        foreach (var column in Forms.Write(Form, $"{stem}_sin", sines).Concat(Forms.Write(Form, $"{stem}_cos", cosines)))
        {
            table.Put(column);
        }
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("period", Period.ToString().ToLowerInvariant());
        writer.WriteString("form", Form.ToString().ToLowerInvariant());
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CyclicalStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredEnum<Period>("period"),
            element.RequiredEnum<Form>("form"));

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
public interface IAddsColumns : IPipelineStep
{
    /// <summary>Works the new columns out and puts them on the table.</summary>
    /// <param name="table">The data, changed in place.</param>
    void AddTo(Table table);
}

/// <summary>
/// Reading a column as numbers, whatever kind of number it holds.
/// </summary>
/// <remarks>
/// Public because a package that adds a verb needs exactly this and would otherwise write its own, and two
/// readings of "what is a number here" is one too many: a boolean counts as one and nought, a gap stays a
/// gap, and words are refused by name.
/// </remarks>
public static class Numbers
{
    /// <summary>The column's values as numbers, with a gap where a cell is a gap.</summary>
    /// <param name="table">The table to look in.</param>
    /// <param name="name">The column's name.</param>
    /// <returns>One value per row.</returns>
    /// <exception cref="InvalidOperationException">The column holds something that is not a number.</exception>
    public static double?[] Of(Table table, string name) => table[name] switch
    {
        Column<double> numbers => [.. Enumerable.Range(0, numbers.Count).Select(row => numbers[row])],
        Column<long> whole => [.. Enumerable.Range(0, whole.Count).Select(row => (double?)whole[row])],
        Column<bool> flags =>
            [.. Enumerable.Range(0, flags.Count).Select(row => flags[row] is { } flag ? flag ? 1 : 0 : (double?)null)],
        var other => throw new InvalidOperationException(
            $"'{name}' holds {other.Kind.ToString().ToLowerInvariant()}, and this step works on numbers."),
    };
}
