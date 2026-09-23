// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How a column of numbers is brought onto a comparable scale.</summary>
public enum Scale
{
    /// <summary>Middle at nothing, spread of one. Pulled around by a single extreme value.</summary>
    Standard,

    /// <summary>Squeezed between nothing and one. One spike leaves everything else in a sliver.</summary>
    MinMax,

    /// <summary>Divided by the largest magnitude, so a zero stays a zero.</summary>
    MaxAbs,

    /// <summary>Middle at the median, spread of the middle half. Unmoved by a few extremes.</summary>
    Robust,
}

/// <summary>What happens to a value outside the range the fit learned.</summary>
public enum OutOfRange
{
    /// <summary>Let it through, outside the range the model was trained on.</summary>
    Pass,

    /// <summary>Hold it at the edge, which hides that the data has moved.</summary>
    Clip,

    /// <summary>Stop, and say the data is outside what this model has seen.</summary>
    Refuse,
}

/// <summary>How a category is written down as numbers.</summary>
public enum As
{
    /// <summary>One column per category, one of them a one and the rest nothing.</summary>
    OneHot,

    /// <summary>One column holding the category's place in the list.</summary>
    Ordinal,
}

/// <summary>What happens to a category the training rows never held.</summary>
public enum Unseen
{
    /// <summary>A place is kept for it, so an unfamiliar value has somewhere to go.</summary>
    Reserve,

    /// <summary>Stop, and say the data holds something this model has never seen.</summary>
    Refuse,
}

/// <summary>How a row is brought onto a comparable scale.</summary>
public enum Norm
{
    /// <summary>Divided by the sum of the magnitudes.</summary>
    L1,

    /// <summary>Divided by the length of the row.</summary>
    L2,

    /// <summary>Divided by the largest magnitude in the row.</summary>
    Max,
}

/// <summary>
/// Brings a column onto a comparable scale, by numbers learned from the training rows.
/// </summary>
/// <remarks>
/// Each kind learns something different — a middle and a spread, two extremes, a magnitude, a median and
/// its quartiles — which is why the kind is declared and what it learned is stored apart from it. Standard
/// and min-max are both moved by a single extreme value, so on prices and volumes the robust form is
/// usually the one describing the data rather than the spike.
/// </remarks>
public sealed record NormaliseStep : IFittedStep, ILearnsFromData, IPipelineStep<NormaliseStep>
{
    /// <summary>Declares that a column is brought onto a comparable scale.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="scale">Which kind of scaling.</param>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public NormaliseStep(string column, Scale scale = Scale.Standard, OutOfRange outOfRange = OutOfRange.Pass)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        Column = column;
        Scale = scale;
        OutOfRange = outOfRange;
    }

    /// <summary>The column being scaled.</summary>
    public string Column { get; }

    /// <summary>Which kind of scaling.</summary>
    public Scale Scale { get; }

    /// <summary>What happens to a value outside the range the fit learned.</summary>
    public OutOfRange OutOfRange { get; }

    /// <inheritdoc />
    public static string Name => "normalise";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Split> splits)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(splits);

        var values = Numbers.Of(table, Column);
        var training = Enumerable.Range(0, values.Length)
            .Where(row => splits[row] == Split.Train && values[row] is not null)
            .Select(row => values[row]!.Value)
            .Order()
            .ToArray();

        if (training.Length == 0)
        {
            throw new InvalidOperationException(
                $"Every training row of '{Column}' is a gap, so there is no scale to learn.");
        }

        var learned = new FittedStepValues();

        switch (Scale)
        {
            case Scale.Standard:
                var mean = training.Average();
                learned.Learned("centre", mean);
                learned.Learned("spread", Spread(Math.Sqrt(training.Average(value => (value - mean) * (value - mean)))));
                break;

            case Scale.MinMax:
                learned.Learned("centre", training[0]);
                learned.Learned("spread", Spread(training[^1] - training[0]));
                break;

            case Scale.MaxAbs:
                learned.Learned("centre", 0);
                learned.Learned("spread", Spread(training.Max(Math.Abs)));
                break;

            default:
                learned.Learned("centre", Quantile(training, 0.5));
                learned.Learned("spread", Spread(Quantile(training, 0.75) - Quantile(training, 0.25)));
                break;
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var centre = fitted.Number("centre");
        var spread = fitted.Number("spread");
        var values = Numbers.Of(table, Column);
        var scaled = new double?[values.Length];

        for (var row = 0; row < values.Length; row++)
        {
            if (values[row] is not { } value)
            {
                continue;
            }

            var next = (value - centre) / spread;

            // Min-max on a price meets this the first time there is a new high, so what happens then is
            // part of the declaration rather than something the library decides on everybody's behalf.
            scaled[row] = Scale == Scale.MinMax || Scale == Scale.MaxAbs
                ? OutOfRange switch
                {
                    OutOfRange.Clip => Math.Clamp(next, Scale == Scale.MaxAbs ? -1 : 0, 1),
                    OutOfRange.Refuse when next < (Scale == Scale.MaxAbs ? -1 : 0) || next > 1 =>
                        throw new InvalidOperationException(
                            $"Row {row + 1} of '{Column}' is outside the range this pipeline was fitted on."),
                    _ => next,
                }
                : next;
        }

        table.Put(new Column<double>(Column, ColumnKind.Number, scaled));
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("scale", Scale.ToString().ToLowerInvariant());
        writer.WriteString("outOfRange", OutOfRange.ToString().ToLowerInvariant());
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NormaliseStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredEnum<Scale>("scale"),
            element.RequiredEnum<OutOfRange>("outOfRange"));

    private static double Spread(double spread) => spread == 0 ? 1 : spread;

    private static double Quantile(double[] sorted, double at)
    {
        var place = at * (sorted.Length - 1);
        var below = (int)Math.Floor(place);
        var above = Math.Min(below + 1, sorted.Length - 1);

        return sorted[below] + ((sorted[above] - sorted[below]) * (place - below));
    }
}

/// <summary>
/// Brings each row onto a comparable scale, learning nothing.
/// </summary>
/// <remarks>
/// A different animal from the rest: it works across a row rather than down a column, so there is nothing
/// to fit and nothing to replay. What you want when the direction of a row matters and its size does not.
/// </remarks>
public sealed record NormaliseRowStep : IPipelineStep<NormaliseRowStep>, IAddsColumns
{
    /// <summary>Declares that these columns are scaled together, row by row.</summary>
    /// <param name="columns">The columns that make up the row.</param>
    /// <param name="norm">How the row's size is measured.</param>
    /// <exception cref="ArgumentException">There are no columns.</exception>
    public NormaliseRowStep(IEnumerable<string> columns, Norm norm = Norm.L2)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = [.. columns];

        if (Columns.Count == 0)
        {
            throw new ArgumentException("Scaling a row needs the columns it is made of.", nameof(columns));
        }

        Norm = norm;
    }

    /// <summary>The columns that make up the row.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How the row's size is measured.</summary>
    public Norm Norm { get; }

    /// <inheritdoc />
    public static string Name => "normalise.row";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public bool Equals(NormaliseRowStep? other) =>
        other is not null && Norm == other.Norm && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Norm);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var values = Columns.Select(column => Numbers.Of(table, column)).ToArray();

        for (var row = 0; row < table.RowCount; row++)
        {
            var present = values.Select(column => column[row]).Where(value => value is not null).ToArray();

            if (present.Length == 0)
            {
                continue;
            }

            var size = Norm switch
            {
                Norm.L1 => present.Sum(value => Math.Abs(value!.Value)),
                Norm.Max => present.Max(value => Math.Abs(value!.Value)),
                _ => Math.Sqrt(present.Sum(value => value!.Value * value!.Value)),
            };

            if (size == 0)
            {
                continue;
            }

            foreach (var column in values)
            {
                column[row] = column[row] is { } value ? value / size : null;
            }
        }

        for (var at = 0; at < Columns.Count; at++)
        {
            table.Put(new Column<double>(Columns[at], ColumnKind.Number, values[at]));
        }
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("norm", Norm.ToString().ToLowerInvariant());
        writer.WriteStartArray("columns");

        foreach (var column in Columns)
        {
            writer.WriteStringValue(column);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NormaliseRowStep ReadFrom(JsonElement element)
    {
        if (!element.TryGetProperty("columns", out var columns) || columns.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException("Scaling a row holds a 'columns' list.");
        }

        return new NormaliseRowStep(
            columns.EnumerateArray().Select(column => column.GetString() ?? string.Empty),
            element.RequiredEnum<Norm>("norm"));
    }
}

/// <summary>
/// Writes a category down as numbers, using the categories the training rows held.
/// </summary>
/// <remarks>
/// The list of categories is learned, which is what makes this a step that belongs after the split. An
/// unfamiliar value will turn up in production sooner or later, so what happens then is declared: a place
/// kept for it, or a refusal saying the data holds something this model has never seen.
/// </remarks>
public sealed record EncodeStep : IFittedStep, ILearnsFromData, IPipelineStep<EncodeStep>
{
    /// <summary>Declares that a column of words is written down as numbers.</summary>
    /// <param name="column">The column of words.</param>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public EncodeStep(string column, As how = As.OneHot, Unseen unseen = Unseen.Reserve)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        Column = column;
        How = how;
        Unseen = unseen;
    }

    /// <summary>The column of words.</summary>
    public string Column { get; }

    /// <summary>How the categories are written down.</summary>
    public As How { get; }

    /// <summary>What happens to a category the training rows never held.</summary>
    public Unseen Unseen { get; }

    /// <inheritdoc />
    public static string Name => "encode";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Split> splits)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(splits);

        var column = table[Column];

        var categories = Enumerable.Range(0, table.RowCount)
            .Where(row => splits[row] == Split.Train && !column.IsMissing(row))
            .Select(row => column.TextAt(row)!)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();

        if (categories.Length == 0)
        {
            throw new InvalidOperationException(
                $"Every training row of '{Column}' is a gap, so there are no categories to learn.");
        }

        var learned = new FittedStepValues();
        learned.Learned("categories", categories);

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var categories = fitted.List("categories");
        var column = table[Column];
        var places = new double?[table.RowCount];

        for (var row = 0; row < table.RowCount; row++)
        {
            if (column.IsMissing(row))
            {
                continue;
            }

            var at = categories.ToList().IndexOf(column.TextAt(row)!);

            places[row] = at >= 0
                ? at
                : Unseen == Unseen.Refuse
                    ? throw new InvalidOperationException(
                        $"Row {row + 1} of '{Column}' holds '{column.TextAt(row)}', which the training rows never held.")
                    : categories.Count;
        }

        table.Remove(Column);

        if (How == As.Ordinal)
        {
            table.Put(new Column<double>(Column, ColumnKind.Number, places));

            return;
        }

        var slots = Unseen == Unseen.Reserve ? categories.Count + 1 : categories.Count;

        for (var slot = 0; slot < slots; slot++)
        {
            var label = slot < categories.Count ? categories[slot] : "other";
            var here = slot;

            table.Put(new Column<double>(
                $"{Column}_{label}", ColumnKind.Number,
                places.Select(place => place is null ? null : (double?)(place == here ? 1 : 0))));
        }
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteString("as", How.ToString().ToLowerInvariant());
        writer.WriteString("unseen", Unseen.ToString().ToLowerInvariant());
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static EncodeStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            element.RequiredEnum<As>("as"),
            element.RequiredEnum<Unseen>("unseen"));
}
