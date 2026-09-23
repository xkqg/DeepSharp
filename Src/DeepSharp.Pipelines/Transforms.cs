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

    /// <summary>By rank: the smallest training value becomes nothing, the largest one, the rest their place
    /// in between. Unmoved by extremes, and it flattens the shape of the distribution along with them.</summary>
    Quantile,

    /// <summary>Reshaped towards a bell curve, then centred. For a column that leans heavily one way.</summary>
    Power,
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
public sealed record NormaliseStep : IFittedStep, ILearnsFromData, IUndoesItself, IPipelineStep<NormaliseStep>
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
    public string Produces => Column;

    /// <inheritdoc />
    public double Undo(double value, FittedStepValues? fitted)
    {
        ArgumentNullException.ThrowIfNull(fitted);

        if (Scale == Scale.Quantile)
        {
            // A rank says where a value sat among the training values, so coming back is reading that
            // place off the same knots. Outside them there is nothing to read, and the edge is the honest
            // answer rather than an extrapolation nobody asked for.
            var knots = fitted.Curve("knots");
            var place = Math.Clamp(value, 0, 1) * (knots.Count - 1);
            var below = (int)Math.Floor(place);
            var above = Math.Min(below + 1, knots.Count - 1);

            return knots[below] + ((knots[above] - knots[below]) * (place - below));
        }

        var plain = (value * fitted.Number("spread")) + fitted.Number("centre");

        return Scale == Scale.Power ? YeoJohnson.Undo(plain, fitted.Number("lambda")) : plain;
    }

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

            case Scale.Robust:
                learned.Learned("centre", Quantile(training, 0.5));
                learned.Learned("spread", Spread(Quantile(training, 0.75) - Quantile(training, 0.25)));
                break;

            case Scale.Quantile:
                // The shape of the training distribution, as a hundred and one steps. A rank transform
                // needs the whole shape, not two numbers, so the whole shape is what gets stored.
                learned.Learned("knots", [.. Enumerable.Range(0, 101).Select(at => Quantile(training, at / 100.0))]);
                break;

            default:
                var lambda = YeoJohnson.Lambda(training);
                learned.Learned("lambda", lambda);
                var shaped = training.Select(value => YeoJohnson.Of(value, lambda)).ToArray();
                var middle = shaped.Average();
                learned.Learned("centre", middle);
                learned.Learned("spread", Spread(Math.Sqrt(shaped.Average(value => (value - middle) * (value - middle)))));
                break;
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var values = Numbers.Of(table, Column);
        var scaled = new double?[values.Length];

        if (Scale == Scale.Quantile)
        {
            var knots = fitted.Curve("knots");

            for (var row = 0; row < values.Length; row++)
            {
                scaled[row] = values[row] is { } value ? Rank(knots, value) : null;
            }

            table.Put(new Column<double>(Column, ColumnKind.Number, scaled));

            return;
        }

        var centre = fitted.Number("centre");
        var spread = fitted.Number("spread");
        var lambda = Scale == Scale.Power ? fitted.Number("lambda") : 0;

        for (var row = 0; row < values.Length; row++)
        {
            if (values[row] is not { } value)
            {
                continue;
            }

            var next = ((Scale == Scale.Power ? YeoJohnson.Of(value, lambda) : value) - centre) / spread;

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

    private static double Rank(IReadOnlyList<double> knots, double value)
    {
        // Where this value sits among the training values, between nothing and one. Outside the range the
        // fit saw, it holds at the edge -- which is what a rank transform can honestly say about a value
        // it has never seen anything like.
        if (value <= knots[0])
        {
            return 0;
        }

        for (var at = 1; at < knots.Count; at++)
        {
            if (value > knots[at])
            {
                continue;
            }

            var width = knots[at] - knots[at - 1];
            var within = width == 0 ? 0 : (value - knots[at - 1]) / width;

            return (at - 1 + within) / (knots.Count - 1);
        }

        return 1;
    }

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

    /// <summary>The column written beside an encoded one, saying where the cell was empty.</summary>
    public string MarkerColumn => $"{Column}_was_missing";

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

        // An empty cell is not a category and not an unfamiliar one either, so it becomes no category at
        // all -- every slot nothing -- and the marking column remembers that it was empty. Leaving a gap
        // in the encoded columns instead would only move the problem to whoever hands the rows over.
        var marker = new Column<double>(
            MarkerColumn, ColumnKind.Number, places.Select(place => (double?)(place is null ? 1 : 0)));

        if (How == As.Ordinal)
        {
            table.Put(new Column<double>(Column, ColumnKind.Number, places.Select(place => (double?)(place ?? 0))));
            table.Put(marker);

            return;
        }

        var slots = Unseen == Unseen.Reserve ? categories.Count + 1 : categories.Count;

        for (var slot = 0; slot < slots; slot++)
        {
            var label = slot < categories.Count ? categories[slot] : "other";
            var here = slot;

            table.Put(new Column<double>(
                $"{Column}_{label}", ColumnKind.Number,
                places.Select(place => (double?)(place == here ? 1 : 0))));
        }

        table.Put(marker);
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

/// <summary>
/// The Yeo-Johnson reshaping, which pulls a lopsided column towards a bell curve.
/// </summary>
/// <remarks>
/// Unlike the older Box-Cox it takes negative values as well, which matters for a column of differences.
/// The one parameter is found by trying a grid of values and keeping the one under which the reshaped
/// column looks most like a bell curve; a finer search buys precision nobody downstream can use.
/// </remarks>
internal static class YeoJohnson
{
    internal static double Of(double value, double lambda) => value >= 0
        ? lambda == 0 ? Math.Log(value + 1) : (Math.Pow(value + 1, lambda) - 1) / lambda
        : lambda == 2 ? -Math.Log(1 - value) : -((Math.Pow(1 - value, 2 - lambda) - 1) / (2 - lambda));

    internal static double Undo(double value, double lambda) => value >= 0
        ? lambda == 0 ? Math.Exp(value) - 1 : Math.Pow((lambda * value) + 1, 1 / lambda) - 1
        : lambda == 2 ? 1 - Math.Exp(-value) : 1 - Math.Pow(1 - ((2 - lambda) * value), 1 / (2 - lambda));

    internal static double Lambda(double[] training)
    {
        var best = 1.0;
        var most = double.NegativeInfinity;

        for (var lambda = -2.0; lambda <= 2.0001; lambda += 0.05)
        {
            var likelihood = Likelihood(training, lambda);

            if (likelihood > most)
            {
                most = likelihood;
                best = lambda;
            }
        }

        return Math.Round(best, 4);
    }

    private static double Likelihood(double[] training, double lambda)
    {
        var shaped = training.Select(value => Of(value, lambda)).ToArray();

        if (shaped.Any(double.IsNaN) || shaped.Any(double.IsInfinity))
        {
            return double.NegativeInfinity;
        }

        var mean = shaped.Average();
        var variance = shaped.Average(value => (value - mean) * (value - mean));

        if (variance <= 0)
        {
            return double.NegativeInfinity;
        }

        return (-0.5 * training.Length * Math.Log(variance))
               + ((lambda - 1) * training.Sum(value => Math.Sign(value) * Math.Log(Math.Abs(value) + 1)));
    }
}
