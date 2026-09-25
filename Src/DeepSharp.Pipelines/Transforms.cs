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
public sealed record NormaliseStep : IFittedStep, IUndoesItself, IPipelineStep<NormaliseStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to bring onto a comparable scale.", "column", ColumnKinds.Numbers);

    private static readonly OneOfParameter<Scale> ScaleKey = new(
        "scale", "Which kind of scaling: what the fit learns from the training rows.", Scale.Standard);

    private static readonly OneOfParameter<OutOfRange> OutOfRangeKey = new(
        "outOfRange", "What happens to a value outside the range the fit learned: let it through, hold it at the edge, or refuse.", OutOfRange.Pass);

    /// <summary>Declares that a column is brought onto a comparable scale.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="scale">Which kind of scaling.</param>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public NormaliseStep(string column, Scale scale = Scale.Standard, OutOfRange outOfRange = OutOfRange.Pass)
    {
        Column = ColumnKey.Require(column);
        Scale = ScaleKey.Require(scale);
        OutOfRange = OutOfRangeKey.Require(outOfRange);
    }

    /// <inheritdoc />
    public static StepParameters<NormaliseStep> Parameters { get; } = new StepParameters<NormaliseStep>()
        .With(ColumnKey, step => step.Column)
        .With(ScaleKey, step => step.Scale)
        .With(OutOfRangeKey, step => step.OutOfRange);

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
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Column, ColumnKind.Number);
    }

    /// <inheritdoc />
    /// <remarks>
    /// A half of a value split by its sign is refused: a scale learned for one half would stretch it by its own
    /// extremes and the other half by its own, and one quantity would come out with two different slopes. The
    /// split sign is the last form a value takes.
    /// </remarks>
    public string? Refusal(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Find(Column) is { HalfOf: { } whole }
            ? $"'{Column}' is one half of '{whole}' split by its sign, and a scale learned for one half alone would "
              + "give the two halves two different slopes. The split sign is the last form a value takes."
            : null;
    }

    /// <inheritdoc />
    public static string Name => "normalise";

    /// <inheritdoc />
    public static string Purpose => "Brings a column onto a comparable scale, by numbers learned from the training rows.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var training = table.TrainingValues(Column, parts).Learnable("a scale");
        var learned = new FittedStepValues();

        switch (Scale)
        {
            case Scale.Standard:
                learned.Learned("centre", training.Mean);
                learned.Learned("spread", Spread(training.StandardDeviation));
                break;

            case Scale.MinMax:
                learned.Learned("centre", training.Finite[0]);
                learned.Learned("spread", Spread(training.Finite[^1] - training.Finite[0]));
                break;

            case Scale.MaxAbs:
                learned.Learned("centre", 0);
                learned.Learned("spread", Spread(training.Finite.Max(Math.Abs)));
                break;

            case Scale.Robust:
                learned.Learned("centre", training.Median);
                learned.Learned("spread", Spread(training.Quantile(0.75) - training.Quantile(0.25)));
                break;

            case Scale.Quantile:
                // The shape of the training distribution, as a hundred and one steps. A rank transform
                // needs the whole shape, not two numbers, so the whole shape is what gets stored.
                learned.Learned("knots", [.. Enumerable.Range(0, 101).Select(at => training.Quantile(at / 100.0))]);
                break;

            default:
                var lambda = YeoJohnson.Lambda([.. training.Finite]);
                learned.Learned("lambda", lambda);
                var shaped = training.Finite.Select(value => YeoJohnson.Of(value, lambda)).ToArray();
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

        var values = table.NumbersOf(Column);
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

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NormaliseStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), ScaleKey.Read(element), OutOfRangeKey.Read(element));

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
}

/// <summary>
/// Brings each row onto a comparable scale, learning nothing.
/// </summary>
/// <remarks>
/// A different animal from the rest: it works across a row rather than down a column, so there is nothing
/// to fit and nothing to replay. What you want when the direction of a row matters and its size does not.
/// </remarks>
public sealed record NormaliseRowStep : IPipelineStep<NormaliseRowStep>, IAddsColumns, IDescribesColumns
{
    private static readonly OneOfParameter<Norm> NormKey = new(
        "norm", "How the row's size is measured: the sum of the magnitudes, the length, or the largest.", Norm.L2);

    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns that make up the row, scaled together.", ["left", "right"], ColumnKinds.Numbers);

    /// <summary>Declares that these columns are scaled together, row by row.</summary>
    /// <param name="columns">The columns that make up the row.</param>
    /// <param name="norm">How the row's size is measured.</param>
    /// <exception cref="ArgumentException">There are no columns, one has no name, or one is named twice.</exception>
    public NormaliseRowStep(IEnumerable<string> columns, Norm norm = Norm.L2)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        Norm = NormKey.Require(norm);
    }

    /// <inheritdoc />
    public static StepParameters<NormaliseRowStep> Parameters { get; } = new StepParameters<NormaliseRowStep>()
        .With(NormKey, step => step.Norm)
        .With(ColumnsKey, step => step.Columns);

    /// <summary>The columns that make up the row.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How the row's size is measured.</summary>
    public Norm Norm { get; }

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Columns.Aggregate(before, (state, column) => state.With(column, ColumnKind.Number));
    }

    /// <inheritdoc />
    public static string Name => "normalise.row";

    /// <inheritdoc />
    public static string Purpose => "Brings each row onto a comparable scale across the columns that make it up, learning nothing.";

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

        var values = Columns.Select(column => table.NumbersOf(column)).ToArray();

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

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NormaliseRowStep ReadFrom(JsonElement element) =>
        new(ColumnsKey.Read(element), NormKey.Read(element));
}

/// <summary>
/// Writes a category down as numbers, using the categories the training rows held.
/// </summary>
/// <remarks>
/// The list of categories is learned, which is what makes this a step that belongs after the split. An
/// unfamiliar value will turn up in production sooner or later, so what happens then is declared: a place
/// kept for it, or a refusal saying the data holds something this model has never seen.
/// </remarks>
public sealed record EncodeStep : IFittedStep, IPipelineStep<EncodeStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column of words to write down as numbers.", "column", ColumnKinds.Any);

    private static readonly OneOfParameter<As> AsKey = new(
        "as", "How the categories are written down: one column per category, or one column of places.", As.OneHot);

    private static readonly OneOfParameter<Unseen> UnseenKey = new(
        "unseen", "What happens to a category the training rows never held: a place kept for it, or a refusal.", Unseen.Reserve);

    /// <summary>Declares that a column of words is written down as numbers.</summary>
    /// <param name="column">The column of words.</param>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public EncodeStep(string column, As how = As.OneHot, Unseen unseen = Unseen.Reserve)
    {
        Column = ColumnKey.Require(column);
        How = AsKey.Require(how);
        Unseen = UnseenKey.Require(unseen);
    }

    /// <inheritdoc />
    public static StepParameters<EncodeStep> Parameters { get; } = new StepParameters<EncodeStep>()
        .With(ColumnKey, step => step.Column)
        .With(AsKey, step => step.How)
        .With(UnseenKey, step => step.Unseen);

    /// <summary>The column of words.</summary>
    public string Column { get; }

    /// <summary>How the categories are written down.</summary>
    public As How { get; }

    /// <summary>What happens to a category the training rows never held.</summary>
    public Unseen Unseen { get; }

    /// <summary>The column written beside an encoded one, saying where the cell was empty.</summary>
    public string MarkerColumn => $"{Column}_was_missing";

    /// <inheritdoc />
    /// <remarks>
    /// One column per category is a family: which categories there are is known once the training rows have
    /// been seen, so its members are known by the start of their names. One column of places keeps the name.
    /// </remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        var without = before.Without(Column);
        var encoded = How == As.Ordinal ? without.With(Column, ColumnKind.Number) : without.WithFamily($"{Column}_");

        return encoded.With(MarkerColumn, ColumnKind.Number);
    }

    /// <inheritdoc />
    public static string Name => "encode";

    /// <inheritdoc />
    public static string Purpose => "Writes a column of words down as numbers, using the categories the training rows held.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var categories = table.CategoriesOf(Column, row => parts[row] == Part.Train);

        if (categories.Count == 0)
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

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static EncodeStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), AsKey.Read(element), UnseenKey.Read(element));
}

/// <summary>
/// Writes every column that stands for a group down as numbers, each by the categories the training rows held.
/// </summary>
/// <remarks>
/// Which columns are categories is said once, where it is a fact: in the schema, or by the step that made the
/// column. This takes every column that is a category where it stands, so marking one more column a category
/// is the whole of the change — nothing further down has to be told. It was a word in the chain that became
/// one encoding step per category at the moment it was written, so a file could not say it and a column
/// marked a category afterwards was never encoded at all.
/// </remarks>
public sealed record EncodeCategoriesStep : IFittedStep, IPipelineStep<EncodeCategoriesStep>, IDescribesColumns
{
    private static readonly OneOfParameter<As> AsKey = new(
        "as", "How each category is written down: one column per category, or one column of places.", As.OneHot);

    private static readonly OneOfParameter<Unseen> UnseenKey = new(
        "unseen", "What happens to a category the training rows never held: a place kept for it, or a refusal.", Unseen.Reserve);

    /// <summary>Declares that every category column is written down as numbers.</summary>
    /// <param name="how">One column per category, or one column of places.</param>
    /// <param name="unseen">What happens to a category the training rows never held.</param>
    public EncodeCategoriesStep(As how = As.OneHot, Unseen unseen = Unseen.Reserve)
    {
        How = AsKey.Require(how);
        Unseen = UnseenKey.Require(unseen);
    }

    /// <summary>How the categories are written down.</summary>
    public As How { get; }

    /// <summary>What happens to a category the training rows never held.</summary>
    public Unseen Unseen { get; }

    /// <inheritdoc />
    /// <remarks>Each category where it stands, encoded exactly as a step for that one column would be.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Columns
            .Where(column => column.Kind == ColumnKind.Category)
            .Aggregate(before, (state, column) => new EncodeStep(column.Name, How, Unseen).After(state));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Refused where provably no column is a category: none is known, and nothing unnamed may be there. The run
    /// still refuses when none is, for the columns a step from elsewhere may have made.
    /// </remarks>
    public string? Refusal(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Open || before.Columns.Any(column => column.Kind == ColumnKind.Category)
            ? null
            : "nothing before it declares a category, so this pipeline has no categories to write down as numbers.";
    }

    /// <inheritdoc />
    public static string Name => "encode.categories";

    /// <inheritdoc />
    public static string Purpose => "Writes every column that stands for a group down as numbers, each by the categories the training rows held.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<EncodeCategoriesStep> Parameters { get; } = new StepParameters<EncodeCategoriesStep>()
        .With(AsKey, step => step.How)
        .With(UnseenKey, step => step.Unseen);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">No column is a category where this step stands.</exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var categories = table.Columns.Where(column => column.Kind == ColumnKind.Category).ToArray();

        if (categories.Length == 0)
        {
            throw new InvalidOperationException(
                "This pipeline has no categories where the encoder stands, so there is nothing to write down as numbers.");
        }

        var learned = new FittedStepValues();

        // One list per column, under the column's name, each learned exactly as a step for that one column
        // would learn it: from the training rows alone.
        foreach (var column in categories)
        {
            learned.Learned(column.Name, new EncodeStep(column.Name, How, Unseen).Fit(table, parts).List("categories"));
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        // In the order the columns stand on the table, which a run and a replay reach the same way, so both
        // hand the encoded columns over in one order.
        var encoded = table.Columns.Select(column => column.Name).Where(fitted.Lists.ContainsKey).ToArray();

        foreach (var column in encoded)
        {
            var one = new FittedStepValues();
            one.Learned("categories", fitted.List(column));

            new EncodeStep(column, How, Unseen).ApplyTo(table, one);
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static EncodeCategoriesStep ReadFrom(JsonElement element) => new(AsKey.Read(element), UnseenKey.Read(element));
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
