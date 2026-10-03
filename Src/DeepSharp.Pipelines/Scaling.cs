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

    /// <summary>
    /// Centred on the middle of the training range and divided by half its width, so the training rows land between minus
    /// one and one, as a network takes them — scikit-learn's MinMaxScaler with a feature range of minus one to one. Moved
    /// by a single extreme value, as min-max is.
    /// </summary>
    MidRange,
}

/// <summary>Where each scale lands the training rows: the one rule a normalise step and a handover both read.</summary>
public static class ScaleExtensions
{
    extension(Scale scale)
    {
        /// <summary>Where a scale lands the training rows, when it lands them in a range.</summary>
        /// <returns>
        /// Between nothing and one for min-max and quantile, between minus one and one for max-abs and midrange; nothing for
        /// standard, robust and power, which centre a column and leave its extremes where they fall.
        /// </returns>
        public Form? Lands() => scale switch
        {
            Scale.MinMax or Scale.Quantile => Form.Unit,
            Scale.MaxAbs or Scale.MidRange => Form.Signed,
            _ => null,
        };
    }

    extension(Form range)
    {
        /// <summary>The scale that lands the training rows in a range, for a scaling that names no kind.</summary>
        /// <returns>Midrange for minus one to one, min-max for nothing to one.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        /// The form writes a value as two columns rather than landing one in a range, so it is no answer to where the
        /// features land.
        /// </exception>
        /// <remarks>
        /// The other direction of <see cref="Lands"/>, kept beside it so the two cannot drift: a range named here is a
        /// range some scale lands its rows in, and that scale is the one a column which names no kind is scaled by.
        /// </remarks>
        internal Scale Landing() => range switch
        {
            Form.Signed => Scale.MidRange,
            Form.Unit => Scale.MinMax,
            _ => throw new ArgumentOutOfRangeException(
                nameof(range),
                range,
                $"{range} writes a value as two columns, each between nothing and one, rather than landing one in a range, "
                + "so it does not say where the features land. Say Signed for minus one to one, or Unit for nothing to one."),
        };
    }
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

/// <summary>
/// The scalings of one line: which kind, and the columns it holds for.
/// </summary>
/// <remarks>
/// The kinds are the methods, as they are in the schema's own builder, so a line reads as the sentence it is and the
/// set stays open-closed: a kind added here adds a method and changes no call site. Each hands back one step a column,
/// which is what the declaration, the file and a notebook's blocks have always held — the line is a door, not a new
/// shape. What happens to a value outside the learned range is said on the kind that can hold one: a scale that lands
/// its rows in no range has nothing to hold a value in, and a rank has no place beyond the training rows to let one
/// through, so neither has a method that offers the choice.
/// </remarks>
public sealed class ScaleBuilder : IDeclaresSteps
{
    private readonly List<IPipelineStep> _steps = [];
    private Form _features = NormaliseStep.DefaultFeatures;

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>Where the pipeline declared its features land, for the columns this line names no kind for.</summary>
    Form IDeclaresSteps.Features
    {
        set => _features = value;
    }

    /// <summary>Columns scaled to land where the pipeline says its features land: between minus one and one, unless it says otherwise.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder Columns(params string[] columns) => Add(columns, _features.Landing(), OutOfRange.Pass);

    /// <summary>Columns centred on the middle of their training range, landing between minus one and one.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MidRange(params string[] columns) => Add(columns, Scale.MidRange, OutOfRange.Pass);

    /// <summary>Columns centred on the middle of their training range, saying what happens outside it.</summary>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MidRange(OutOfRange outOfRange, params string[] columns) => Add(columns, Scale.MidRange, outOfRange);

    /// <summary>Columns squeezed between nothing and one by their training extremes.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MinMax(params string[] columns) => Add(columns, Scale.MinMax, OutOfRange.Pass);

    /// <summary>Columns squeezed between nothing and one, saying what happens outside that range.</summary>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MinMax(OutOfRange outOfRange, params string[] columns) => Add(columns, Scale.MinMax, outOfRange);

    /// <summary>Columns divided by their largest magnitude, so a nought stays a nought.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MaxAbs(params string[] columns) => Add(columns, Scale.MaxAbs, OutOfRange.Pass);

    /// <summary>Columns divided by their largest magnitude, saying what happens outside that range.</summary>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    public ScaleBuilder MaxAbs(OutOfRange outOfRange, params string[] columns) => Add(columns, Scale.MaxAbs, outOfRange);

    /// <summary>Columns centred on their median and divided by the spread of their middle half.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>Unmoved by a few extremes, and it lands its rows in no range, so nothing is held or refused outside one.</remarks>
    public ScaleBuilder Robust(params string[] columns) => Add(columns, Scale.Robust, OutOfRange.Pass);

    /// <summary>Columns centred on their mean and divided by their spread.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>Moved by a single extreme value, and it lands its rows in no range.</remarks>
    public ScaleBuilder Standard(params string[] columns) => Add(columns, Scale.Standard, OutOfRange.Pass);

    /// <summary>Columns reshaped towards a bell curve and then centred.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>For a column that leans heavily one way; it lands its rows in no range.</remarks>
    public ScaleBuilder Power(params string[] columns) => Add(columns, Scale.Power, OutOfRange.Pass);

    /// <summary>Columns written as where each value sat among the training values, saying what happens beyond them.</summary>
    /// <param name="outOfRange">What happens to a value beyond the training rows: held at the edge, or refused.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>
    /// A rank has no place beyond the training rows for a value to pass to, which is why this is the one kind with no
    /// line that leaves the choice unsaid.
    /// </remarks>
    public ScaleBuilder Quantile(OutOfRange outOfRange, params string[] columns) => Add(columns, Scale.Quantile, outOfRange);

    private ScaleBuilder Add(string[] columns, Scale scale, OutOfRange outOfRange)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            _steps.Add(new NormaliseStep(column, scale, outOfRange));
        }

        return this;
    }
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
/// <para>
/// What happens to a value outside the range the fit learned is declared with it, and a pair the scale cannot
/// honour is refused where it is written: a scale that lands its rows in no range — standard, robust, power —
/// has nothing to hold a value in or refuse it outside, and a rank has no place beyond the training rows for a
/// value to pass to, so a quantile scale holds it at the edge or refuses it. A value within the rounding that a
/// scale's own centre and spread carry, of an end its training rows reached, is read as that end rather than as
/// outside it; a value truly beyond it is still refused.
/// </para>
/// </remarks>
public sealed record NormaliseStep : IFittedStep, IUndoesItself, IPipelineStep<NormaliseStep>, IDescribesColumns, IMeetsANeed
{
    /// <summary>
    /// The scaling a caller who names none means: the training rows land between minus one and one, which is where a
    /// network takes its features.
    /// </summary>
    /// <remarks>
    /// Said once, here, and read by every door that lets the scale be left out — this step's own constructor, the
    /// chain's verb, and the value a new block starts with. A default is a compile-time constant in C#, so it cannot
    /// be read from the parameter at the signature; the parameter reads it from here instead, and a test holds every
    /// door to this one value.
    /// </remarks>
    public const Scale DefaultScale = Scale.MidRange;

    /// <summary>Where a pipeline that does not say lands its features: between minus one and one.</summary>
    /// <remarks>
    /// The range <see cref="DefaultScale"/> lands its rows in, said as the range rather than as the scale, because that
    /// is what <see cref="PipelineBuilder.DefaultFeatures"/> takes and what a reader of a chain asks about.
    /// </remarks>
    public const Form DefaultFeatures = Form.Signed;

    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to bring onto a comparable scale.", "column", ColumnKinds.Numbers);

    private static readonly OneOfParameter<Scale> ScaleKey = new(
        "scale", "Which kind of scaling: what the fit learns from the training rows.", DefaultScale);

    private static readonly OneOfParameter<OutOfRange> OutOfRangeKey = new(
        "outOfRange", "What happens to a value outside the range the fit learned: let it through, hold it at the edge, or refuse.", OutOfRange.Pass);

    /// <summary>Declares that a column is brought onto a comparable scale.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="scale">Which kind of scaling.</param>
    /// <param name="outOfRange">What happens to a value outside the range the fit learned.</param>
    /// <exception cref="ArgumentException">
    /// The column has no name, or the scale cannot do what is said of a value outside its range: hold it or refuse it
    /// with no range to hold it in, or let it through a rank.
    /// </exception>
    public NormaliseStep(string column, Scale scale = DefaultScale, OutOfRange outOfRange = OutOfRange.Pass)
    {
        Column = ColumnKey.Require(column);
        Scale = ScaleKey.Require(scale);
        OutOfRange = OutOfRangeKey.Require(outOfRange);

        // A rule between two parameters, so it lives with the step that has both.
        var word = Vocabulary<Scale>.WordFor(Scale, ScaleKey.Key);

        if (OutOfRange != OutOfRange.Pass && Scale.Lands() is null)
        {
            throw new ArgumentException(
                $"A {word} scale lands its rows in no range, so there is nothing to hold a value in or to refuse one outside: it lets every value through, which is pass.",
                nameof(outOfRange));
        }

        if (OutOfRange == OutOfRange.Pass && Scale == Scale.Quantile)
        {
            throw new ArgumentException(
                $"A {word} scale ranks a value among the training rows and has no place beyond them to let one through: it holds it at the edge, which is clip, or refuses it.",
                nameof(outOfRange));
        }
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
    /// <remarks>
    /// A learner that takes numbers of any size does without a scale: a tree splits between two training values, and a scale
    /// moves no row to the other side of one, nor does holding a value at the training rows' edge. A scale that refuses what
    /// it was not fitted on is taken for every learner, since the refusal is a check somebody declared.
    /// </remarks>
    public bool NeededBy(Needs needs) => !needs.DoesWithoutScaling() || OutOfRange == OutOfRange.Refuse;

    /// <inheritdoc />
    /// <remarks>Nothing: a scale left out is not fitted, not applied, and writes no entry.</remarks>
    IFittedStep? IMeetsANeed.Instead => null;

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

        return before.With(Column, ColumnKind.Number, Scale.Lands());
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
                learned.Learned("spread", Spread(training.StandardDeviation, training.Finite));
                break;

            case Scale.MinMax:
                learned.Learned("centre", training.Finite[0]);
                learned.Learned("spread", Spread(training.Finite[^1] - training.Finite[0], training.Finite));
                break;

            case Scale.MidRange:
                learned.Learned("centre", (training.Finite[0] + training.Finite[^1]) / 2);
                learned.Learned("spread", Spread((training.Finite[^1] - training.Finite[0]) / 2, training.Finite));
                break;

            case Scale.MaxAbs:
                learned.Learned("centre", 0);
                learned.Learned("spread", Spread(training.Finite.Max(Math.Abs), training.Finite));
                break;

            case Scale.Robust:
                learned.Learned("centre", training.Median);
                learned.Learned("spread", Spread(training.Quantile(0.75) - training.Quantile(0.25), training.Finite));
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
                learned.Learned("spread", Spread(Math.Sqrt(shaped.Average(value => (value - middle) * (value - middle))), shaped));
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
                scaled[row] = values[row] is not { } value ? null
                    : OutOfRange == OutOfRange.Refuse && (value < knots[0] || value > knots[^1]) ? throw Outside(table, row)
                    : Rank(knots, value);
            }

            table.Put(new Column<double>(Column, ColumnKind.Number, scaled));

            return;
        }

        var centre = fitted.Number("centre");
        var spread = fitted.Number("spread");
        var lambda = Scale == Scale.Power ? fitted.Number("lambda") : 0;

        // Where the training rows land, for a scale that lands them in a range; a scale that does not only passes.
        var floor = Scale.Lands()?.Floor() ?? double.NegativeInfinity;
        var ceiling = Scale.Lands() is null ? double.PositiveInfinity : 1;
        var edge = Edge(centre, spread);

        for (var row = 0; row < values.Length; row++)
        {
            if (values[row] is not { } value)
            {
                continue;
            }

            var next = ((Scale == Scale.Power ? YeoJohnson.Of(value, lambda) : value) - centre) / spread;

            // A value within the rounding that building centre and spread out of the training extremes
            // carries, of an end the training rows reached, is that end: min-max reaches its ends exactly,
            // because a value there and centre share one of the two training extremes bit for bit, but
            // midrange builds centre as their average and spread as their half-difference -- two roundings
            // that do not cancel (measured on AAPL.Close: the training maximum came back
            // 1.0000000000000004). A genuinely outside value stays outside; a new high is nowhere near this close.
            next = Math.Abs(next - floor) <= edge ? floor : Math.Abs(next - ceiling) <= edge ? ceiling : next;

            // Min-max on a price meets this the first time there is a new high, so what happens then is
            // part of the declaration rather than something the library decides on everybody's behalf.
            scaled[row] = OutOfRange switch
            {
                OutOfRange.Clip => Math.Clamp(next, floor, ceiling),
                OutOfRange.Refuse when next < floor || next > ceiling => throw Outside(table, row),
                _ => next,
            };
        }

        table.Put(new Column<double>(Column, ColumnKind.Number, scaled));
    }

    // A refusal names the row as it was read, which is the row a person can find in their file -- the same
    // rule Handover.cs uses for the rows it hands over, rather than where a row now stands in a table a split
    // may have put in a different order.
    private InvalidOperationException Outside(Table table, int row) =>
        new($"Row {table.Identities[row].ReadAt + 1} of '{Column}' is outside the range this pipeline was fitted on.");

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static NormaliseStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), ScaleKey.Read(element), OutOfRangeKey.Read(element));

    // The rounding one sum of doubles carries, relative to the largest of them: the step from one to the next double.
    private static readonly double Rounding = Math.BitIncrement(1.0) - 1.0;

    // A spread no larger than the rounding its own arithmetic carries is nothing, and a column is then divided by one: as
    // scikit-learn's StandardScaler decides a feature is constant. Two hundred of 0.1 average to 0.10000000000000007, and
    // the deviation around that came out as 6.9e-17, which divided every other value into the quadrillions.
    private static double Spread(double spread, IReadOnlyList<double> from) =>
        spread <= from.Count * Rounding * from.Max(Math.Abs) ? 1 : spread;

    // The rounding fitting a boundary from the two training extremes carries, on top of placing a value against
    // it afterwards: the centre's own rounding, scaled by how far it sits from nothing relative to the spread,
    // plus one apiece for the centre, the spread, the subtraction and the division that follow -- the four
    // roundings between the two training extremes and a ratio of them.
    private static double Edge(double centre, double spread) => Rounding * ((Math.Abs(centre) / spread) + 4);

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

            // The first knot the value does not pass, after one it passed: the two are never the same.
            var within = (value - knots[at - 1]) / (knots[at] - knots[at - 1]);

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

        // Each value divided by the row's size lies between minus one and one, whichever size is taken.
        return Columns.Aggregate(before, (state, column) => state.With(column, ColumnKind.Number, Form.Signed));
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

            // Worked out around the largest value in the row, as every library that measures a length does: squaring
            // the values themselves loses a row of small ones to nothing and sends a row of large ones to infinity.
            // Measured on this code before the scaling: a row of 1e200 came back as noughts, and a row of 1e-160 had
            // its length out by a relative 5.6e-6.
            var largest = present.Max(value => Math.Abs(value!.Value));

            // A row of noughts has no size to divide by, and nothing to change.
            if (largest <= 0)
            {
                continue;
            }

            // Every size is at least the largest value, since that value divided by itself is one, so there is no
            // second guard against dividing by nothing.
            var size = Norm switch
            {
                Norm.L1 => largest * present.Sum(value => Math.Abs(value!.Value) / largest),
                Norm.Max => largest,
                _ => largest * Math.Sqrt(present.Sum(value => (value!.Value / largest) * (value!.Value / largest))),
            };

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

    // Every twentieth from minus two to two, each worked out from its whole count rather than added up: added up, the points
    // near nought and two came out as 1.2e-15 and 2.000000000000002, so the search never measured the shaping it keeps
    // there, and dividing by what rounding left of them took 2 for the best shaping of values scipy puts at 0.71.
    internal static double Lambda(double[] training)
    {
        var best = 1.0;
        var most = double.NegativeInfinity;

        for (var step = 0; step <= 80; step++)
        {
            var lambda = Math.Round(-2 + (0.05 * step), 4);
            var likelihood = Likelihood(training, lambda);

            if (likelihood > most)
            {
                most = likelihood;
                best = lambda;
            }
        }

        return best;
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
