// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using MatPlotLibNet;
using MatPlotLibNet.Indicators;
using Microsoft.Data.Analysis;

namespace DeepSharp.Pipelines;

/// <summary>Which indicator a step asks for.</summary>
/// <remarks>
/// A name rather than a piece of code, because a piece of code cannot be written into a file and read back
/// a year later. Adding one is a case in this list and a case in the adapter; the arithmetic itself is
/// borrowed and stays borrowed.
/// </remarks>
public enum Indicator
{
    /// <summary>Simple moving average of a price column.</summary>
    Sma,

    /// <summary>Exponential moving average of a price column.</summary>
    Ema,

    /// <summary>Relative strength index of a price column.</summary>
    Rsi,

    /// <summary>Average true range, from high, low and close.</summary>
    Atr,

    /// <summary>Average directional index, from high, low and close.</summary>
    Adx,

    /// <summary>Commodity channel index, from high, low and close.</summary>
    Cci,

    /// <summary>Williams %R, from high, low and close.</summary>
    WilliamsR,

    /// <summary>On-balance volume, from close and volume.</summary>
    Obv,

    /// <summary>Moving average convergence divergence: three columns.</summary>
    Macd,

    /// <summary>Bollinger bands: three columns.</summary>
    BollingerBands,

    /// <summary>Stochastic oscillator: two columns.</summary>
    Stochastic,

    /// <summary>Volume-weighted average price, from high, low, close and volume.</summary>
    Vwap,
}

/// <summary>
/// Adds an indicator as one or more columns, worked out from the rows that came before.
/// </summary>
/// <remarks>
/// An indicator is a feature with a memory, so it belongs before the split: it learns nothing from the data
/// as a whole, it is arithmetic over a row and its predecessors. Two properties are not negotiable and both
/// are tested rather than promised.
/// <para>
/// <b>The window only looks backwards.</b> A centred average, or anything that reaches forward to smooth,
/// is a leak in mathematical dress — the row would carry what had not happened yet. Every indicator here is
/// held to an impulse test: change one row, and no earlier row may move.
/// </para>
/// <para>
/// <b>The warm-up is a gap, not a fault.</b> An indicator of period N has no value for its first rows and
/// says so with a not-a-number. Here that becomes an absence, because it is one — the value was never
/// computed, rather than computed wrongly. A not-a-number appearing after the first real value is left
/// alone, so the step that refuses those can still see it.
/// </para>
/// </remarks>
public sealed record AddIndicatorStep : IPipelineStep<AddIndicatorStep>, IAddsColumns, IReadsRowOrder, IDescribesColumns
{
    private static readonly NewColumnParameter ColumnKey = new(
        "column", "What the new column is called; an indicator with several parts adds a suffix for each.", "indicator");

    private static readonly OneOfParameter<Indicator> IndicatorKey = new(
        "indicator", "Which indicator, worked out from the rows that came before.", Indicator.Sma);

    private static readonly WholeNumberParameter PeriodKey = new(
        "period", "The look-back in rows, for an indicator that takes one.", 14, atLeast: 1);

    // Each place is a role — high, low, close — so rows with one price may name it in every place.
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns it reads, in the order the indicator expects them.", ["close"], ColumnKinds.Numbers, repeatable: true);

    /// <summary>Declares an indicator over the named columns.</summary>
    /// <param name="name">What the new column is called; a many-valued indicator adds a suffix per part.</param>
    /// <param name="indicator">Which indicator.</param>
    /// <param name="columns">The columns it reads, in the order the indicator expects them.</param>
    /// <param name="period">The look-back, for an indicator that takes one.</param>
    /// <exception cref="ArgumentException">A name is empty, or the columns are not what the indicator needs.</exception>
    public AddIndicatorStep(string name, Indicator indicator, IEnumerable<string> columns, int period = 14)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Column = ColumnKey.Require(name)!;
        Kind = IndicatorKey.Require(indicator);
        Period = PeriodKey.Require(period);

        IReadOnlyList<string> given = [.. columns];
        var wanted = Needs(Kind);

        // A rule between two parameters, so it lives with the step that has both.
        if (given.Count != wanted)
        {
            throw new ArgumentException(
                $"{indicator} reads {wanted} column{(wanted == 1 ? string.Empty : "s")} and was given {given.Count}.",
                nameof(columns));
        }

        Columns = ColumnsKey.Require(given);
    }

    /// <inheritdoc />
    public static StepParameters<AddIndicatorStep> Parameters { get; } = new StepParameters<AddIndicatorStep>()
        .With(ColumnKey, step => step.Column)
        .With(IndicatorKey, step => step.Kind)
        .With(PeriodKey, step => step.Period)
        .With(ColumnsKey, step => step.Columns);

    /// <summary>What the new column is called.</summary>
    public string Column { get; }

    /// <summary>Which indicator.</summary>
    public Indicator Kind { get; }

    /// <summary>The columns it reads, in the order the indicator expects them.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The look-back, for an indicator that takes one.</summary>
    public int Period { get; }

    /// <inheritdoc />
    public static string Name => "feature.indicator";

    /// <inheritdoc />
    public static string Purpose => "Adds a market indicator worked out from the rows that came before: an average, a strength index, a band.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public bool Equals(AddIndicatorStep? other) =>
        other is not null
        && Column == other.Column
        && Kind == other.Kind
        && Period == other.Period
        && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Column);
        hash.Add(Kind);
        hash.Add(Period);

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

        var frame = new DataFrame(
            Columns.Select(name => (DataFrameColumn)new PrimitiveDataFrameColumn<double>(
                name, table.NumbersOf(name).Select(value => value ?? double.NaN))));

        foreach (var output in Compute(frame))
        {
            table.Put(new Column<double>(output.Column, ColumnKind.Number, WarmUpIsAGap(output.Values, table.RowCount)));
        }
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return Made.Aggregate(before, (state, column) => state.With(column, ColumnKind.Number));
    }

    /// <summary>The columns this indicator makes: its name, or its name and a suffix for each of its parts.</summary>
    public IReadOnlyList<string> Made => [.. Parts(Kind).Select(Named)];

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or names an indicator nobody defined.</exception>
    public static AddIndicatorStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element)!, IndicatorKey.Read(element), ColumnsKey.Read(element), PeriodKey.Read(element));

    /// <summary>How many columns an indicator reads.</summary>
    /// <param name="indicator">The indicator.</param>
    /// <returns>The number of columns it expects.</returns>
    public static int Needs(Indicator indicator) => indicator switch
    {
        Indicator.Sma or Indicator.Ema or Indicator.Rsi or Indicator.Macd or Indicator.BollingerBands => 1,
        Indicator.Obv => 2,
        Indicator.Vwap => 4,
        _ => 3,
    };

    // The parts an indicator has, by the suffix each is written under; one part, unnamed, for most of them.
    // The act and the description both read this, so the columns an indicator makes are said in one place.
    private static IReadOnlyList<string> Parts(Indicator indicator) => indicator switch
    {
        Indicator.Macd => ["line", "signal", "histogram"],
        Indicator.BollingerBands => ["upper", "middle", "lower"],
        Indicator.Stochastic => ["k", "d"],
        _ => [string.Empty],
    };

    private string Named(string part) => part.Length == 0 ? Column : $"{Column}_{part}";

    private IEnumerable<IndicatorOutput> Compute(DataFrame frame)
    {
        var one = Columns[0];

        IEnumerable<double[]> values = Kind switch
        {
            Indicator.Sma => [frame.Sma(one, Period)],
            Indicator.Ema => [frame.Ema(one, Period)],
            Indicator.Rsi => [frame.Rsi(one, Period)],
            Indicator.Atr => [frame.Atr(Columns[0], Columns[1], Columns[2], Period)],
            Indicator.Adx => [frame.Adx(Columns[0], Columns[1], Columns[2], Period)],
            Indicator.Cci => [frame.Cci(Columns[0], Columns[1], Columns[2], Period)],
            Indicator.WilliamsR => [frame.WilliamsR(Columns[0], Columns[1], Columns[2], Period)],
            Indicator.Obv => [frame.Obv(Columns[0], Columns[1])],
            Indicator.Macd => Macd(frame.Macd(one)),
            Indicator.BollingerBands => Bands(frame.BollingerBands(one, Period)),
            Indicator.Stochastic => Stochastic(frame.Stochastic(Columns[0], Columns[1], Columns[2], Period)),
            _ => [frame.Vwap(Columns[0], Columns[1], Columns[2], Columns[3])],
        };

        return values.Zip(Made, (series, column) => new IndicatorOutput(column, series));
    }

    private static IEnumerable<double[]> Macd(MacdResult macd) => [macd.MacdLine, macd.SignalLine, macd.Histogram];

    private static IEnumerable<double[]> Bands(BandsResult bands) => [bands.Upper, bands.Middle, bands.Lower];

    private static IEnumerable<double[]> Stochastic(StochasticResult stochastic) => [stochastic.K, stochastic.D];

    private static IEnumerable<double?> WarmUpIsAGap(double[] values, int rows)
    {
        // An indicator says its warm-up in one of two ways, and which one is not something the caller can
        // see: some pad the front with a not-a-number and keep the length, others hand back a shorter array
        // and leave the alignment to whoever asked. Measured on a twenty-period average over 506 rows: 487
        // values came back. Laying a short result against row nought would shift every value backwards by
        // the length of the warm-up, which changes nothing visible and corrupts everything afterwards.
        if (values.Length > rows)
        {
            throw new InvalidOperationException(
                $"The indicator returned {values.Length} values for {rows} rows, which is more than there are.");
        }

        for (var missing = rows - values.Length; missing > 0; missing--)
        {
            yield return null;
        }

        var started = false;

        foreach (var value in values)
        {
            if (!started && double.IsNaN(value))
            {
                // No value yet rather than a value that went wrong: the two are different answers and the
                // library keeps them apart everywhere else, so it keeps them apart here.
                yield return null;

                continue;
            }

            started = true;

            yield return value;
        }
    }
}

/// <summary>
/// One column an indicator makes, and its values as the arithmetic handed them back.
/// </summary>
/// <param name="Column">The column's name.</param>
/// <param name="Values">The values, perhaps fewer than there are rows.</param>
internal readonly record struct IndicatorOutput(string Column, double[] Values);

/// <summary>
/// Adding an indicator to a pipeline.
/// </summary>
public static class IndicatorExtensions
{
    /// <summary>Adds an indicator worked out from the rows that came before.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="name">What the new column is called.</param>
    /// <param name="indicator">Which indicator.</param>
    /// <param name="columns">The columns it reads, in the order the indicator expects them.</param>
    /// <param name="period">The look-back, for an indicator that takes one.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    public static PipelineBuilder AddIndicator(
        this PipelineBuilder pipeline, string name, Indicator indicator, string[] columns, int period = 14)
    {
        ArgumentNullException.ThrowIfNull(pipeline);

        return pipeline.Add(new AddIndicatorStep(name, indicator, columns, period));
    }

    /// <summary>Teaches a catalog to read the indicator verb back out of a file.</summary>
    /// <param name="catalog">The catalog being assembled.</param>
    /// <returns>The same catalog, so registration reads as one sentence.</returns>
    public static StepCatalog WithIndicators(this StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        catalog.Register<AddIndicatorStep>();

        return catalog;
    }
}

/// <summary>
/// The indicator verb, offered to whichever catalog an application builds.
/// </summary>
public sealed class IndicatorSteps : IStepContribution
{
    /// <inheritdoc />
    public void AddTo(StepCatalog catalog) => catalog.WithIndicators();
}
