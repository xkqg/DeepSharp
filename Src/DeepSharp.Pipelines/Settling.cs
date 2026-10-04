// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Settles the gaps in a column with a value no row decided, where the features are worked out.
/// </summary>
/// <remarks>
/// <para>
/// What fills a gap is usually learned from the training rows — a mean, a median — and a step that learns stands below
/// the split. That leaves one thing unreachable: a column worked out above the split from a column with a gap is itself a
/// gap, and filling what it was made from afterwards does not reach back into it. Settling is the other half of the same
/// work. A nought, a number you choose, or a refusal: none of them is a value any row could have told, so no row of
/// validation or test taught it anything, and it may stand where the features are.
/// </para>
/// <para>
/// It learns nothing, so it writes nothing down: there is no entry for it in the fitted half of the file, and a replay
/// does what the run did by doing it again. It marks where the gaps were exactly as a fill does, and for the same reason
/// — filling destroys the difference between absent and measured whichever value goes in, so the difference is written
/// down before it goes.
/// </para>
/// </remarks>
public sealed record SettleGapsStep : IPipelineStep<SettleGapsStep>, IAddsColumns, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column with gaps in it.", "column", ColumnKinds.Fillable);

    private static readonly FillStrategyParameter WithKey = new(
        "with",
        "What goes in the gaps, decided by nobody but you: zero, a constant, or refuse for a column that is not supposed to have gaps at all.",
        With.Zero,
        ["zero", "constant", "refuse"])
    {
        What = "settling a gap",
    };

    /// <summary>Declares that the gaps in a column are settled, above the split, with a value no row decided.</summary>
    /// <param name="column">The column with gaps in it.</param>
    /// <param name="strategy">What goes in them: <see cref="With.Zero"/>, <see cref="With.Constant"/> or <see cref="With.Refuse"/>.</param>
    /// <exception cref="ArgumentException">
    /// The column has no name, or the strategy is one a row decides — a mean or a median is learned from the training
    /// rows, and carrying the value before a gap forward reads the rows in their order; both of those are
    /// <see cref="FillMissingStep"/>, below the split.
    /// </exception>
    public SettleGapsStep(string column, FillStrategy strategy = default)
    {
        Column = ColumnKey.Require(column);
        Strategy = WithKey.Require(string.IsNullOrWhiteSpace(strategy.Name) ? With.Zero : strategy);
    }

    /// <inheritdoc />
    public static StepParameters<SettleGapsStep> Parameters { get; } = new StepParameters<SettleGapsStep>()
        .With(ColumnKey, step => step.Column)
        .With(WithKey, step => step.Strategy);

    /// <summary>The column with gaps in it.</summary>
    public string Column { get; }

    /// <summary>What goes in them.</summary>
    public FillStrategy Strategy { get; }

    /// <summary>The column written beside a settled one, saying where the gaps were.</summary>
    /// <remarks>The same name a fill writes, because it says the same thing about the same column.</remarks>
    public string MarkerColumn => Column.Marked;

    /// <inheritdoc />
    public static string Name => "settle.gaps";

    /// <inheritdoc />
    public static string Purpose =>
        "Settles the gaps in a column with a value no row decided, so a feature worked out from it is not a gap.";

    /// <summary>The first version of the pipeline file that has this verb.</summary>
    public static int Since => 6;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Filled(Column, Strategy).With(MarkerColumn, ColumnKind.Number, Form.Unit);
    }

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var column = table[Column];

        table.Marks(Column);

        switch (column)
        {
            case Column<double> numbers:
                Settle(numbers, value => value);
                break;

            case Column<long> whole:
                Settle(whole, value => (long)Math.Round(value, MidpointRounding.AwayFromZero));
                break;

            default:
                throw new InvalidOperationException(
                    $"'{Column}' holds {column.Kind.ToString().ToLowerInvariant()}, and a gap in it is not settled with a number.");
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or is not what it should be.</exception>
    public static SettleGapsStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), WithKey.Read(element));

    // Every gap in the column, settled where it stands: the value is the same whatever else the table holds, which is
    // what lets this stand above the split at all.
    private void Settle<T>(Column<T> column, Func<double, T> asValue)
        where T : struct
    {
        for (var row = 0; row < column.Count; row++)
        {
            if (!column.IsMissing(row))
            {
                continue;
            }

            column[row] = Strategy.Name == With.Refuse.Name
                ? throw new InvalidOperationException(
                    string.Create(CultureInfo.InvariantCulture, $"Row {row + 1} of '{Column}' is a gap, and this pipeline says there should be none."))
                : asValue(Strategy.Name == With.Zero.Name ? 0 : Strategy.Value!.Value);
        }
    }
}

/// <summary>
/// The gaps one line settles, each with a value no row decided.
/// </summary>
/// <remarks>
/// The kinds are the methods, as everywhere else a line is written, so the line reads as the sentence it is and offers
/// only what its verb takes. The ways a fill learns from the training rows — a mean, a median — are not here, and neither
/// is carrying the value before a gap forward: each of those is decided by rows, so each belongs below the split, where
/// <see cref="GapLine"/> offers it.
/// </remarks>
public sealed class SettleLine : IDeclaresSteps
{
    private readonly List<IPipelineStep> _steps = [];

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>Columns whose gaps are settled with nought, which is a measurement and not an absence.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>The marking column beside each says which it was.</remarks>
    public SettleLine Zero(params string[] columns) => Add(columns, With.Zero);

    /// <summary>Columns whose gaps are settled with a number you choose.</summary>
    /// <param name="value">The number to put in every gap.</param>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>For a column where absence has a meaning you already know.</remarks>
    public SettleLine Constant(double value, params string[] columns) => Add(columns, With.Constant(value));

    /// <summary>Columns where a gap stops the run.</summary>
    /// <param name="columns">The columns.</param>
    /// <returns>This line, so the next kind can be written after it.</returns>
    /// <remarks>For a column that is not supposed to have anything wrong with it.</remarks>
    public SettleLine Refuse(params string[] columns) => Add(columns, With.Refuse);

    private SettleLine Add(string[] columns, FillStrategy strategy)
    {
        ArgumentNullException.ThrowIfNull(columns);

        foreach (var column in columns)
        {
            _steps.Add(new SettleGapsStep(column, strategy));
        }

        return this;
    }
}
