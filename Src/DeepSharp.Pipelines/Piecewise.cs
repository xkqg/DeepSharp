// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Writes a column of numbers down as how far each value fills a few bins: Gorishniy, Rubachev and Babenko's piecewise linear
/// encoding, its edges the quantiles of the training rows alone.
/// </summary>
/// <remarks>
/// <para>
/// The edges are the quantiles at every multiple of one over <see cref="Bins"/> of the training rows' values, the smallest and
/// the largest among them, and edges that fall together are one, so a column whose training rows repeat a value has fewer bins
/// than asked for and no bin that is empty on every row. A value fills every bin wholly below it, the share of its own bin it
/// reaches, and none above, so the bins, each times its width, added to the first edge, are the value again. The column is
/// written as one column a bin, <c>{column}_bin1</c> onwards, in its place.
/// </para>
/// <para>
/// A value below the first edge or above the last is what <see cref="OutOfRange"/> says: let through, the first bin going
/// below nothing and the last above one, as the encoding is written; held at the edges; or refused, naming the row. The
/// training rows lie between the edges, so they land between nothing and one.
/// </para>
/// </remarks>
public sealed record PiecewiseLinearStep : IFittedStep, IPipelineStep<PiecewiseLinearStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column of numbers to write down as bins.", "column", ColumnKinds.Numbers);

    private static readonly WholeNumberParameter BinsKey = new(
        "bins", "How many bins the training rows' quantiles divide the values into, at least two; fewer where edges fall together.", 8)
    {
        AtLeast = 2,
    };

    private static readonly OneOfParameter<OutOfRange> OutOfRangeKey = new(
        "outOfRange", "What happens to a value beyond the edges the training rows set: let through the end bins, held at them, or refused.", OutOfRange.Pass);

    /// <summary>Declares that a column of numbers is written down as bins.</summary>
    /// <param name="column">The column of numbers.</param>
    /// <param name="bins">How many bins, at least two.</param>
    /// <param name="outOfRange">What happens to a value beyond the edges.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two bins.</exception>
    public PiecewiseLinearStep(string column, int bins, OutOfRange outOfRange = OutOfRange.Pass)
    {
        Column = ColumnKey.Require(column);
        Bins = BinsKey.Require(bins);
        OutOfRange = OutOfRangeKey.Require(outOfRange);
    }

    /// <inheritdoc />
    public static StepParameters<PiecewiseLinearStep> Parameters { get; } = new StepParameters<PiecewiseLinearStep>()
        .With(ColumnKey, step => step.Column)
        .With(BinsKey, step => step.Bins)
        .With(OutOfRangeKey, step => step.OutOfRange);

    /// <summary>The column of numbers.</summary>
    public string Column { get; }

    /// <summary>How many bins the training rows' quantiles divide the values into, at most.</summary>
    public int Bins { get; }

    /// <summary>What happens to a value beyond the edges the training rows set.</summary>
    public OutOfRange OutOfRange { get; }

    /// <inheritdoc />
    public static string Name => "encode.piecewise";

    /// <inheritdoc />
    public static string Purpose => "Writes a column of numbers down as how far each value fills a few bins, their edges the training rows' quantiles.";

    /// <inheritdoc />
    public static int Since => 10;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>The bins are a family: how many there are is known once the training rows have been seen, so they are known by the start of their names.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.Without(Column).WithFamily($"{Column}_bin", Form.Unit);
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">The training rows hold no number of the column, or only one value of it, and so no edges to place.</exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var training = table.TrainingValues(Column, parts).Learnable("the edges of bins");
        var edges = Enumerable.Range(0, Bins + 1).Select(at => training.Quantile((double)at / Bins)).Distinct().ToArray();

        if (edges.Length < 2)
        {
            throw new InvalidOperationException(
                string.Create(CultureInfo.InvariantCulture, $"The training rows of '{Column}' hold one value, {edges[0]}, and one value has no edges between which to place a bin."));
        }

        var learned = new FittedStepValues();
        learned.Learned("edges", edges);

        return learned;
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A value lies beyond the edges, and such a value is refused.</exception>
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var edges = fitted.Curve("edges");
        var values = table.NumbersOf(Column);
        var last = edges.Count - 1;

        if (OutOfRange == OutOfRange.Refuse
            && Enumerable.Range(0, values.Length).FirstOrDefault(row => values[row] is { } value && (value < edges[0] || value > edges[last]), -1) is var beyond and >= 0)
        {
            throw new InvalidOperationException($"Row {table.Identities[beyond].ReadAt + 1} of '{Column}' is beyond the edges this pipeline was fitted on.");
        }

        table.Remove(Column);

        for (var bin = 1; bin <= last; bin++)
        {
            var here = new Bin(edges[bin - 1], edges[bin], First: bin == 1, Last: bin == last);

            table.Put(new Column<double>(
                string.Create(CultureInfo.InvariantCulture, $"{Column}_bin{bin}"),
                ColumnKind.Number,
                values.Select(value => value is { } x ? (double?)Filled(x, here) : null)));
        }
    }

    // How far a value fills one bin: none below it and wholly above it — but for the end bins, which a value beyond the edges
    // runs on through, unless it is held at them.
    private double Filled(double x, Bin bin)
    {
        var share = !bin.First && x < bin.Low ? 0
            : !bin.Last && x >= bin.High ? 1
            : (x - bin.Low) / (bin.High - bin.Low);

        return OutOfRange == OutOfRange.Clip ? Math.Clamp(share, 0, 1) : share;
    }

    // One bin: its two edges, and whether it is the first or the last, which a value beyond the edges runs on through.
    private readonly record struct Bin(double Low, double High, bool First, bool Last);

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static PiecewiseLinearStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), BinsKey.Read(element), OutOfRangeKey.Read(element));
}
