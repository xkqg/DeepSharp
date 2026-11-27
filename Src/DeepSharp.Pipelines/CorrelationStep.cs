// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Sets out the rows a correlation between columns is drawn from, on the rows the split trains on.
/// </summary>
/// <remarks>
/// Only the rows with a finite number in every column named, and it says how many of the measured rows that is
/// and by which rule: a correlation over the rows that happened to be complete is a different number from one
/// over all of them, and a picture that does not say so is a picture of something else. The coefficient shown is declared
/// too: Pearson's, how well two columns lie on a line, unless it says Spearman's, how well they keep the same order —
/// which does not mind a curve that only ever rises, and is a file's <c>coefficient</c> from version 8, left out as
/// Pearson's always was.
/// </remarks>
public sealed record CorrelationStep : IPipelineStep<CorrelationStep>, IProducesEvidence, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to correlate with one another: two or more, each holding numbers.", ["left", "right"], ColumnKinds.Numbers);

    private static readonly OneOfParameter<Shown> ShownKey = new(
        "shown", "How it is shown: drawn as a coloured grid, or as the numbers themselves.", Shown.Drawn);

    private static readonly OneOfParameter<Coefficient> CoefficientKey = new(
        "coefficient",
        "Which coefficient is shown: how well two columns lie on a line (pearson), or how well they keep the same order (spearman). "
        + "Left out, pearson.",
        Coefficient.Pearson)
    {
        LeftOut = Coefficient.Pearson,
    };

    /// <summary>Declares the rows a correlation between these columns is drawn from.</summary>
    /// <param name="columns">The columns: two or more, each holding numbers.</param>
    /// <param name="shown">How it is shown.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    public CorrelationStep(IEnumerable<string> columns, Shown shown = Shown.Drawn)
        : this(columns, shown, Coefficient.Pearson)
    {
    }

    /// <summary>Declares the rows a correlation between these columns is drawn from, and which coefficient is shown.</summary>
    /// <param name="columns">The columns: two or more, each holding numbers.</param>
    /// <param name="shown">How it is shown.</param>
    /// <param name="coefficient">Pearson's, which is how well two columns lie on a line, or Spearman's, how well they keep the same order.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    public CorrelationStep(IEnumerable<string> columns, Shown shown, Coefficient coefficient)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        Shown = ShownKey.Require(shown);
        Coefficient = CoefficientKey.Require(coefficient);

        // A rule on the number of columns, so it lives with the step.
        if (Columns.Count < 2)
        {
            throw new ArgumentException("A correlation is between two columns or more.", nameof(columns));
        }
    }

    /// <summary>The columns.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>How it is shown.</summary>
    public Shown Shown { get; }

    /// <summary>Which coefficient is shown: Pearson's, or Spearman's on the ranks.</summary>
    public Coefficient Coefficient { get; }

    /// <inheritdoc />
    public static string Name => "evidence.correlation";

    /// <inheritdoc />
    public static string Purpose => "Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<CorrelationStep> Parameters { get; } = new StepParameters<CorrelationStep>()
        .With(ColumnsKey, step => step.Columns)
        .With(ShownKey, step => step.Shown)
        .With(CoefficientKey, step => step.Coefficient);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public Evidence Produce(PipelineView view)
    {
        ArgumentNullException.ThrowIfNull(view);

        var table = view.Table;
        var values = Columns.Select(table.NumbersOf).ToArray();
        var measured = view.MeasuredRows();

        var complete = measured
            .Where(row => values.All(column => column[row] is { } value && double.IsFinite(value)))
            .Select(row => values.Select(column => column[row]!.Value).ToArray())
            .ToArray();

        return new CorrelationInput(view, Columns, complete, Shown) { Coefficient = Coefficient };
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CorrelationStep ReadFrom(JsonElement element) =>
        new(ColumnsKey.Read(element), ShownKey.Read(element), CoefficientKey.Read(element));
}
