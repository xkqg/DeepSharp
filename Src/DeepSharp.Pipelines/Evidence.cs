// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How a piece of evidence is shown: drawn, or as a grid of numbers.</summary>
public enum Shown
{
    /// <summary>Drawn: a correlation as a coloured grid, a profile as bars.</summary>
    Drawn,

    /// <summary>As the numbers themselves.</summary>
    Numbers,
}

/// <summary>
/// A step that produces evidence about the rows where it stands, and changes nothing.
/// </summary>
/// <remarks>
/// Declared with the pipeline, before the numbers exist, so every run produces the same evidence and the
/// report cannot shrink to whatever happened to look good. It is measured when the pipeline is fitted, on the
/// rows the split trains on — the split below it as much as one above — and not again when the pipeline is
/// replayed. What it produces is output, kept with the run and never written into the pipeline's file.
/// </remarks>
public interface IProducesEvidence : IActsInAWalk
{
    /// <summary>The evidence about the rows as they stand here.</summary>
    /// <param name="view">The rows here, and where each stands.</param>
    /// <returns>The evidence.</returns>
    Evidence Produce(PipelineView view);

    /// <inheritdoc />
    void IActsInAWalk.ActOn(Walk walk) => walk.Evidence(Produce);
}

/// <summary>
/// Something a run shows as proof: a profile of the columns, the rows a correlation is drawn from.
/// </summary>
public abstract class Evidence
{
    private protected Evidence()
    {
    }

    /// <summary>Hands this evidence to whatever shows it, as the kind it is.</summary>
    /// <typeparam name="TResult">What the visitor builds.</typeparam>
    /// <param name="visitor">The visitor.</param>
    /// <returns>What it built.</returns>
    public abstract TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor);
}

/// <summary>Something built from each kind of evidence: a picture, a table of numbers.</summary>
/// <typeparam name="TResult">What it builds.</typeparam>
public interface IEvidenceVisitor<out TResult>
{
    /// <summary>A profile of the columns.</summary>
    /// <param name="profile">The profile.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(DataProfile profile);

    /// <summary>The rows a correlation is drawn from.</summary>
    /// <param name="correlation">The rows, and how many were kept.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(CorrelationInput correlation);
}

/// <summary>One column, as a profile measures it on the measured rows.</summary>
/// <param name="Name">The column's name.</param>
/// <param name="Kind">What it holds.</param>
/// <param name="Rows">How many rows were measured.</param>
/// <param name="Gaps">How many of them are gaps.</param>
/// <param name="NotFinite">How many hold a value that is not a finite number.</param>
/// <param name="Distinct">How many different values there are, a gap not counted.</param>
/// <param name="Constant">Whether every value there is is the same one.</param>
/// <param name="Min">The smallest number, for a column of numbers with any.</param>
/// <param name="Max">The largest number, for a column of numbers with any.</param>
/// <param name="Mean">The average, for a column of numbers with any.</param>
/// <param name="Median">The middle value, for a column of numbers with any.</param>
public readonly record struct ColumnProfile(
    string Name,
    ColumnKind Kind,
    int Rows,
    int Gaps,
    int NotFinite,
    int Distinct,
    bool Constant,
    double? Min,
    double? Max,
    double? Mean,
    double? Median);

/// <summary>Something a profile found, and the step that answers it.</summary>
/// <param name="Column">The column it is about.</param>
/// <param name="Says">What it found, in words.</param>
/// <param name="Verb">The step that deals with it.</param>
public readonly record struct ProfileAlert(string Column, string Says, string Verb);

/// <summary>
/// A profile of the columns where it stands: what they hold, what is wrong with them, and which step answers it.
/// </summary>
public sealed class DataProfile : Evidence
{
    internal DataProfile(
        Standing over, int rows, IReadOnlyList<ColumnProfile> columns, IReadOnlyList<ProfileAlert> alerts, DuplicateRows duplicates)
    {
        Over = over;
        Rows = rows;
        Columns = columns;
        Alerts = alerts;
        Duplicates = duplicates;
    }

    /// <summary>The rows it measured: the training rows, or the undivided ones in a pipeline with no split.</summary>
    public Standing Over { get; }

    /// <summary>How many rows it measured.</summary>
    public int Rows { get; }

    /// <summary>Each column it profiled.</summary>
    public IReadOnlyList<ColumnProfile> Columns { get; }

    /// <summary>What it found, each with the step that answers it.</summary>
    public IReadOnlyList<ProfileAlert> Alerts { get; }

    /// <summary>The rows that are there more than once, among all the rows where it stands.</summary>
    public DuplicateRows Duplicates { get; }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>
/// The rows a correlation is drawn from: the measured rows with a finite number in every column named.
/// </summary>
/// <remarks>
/// The correlation itself is worked out by whatever draws it. What the pipeline says is which rows it may be
/// worked out from, how many of the measured rows that is, and the rule that left the others out — because a
/// correlation over the rows that happened to be complete is a different number from one over all of them.
/// </remarks>
public sealed class CorrelationInput : Evidence
{
    internal CorrelationInput(Standing over, IReadOnlyList<string> columns, IReadOnlyList<double[]> rows, int total, Shown shown)
    {
        Over = over;
        Columns = columns;
        Rows = rows;
        Total = total;
        Shown = shown;
    }

    /// <summary>Which rows it was drawn from: the training rows, or the undivided ones when nothing divides them.</summary>
    public Standing Over { get; }

    /// <summary>The columns, in the order every row lists them.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The complete rows, each with one number per column.</summary>
    public IReadOnlyList<double[]> Rows { get; }

    /// <summary>How many rows were kept.</summary>
    public int Kept => Rows.Count;

    /// <summary>How many rows were measured, kept or not.</summary>
    public int Total { get; }

    /// <summary>The rule that decided which rows were kept.</summary>
    public string Policy => "complete rows";

    /// <summary>How it is shown.</summary>
    public Shown Shown { get; }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>
/// Profiles the columns where it stands, on the rows the split trains on.
/// </summary>
/// <remarks>
/// For each column: how many rows, gaps and values that are not numbers; how many different values; for
/// numbers, the smallest, the largest, the average and the middle. And what is wrong, each thing with the step
/// that answers it — a gap with <c>fill.missing</c>, a value that is not a number with <c>fill.nan</c>, a column
/// that never changes with <c>drop.columns</c>, words with an encoder.
/// </remarks>
public sealed record ProfileStep : IPipelineStep<ProfileStep>, IProducesEvidence, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to profile; left out, every column where the step stands.", ["column"], ColumnKinds.Any, optional: true);

    /// <summary>Declares a profile of these columns, or of every column where it stands.</summary>
    /// <param name="columns">The columns to profile; none, for every column.</param>
    /// <exception cref="ArgumentException">A column has no name, or one is named twice.</exception>
    public ProfileStep(IEnumerable<string>? columns = null) =>
        Columns = ColumnsKey.Require([.. columns ?? []]);

    /// <summary>The columns to profile; empty for every column where the step stands.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "evidence.profile";

    /// <inheritdoc />
    public static string Purpose => "Profiles the columns where it stands, on the rows the split trains on, and names the step that answers each thing it finds.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<ProfileStep> Parameters { get; } =
        new StepParameters<ProfileStep>().With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(ProfileStep? other) => other is not null && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public Evidence Produce(PipelineView view)
    {
        ArgumentNullException.ThrowIfNull(view);

        var table = view.Table;
        var measured = Enumerable.Range(0, table.RowCount).Where(row => view.Standings[row] == view.Measured).ToArray();
        var names = Columns.Count > 0 ? Columns : [.. table.Columns.Select(column => column.Name)];
        var profiles = names.Select(name => Profiled(view, table[name], measured)).ToArray();

        return new DataProfile(view.Measured, measured.Length, profiles, [.. profiles.SelectMany(Alerts)], Duplicates(view));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ProfileStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element));

    private static ColumnProfile Profiled(PipelineView view, IColumn column, int[] measured)
    {
        var gaps = measured.Count(column.IsMissing);
        var distinct = measured.Where(row => !column.IsMissing(row)).Select(column.TextAt).Distinct(StringComparer.Ordinal).Count();

        if (column.Kind is ColumnKind.Text or ColumnKind.Category or ColumnKind.Timestamp)
        {
            return new ColumnProfile(column.Name, column.Kind, measured.Length, gaps, 0, distinct, distinct == 1, null, null, null, null);
        }

        var values = view.MeasuredValues(column.Name);
        var any = values.Finite.Count > 0;

        return new ColumnProfile(
            column.Name, column.Kind, measured.Length, gaps, values.NotFinite, distinct, distinct == 1,
            any ? values.Finite[0] : null,
            any ? values.Finite[^1] : null,
            any ? values.Mean : null,
            any ? values.Median : null);
    }

    private static IEnumerable<ProfileAlert> Alerts(ColumnProfile column)
    {
        if (column.Gaps > 0)
        {
            yield return new ProfileAlert(column.Name, $"{column.Gaps} of {column.Rows} rows are gaps.", "fill.missing");
        }

        if (column.NotFinite > 0)
        {
            yield return new ProfileAlert(column.Name, $"{column.NotFinite} of {column.Rows} rows hold a value that is not a finite number.", "fill.nan");
        }

        if (column.Constant)
        {
            yield return new ProfileAlert(column.Name, "Every row holds the same value, so it tells a model nothing.", "drop.columns");
        }

        if (column.Kind == ColumnKind.Category)
        {
            yield return new ProfileAlert(column.Name, "It holds groups as words, which a model cannot take as they are.", "encode.categories");
        }

        if (column.Kind == ColumnKind.Text)
        {
            yield return new ProfileAlert(column.Name, "It holds words, which a model cannot take as they are.", "encode");
        }
    }

    // The part of each standing: the one of the same name, and none for a row dropped before the split reaches it.
    private static readonly Part[] PartOf =
        [.. Enum.GetValues<Standing>().Select(standing => Enum.TryParse<Part>(standing.ToString(), out var part) ? part : Part.Undivided)];

    // Among every row where it stands; across parts only where there are parts to be across.
    private static DuplicateRows Duplicates(PipelineView view) =>
        view.Measured == Standing.Undivided
            ? view.Table.Duplicates()
            : view.Table.Duplicates([.. view.Standings.Select(standing => PartOf[(int)standing])]);
}

/// <summary>
/// Sets out the rows a correlation between columns is drawn from, on the rows the split trains on.
/// </summary>
/// <remarks>
/// Only the rows with a finite number in every column named, and it says how many of the measured rows that is
/// and by which rule: a correlation over the rows that happened to be complete is a different number from one
/// over all of them, and a picture that does not say so is a picture of something else.
/// </remarks>
public sealed record CorrelationStep : IPipelineStep<CorrelationStep>, IProducesEvidence, IDescribesColumns
{
    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to correlate with one another: two or more, each holding numbers.", ["left", "right"], ColumnKinds.Numbers);

    private static readonly OneOfParameter<Shown> ShownKey = new(
        "shown", "How it is shown: drawn as a coloured grid, or as the numbers themselves.", Shown.Drawn);

    /// <summary>Declares the rows a correlation between these columns is drawn from.</summary>
    /// <param name="columns">The columns: two or more, each holding numbers.</param>
    /// <param name="shown">How it is shown.</param>
    /// <exception cref="ArgumentException">There are fewer than two columns, one has no name, or one is named twice.</exception>
    public CorrelationStep(IEnumerable<string> columns, Shown shown = Shown.Drawn)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        Shown = ShownKey.Require(shown);

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

    /// <inheritdoc />
    public static string Name => "evidence.correlation";

    /// <inheritdoc />
    public static string Purpose => "Sets out the rows a correlation between columns is drawn from, on the rows the split trains on, and how many it kept.";

    /// <inheritdoc />
    public static int Since => 2;

    /// <inheritdoc />
    public static StepParameters<CorrelationStep> Parameters { get; } = new StepParameters<CorrelationStep>()
        .With(ColumnsKey, step => step.Columns)
        .With(ShownKey, step => step.Shown);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(CorrelationStep? other) => other is not null && Shown == other.Shown && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Shown);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public Evidence Produce(PipelineView view)
    {
        ArgumentNullException.ThrowIfNull(view);

        var table = view.Table;
        var values = Columns.Select(table.NumbersOf).ToArray();
        var measured = Enumerable.Range(0, table.RowCount).Where(row => view.Standings[row] == view.Measured).ToArray();

        var complete = measured
            .Where(row => values.All(column => column[row] is { } value && double.IsFinite(value)))
            .Select(row => values.Select(column => column[row]!.Value).ToArray())
            .ToArray();

        return new CorrelationInput(view.Measured, Columns, complete, measured.Length, Shown);
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static CorrelationStep ReadFrom(JsonElement element) => new(ColumnsKey.Read(element), ShownKey.Read(element));
}
