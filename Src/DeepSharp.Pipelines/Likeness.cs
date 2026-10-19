// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>How far apart the values of one column are in two parts.</summary>
public enum Distance
{
    /// <summary>
    /// For numbers, the two-sample Kolmogorov–Smirnov distance: the most the share of one part at or below a value differs from
    /// the other's, from nought, the same spread, to one, apart.
    /// </summary>
    KolmogorovSmirnov,

    /// <summary>
    /// For words, the total variation distance: half the sum of how far each word's share in one part differs from its share in
    /// the other, from nought to one.
    /// </summary>
    TotalVariation,
}

/// <summary>How far apart one column is in two parts, and whether that is above the bound the person set.</summary>
/// <param name="First">The first part.</param>
/// <param name="Second">The second part.</param>
/// <param name="Apart">The distance, from nought to one.</param>
/// <param name="Above">Whether it is above the bound.</param>
public readonly record struct PartsApart(Standing First, Standing Second, double Apart, bool Above)
{
    /// <summary>How many rows of the first part hold a value of the column.</summary>
    public int FirstRows { get; init; }

    /// <summary>How many rows of the second part hold a value of the column.</summary>
    public int SecondRows { get; init; }
}

/// <summary>How alike one column is across the parts, measured by the distance its kind takes.</summary>
/// <param name="Column">The column.</param>
/// <param name="Distance">The distance its kind takes.</param>
/// <param name="Pairs">Every two parts that hold rows: training and validation, training and test, validation and test.</param>
public sealed record ColumnLikeness(string Column, Distance Distance, IReadOnlyList<PartsApart> Pairs);

/// <summary>
/// How alike the parts a split made are, column by column, each distance beside the bound above which it is flagged.
/// </summary>
/// <remarks>
/// A part the model is measured on that is unlike the rows it learned from explains a measurement that disappoints, and a
/// validation part unlike the training rows is a poorer guide to stop on — as a split in time makes them, where the latest rows
/// differ from the earlier ones. The distances are worked out on the rows of each part as they stand where the step does; a
/// distance of a few rows is noise as much as anything, which is why each pair says how many rows it was worked out from.
/// </remarks>
public sealed class PartLikeness : Evidence
{
    internal PartLikeness(double warnAbove, IReadOnlyList<ColumnLikeness> columns)
    {
        WarnAbove = warnAbove;
        Columns = columns;
    }

    /// <summary>The bound above which a distance is flagged.</summary>
    public double WarnAbove { get; }

    /// <summary>Every column measured, in the order the rows hold them.</summary>
    public IReadOnlyList<ColumnLikeness> Columns { get; }

    /// <summary>
    /// The two-sample Kolmogorov–Smirnov distance between two sets of values: the most the share of one at or below a value
    /// differs from the other's, values that tie taken whole, as scipy's <c>ks_2samp</c> takes them.
    /// </summary>
    /// <param name="first">The values of the first set, each a finite number; at least one.</param>
    /// <param name="second">The values of the second set; at least one.</param>
    /// <returns>The distance, from nought to one.</returns>
    /// <exception cref="ArgumentException">A set holds no value.</exception>
    public static double KolmogorovSmirnov(IReadOnlyList<double> first, IReadOnlyList<double> second)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);

        if (first.Count == 0 || second.Count == 0)
        {
            throw new ArgumentException("A distance is between two sets that each hold a value.");
        }

        return NumbersTallied.Of(first, gaps: 0).ApartFrom(NumbersTallied.Of(second, gaps: 0));
    }

    /// <summary>The total variation distance between two sets of words: half the sum of how far each word's share differs.</summary>
    /// <param name="first">The words of the first set; at least one.</param>
    /// <param name="second">The words of the second set; at least one.</param>
    /// <returns>The distance, from nought to one.</returns>
    /// <exception cref="ArgumentException">A set holds no word.</exception>
    public static double TotalVariation(IReadOnlyList<string> first, IReadOnlyList<string> second)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);

        if (first.Count == 0 || second.Count == 0)
        {
            throw new ArgumentException("A distance is between two sets that each hold a word.");
        }

        return WordsTallied.Of(first, gaps: 0).ApartFrom(WordsTallied.Of(second, gaps: 0));
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IEvidenceVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>
/// Sets out how alike the parts a split made are, column by column, and flags every distance above a bound the person sets.
/// </summary>
/// <remarks>
/// Every other piece of evidence is measured on the training rows alone, because a profile over every row lets the rows a model
/// is measured on shape what it is shown. This one compares the parts with one another, so it reads the validation and the test
/// rows — the features only, never the answer, and it changes nothing: it warns, and a bound above which it warns has no
/// default, since how far apart is too far depends on how many rows a part holds and on what the data is.
/// </remarks>
public sealed record LikenessStep : IPipelineStep<LikenessStep>, IProducesEvidence, IDescribesColumns
{
    internal static readonly NumberParameter WarnAboveKey = new(
        "warnAbove", "The distance above which two parts are flagged as unlike: a share above nought, at most one.", 0.1)
    {
        AtLeast = 0,
    };

    private static readonly ColumnsParameter ColumnsKey = new(
        "columns", "The columns to compare across the parts; left out, every column where it stands but the answer.", ["column"], ColumnKinds.Any)
    {
        Optional = true,
    };

    /// <summary>Declares a comparison of the parts, of these columns or of every column where it stands but the answer.</summary>
    /// <param name="warnAbove">The distance above which two parts are flagged: above nought, at most one.</param>
    /// <param name="columns">The columns to compare; none, for every column but the answer.</param>
    /// <exception cref="ArgumentOutOfRangeException">The bound is not above nought and at most one.</exception>
    /// <exception cref="ArgumentException">A column has no name, or one is named twice.</exception>
    public LikenessStep(double warnAbove, IEnumerable<string>? columns = null)
    {
        WarnAbove = warnAbove > 0 && warnAbove <= 1
            ? WarnAboveKey.Require(warnAbove)
            : throw new ArgumentOutOfRangeException(nameof(warnAbove), warnAbove, "Two parts are flagged above a distance that is above nought and at most one.");
        Columns = ColumnsKey.Require([.. columns ?? []]);
    }

    /// <summary>The distance above which two parts are flagged.</summary>
    public double WarnAbove { get; }

    /// <summary>The columns compared; none, for every column but the answer.</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <inheritdoc />
    public static string Name => "evidence.likeness";

    /// <inheritdoc />
    public static string Purpose => "Sets out how alike the parts a split made are, column by column, and flags the ones further apart than a bound.";

    /// <inheritdoc />
    public static int Since => 10;

    /// <inheritdoc />
    public static StepParameters<LikenessStep> Parameters { get; } = new StepParameters<LikenessStep>()
        .With(WarnAboveKey, step => step.WarnAbove)
        .With(ColumnsKey, step => step.Columns);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(LikenessStep? other) => other is not null && WarnAbove.Equals(other.WarnAbove) && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(WarnAbove);

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
        var named = Columns.Count > 0
            ? Columns
            : [.. table.Columns.Where(column => column.Kind != ColumnKind.Media).Select(column => column.Name).Where(name => name != table.Id && !view.Answers.Contains(name, StringComparer.Ordinal))];
        Standing[] parts = [Standing.Train, Standing.Validation, Standing.Calibration, Standing.Test];
        var present = parts.Where(part => view.Standings.Contains(part)).ToArray();
        var pairs = present.SelectMany((first, at) => present.Skip(at + 1).Select(second => new Pair(first, second))).ToArray();

        return new PartLikeness(WarnAbove, [.. named.Select(name => Measured(table, view.Standings, name, pairs))]);
    }

    private ColumnLikeness Measured(Table table, IReadOnlyList<Standing> standings, string name, Pair[] pairs)
    {
        Tally In(Standing part) => Tally.Of(table, name, [.. Enumerable.Range(0, table.RowCount).Where(row => standings[row] == part)]);

        var tallies = pairs.SelectMany(pair => new[] { pair.First, pair.Second }).Distinct().ToDictionary(part => part, In);
        var distance = table[name] is TextColumn ? Distance.TotalVariation : Distance.KolmogorovSmirnov;

        return new ColumnLikeness(name, distance, [.. pairs.Select(pair => Apart(tallies[pair.First], tallies[pair.Second], pair))]);
    }

    private PartsApart Apart(Tally first, Tally second, Pair pair)
    {
        var apart = first.ApartFrom(second);

        return new PartsApart(pair.First, pair.Second, apart, apart > WarnAbove) { FirstRows = first.Held, SecondRows = second.Held };
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static LikenessStep ReadFrom(JsonElement element) => new(WarnAboveKey.Read(element), ColumnsKey.Read(element));

    // Two parts a column is compared across, the earlier first.
    private readonly record struct Pair(Standing First, Standing Second);
}
