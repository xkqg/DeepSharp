// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>What the cells of one column say it holds, proposed for a person to accept or change.</summary>
/// <param name="Name">The column's name in the source.</param>
/// <param name="Kind">
/// The kind proposed: the one the source states, when it states one; else the first every cell that holds a value reads
/// as — true or false, a whole number, a number, a moment — and words when none does, a category when they are few.
/// </param>
/// <param name="Offered">
/// Another kind the cells can be taken as, offered beside the one proposed and never in its place: a category for whole
/// numbers few enough to be groups, since a class and a count of siblings look alike; a timestamp for moments whose order
/// of day and month the cells cannot settle. Nothing otherwise.
/// </param>
/// <param name="Formats">
/// The formats, as .NET writes a date format, that each read every cell: the one a timestamp is proposed with; the several
/// a timestamp is offered with, which read the cells day first and month first alike; none otherwise.
/// </param>
/// <param name="Stated">The kind the source states for the column, when it states one: it outranks what the cells look like.</param>
/// <param name="Values">How many cells hold a value, for the kind proposed.</param>
/// <param name="Gaps">How many cells are gaps, for the kind proposed: an empty cell for words, and a cell of spaces too for anything else.</param>
/// <param name="Distinct">How many different values the cells hold, when that is at most 64; nothing when there are more.</param>
public readonly record struct ColumnProposal(
    string Name, ColumnKind Kind, ColumnKind? Offered, IReadOnlyList<string> Formats, ColumnKind? Stated, int Values, int Gaps, int? Distinct)
{
    /// <summary>The format a timestamp is proposed with; nothing when its moments are ISO 8601, or it is proposed as no timestamp.</summary>
    public string? Format => Kind == ColumnKind.Timestamp && Formats.Count == 1 ? Formats[0] : null;

    /// <inheritdoc />
    public bool Equals(ColumnProposal other) =>
        Name == other.Name && Kind == other.Kind && Offered == other.Offered && Formats.SequenceEqual(other.Formats) && Stated == other.Stated
        && Values == other.Values && Gaps == other.Gaps && Distinct == other.Distinct;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Name, Kind, Offered, Formats.Count, Stated, Values, Gaps, Distinct);
}

/// <summary>
/// What a source's cells say each of its columns holds: a proposal a person accepts or changes, never a schema.
/// </summary>
/// <remarks>
/// Every cell of every row is read, before anything divides the rows — a schema comes before a split, and a kind is a
/// declaration rather than something learned — and by the one reading the schema binds with, so every kind proposed is a
/// kind every row binds as. It is a value to decide from and nothing takes it in: a person writes the schema, once, and a
/// pipeline replays what was written rather than working a schema out again from whatever the file holds on the day it
/// runs. What only the training rows may say — which categories there are, where the extremes lie — is still learned
/// from them alone.
/// <para>
/// Words are proposed as a category when there are at most 64 different ones and each stands for twenty rows or more;
/// libraries that do this disagree sevenfold on the share, so the rule is DeepSharp's own, measured on the samples. Whole
/// numbers that few are offered as a category and never proposed as one, since nothing in the cells tells a class from a
/// count. A column of noughts and ones is whole numbers rather than true or false, which could not be filled where a gap
/// is. Moments are read as ISO 8601 writes them, or by the one format among the usual ones that reads every cell; every
/// format names a year, a month and a day, so nothing is filled in from the day the proposal is made, and cells that read
/// day first and month first alike are offered as moments with both formats named, never proposed as one of them.
/// </para>
/// </remarks>
public sealed class KindProposal
{
    // At most this many different words make a category.
    private const int MostCategories = 64;

    // A category stands for at least this many rows, on average.
    private const int RowsPerCategory = 20;

    // The usual ways of writing a moment other than ISO 8601's, each naming a year, a month and a day. Two of them read
    // the same cells only when the cells cannot say whether the day or the month comes first.
    private static readonly string[] Written =
    [
        "d/M/yyyy", "d/M/yyyy H:mm", "d/M/yyyy H:mm:ss",
        "M/d/yyyy", "M/d/yyyy H:mm", "M/d/yyyy H:mm:ss", "M/d/yyyy h:mm tt", "M/d/yyyy h:mm:ss tt",
        "d-M-yyyy", "d-M-yyyy H:mm", "d-M-yyyy H:mm:ss",
        "M-d-yyyy", "M-d-yyyy H:mm", "M-d-yyyy H:mm:ss",
        "d.M.yyyy", "d.M.yyyy H:mm", "d.M.yyyy H:mm:ss",
        "yyyy/M/d", "yyyy/M/d H:mm", "yyyy/M/d H:mm:ss", "yyyy-M-d", "yyyy.M.d",
        "d MMM yyyy", "d MMMM yyyy", "MMM d, yyyy", "MMMM d, yyyy", "MMM d yyyy",
    ];

    private static readonly IReadOnlyDictionary<string, ColumnKind> NoneStated = new Dictionary<string, ColumnKind>();

    private KindProposal(IReadOnlyList<ColumnProposal> columns) => Columns = columns;

    /// <summary>Every column of the source, in its order, with what its cells say it holds.</summary>
    public IReadOnlyList<ColumnProposal> Columns { get; }

    /// <summary>What one column's cells say it holds.</summary>
    /// <param name="column">The column's name in the source.</param>
    /// <returns>Its proposal.</returns>
    /// <exception cref="ArgumentException">The source has no such column.</exception>
    public ColumnProposal this[string column]
    {
        get
        {
            foreach (var proposed in Columns)
            {
                if (proposed.Name == column)
                {
                    return proposed;
                }
            }

            throw new ArgumentException($"The source has no column called '{column}', so nothing is proposed for it.", nameof(column));
        }
    }

    /// <summary>What every column of a source holds, as its cells say and as the source states.</summary>
    /// <param name="source">The rows, every one of which is read.</param>
    /// <returns>The proposal.</returns>
    public static KindProposal Of(IRowSource source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var rows = source.Rows.ToArray();
        var stated = source is IStatesKinds states ? states.StatedKinds : NoneStated;

        return new([.. source.ColumnNames.Select((name, at) => Proposed(
            name, [.. rows.Select(row => at < row.Count ? row[at] : null)], stated.TryGetValue(name, out var kind) ? kind : null))]);
    }

    private static ColumnProposal Proposed(string name, string?[] cells, ColumnKind? stated)
    {
        if (stated is { } kind)
        {
            return Counted(name, cells, kind, stated);
        }

        string[] filled = [.. cells.Where(cell => !cell.IsGap()).Select(cell => cell!)];

        if (filled.Length == 0)
        {
            return Counted(name, cells, ColumnKind.Text, null);
        }

        // Every spelling of true and false, except noughts and ones alone, which are whole numbers a gap can be filled in.
        if (filled.All(cell => cell.AsTrueOrFalse().Refusal is null) && !filled.All(cell => cell.Trim() is "0" or "1"))
        {
            return Counted(name, cells, ColumnKind.Boolean, null);
        }

        if (filled.All(cell => cell.AsWholeNumber().Refusal is null))
        {
            return Counted(name, cells, ColumnKind.Integer, null);
        }

        if (filled.All(cell => cell.AsNumber().Refusal is null))
        {
            return Counted(name, cells, ColumnKind.Number, null);
        }

        if (filled.All(cell => cell.AsMoment(null).Refusal is null))
        {
            return Counted(name, cells, ColumnKind.Timestamp, null);
        }

        string[] formats = [.. Written.Where(format => filled.All(cell => cell.AsMoment(format).Refusal is null))];

        return formats.Length == 1
            ? Counted(name, cells, ColumnKind.Timestamp, null) with { Formats = formats }
            : Counted(name, cells, ColumnKind.Text, null) with { Formats = formats, Offered = formats.Length > 1 ? ColumnKind.Timestamp : null };
    }

    // The counts of a column for a kind: what is a gap is the kind's to say, and few enough words are a category.
    private static ColumnProposal Counted(string name, string?[] cells, ColumnKind kind, ColumnKind? stated)
    {
        var words = kind is ColumnKind.Text or ColumnKind.Category;
        string[] values = [.. cells.Where(cell => words ? !string.IsNullOrEmpty(cell) : !cell.IsGap()).Select(cell => cell!)];
        var distinct = Distinct(values);
        var few = distinct is { } count && count * RowsPerCategory <= values.Length && values.Length > 0;

        return new(
            name,
            kind == ColumnKind.Text && few ? ColumnKind.Category : kind,
            kind == ColumnKind.Integer && few ? ColumnKind.Category : null,
            [],
            stated,
            values.Length,
            cells.Length - values.Length,
            distinct);
    }

    // How many different values there are, counted as far as a category could hold them and no further.
    private static int? Distinct(string[] values)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var value in values)
        {
            if (seen.Add(value) && seen.Count > MostCategories)
            {
                return null;
            }
        }

        return seen.Count;
    }
}
