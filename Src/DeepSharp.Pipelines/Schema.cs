// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>What a column holds.</summary>
public enum ColumnKind
{
    /// <summary>Words: a name, a category, a code.</summary>
    Text,

    /// <summary>A number that can have a fraction.</summary>
    Number,

    /// <summary>A whole number: a count, an identifier.</summary>
    Integer,

    /// <summary>True or false, in whichever of the usual spellings the file happens to use.</summary>
    Boolean,

    /// <summary>A moment in time.</summary>
    Timestamp,

    /// <summary>Words that stand for a group rather than for themselves: a port, a class, a symbol.</summary>
    /// <remarks>
    /// Said at the top, where the columns are declared, because it is a fact about the data and not a
    /// decision about the model. Everything downstream can then act on it: encoding takes them by name
    /// without being told twice, and a category that never reached a model as numbers is refused at the
    /// handover rather than silently dropped.
    /// </remarks>
    Category,
}

/// <summary>What happens to the columns the schema does not mention.</summary>
public enum Remainder
{
    /// <summary>They are left behind. A column nobody thought about is a column nobody checked.</summary>
    Drop,

    /// <summary>They come along as text, for a pipeline that is still finding out what is in the file.</summary>
    Keep,

    /// <summary>The run stops, for data whose shape is supposed to be fixed and is not.</summary>
    Refuse,
}

/// <summary>One column, as the pipeline expects to find it.</summary>
/// <param name="Name">The column's name in the source.</param>
/// <param name="Kind">What it holds.</param>
/// <param name="Optional">Whether the source is allowed not to have it at all.</param>
public sealed record ColumnDeclaration(string Name, ColumnKind Kind, bool Optional)
{
    /// <summary>
    /// Whether the schema leaves the column out while it keeps what the column holds: named, read by nothing, and
    /// taken in again as it was by clearing this.
    /// </summary>
    public bool Excluded { get; init; }

    /// <summary>The kind a category column held before it was made one, so it can go back to it; nothing otherwise.</summary>
    public ColumnKind? Was { get; init; }
}

/// <summary>
/// Names the columns that take part, and says what they hold.
/// </summary>
/// <remarks>
/// The columns are written in the order they are declared, which is also the order they reach a model.
/// Nothing here opens the data: this says what the pipeline expects, and the expectation is checked against
/// the source when the pipeline runs.
/// </remarks>
public sealed class SchemaBuilder
{
    private readonly List<ColumnDeclaration> _columns = [];

    /// <summary>A schema with nothing in it yet.</summary>
    public SchemaBuilder()
    {
    }

    /// <summary>Columns holding words.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    public SchemaBuilder Text(params string[] names) => Add(names, ColumnKind.Text);

    /// <summary>Columns holding numbers that can have a fraction.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    public SchemaBuilder Number(params string[] names) => Add(names, ColumnKind.Number);

    /// <summary>Columns holding whole numbers.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    public SchemaBuilder Integer(params string[] names) => Add(names, ColumnKind.Integer);

    /// <summary>Columns holding true or false.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    public SchemaBuilder Boolean(params string[] names) => Add(names, ColumnKind.Boolean);

    /// <summary>Columns holding a moment in time.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    public SchemaBuilder Timestamp(params string[] names) => Add(names, ColumnKind.Timestamp);

    /// <summary>Columns whose words stand for a group rather than for themselves.</summary>
    /// <param name="names">The column names.</param>
    /// <returns>This schema, so the next kind can be written after it.</returns>
    /// <remarks>
    /// The same words a text column holds, said to be a category here so that everything after knows it.
    /// The list of categories is still learned from the training rows alone, when they are encoded.
    /// </remarks>
    public SchemaBuilder Category(params string[] names) => Add(names, ColumnKind.Category);

    /// <summary>A column the source is allowed not to have at all.</summary>
    /// <param name="name">The column name.</param>
    /// <param name="kind">What it holds, when it is there.</param>
    /// <returns>This schema, so the next column can be written after it.</returns>
    /// <remarks>
    /// Absent from the file is not the same as empty in a row. A column that is simply not there is what
    /// this allows; an empty value in a column that is there is a gap, and gaps are filled after the split.
    /// </remarks>
    public SchemaBuilder Optional(string name, ColumnKind kind) => Add([name], kind, optional: true);

    /// <summary>The columns declared so far, in the order they were written.</summary>
    public IReadOnlyList<ColumnDeclaration> Columns => _columns;

    /// <summary>One column, said in full.</summary>
    /// <param name="name">The column's name in the source.</param>
    /// <param name="kind">What it holds.</param>
    /// <param name="optional">Whether the source is allowed not to have it at all.</param>
    /// <returns>This schema, so the next column can be written after it.</returns>
    public SchemaBuilder Column(string name, ColumnKind kind, bool optional = false) =>
        Add([name], kind, optional);

    private SchemaBuilder Add(string[] names, ColumnKind kind, bool optional = false)
    {
        ArgumentNullException.ThrowIfNull(names);

        foreach (var name in names)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("A column needs a name.", nameof(names));
            }

            if (_columns.Any(column => column.Name == name))
            {
                throw new ArgumentException($"The column '{name}' is declared twice.", nameof(names));
            }

            _columns.Add(new ColumnDeclaration(name, kind, optional));
        }

        return this;
    }
}

/// <summary>
/// Declares which columns take part, what they hold, and what becomes of the rest.
/// </summary>
/// <remarks>
/// The rest is the part worth saying out loud. A published dataset can carry the answer in a column nobody
/// thought about — a survival flag written as a word beside the number it repeats — and a pipeline that
/// carries everything it finds hands a model its own answer. Nothing splits its way out of that, so the
/// defence is that a column nobody declared does not come along.
/// </remarks>
public sealed record DeclareStep : IPipelineStep<DeclareStep>, IBindsColumns, IDeclaresCategories, IDescribesColumns
{
    private static readonly OneOfParameter<Remainder> RemainderKey = new(
        "remainder", "What becomes of the columns the schema does not name: dropped, kept as text, or refused.", Remainder.Drop);

    private static readonly ColumnDeclarationsParameter ColumnsKey = new(
        "columns", "The columns that take part, what each holds, and whether the source may lack it.",
        [new ColumnDeclaration("column", ColumnKind.Number, false)]);

    /// <summary>Declares the columns and the policy for everything else.</summary>
    /// <param name="columns">The columns the schema names, in the order they were written: those that take part, and those it excludes.</param>
    /// <param name="remainder">What becomes of the columns not named here.</param>
    /// <exception cref="ArgumentException">
    /// There are no columns, one is declared twice, a column says which kind it was without being a category, or every
    /// column is excluded while the rest is not kept, so none would take part.
    /// </exception>
    public DeclareStep(IEnumerable<ColumnDeclaration> columns, Remainder remainder = Remainder.Drop)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = ColumnsKey.Require([.. columns]);
        Remainder = RemainderKey.Require(remainder);
        Taking = [.. Columns.Where(column => !column.Excluded)];

        if (Taking.Count == 0 && Remainder != Remainder.Keep)
        {
            throw new ArgumentException(
                "The schema excludes every column it names and keeps none of the rest, so no column would take part.", nameof(columns));
        }
    }

    /// <inheritdoc />
    public static StepParameters<DeclareStep> Parameters { get; } = new StepParameters<DeclareStep>()
        .With(RemainderKey, step => step.Remainder)
        .With(ColumnsKey, step => step.Columns);

    /// <summary>
    /// Every column the schema names, in the order they were written: those that take part, and those it excludes,
    /// each with its kind. This is what the schema is written as, and what two schemas are compared by.
    /// </summary>
    public IReadOnlyList<ColumnDeclaration> Columns { get; }

    /// <summary>The columns that take part: every one the schema names, except those it excludes.</summary>
    /// <remarks>These are read from the source, and required of it unless they may be absent.</remarks>
    public IReadOnlyList<ColumnDeclaration> Taking { get; }

    /// <summary>This schema with a column taken in.</summary>
    /// <param name="name">The column.</param>
    /// <param name="kind">The kind it takes when the schema does not name it yet.</param>
    /// <param name="header">The source's columns, in their order: a new column stands where the source has it.</param>
    /// <returns>
    /// The schema with the column taking part: an excluded one brought back with the kind it had, a new one before the
    /// first declared column the source has after it, or last; this schema when the column takes part already.
    /// </returns>
    public DeclareStep WithColumn(string name, ColumnKind kind, IReadOnlyList<string> header)
    {
        ArgumentNullException.ThrowIfNull(header);

        var at = IndexOf(name);

        if (at >= 0)
        {
            return Columns[at].Excluded ? Replaced(at, Columns[at] with { Excluded = false }) : this;
        }

        var place = InSourceOrder(name, header);

        return Made([.. Columns.Take(place), new ColumnDeclaration(name, kind, Optional: false), .. Columns.Skip(place)]);
    }

    /// <summary>This schema with a column excluded: still named, with its kind, and read by nothing.</summary>
    /// <param name="name">The column.</param>
    /// <returns>The schema with the column excluded; this schema when it is excluded already or not named at all.</returns>
    /// <exception cref="ArgumentException">
    /// The column is the last one the schema takes, and the schema does not keep the rest: it would take no column.
    /// </exception>
    public DeclareStep WithColumnExcluded(string name)
    {
        var at = IndexOf(name);

        return at < 0 || Columns[at].Excluded ? this : Replaced(at, Columns[at] with { Excluded = true });
    }

    /// <summary>This schema with a column of another kind.</summary>
    /// <param name="name">The column.</param>
    /// <param name="kind">The kind.</param>
    /// <returns>
    /// The schema with the column of that kind: made a category, it remembers the kind it was; given any other kind,
    /// it forgets it. This schema when the column is of that kind already.
    /// </returns>
    /// <exception cref="ArgumentException">The schema does not name the column; taking one in is <see cref="WithColumn"/>.</exception>
    public DeclareStep WithColumnKind(string name, ColumnKind kind)
    {
        var at = IndexOf(name);

        if (at < 0)
        {
            throw new ArgumentException($"The schema does not name '{name}'; a column is taken in with its kind, not given one.", nameof(name));
        }

        var column = Columns[at];

        return column.Kind == kind
            ? this
            : Replaced(at, kind == ColumnKind.Category ? column with { Kind = kind, Was = column.Kind } : column with { Kind = kind, Was = null });
    }

    private int IndexOf(string name)
    {
        for (var at = 0; at < Columns.Count; at++)
        {
            if (Columns[at].Name == name)
            {
                return at;
            }
        }

        return -1;
    }

    private DeclareStep Replaced(int at, ColumnDeclaration column) => Made([.. Columns.Take(at), column, .. Columns.Skip(at + 1)]);

    // A schema one of this schema's operations makes, refusing what a schema cannot be in the words a file shows.
    private DeclareStep Made(IEnumerable<ColumnDeclaration> columns)
    {
        try
        {
            return new(columns, Remainder);
        }
        catch (ArgumentException refused)
        {
            throw new ArgumentException(StepCatalog.InTheFilesWords(refused), refused);
        }
    }

    // Where a new column goes: before the first declared column the source has after it; last when the source does not
    // have it, or has none after it.
    private int InSourceOrder(string name, IReadOnlyList<string> header)
    {
        var at = IndexIn(header, name);

        for (var place = 0; at >= 0 && place < Columns.Count; place++)
        {
            if (IndexIn(header, Columns[place].Name) > at)
            {
                return place;
            }
        }

        return Columns.Count;
    }

    private static int IndexIn(IReadOnlyList<string> names, string name)
    {
        for (var at = 0; at < names.Count; at++)
        {
            if (names[at] == name)
            {
                return at;
            }
        }

        return -1;
    }

    /// <summary>The columns taking part that are declared as standing for a group rather than for themselves.</summary>
    public IEnumerable<string> Categories =>
        Taking.Where(column => column.Kind == ColumnKind.Category).Select(column => column.Name);

    /// <summary>What becomes of the columns the schema does not name.</summary>
    public Remainder Remainder { get; }

    /// <inheritdoc />
    public static string Name => "declare";

    /// <inheritdoc />
    public static string Purpose => "Names the columns that take part, says what each holds, and decides what becomes of the rest.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public Table Bind(IRowSource source) => SchemaBinding.Bind(this, source);

    /// <inheritdoc />
    /// <remarks>
    /// The columns taking part, whatever there was before; a schema that keeps the rest leaves any other possible, except
    /// a column it excludes, which nothing below may read.
    /// </remarks>
    public ColumnState After(ColumnState before) => ColumnState.Declared(Columns, Remainder);

    /// <inheritdoc />
    public bool Equals(DeclareStep? other) =>
        other is not null && Remainder == other.Remainder && Columns.SequenceEqual(other.Columns);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Remainder);

        foreach (var column in Columns)
        {
            hash.Add(column);
        }

        return hash.ToHashCode();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or names a kind or a policy nobody defined.</exception>
    public static DeclareStep ReadFrom(JsonElement element) =>
        new(ColumnsKey.Read(element), RemainderKey.Read(element));
}
