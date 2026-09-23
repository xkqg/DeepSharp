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
public sealed record ColumnDeclaration(string Name, ColumnKind Kind, bool Optional);

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
public sealed record DeclareStep : IPipelineStep<DeclareStep>, IBindsColumns, IDeclaresCategories
{
    /// <summary>Declares the columns and the policy for everything else.</summary>
    /// <param name="columns">The columns that take part, in the order they were written.</param>
    /// <param name="remainder">What becomes of the columns not named here.</param>
    /// <exception cref="ArgumentException">There are no columns, or one is declared twice.</exception>
    public DeclareStep(IEnumerable<ColumnDeclaration> columns, Remainder remainder = Remainder.Drop)
    {
        ArgumentNullException.ThrowIfNull(columns);

        Columns = [.. columns];

        if (Columns.Count == 0)
        {
            throw new ArgumentException("A schema names at least one column.", nameof(columns));
        }

        var duplicate = Columns.GroupBy(column => column.Name).FirstOrDefault(group => group.Count() > 1);

        if (duplicate is not null)
        {
            throw new ArgumentException($"The column '{duplicate.Key}' is declared twice.", nameof(columns));
        }

        Remainder = remainder;
    }

    /// <summary>The columns that take part, in the order they were written.</summary>
    public IReadOnlyList<ColumnDeclaration> Columns { get; }

    /// <summary>The columns declared as standing for a group rather than for themselves.</summary>
    public IEnumerable<string> Categories =>
        Columns.Where(column => column.Kind == ColumnKind.Category).Select(column => column.Name);

    /// <summary>What becomes of the columns the schema does not name.</summary>
    public Remainder Remainder { get; }

    /// <inheritdoc />
    public static string Name => "declare";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public Table Bind(IRowSource source) => SchemaBinding.Bind(this, source);

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

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("remainder", Remainder.ToString().ToLowerInvariant());
        writer.WriteStartArray("columns");

        foreach (var column in Columns)
        {
            writer.WriteStartObject();
            writer.WriteString("name", column.Name);
            writer.WriteString("kind", column.Kind.ToString().ToLowerInvariant());
            writer.WriteBoolean("optional", column.Optional);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or names a kind or a policy nobody defined.</exception>
    public static DeclareStep ReadFrom(JsonElement element)
    {
        var remainder = element.RequiredEnum<Remainder>("remainder");

        if (!element.TryGetProperty("columns", out var columns) || columns.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException("A declare step holds a 'columns' list.");
        }

        return new DeclareStep(
            columns.EnumerateArray().Select(column => new ColumnDeclaration(
                column.RequiredString("name"),
                column.RequiredEnum<ColumnKind>("kind"),
                column.RequiredBoolean("optional"))),
            remainder);
    }
}
