// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What a pipeline decided about its columns, saved as a file of its own: the schema, the columns dropped after the
/// steps that read them, the output, and the source's columns as they were last shown.
/// </summary>
/// <remarks>
/// Nothing fitted and no rows — only what a person decided, so a pipeline written again next month, over rows that
/// did not exist yet, can take it over. It is written and read through the same door as a pipeline file: with the
/// catalog of the verbs it may hold, every fault at its line and column, and a file from a newer version refused
/// whole.
/// </remarks>
public sealed record PipelinePreset
{
    /// <summary>A preset of these decisions.</summary>
    /// <param name="declare">The schema: every column it names, those it excludes too, with their kinds.</param>
    /// <param name="drop">The columns left out after the steps that read them, in the order the steps stand.</param>
    /// <param name="output">What a model is asked to predict, or nothing when no output was decided.</param>
    /// <param name="source">The source's columns as they were last shown, or nothing when they never were.</param>
    /// <exception cref="ArgumentException">A column dropped has no name, or is named twice.</exception>
    public PipelinePreset(
        DeclareStep declare, IEnumerable<string>? drop = null, INamesTheAnswer? output = null, IEnumerable<string>? source = null)
    {
        ArgumentNullException.ThrowIfNull(declare);

        Declare = declare;
        Drop = NamedOnce(drop ?? []);
        Output = output;
        Source = source is null ? null : [.. source];
    }

    /// <summary>The schema: every column it names, those it excludes too, with their kinds.</summary>
    public DeclareStep Declare { get; }

    /// <summary>The columns left out after the steps that read them, in the order the steps stand.</summary>
    public IReadOnlyList<string> Drop { get; }

    /// <summary>What a model is asked to predict, or nothing when no output was decided.</summary>
    public INamesTheAnswer? Output { get; }

    /// <summary>The source's columns as they were last shown, or nothing when they never were.</summary>
    public IReadOnlyList<string>? Source { get; }

    /// <summary>What a pipeline decided about its columns, as its steps say it.</summary>
    /// <param name="declaration">The pipeline.</param>
    /// <param name="header">The source's columns as they are shown, or nothing when they are not known.</param>
    /// <returns>The preset: its schema, every column a drop names, its output, and the header.</returns>
    /// <exception cref="ArgumentException">The pipeline has no schema, so it decided nothing about its columns.</exception>
    public static PipelinePreset Of(PipelineDeclaration declaration, IReadOnlyList<string>? header)
    {
        ArgumentNullException.ThrowIfNull(declaration);

        var declare = declaration.Steps.OfType<DeclareStep>().FirstOrDefault()
            ?? throw new ArgumentException("A pipeline without a schema has decided nothing about its columns.", nameof(declaration));

        return new(declare, declaration.Steps.OfType<DropColumnsStep>().SelectMany(drop => drop.Columns), declaration.Output, header);
    }

    /// <summary>Takes the decisions over into a pipeline, listing what that changes before anything is applied.</summary>
    /// <param name="into">The pipeline the decisions are taken into: a notebook's blocks, or a chain written in code.</param>
    /// <param name="header">The source's columns as they are now, or nothing when they are not known.</param>
    /// <returns>
    /// The steps the decisions make of the pipeline and every change they make: each column whose decision changes, the
    /// output, the schema's order, and the source's columns the decisions never showed. When those steps would break a
    /// rule, every fault instead, and no steps.
    /// </returns>
    /// <remarks>
    /// The schema is taken whole — in place of the one there, or directly after the source when there is none; the
    /// drops are made these; and the output is this one, placed whole, or none. A column is listed when how it stands,
    /// its kind or the kind it was changes: what it offers, or what it is to the output, follows from those, and the
    /// output is listed once. The same pipeline, decisions and header give the same answer, whoever asks.
    /// </remarks>
    public PresetTakeOver TakeOver(PipelineDeclaration into, IReadOnlyList<string>? header)
    {
        ArgumentNullException.ThrowIfNull(into);

        return Listed(into, Taken(into), header);
    }

    /// <summary>Takes the schema alone over into a pipeline, listing what it decides.</summary>
    /// <param name="into">The pipeline: a chain at its source, before the steps that read the drops are written.</param>
    /// <param name="header">The source's columns as they are now, or nothing when they are not known.</param>
    /// <returns>
    /// The steps with this schema in place of the one there, or directly after the source; every column whose decision
    /// that changes; and the source's columns the decisions — all of them, the drops and the output too — never showed.
    /// </returns>
    internal PresetTakeOver TakeOverOfTheSchema(PipelineDeclaration into, IReadOnlyList<string>? header) =>
        Listed(into, WithSchema(into), header);

    /// <summary>The columns of a source these decisions never showed.</summary>
    /// <param name="header">The source's columns as they are now.</param>
    /// <returns>
    /// Those the source's columns as last shown lack; when they were never shown, those the decisions do not name — in
    /// the schema, excluded ones too, in a drop, or as the output's answer. In the source's order.
    /// </returns>
    public IReadOnlyList<string> NewColumns(IReadOnlyList<string> header)
    {
        ArgumentNullException.ThrowIfNull(header);

        return [.. header.Except(Source ?? Named(), StringComparer.Ordinal)];
    }

    // What a take-over made of a pipeline, listed against how it stood: or every fault, and no steps.
    private PresetTakeOver Listed(PipelineDeclaration into, TakenSteps taken, IReadOnlyList<string>? header)
    {
        var newColumns = header is null ? null : NewColumns(header);

        if (taken.Result is not { } result)
        {
            return new PresetTakeOver([], taken.Faults, newColumns, [], null, null);
        }

        IReadOnlyList<string> asked = [.. (header ?? []).Concat(NamedBy(into)).Concat(NamedBy(result)).Distinct(StringComparer.Ordinal)];
        var before = into.ChoicesFor(asked).Rows;
        var after = result.ChoicesFor(asked).Rows;
        List<ColumnChange> changes = [];

        for (var at = 0; at < asked.Count; at++)
        {
            if (Decided(before[at]) != Decided(after[at]))
            {
                var made = before[at].Standing == ColumnStanding.Made || after[at].Standing == ColumnStanding.Made;

                changes.Add(new ColumnChange(asked[at], before[at], after[at], header is null || made ? null : header.Contains(asked[at], StringComparer.Ordinal)));
            }
        }

        return new PresetTakeOver(result.Steps, [], newColumns, changes, OutputChanged(into.Output, result.Output), OrderChanged(into, result));
    }

    /// <summary>Writes the preset as JSON.</summary>
    /// <returns>The version it is written against, then what it holds; a part it does not hold is not written.</returns>
    public string ToJson() => PipelineDocument.Write(this);

    /// <summary>Reads a preset back, using the verbs a given catalog knows.</summary>
    /// <param name="json">The file a preset was written as.</param>
    /// <param name="catalog">The verbs its schema and its output may use.</param>
    /// <returns>The preset the file describes.</returns>
    /// <exception cref="PipelineFileException">
    /// Anything in the file is wrong, every fault named at its line and column: text that is not JSON, a key written
    /// twice, no version or a newer one, a key a preset does not have, a schema or an output the catalog cannot read,
    /// or a list of columns that is not one.
    /// </exception>
    public static PipelinePreset FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);

        return PipelineDocument.ReadPreset(json, catalog);
    }

    /// <summary>Whether two presets say the same thing.</summary>
    /// <param name="other">The preset to compare with.</param>
    /// <returns><see langword="true"/> when both hold the same schema, drops, output and source columns.</returns>
    public bool Equals(PipelinePreset? other) =>
        other is not null
        && Declare.Equals(other.Declare)
        && Drop.SequenceEqual(other.Drop, StringComparer.Ordinal)
        && Equals(Output, other.Output)
        && (Source is null ? other.Source is null : other.Source is not null && Source.SequenceEqual(other.Source, StringComparer.Ordinal));

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Declare);
        hash.Add(Output);

        foreach (var column in Drop.Concat(Source ?? []))
        {
            hash.Add(column, StringComparer.Ordinal);
        }

        return hash.ToHashCode();
    }

    // The steps these decisions make of a pipeline, a stage at a time and each kept to the rules before the next: the
    // drops they do not make taken back in, under the schema they were made under; the schema this one, in place of the
    // one there or directly after the source; the drops they make; the output this one, or none.
    private TakenSteps Taken(PipelineDeclaration into)
    {
        var taken = new TakenSteps(into, []);

        foreach (var column in into.Steps.OfType<DropColumnsStep>().SelectMany(drop => drop.Columns).Except(Drop, StringComparer.Ordinal).ToArray())
        {
            taken = Kept(() => taken.Result!.Including(column, ColumnKind.Text, []));

            if (taken.Result is null)
            {
                return taken;
            }
        }

        taken = WithSchema(taken.Result!);

        foreach (var column in Drop)
        {
            if (taken.Result is null)
            {
                return taken;
            }

            taken = Kept(() => taken.Result.Excluding(column));
        }

        return taken.Result is { } decided
            ? Kept(() => Output is { } output ? decided.WithOutput(output) : decided.WithoutOutput())
            : taken;
    }

    // This schema, in place of the one there or directly after the source.
    private TakenSteps WithSchema(PipelineDeclaration into)
    {
        var steps = into.Steps;
        var at = into.ColumnsAt;

        return Kept(() => at >= 0 ? [.. steps.Take(at), Declare, .. steps.Skip(at + 1)] : [.. steps.Take(1), Declare, .. steps.Skip(1)]);
    }

    // The steps an operation makes as a declaration, or every rule they break; the schema's own refusal among them.
    private static TakenSteps Kept(Func<IReadOnlyList<IPipelineStep>> operation)
    {
        try
        {
            var steps = operation();
            var faults = PipelineDeclaration.FaultsIn(steps);

            return faults.Count == 0 ? new TakenSteps(new PipelineDeclaration(steps), []) : new TakenSteps(null, faults);
        }
        catch (DeclarationException refused)
        {
            return new TakenSteps(null, refused.Faults);
        }
    }

    // Every column these decisions name: the schema's, excluded ones too, the drops and the output's answers.
    private IEnumerable<string> Named() =>
        Declare.Columns.Select(column => column.Name).Concat(Drop).Concat(Output?.Answers ?? []);

    // Every column a pipeline names: its schema's, every one known after any of its steps, its drops, and every one the
    // ways back of its answers read.
    private static IEnumerable<string> NamedBy(PipelineDeclaration declaration) =>
        SchemaNames(declaration)
            .Concat(Enumerable.Range(1, declaration.Steps.Count).SelectMany(steps => declaration.ColumnsBefore(steps).Columns.Select(column => column.Name)))
            .Concat(declaration.Steps.OfType<DropColumnsStep>().SelectMany(drop => drop.Columns))
            .Concat(UndoChain.ReadBy(declaration));

    private static IEnumerable<string> SchemaNames(PipelineDeclaration declaration) =>
        declaration.Steps.OfType<DeclareStep>().SelectMany(declare => declare.Columns.Select(column => column.Name));

    // What decides a column: how it stands, its kind, and the kind a category was.
    private static ColumnDecision Decided(ColumnChoice choice) => new(choice.Standing, choice.Kind, choice.Was);

    // The output, when it changes: by what each writes, as placing one decides.
    private static OutputChange? OutputChanged(INamesTheAnswer? before, INamesTheAnswer? after)
    {
        if (before is null && after is null)
        {
            return null;
        }

        return before is not null && after is not null && before.Canonical().AsSpan().SequenceEqual(after.Canonical())
            ? null
            : new OutputChange(before, after);
    }

    // The schemas' order, when the names both declare stand in another; a name only one declares is a change of its own.
    private static OrderChange? OrderChanged(PipelineDeclaration into, PipelineDeclaration result)
    {
        string[] before = [.. SchemaNames(into)];
        string[] after = [.. SchemaNames(result)];
        var both = before.Intersect(after, StringComparer.Ordinal).ToHashSet(StringComparer.Ordinal);

        return before.Where(both.Contains).SequenceEqual(after.Where(both.Contains), StringComparer.Ordinal) ? null : new OrderChange(before, after);
    }

    /// <summary>The columns a preset drops, each with a name and each once.</summary>
    /// <param name="drop">The columns.</param>
    /// <returns>The columns.</returns>
    /// <exception cref="ArgumentException">A column has no name, or is named twice.</exception>
    internal static string[] NamedOnce(IEnumerable<string> drop)
    {
        var named = new HashSet<string>(StringComparer.Ordinal);

        foreach (var column in drop)
        {
            if (string.IsNullOrWhiteSpace(column))
            {
                throw new ArgumentException("A column dropped needs a name.", nameof(drop));
            }

            if (!named.Add(column))
            {
                throw new ArgumentException($"The drop names '{column}' twice; name each column once.", nameof(drop));
            }
        }

        return [.. drop];
    }

    /// <summary>The steps a take-over has made so far, or every rule they break.</summary>
    /// <param name="Result">The declaration, while the steps keep the rules.</param>
    /// <param name="Faults">Every rule broken, once one is.</param>
    private readonly record struct TakenSteps(PipelineDeclaration? Result, IReadOnlyList<DeclarationFault> Faults);

    /// <summary>What decides a column in a listing.</summary>
    /// <param name="Standing">How it stands.</param>
    /// <param name="Kind">Its kind.</param>
    /// <param name="Was">The kind a category was.</param>
    private readonly record struct ColumnDecision(ColumnStanding Standing, ColumnKind? Kind, ColumnKind? Was);
}
