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
}
