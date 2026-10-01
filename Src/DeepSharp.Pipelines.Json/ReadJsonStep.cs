// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from a JSON file holding an array of records.
/// </summary>
/// <remarks>
/// A file the pipeline reads, as a comma-separated one is: named in the declaration, opened only when the pipeline runs,
/// read from the pipeline's folder when its path is relative, and read again whenever the pipeline is.
/// </remarks>
public sealed record ReadJsonStep : IPipelineStep<ReadJsonStep>, IOpensRows, IReadsAFile, IDescribesColumns
{
    private static readonly FilePathParameter PathKey = new(
        "path", "Where the JSON file will be, when the pipeline runs: an array of records, one object a row.", "data.json");

    /// <summary>Declares that the rows come from the JSON file at this path.</summary>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadJsonStep(string path) => Path = PathKey.Require(path);

    /// <summary>Where the file will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.json";

    /// <inheritdoc />
    public static string Purpose => "Reads the rows from a JSON file holding an array of records, one object a row, every value as the file writes it.";

    /// <inheritdoc />
    public static StepParameters<ReadJsonStep> Parameters { get; } =
        new StepParameters<ReadJsonStep>().With(PathKey, step => step.Path);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>
    /// A relative path is read from the pipeline's folder; the path stays as it was written. The file's bytes are read once
    /// and parsed as <see cref="Open(byte[], string)"/> parses them.
    /// </remarks>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">
    /// The file is not JSON, is not an array of records, or holds a value that is not one value to a cell.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        var file = folder.Resolve(Path);

        return Open(File.ReadAllBytes(file), file);
    }

    /// <inheritdoc />
    /// <remarks>Read as text as the file is, in the encoding its byte-order mark names, UTF-8 when it names none.</remarks>
    /// <exception cref="FormatException">
    /// The bytes are not JSON, not an array of records, or hold a value that is not one value to a cell.
    /// </exception>
    public IRowSource Open(byte[] bytes, string file)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(file);

        return new JsonRowSource(bytes, file);
    }

    /// <inheritdoc />
    /// <remarks>The file's columns as the rows open them.</remarks>
    public IReadOnlyList<string> ColumnNamesIn(SourceFolder folder) => Open(folder).ColumnNames;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">The path is missing or is not text.</exception>
    public static ReadJsonStep ReadFrom(JsonElement element) => new(PathKey.Read(element));
}
