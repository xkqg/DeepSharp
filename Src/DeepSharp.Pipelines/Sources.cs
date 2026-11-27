// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from a comma-separated file.
/// </summary>
/// <remarks>
/// Declaring where the data comes from is not the same act as going to get it: nothing is opened until the
/// pipeline runs, so a declaration can be written, saved and checked on a machine that has no data on it.
/// </remarks>
public sealed record ReadCsvStep : IPipelineStep<ReadCsvStep>, IOpensRows, IReadsAFile, IDescribesColumns
{
    private static readonly FilePathParameter PathKey = new(
        "path", "Where the comma-separated file will be, when the pipeline runs.", "data.csv");

    /// <summary>Declares that the rows come from the file at this path.</summary>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadCsvStep(string path) => Path = PathKey.Require(path);

    /// <summary>Where the file will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.csv";

    /// <inheritdoc />
    public static string Purpose => "Reads the rows from a comma-separated file.";

    /// <inheritdoc />
    public static StepParameters<ReadCsvStep> Parameters { get; } =
        new StepParameters<ReadCsvStep>().With(PathKey, step => step.Path);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>
    /// A relative path is read from the pipeline's folder; the path stays as it was written. The file's bytes are read once
    /// and parsed as <see cref="Open(byte[], string)"/> parses them.
    /// </remarks>
    public IRowSource Open(SourceFolder folder) => this.OpenIn(folder);

    /// <inheritdoc />
    /// <remarks>Read as text as the file is, in the encoding its byte-order mark names, UTF-8 when it names none.</remarks>
    /// <exception cref="FormatException">The text has no header, or a row has the wrong number of cells.</exception>
    public IRowSource Open(byte[] bytes, string file)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(file);

        return CsvRowSource.FromText(bytes.AsText(), file);
    }

    /// <inheritdoc />
    /// <remarks>The file's first line alone, read as the whole file is, so a large file costs that line.</remarks>
    public IReadOnlyList<string> ColumnNamesIn(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return CsvRowSource.HeaderOf(folder.Resolve(Path));
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is not text.</exception>
    public static ReadCsvStep ReadFrom(JsonElement element) => new(PathKey.Read(element));
}

/// <summary>
/// The rows are handed in rather than opened.
/// </summary>
/// <remarks>
/// A source that lives outside the file: a table already in memory, a reader over a query, anything a
/// package turns into rows. The declaration says so and says what it was, and the rows themselves are
/// handed to the pipeline when it runs — which is also exactly how serving works, so the same declaration
/// covers both without a second shape.
/// </remarks>
public sealed record ReadRowsStep : IPipelineStep<ReadRowsStep>, IOpensRows, IDescribesColumns
{
    private static readonly TextParameter DescriptionKey = new(
        "description", "What the rows are, for whoever reads the file later.", "rows handed in");

    /// <summary>Declares that the rows are handed in.</summary>
    /// <param name="description">What the rows are, for whoever reads the file later.</param>
    /// <exception cref="ArgumentException">The description is empty.</exception>
    public ReadRowsStep(string description) => Description = DescriptionKey.Require(description);

    /// <summary>What the rows are, for whoever reads the file later.</summary>
    public string Description { get; }

    /// <inheritdoc />
    public static string Name => "read.rows";

    /// <inheritdoc />
    public static string Purpose => "Takes rows that are handed in rather than opened: a table already in memory, a reader over a query.";

    /// <inheritdoc />
    public static StepParameters<ReadRowsStep> Parameters { get; } =
        new StepParameters<ReadRowsStep>().With(DescriptionKey, step => step.Description);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public IRowSource Open(SourceFolder folder) =>
        throw new InvalidOperationException(
            $"This pipeline reads rows that are handed in ({Description}). "
            + "Hand them to Run or Prepare, or to Replay when the pipeline is already fitted.");

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static ReadRowsStep ReadFrom(JsonElement element) => new(DescriptionKey.Read(element));
}
