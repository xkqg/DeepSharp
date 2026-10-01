// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from an Apache Parquet file.
/// </summary>
/// <remarks>
/// A file the pipeline reads, as a comma-separated one is: named in the declaration, opened only when the pipeline runs,
/// read from the pipeline's folder when its path is relative, and read again whenever the pipeline is. A Parquet file
/// types its columns, so the rows it opens say what each column holds, and the proposal of kinds takes what they say.
/// </remarks>
public sealed record ReadParquetStep : IPipelineStep<ReadParquetStep>, IOpensRows, IReadsAFile, IDescribesColumns
{
    private static readonly FilePathParameter PathKey = new(
        "path", "Where the Parquet file will be, when the pipeline runs.", "data.parquet");

    /// <summary>Declares that the rows come from the Parquet file at this path.</summary>
    /// <param name="path">Where the file will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadParquetStep(string path) => Path = PathKey.Require(path);

    /// <summary>Where the file will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.parquet";

    /// <inheritdoc />
    public static string Purpose => "Reads the rows from an Apache Parquet file, which says what each of its columns holds.";

    /// <inheritdoc />
    public static StepParameters<ReadParquetStep> Parameters { get; } =
        new StepParameters<ReadParquetStep>().With(PathKey, step => step.Path);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>
    /// A relative path is read from the pipeline's folder; the path stays as it was written. The file's bytes are read once
    /// and parsed as <see cref="Open(byte[], string)"/> parses them.
    /// </remarks>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">
    /// The file is not a Parquet file, or a column holds something other than one value to a row: a list, a group of
    /// fields, raw bytes.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        var file = folder.Resolve(Path);

        return Open(File.ReadAllBytes(file), file);
    }

    /// <inheritdoc />
    /// <remarks>Every row group, in the order the file holds them, with what the file states each column holds.</remarks>
    /// <exception cref="FormatException">
    /// The bytes are not a Parquet file, or a column holds something other than one value to a row: a list, a group of
    /// fields, raw bytes.
    /// </exception>
    public IRowSource Open(byte[] bytes, string file)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(file);

        return new ParquetRowSource(bytes, file);
    }

    /// <inheritdoc />
    /// <remarks>The file's schema alone: its columns are named without reading a row.</remarks>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">The file is not a Parquet file, or a column holds no one value to a row.</exception>
    public IReadOnlyList<string> ColumnNamesIn(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return ParquetRowSource.ColumnNamesOf(folder.Resolve(Path));
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">The path is missing or is not text.</exception>
    public static ReadParquetStep ReadFrom(JsonElement element) => new(PathKey.Read(element));
}
