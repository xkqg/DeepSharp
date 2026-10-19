// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Reads the rows from a folder of text files, a folder to a class: a file a row, the folder it stands in the answer.
/// </summary>
/// <remarks>
/// The columns are <c>text</c> — the file as it is written, in UTF-8 — and <c>label</c>, the name of the folder, and then a
/// column of nought and one for each folder, named for it, which a learner is handed the answer by. A row is known by what the file
/// says and by its class, never by the path it lies at. Nothing is opened until the pipeline runs.
/// </remarks>
public sealed record ReadTextsStep : IPipelineStep<ReadTextsStep>, IOpensRows, IDescribesColumns
{
    /// <summary>The name of the column that holds each file's text.</summary>
    public const string TextColumn = "text";

    private static readonly FilePathParameter PathKey = new(
        "path", "The folder the text files will be in, when the pipeline runs: a folder to each class, files of .txt in each.", "texts");

    /// <summary>Declares that the rows come from the text files of this folder.</summary>
    /// <param name="path">Where the folder will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadTextsStep(string path) => Path = PathKey.Require(path);

    /// <summary>Where the folder will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <inheritdoc />
    public static string Name => "read.texts";

    /// <inheritdoc />
    public static string Purpose =>
        "Reads the rows from a folder of text files, a folder to a class: a file a row, the folder it stands in the answer, and a column of nought and one for each class.";

    /// <inheritdoc />
    public static StepParameters<ReadTextsStep> Parameters { get; } =
        new StepParameters<ReadTextsStep>().With(PathKey, step => step.Path);

    /// <inheritdoc />
    public static int Since => 12;

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <exception cref="DirectoryNotFoundException">There is no such folder.</exception>
    /// <exception cref="FormatException">A file stands beside the folders, none is a text file, or a class is named as a column of the rows is.</exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        return new FilesAsRows(folder.Resolve(Path), TextColumn, [".txt"], file => File.ReadAllText(file, System.Text.Encoding.UTF8));
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">The path is missing or is not text.</exception>
    public static ReadTextsStep ReadFrom(JsonElement element) => new(PathKey.Read(element));
}

/// <summary>Reading the rows from a folder of text files.</summary>
public static class TextsSourceExtensions
{
    extension(PipelineBuilder pipeline)
    {
        /// <summary>Declares that the rows come from a folder of text files, a folder to each class.</summary>
        /// <param name="path">Where the folder will be, when the pipeline runs; a relative path is read from the pipeline's folder.</param>
        /// <returns>The pipeline, so the next verb can be written after it.</returns>
        /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
        public PipelineBuilder ReadTexts(string path)
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            return pipeline.Add(new ReadTextsStep(path));
        }
    }
}
