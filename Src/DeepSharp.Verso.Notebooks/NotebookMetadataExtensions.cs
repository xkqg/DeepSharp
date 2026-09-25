// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>Where a notebook reads a relative path from, by the rule a pipeline file is read by.</summary>
internal static class NotebookMetadataExtensions
{
    /// <summary>The folder a relative path in the notebook's blocks is read from.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>The folder the notebook is saved in; the working directory for a notebook never saved.</returns>
    public static SourceFolder SourceFolder(this INotebookMetadata notebook) =>
        notebook.FolderPath() is null
            ? Pipelines.SourceFolder.WorkingDirectory
            : Pipelines.SourceFolder.OfDocument(notebook.FilePath!);

    /// <summary>The folder the notebook is saved in, as a path.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>The folder, or nothing for a notebook never saved.</returns>
    public static string? FolderPath(this INotebookMetadata notebook) =>
        string.IsNullOrWhiteSpace(notebook.FilePath) ? null : Path.GetDirectoryName(Path.GetFullPath(notebook.FilePath));

    /// <summary>Where what the notebook decided about its columns is saved: beside it, named after it.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>The file's path, or nothing for a notebook never saved, which has nowhere beside it.</returns>
    public static string? ColumnsFilePath(this INotebookMetadata notebook) =>
        notebook.FolderPath() is { } folder ? Path.Join(folder, $"{Path.GetFileNameWithoutExtension(notebook.FilePath)}.columns.json") : null;

    /// <summary>The name the notebook's pipeline is exported under: named after the notebook.</summary>
    /// <param name="notebook">The notebook.</param>
    /// <returns>The file name; a name of its own for a notebook never saved.</returns>
    public static string PipelineFileName(this INotebookMetadata notebook) =>
        string.IsNullOrWhiteSpace(notebook.FilePath) ? "pipeline.json" : $"{Path.GetFileNameWithoutExtension(notebook.FilePath)}.pipeline.json";
}
