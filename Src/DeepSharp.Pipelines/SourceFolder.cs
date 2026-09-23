// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Where a relative path in a pipeline points: the one rule, for every source and every door.
/// </summary>
/// <remarks>
/// A path is kept as it was written, and resolved when a source is opened. An absolute path is used as it
/// stands. A relative one is read from the folder the pipeline sits in — the folder of the file it was read
/// from, or of the notebook it was written in — and from the working directory for a pipeline written in code.
/// So a pipeline file moved together with its data still finds it, whichever folder the program reading it
/// happens to have been started in.
/// </remarks>
public sealed class SourceFolder
{
    private readonly string? _folder;

    private SourceFolder(string? folder) => _folder = folder;

    /// <summary>Relative paths read from the working directory, as it is when a source is opened: a pipeline written in code.</summary>
    public static SourceFolder WorkingDirectory { get; } = new(null);

    /// <summary>Relative paths read from this folder.</summary>
    /// <param name="folder">The folder the pipeline sits in.</param>
    /// <returns>The folder.</returns>
    /// <exception cref="ArgumentException">The folder is not named.</exception>
    public static SourceFolder Of(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);

        return new SourceFolder(Path.GetFullPath(folder));
    }

    /// <summary>Relative paths read from the folder a document sits in: a pipeline file, a notebook.</summary>
    /// <param name="document">The document's path.</param>
    /// <returns>The document's folder.</returns>
    /// <exception cref="ArgumentException">The document is not named.</exception>
    public static SourceFolder OfDocument(string document)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(document);

        return Of(Path.GetDirectoryName(Path.GetFullPath(document))!);
    }

    /// <summary>The path a source opens.</summary>
    /// <param name="path">The path as it was written.</param>
    /// <returns>The path as written when it is absolute; otherwise the path read from this folder.</returns>
    /// <exception cref="ArgumentException">The path is not named.</exception>
    public string Resolve(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        return Path.IsPathFullyQualified(path)
            ? path
            : Path.GetFullPath(Path.Join(_folder ?? Environment.CurrentDirectory, path));
    }
}
