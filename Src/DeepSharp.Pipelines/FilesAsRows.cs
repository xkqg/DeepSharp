// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>
/// A folder of files read as rows: a file a row, the folder it stands in the answer.
/// </summary>
/// <remarks>
/// <para>
/// The way images, sounds and texts are kept for learning from: one folder to a class, one file to an example. The rows are the
/// files in the order of their paths, so the same folder gives the same rows on any machine; each row is the file's cell, as the
/// reader that opened it writes one, and the name of the folder it stood in; and beside them a column of nought and one for each
/// class, named for the folder, which is how the answer is handed to a learner (<c>Labels</c>).
/// </para>
/// <para>
/// A row is known by what its file holds and not by where the file lies: the key of a row is made from its cells, so two files
/// with the same content in the same folder are the same record wherever they are kept. Every file must stand in a folder of a
/// class — a file beside the folders has no class to be — and a class is not named as a column of the rows is.
/// </para>
/// </remarks>
public sealed class FilesAsRows : IRowSource
{
    /// <summary>The name of the column that holds the name of each file's folder.</summary>
    public const string LabelColumn = "label";

    private readonly List<IReadOnlyList<string?>> _rows = [];

    /// <summary>The files of a folder, read as rows.</summary>
    /// <param name="folder">The folder: a folder to each class, files in each.</param>
    /// <param name="column">The name of the column that holds what each file holds.</param>
    /// <param name="extensions">The extensions of the files that are rows, with the dot; any other file is not one.</param>
    /// <param name="cellOf">What a file is as a cell: the text of a text, the media cell of an image.</param>
    /// <exception cref="DirectoryNotFoundException">There is no such folder.</exception>
    /// <exception cref="FormatException">
    /// A file stands beside the folders instead of in one, no file has one of the extensions, a class is named as the rows' own
    /// columns are, or a file cannot be read as a cell — each named with the file.
    /// </exception>
    public FilesAsRows(string folder, string column, IReadOnlyList<string> extensions, Func<string, string> cellOf)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        ArgumentException.ThrowIfNullOrWhiteSpace(column);
        ArgumentNullException.ThrowIfNull(extensions);
        ArgumentNullException.ThrowIfNull(cellOf);

        if (!Directory.Exists(folder))
        {
            throw new DirectoryNotFoundException($"There is no folder '{folder}' to read the files of.");
        }

        var root = Path.GetFullPath(folder);
        string[] files =
        [
            .. Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(file => extensions.Contains(Path.GetExtension(file), StringComparer.OrdinalIgnoreCase))
                .OrderBy(file => Path.GetRelativePath(root, file), StringComparer.Ordinal),
        ];

        if (files.Length == 0)
        {
            throw new FormatException($"'{folder}' holds no file of {string.Join(" or ", extensions.Select(extension => $"'{extension}'"))}: there is no row to read.");
        }

        string[] classes = [.. files.Select(file => ClassOf(root, file)).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)];

        foreach (var name in classes.Where(name => name == column || name == LabelColumn))
        {
            throw new FormatException($"The folder '{name}' is named as a column of the rows is ('{column}', '{LabelColumn}'): rename the folder.");
        }

        ColumnNames = [column, LabelColumn, .. classes];

        foreach (var file in files)
        {
            var label = ClassOf(root, file);
            string cell;

            try
            {
                cell = cellOf(file);
            }
            catch (Exception fault) when (fault is FormatException or ArgumentException or InvalidDataException or NotSupportedException)
            {
                throw new FormatException($"'{Path.GetRelativePath(root, file)}' cannot be read as a row: {fault.Message}", fault);
            }

            _rows.Add([cell, label, .. classes.Select(name => name == label ? "1" : "0")]);
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => _rows;

    /// <summary>The name of the class a file stands in: the folder it stands directly in.</summary>
    private static string ClassOf(string root, string file)
    {
        var folder = Path.GetDirectoryName(file)!;

        return string.Equals(folder, root.TrimEnd(Path.DirectorySeparatorChar), StringComparison.Ordinal)
            ? throw new FormatException(string.Create(CultureInfo.InvariantCulture, $"'{Path.GetRelativePath(root, file)}' stands beside the folders and not in one: a file's class is the name of the folder it stands in."))
            : Path.GetFileName(folder);
    }
}
