// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Read the rows from a sheet of an Excel workbook.
/// </summary>
/// <remarks>
/// A file the pipeline reads, as a comma-separated one is: named in the declaration, opened only when the pipeline runs,
/// read from the pipeline's folder when its path is relative, and read again whenever the pipeline is. The first sheet
/// is read unless the step names another, and the sheet's first row names its columns.
/// </remarks>
public sealed record ReadExcelStep : IPipelineStep<ReadExcelStep>, IOpensRows, IReadsAFile, IDescribesColumns
{
    private static readonly FilePathParameter PathKey = new(
        "path", "Where the workbook will be, when the pipeline runs: .xlsx, .xls or .xlsb.", "data.xlsx");

    private static readonly TextParameter SheetKey = new(
        "sheet", "The sheet the rows are on. Left out, the first sheet.", "Sheet1", optional: true);

    /// <summary>Declares that the rows come from the first sheet of the workbook at this path.</summary>
    /// <param name="path">Where the workbook will be, when the pipeline runs.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadExcelStep(string path)
        : this(path, string.Empty)
    {
    }

    /// <summary>Declares that the rows come from a named sheet of the workbook at this path.</summary>
    /// <param name="path">Where the workbook will be, when the pipeline runs.</param>
    /// <param name="sheet">The sheet the rows are on; empty, or nothing but spaces, for the first.</param>
    /// <exception cref="ArgumentException">The path is empty or nothing but spaces.</exception>
    public ReadExcelStep(string path, string sheet)
    {
        Path = PathKey.Require(path);

        var named = SheetKey.Require(sheet);

        Sheet = named.Length == 0 ? null : named;
    }

    /// <summary>Where the workbook will be, when the pipeline runs.</summary>
    public string Path { get; }

    /// <summary>The sheet the rows are on; nothing for the first.</summary>
    public string? Sheet { get; }

    /// <inheritdoc />
    public static string Name => "read.excel";

    /// <inheritdoc />
    public static string Purpose => "Reads the rows from a sheet of an Excel workbook, the first unless one is named, its first row naming the columns.";

    /// <inheritdoc />
    public static StepParameters<ReadExcelStep> Parameters { get; } = new StepParameters<ReadExcelStep>()
        .With(PathKey, step => step.Path)
        .With(SheetKey, step => step.Sheet ?? string.Empty);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>
    /// A relative path is read from the pipeline's folder; the path stays as it was written. The file's bytes are read once
    /// and parsed as <see cref="Open(byte[], string)"/> parses them.
    /// </remarks>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">
    /// The file is not a workbook, it has no sheet of the name the step gives, or the sheet has no row to name its columns.
    /// </exception>
    public IRowSource Open(SourceFolder folder)
    {
        ArgumentNullException.ThrowIfNull(folder);

        var file = folder.Resolve(Path);

        // Read whole and opened for sharing, as a file's bytes are read, so another pipeline reading the same workbook at
        // that moment is not refused by a lock.
        return Open(File.ReadAllBytes(file), file);
    }

    /// <inheritdoc />
    /// <remarks>The sheet this step names, or the first.</remarks>
    /// <exception cref="FormatException">
    /// The bytes are not a workbook, it has no sheet of the name the step gives, or the sheet has no row to name its columns.
    /// </exception>
    public IRowSource Open(byte[] bytes, string file)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(file);

        return new ExcelRowSource(bytes, file, Sheet);
    }

    /// <inheritdoc />
    /// <remarks>The file's columns as the rows open them.</remarks>
    public IReadOnlyList<string> ColumnNamesIn(SourceFolder folder) => Open(folder).ColumnNames;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">The path is missing, or the path or the sheet is not text.</exception>
    public static ReadExcelStep ReadFrom(JsonElement element) => new(PathKey.Read(element), SheetKey.Read(element));
}
