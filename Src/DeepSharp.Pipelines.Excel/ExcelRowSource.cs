// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using ExcelDataReader;
using ExcelDataReader.Exceptions;

namespace DeepSharp.Pipelines;

/// <summary>
/// Rows read from one sheet of an Excel workbook: its first row names the columns, and every row below it is a record.
/// </summary>
/// <remarks>
/// A cell is read as the sheet types it, never as the text a culture shows it as: a number as the number it holds, true or
/// false, a date as the moment it is, words as they are — by the one rule every typed source hands a value over in
/// (<see cref="TypedValueExtensions.AsCell"/>). A grid has every field, so an empty cell is empty text, as the same
/// sheet saved as comma-separated text holds it, and a formula's error is spelled as Excel spells it there. The whole
/// sheet is read when it is opened, so its rows are the ones the file held at that moment however often they are asked
/// for.
/// </remarks>
internal sealed class ExcelRowSource : IRowSource
{
    // Excel's own spelling of each error a formula can leave in a cell, as a sheet saved as comma-separated text writes it.
    private static readonly Dictionary<CellError, string> Errors = new()
    {
        [CellError.NULL] = "#NULL!",
        [CellError.DIV0] = "#DIV/0!",
        [CellError.VALUE] = "#VALUE!",
        [CellError.REF] = "#REF!",
        [CellError.NAME] = "#NAME?",
        [CellError.NUM] = "#NUM!",
        [CellError.NA] = "#N/A",
        [CellError.GETTING_DATA] = "#GETTING_DATA",
    };

    private readonly List<IReadOnlyList<string?>> _rows;

    /// <summary>Reads a sheet of a workbook's bytes.</summary>
    /// <param name="bytes">Every byte of the workbook: .xlsx, .xls or .xlsb.</param>
    /// <param name="file">What a refusal names the workbook as.</param>
    /// <param name="sheet">The sheet; nothing for the first.</param>
    /// <exception cref="FormatException">
    /// The bytes are not a workbook, it has no sheet of that name, or the sheet has no row to name its columns.
    /// </exception>
    public ExcelRowSource(byte[] bytes, string file, string? sheet)
    {
        var read = Read(bytes, file, sheet);

        ColumnNames = read.ColumnNames;
        _rows = read.Rows;
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => _rows;

    private static SheetContent Read(byte[] bytes, string file, string? sheet)
    {
        // The reader asks .NET for the Windows code page 1252 before it reads any workbook — one that keeps no text in a code
        // page too — and .NET carries the code pages without offering them. Registering them offers the ones .NET leaves
        // out, once for the whole program, and changes no encoding the program already had: UTF-8, UTF-16, ASCII and
        // Latin-1 stay the very ones they were.
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

        using var stream = new MemoryStream(bytes, writable: false);

        try
        {
            using var reader = ExcelReaderFactory.CreateReader(stream);
            var sheets = new List<string>();

            do
            {
                if (sheet is null || reader.Name == sheet)
                {
                    return Sheet(file, reader);
                }

                sheets.Add(reader.Name);
            }
            while (reader.NextResult());

            throw new FormatException($"{file} has no sheet called '{sheet}'. Its sheets are: {string.Join(", ", sheets)}.");
        }
        catch (Exception unreadable) when (unreadable is ExcelReaderException or InvalidDataException)
        {
            throw new FormatException($"{file} cannot be read as an Excel workbook: {unreadable.Message}", unreadable);
        }
    }

    private static SheetContent Sheet(string file, IExcelDataReader reader)
    {
        if (!reader.Read())
        {
            throw new FormatException($"The sheet '{reader.Name}' of {file} has no header row, so its columns have no names.");
        }

        IReadOnlyList<string> names = [.. Cells(reader)];
        var rows = new List<IReadOnlyList<string?>>();

        while (reader.Read())
        {
            rows.Add([.. Cells(reader)]);
        }

        return new SheetContent(names, rows);
    }

    private static IEnumerable<string> Cells(IExcelDataReader reader) =>
        Enumerable.Range(0, reader.FieldCount).Select(at =>
            reader.GetValue(at) is { } value ? value.AsCell()!
            : reader.GetCellError(at) is { } error ? Errors[error]
            : string.Empty);

    /// <summary>What a sheet held.</summary>
    /// <param name="ColumnNames">The names its first row gives the columns.</param>
    /// <param name="Rows">Every row below the first.</param>
    private readonly record struct SheetContent(IReadOnlyList<string> ColumnNames, List<IReadOnlyList<string?>> Rows);
}
