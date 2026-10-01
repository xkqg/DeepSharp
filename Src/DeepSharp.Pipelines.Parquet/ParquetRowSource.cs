// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Parquet;
using Parquet.Schema;

namespace DeepSharp.Pipelines;

/// <summary>
/// Rows read from an Apache Parquet file, with what the file says each column holds.
/// </summary>
/// <remarks>
/// The whole file is read when it is opened, every row group in the order the file holds them, so the rows are the ones
/// its bytes held at that moment however often they are asked for. Each value is handed over by the one rule every typed
/// source hands a value over in (<see cref="TypedValueExtensions.AsCell"/>), and each column's type is what the file
/// states it holds. A column of lists, of groups of fields, of maps or of raw bytes holds no one value to a row, and it is
/// refused by name rather than flattened by a guess about which of its values a cell was meant to be.
/// </remarks>
internal sealed class ParquetRowSource : IStatesKinds
{
    // How each type of value a Parquet file holds is read, as the cells of one row group's column.
    private static readonly Dictionary<Type, Func<ParquetRowGroupReader, DataField, int, Task<string?[]>>> Readers = new()
    {
        [typeof(ReadOnlyMemory<char>)] = Words,
        [typeof(bool)] = Values<bool>,
        [typeof(sbyte)] = Values<sbyte>,
        [typeof(byte)] = Values<byte>,
        [typeof(short)] = Values<short>,
        [typeof(ushort)] = Values<ushort>,
        [typeof(int)] = Values<int>,
        [typeof(uint)] = Values<uint>,
        [typeof(long)] = Values<long>,
        [typeof(ulong)] = Values<ulong>,
        [typeof(float)] = Values<float>,
        [typeof(double)] = Values<double>,
        [typeof(decimal)] = Values<decimal>,
        [typeof(DateTime)] = Values<DateTime>,
        [typeof(DateOnly)] = Values<DateOnly>,
        [typeof(Guid)] = Values<Guid>,
    };

    // What a column that holds no one value to a row holds instead, in words.
    private static readonly Dictionary<SchemaType, string> Nested = new()
    {
        [SchemaType.Struct] = "a group of fields",
        [SchemaType.Map] = "a map of keys to values",
    };

    // What a column of one value to a row, of a type no cell holds, holds, in words.
    private static readonly Dictionary<Type, string> Unread = new()
    {
        [typeof(ReadOnlyMemory<byte>)] = "raw bytes",
    };

    private readonly List<IReadOnlyList<string?>> _rows;

    /// <summary>Reads a Parquet file's bytes.</summary>
    /// <param name="bytes">Every byte of the file.</param>
    /// <param name="file">What a refusal names the file as.</param>
    /// <exception cref="FormatException">The bytes are not a Parquet file, or a column holds no one value to a row.</exception>
    public ParquetRowSource(byte[] bytes, string file)
    {
        // The reader is asynchronous and a pipeline opens its source where it stands, so the file is read on the thread
        // pool and waited for there: nothing the caller's context holds is waited on, so nothing can wait on the wait.
        var read = Task.Run(() => Read(bytes, file)).GetAwaiter().GetResult();

        ColumnNames = read.ColumnNames;
        StatedKinds = read.StatedKinds;
        _rows = read.Rows;
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => _rows;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, ColumnKind> StatedKinds { get; }

    /// <summary>The names of a Parquet file's columns, read from its schema alone: not one of its rows is read.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The names, in the file's order.</returns>
    /// <exception cref="FileNotFoundException">There is no file there.</exception>
    /// <exception cref="FormatException">The file is not a Parquet file, or a column holds no one value to a row.</exception>
    /// <remarks>
    /// The schema sits in the file's footer, which is all the reader reads to open it, so a large file costs its footer.
    /// A column the rows could not be read as is refused here as it is when the rows are read.
    /// </remarks>
    internal static IReadOnlyList<string> ColumnNamesOf(string path) =>
        Task.Run(() => NamesOf(path)).GetAwaiter().GetResult();

    private static async Task<IReadOnlyList<string>> NamesOf(string path)
    {
        await using var file = File.OpenRead(path);
        var reader = await Opened(file, path).ConfigureAwait(false);

        await using (reader.ConfigureAwait(false))
        {
            return [.. Columns(path, reader.Schema).Select(field => field.Name)];
        }
    }

    // A reader over the file's footer; a file that has none is refused with the reader's reason.
    private static async Task<ParquetReader> Opened(Stream stream, string file)
    {
        try
        {
            // A date is a day, so it is read as one rather than as midnight of it.
            return await ParquetReader.CreateAsync(stream, new ParquetOptions { UseDateOnlyTypeForDates = true }).ConfigureAwait(false);
        }
        catch (IOException unreadable)
        {
            throw new FormatException($"{file} cannot be read as a Parquet file: {unreadable.Message}", unreadable);
        }
    }

    private static async Task<ParquetContent> Read(byte[] bytes, string file)
    {
        await using var stream = new MemoryStream(bytes, writable: false);
        var reader = await Opened(stream, file).ConfigureAwait(false);

        await using (reader.ConfigureAwait(false))
        {
            var fields = Columns(file, reader.Schema);
            var rows = new List<IReadOnlyList<string?>>();

            for (var at = 0; at < reader.RowGroupCount; at++)
            {
                using var group = reader.OpenRowGroupReader(at);
                var count = checked((int)group.RowCount);
                var cells = new string?[fields.Length][];

                for (var column = 0; column < fields.Length; column++)
                {
                    cells[column] = await Cells(group, fields[column], count, file).ConfigureAwait(false);
                }

                for (var row = 0; row < count; row++)
                {
                    rows.Add([.. cells.Select(column => column[row])]);
                }
            }

            return new ParquetContent(
                [.. fields.Select(field => field.Name)],
                fields
                    .Where(field => Stated(field) is not null)
                    .ToDictionary(field => field.Name, field => Stated(field)!.Value, StringComparer.Ordinal),
                rows);
        }
    }

    // One column of a row group, as cells. A file whose footer is whole can still hold rows its library cannot decode, and it
    // says so in an exception of its own, which no door of the pipeline knows: such a file is refused as every file that
    // cannot be read as Parquet is.
    private static async Task<string?[]> Cells(ParquetRowGroupReader group, DataField field, int count, string file)
    {
        try
        {
            return await Readers[field.ClrType](group, field, count).ConfigureAwait(false);
        }
        catch (Exception unreadable) when (unreadable is InvalidDataException or IOException)
        {
            throw new FormatException($"{file} cannot be read as a Parquet file: {unreadable.Message}", unreadable);
        }
    }

    // The file's columns, each one value to a row of a type it can be read as; a column that is not is refused by name.
    private static DataField[] Columns(string file, ParquetSchema schema)
    {
        var columns = new List<DataField>();

        foreach (var field in schema.Fields)
        {
            if (field is not DataField { MaxRepetitionLevel: 0 } column)
            {
                throw new FormatException(
                    $"{file}: '{field.Name}' holds {Nested.GetValueOrDefault(field.SchemaType, "a list")} rather than one value to a row, "
                    + "and a cell holds one value. Write it as columns of one value each, or leave it out of the file.");
            }

            if (!Readers.ContainsKey(column.ClrType))
            {
                throw new FormatException(
                    $"{file}: '{field.Name}' holds {Unread.GetValueOrDefault(column.ClrType, $"values of the type {column.ClrType.Name}")}, "
                    + "which are not a cell a pipeline reads. Write them as text, or leave the column out of the file.");
            }

            columns.Add(column);
        }

        return [.. columns];
    }

    // The kind a column states: words for a column of text, which the file holds as characters, and otherwise what the
    // type of its values says.
    private static ColumnKind? Stated(DataField field) =>
        (field.ClrType == typeof(ReadOnlyMemory<char>) ? typeof(string) : field.ClrType).AsColumnKind();

    // Every column is read into memory of this reader's own. The library's convenient reads put a column's values and the
    // marks of its empty cells in buffers rented from the pool the whole process shares, and leave the marks of a column
    // every row holds unwritten: read beside Parquet files written on other threads, such reads were measured to find cells
    // marked empty that the file holds, with every value below them moved up a row (3 runs in 6 of the test that reads and
    // writes at once). The raw read into arrays nothing else can reach, paired here, never did (20 runs in 20).
    private static async Task<string?[]> Words(ParquetRowGroupReader group, DataField field, int count)
    {
        var column = await Raw<ReadOnlyMemory<char>>(group, field, count).ConfigureAwait(false);

        return column.Cells(value => new string(value.Span));
    }

    private static async Task<string?[]> Values<T>(ParquetRowGroupReader group, DataField field, int count)
        where T : struct
    {
        var column = await Raw<T>(group, field, count).ConfigureAwait(false);

        return column.Cells(value => ((object)value).AsCell());
    }

    private static async Task<RawColumn<T>> Raw<T>(ParquetRowGroupReader group, DataField field, int count)
        where T : struct
    {
        var values = new T[count];
        var marks = field.MaxDefinitionLevel > 0 ? new int[count] : null;

        // No marks at all for a column every row holds, said as no memory: a null array would turn into memory of no length,
        // marks for no row, which the library refuses.
        await group.ReadRawAsync<T>(field, values, marks is null ? (Memory<int>?)null : marks, null).ConfigureAwait(false);

        return new RawColumn<T>(values, marks, field.MaxDefinitionLevel);
    }

    /// <summary>A column as the file stores it: the values its rows hold, one after another, and a mark per row.</summary>
    /// <param name="Values">The values, one for each row that holds one, in the rows' order.</param>
    /// <param name="Marks">A row's mark, below <paramref name="Held"/> when the row holds nothing; none when every row holds a value.</param>
    /// <param name="Held">The mark of a row that holds a value.</param>
    private readonly record struct RawColumn<T>(T[] Values, int[]? Marks, int Held)
    {
        // Each row's cell: nothing where the row holds nothing, else the next value, as the given rule writes it.
        public string?[] Cells(Func<T, string?> written)
        {
            var cells = new string?[Values.Length];

            for (int row = 0, value = 0; row < cells.Length; row++)
            {
                cells[row] = Marks is not null && Marks[row] < Held ? null : written(Values[value++]);
            }

            return cells;
        }
    }

    /// <summary>What a Parquet file held.</summary>
    /// <param name="ColumnNames">Its columns, in its order.</param>
    /// <param name="StatedKinds">What each column says it holds, for each that says anything.</param>
    /// <param name="Rows">Its rows, every row group in the order the file holds them.</param>
    private readonly record struct ParquetContent(
        IReadOnlyList<string> ColumnNames, IReadOnlyDictionary<string, ColumnKind> StatedKinds, List<IReadOnlyList<string?>> Rows);
}
