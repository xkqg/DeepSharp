// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Data.Common;
using System.Globalization;
using Microsoft.Data.Analysis;

namespace DeepSharp.Pipelines;

/// <summary>
/// Rows read from a <see cref="DataFrame"/>.
/// </summary>
/// <remarks>
/// One reader for everything a DataFrame can be filled from: a comma-separated file, a database query, a
/// stream of rows somebody already had. That is the whole reason the pipeline takes rows rather than files
/// — the long tail of formats is somebody else's solved problem, and this is where it is borrowed.
/// <para>
/// Cells arrive as text and stay text until the schema says what they are, and an absent value stays
/// absent. A reader that turned a null into a not-a-number would be convenient for drawing and wrong here:
/// never-there and arithmetic-went-wrong are different answers, and a pipeline that confuses them fills a
/// column with a number nobody measured.
/// </para>
/// </remarks>
public sealed class DataFrameRowSource : IRowSource
{
    private readonly DataFrame _frame;

    /// <summary>Reads the rows of a data frame.</summary>
    /// <param name="frame">The frame to read.</param>
    public DataFrameRowSource(DataFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);

        _frame = frame;
        ColumnNames = [.. frame.Columns.Select(column => column.Name)];
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows
    {
        get
        {
            for (long row = 0; row < _frame.Rows.Count; row++)
            {
                yield return [.. _frame.Columns.Select(column => AsText(column[row]))];
            }
        }
    }

    private static string? AsText(object? value) => value switch
    {
        null => null,
        DateTime moment => moment.ToString("O", CultureInfo.InvariantCulture),
        _ => Convert.ToString(value, CultureInfo.InvariantCulture),
    };
}

/// <summary>
/// Reading a pipeline's rows out of a data frame, or out of anything that can fill one.
/// </summary>
public static class DataFrameSourceExtensions
{
    /// <summary>Declares that the rows come from a data frame already in hand.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="frame">The frame to read.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    public static PipelineBuilder ReadDataFrame(this PipelineBuilder pipeline, DataFrame frame)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(frame);

        return pipeline.Read(new DataFrameRowSource(frame), "a data frame");
    }

    /// <summary>Declares that the rows come from a database query.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="reader">A reader over the query's results, from whichever provider you already use.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    /// <remarks>
    /// The query runs here and the rows are read into memory before anything else happens, which is what a
    /// pipeline wants: a source that answers differently each time it is asked is the one thing a
    /// declaration cannot promise anything about. Reading a database is waiting, so this is the one verb in
    /// the chain that is awaited — write it as <c>(await pipeline.ReadDbAsync(reader)).Declare(...)</c>.
    /// </remarks>
    public static async Task<PipelineBuilder> ReadDbAsync(this PipelineBuilder pipeline, DbDataReader reader)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(reader);

        var frame = await DataFrame.LoadFrom(reader).ConfigureAwait(false);

        return pipeline.Read(new DataFrameRowSource(frame), "a database query");
    }

    /// <summary>Declares that the rows come from a comma-separated file, read by the data frame.</summary>
    /// <param name="pipeline">The pipeline being written.</param>
    /// <param name="path">The file to read.</param>
    /// <returns>The pipeline, so the next verb can be written after it.</returns>
    /// <remarks>
    /// The pipeline has a reader of its own for this; it is here so that a file, a query and a frame all
    /// arrive through the same door when a project has already chosen the data frame for everything else.
    /// <para>
    /// Two things are said here that the shorter call does not say, and both were measured rather than
    /// guessed. The file is opened for <b>sharing</b>: handed a path, the data frame opens it exclusively,
    /// and a second pipeline reading the same file at that moment fails on a lock rather than on anything
    /// to do with the data. And the numbers are read with the <b>invariant culture</b>: left to the
    /// machine's own, a fare of <c>7.25</c> was read as <c>725</c> on a culture where a dot groups
    /// thousands — the same file, a hundred times wrong, with nothing going red.
    /// </para>
    /// </remarks>
    public static PipelineBuilder ReadCsvFrame(this PipelineBuilder pipeline, string path)
    {
        ArgumentNullException.ThrowIfNull(pipeline);

        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var frame = DataFrame.LoadCsv(file, cultureInfo: CultureInfo.InvariantCulture);

        return pipeline.Read(new DataFrameRowSource(frame), $"the data frame from {path}");
    }
}
