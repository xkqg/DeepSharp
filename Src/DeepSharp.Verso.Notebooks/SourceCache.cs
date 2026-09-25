// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The rows a notebook's source opens, and the bytes they were read from.</summary>
/// <param name="Rows">The rows as the file reads them.</param>
/// <param name="Fingerprint">A SHA-256 of the file's bytes.</param>
internal readonly record struct SourceRows(IRowSource Rows, string Fingerprint);

/// <summary>
/// The rows a notebook's source opened last, kept for the next view.
/// </summary>
/// <remarks>
/// A view is worked out from the source up, and parsing the file was most of that work on a file of any size. Two
/// things are kept, apart. The session keeps the last view worked out — one, under the key of the steps it comes
/// from and a SHA-256 of the bytes it read, in memory only. This cache keeps the rows as the source reads them: the
/// input to the first step and nothing more, so a view of other steps over the same bytes parses nothing again.
/// Anything a step made of the rows, other than the one view kept, is worked out from the declaration each time.
/// <para>
/// One entry, keyed by the read step itself, the path the core's one rule resolves, and a SHA-256 of the file's
/// bytes. The bytes are read and hashed on every use and the rows are parsed from those same bytes, so a file
/// changed on disk is opened again and the rows kept are always the ones the fingerprint names. The entry lives as
/// long as the notebook's session and is shared with nothing else; one parse per key is kept under the entry's own
/// lock.
/// </para>
/// <para>
/// A notebook's rows come from a file: nothing hands rows in to a notebook, the way code hands them to a pipeline
/// whose rows are handed in. So a pipeline that reads no file has no rows here, and is told so in a notebook's words.
/// </para>
/// </remarks>
internal sealed class SourceCache
{
    private readonly object _lock = new();
    private Kept? _kept;

    /// <summary>How many times a source was parsed: the number the cache exists to keep down.</summary>
    public int Parsed { get; private set; }

    /// <summary>The rows the declaration's source opens, from memory when the same step read the same bytes last.</summary>
    /// <param name="declaration">The declaration whose source is read.</param>
    /// <param name="folder">Where a relative path in it is read from.</param>
    /// <returns>The rows as read and the fingerprint of the bytes they were read from.</returns>
    /// <exception cref="InvalidOperationException">The pipeline reads no file: its rows are handed in, or it has no source.</exception>
    /// <exception cref="IOException">The file cannot be read: it is not there, say.</exception>
    /// <exception cref="FormatException">The file has no header, or a row has the wrong number of cells.</exception>
    public SourceRows RowsFor(PipelineDeclaration declaration, SourceFolder folder)
    {
        if (declaration.Steps is not [ReadCsvStep read, ..])
        {
            throw new InvalidOperationException(
                $"A notebook reads its rows from a file, with '{ReadCsvStep.Name}' as its first block, and hands none in; "
                + "these blocks read no file, so there are no rows to show.");
        }

        var path = folder.Resolve(read.Path);
        var bytes = File.ReadAllBytes(path);
        var fingerprint = Fingerprint(bytes);

        lock (_lock)
        {
            if (_kept is { } kept && kept.Read == read && kept.Path == path && kept.Fingerprint == fingerprint)
            {
                return new SourceRows(kept.Rows, fingerprint);
            }

            var rows = CsvRowSource.FromText(Decoded(bytes), path);

            _kept = new Kept(read, path, fingerprint, rows);
            Parsed++;

            return new SourceRows(rows, fingerprint);
        }
    }

    /// <summary>What is known of the bytes the declaration's source holds now: read and hashed, never parsed.</summary>
    /// <param name="declaration">The declaration whose source is read.</param>
    /// <param name="folder">Where a relative path in it is read from.</param>
    /// <returns>Their fingerprint; unreadable when the file cannot be read; unknown when the source is not a file.</returns>
    public static SourceBytes BytesOf(PipelineDeclaration declaration, SourceFolder folder)
    {
        if (declaration.Steps is not [ReadCsvStep read, ..])
        {
            return SourceBytes.Unknown;
        }

        try
        {
            return SourceBytes.Of(Fingerprint(File.ReadAllBytes(folder.Resolve(read.Path))));
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException)
        {
            return SourceBytes.Unreadable;
        }
    }

    /// <summary>The rows kept for a read step, and the fingerprint of the bytes they were read from.</summary>
    /// <param name="read">The read step.</param>
    /// <returns>The rows and their fingerprint, one entry, or nothing when no rows are kept for that step.</returns>
    /// <remarks>
    /// What a gesture knows of the source: it is handed no file, so the rows this session read last are the source's
    /// columns as the person saw them.
    /// </remarks>
    public SourceRows? KeptFor(ReadCsvStep read)
    {
        lock (_lock)
        {
            return _kept is { } kept && kept.Read == read ? new SourceRows(kept.Rows, kept.Fingerprint) : null;
        }
    }

    private static string Fingerprint(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    // As the file door reads text: the encoding its byte-order mark names, and UTF-8 when it names none.
    private static string Decoded(byte[] bytes)
    {
        using var reader = new StreamReader(new MemoryStream(bytes), Encoding.UTF8, detectEncodingFromByteOrderMarks: true);

        return reader.ReadToEnd();
    }

    /// <summary>The one source kept, and what it was kept under.</summary>
    private sealed record Kept(ReadCsvStep Read, string Path, string Fingerprint, IRowSource Rows);
}
