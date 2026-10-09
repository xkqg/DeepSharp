// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The rows a notebook's source opens, the bytes they were read from, and what their cells say each column holds.</summary>
/// <param name="Rows">The rows as the file reads them.</param>
/// <param name="Fingerprint">
/// A SHA-256 of the file's bytes; of a source of several files, a SHA-256 over each file's own, in the order the step reads them.
/// </param>
/// <param name="Proposal">What the rows' cells say each column holds, proposed once for these bytes.</param>
internal readonly record struct SourceRows(IRowSource Rows, string Fingerprint, KindProposal Proposal);

/// <summary>
/// The rows a notebook's source opened last, kept for the next view.
/// </summary>
/// <remarks>
/// A view is worked out from the source up, and parsing the file was most of that work on a file of any size. Two
/// things are kept, apart. The session keeps the last view worked out — one, under the key of the steps it comes
/// from and a SHA-256 of the bytes it read, in memory only. This cache keeps the rows as the source reads them: the
/// input to the first step, and what their cells say each column holds, so a view of other steps over the same bytes
/// parses nothing again and a list drawn again proposes nothing again. Anything a step made of the rows, other than
/// the one view kept, is worked out from the declaration each time.
/// <para>
/// One entry, keyed by the read step itself — compared by what it says — the path the core's one rule resolves, and a
/// SHA-256 of the file's bytes. A source of several files — two files joined into one — is keyed on every path and on one
/// SHA-256 over each file's own, in the order the step reads them: a change to any one file is another key, and two files
/// split another way over the same bytes are too, which a digest of the files laid end to end could not tell apart. The bytes are read and hashed on every use and the step that reads the file parses the
/// rows from those same bytes, whichever of the files a pipeline file names it reads, so a file changed on disk is
/// opened again and the rows kept are always the ones the fingerprint names. The entry lives as
/// long as the notebook's session and is shared with nothing else. It is one value, put in place whole — the rows and
/// the proposal of their kinds together, so the two never come from different bytes: two readers of new bytes at the
/// same moment may both parse them, and the entry one of them made stays — the same rows either way.
/// </para>
/// <para>
/// A notebook's rows come from a file: nothing hands rows in to a notebook, the way code hands them to a pipeline
/// whose rows are handed in. So a pipeline that reads no file has no rows here, and is told so in a notebook's words.
/// </para>
/// </remarks>
internal sealed class SourceCache
{
    private Kept? _kept;
    private int _parsed;

    /// <summary>How many times a source was parsed: the number the cache exists to keep down.</summary>
    public int Parsed => Volatile.Read(ref _parsed);

    /// <summary>The rows the declaration's source opens, from memory when the same step read the same bytes last.</summary>
    /// <param name="declaration">The declaration whose source is read.</param>
    /// <param name="folder">Where a relative path in it is read from.</param>
    /// <returns>The rows as read and the fingerprint of the bytes they were read from.</returns>
    /// <exception cref="InvalidOperationException">The pipeline reads no file: its rows are handed in, or it has no source.</exception>
    /// <exception cref="IOException">The file cannot be read: it is not there, say.</exception>
    /// <exception cref="FormatException">The file cannot be read as its first block reads it.</exception>
    /// <remarks>
    /// The file's bytes are read once, hashed, and handed to the first block to parse, whichever of the files a pipeline
    /// file names it reads: the rows kept are always the ones the fingerprint names.
    /// </remarks>
    public SourceRows RowsFor(PipelineDeclaration declaration, SourceFolder folder)
    {
        if (FilesOf(declaration) is not { } read)
        {
            throw new InvalidOperationException(
                $"A notebook reads its rows from a file, with {Readers()} as its first block, and hands none in; "
                + "these blocks read no file, so there are no rows to show.");
        }

        string[] paths = [.. read.Paths.Select(folder.Resolve)];
        byte[][] bytes = [.. paths.Select(File.ReadAllBytes)];
        var fingerprint = Fingerprint(bytes);

        // A step is known by what it says, never by which object says it: every gesture reads the blocks afresh.
        if (Volatile.Read(ref _kept) is { } kept && kept.Read.Equals(read.Step) && kept.Paths.SequenceEqual(paths) && kept.Fingerprint == fingerprint)
        {
            return new SourceRows(kept.Rows, fingerprint, kept.Proposal);
        }

        var rows = read.Open(bytes, paths);
        var proposal = KindProposal.Of(rows);

        Volatile.Write(ref _kept, new Kept(read.Step, paths, fingerprint, rows, proposal));
        Interlocked.Increment(ref _parsed);

        return new SourceRows(rows, fingerprint, proposal);
    }

    /// <summary>What is known of the bytes the declaration's source holds now: read and hashed, never parsed.</summary>
    /// <param name="declaration">The declaration whose source is read.</param>
    /// <param name="folder">Where a relative path in it is read from.</param>
    /// <returns>Their fingerprint; unreadable when the file cannot be read; unknown when the source is not a file.</returns>
    public static SourceBytes BytesOf(PipelineDeclaration declaration, SourceFolder folder)
    {
        if (FilesOf(declaration) is not { } read)
        {
            return SourceBytes.Unknown;
        }

        try
        {
            return SourceBytes.Of(Fingerprint([.. read.Paths.Select(path => File.ReadAllBytes(folder.Resolve(path)))]));
        }
        catch (Exception unreadable) when (unreadable is IOException or UnauthorizedAccessException)
        {
            return SourceBytes.Unreadable;
        }
    }

    /// <summary>The rows kept for the blocks' source, and the fingerprint of the bytes they were read from.</summary>
    /// <param name="declaration">The blocks, as the notebook reads them now.</param>
    /// <returns>
    /// The rows, their fingerprint and their proposal, one entry, or nothing when no rows are kept for the step that reads
    /// the file, or the blocks read no file.
    /// </returns>
    /// <remarks>
    /// What a gesture knows of the source: it is handed no file, so the rows this session read last are the source's
    /// columns as the person saw them.
    /// </remarks>
    public SourceRows? KeptFor(PipelineDeclaration declaration) =>
        FilesOf(declaration) is { } read && Volatile.Read(ref _kept) is { } kept && kept.Read.Equals(read.Step)
            ? new SourceRows(kept.Rows, kept.Fingerprint, kept.Proposal)
            : null;

    /// <summary>The block that reads the notebook's files: its first, when that reads a file or several.</summary>
    /// <param name="declaration">The blocks, as the notebook reads them.</param>
    /// <returns>What it reads and how, or nothing when the rows are handed in or there is no step at all.</returns>
    internal static SourceFiles? FilesOf(PipelineDeclaration declaration) => declaration.Steps switch
    {
        [IReadsAFile read, ..] => new SourceFiles(read, [read.Path], (bytes, paths) => read.Open(bytes[0], paths[0])),
        [IReadsFiles read, ..] => new SourceFiles(read, read.Paths, read.Open),
        _ => null,
    };

    // One file's fingerprint as it always was; several files' as one SHA-256 over each file's own, in order — never over
    // the files laid end to end, which two files split another way over the same bytes would share.
    private static string Fingerprint(IReadOnlyList<byte[]> files) =>
        files is [var one]
            ? one.Fingerprint()
            : Convert.ToHexString(SHA256.HashData([.. files.SelectMany(file => SHA256.HashData(file))])).ToLowerInvariant();

    // Every verb of the notebook's own that reads a file or several, as its catalog knows them, in the order of their names.
    private static string Readers()
    {
        var catalog = NotebookVerbs.Catalog();
        string[] verbs = [.. catalog.Descriptions.Where(each => catalog.ReadStep(each.Template) is IReadsAFile or IReadsFiles).Select(each => $"'{each.Verb}'")];

        return $"{string.Join(", ", verbs[..^1])} or {verbs[^1]}";
    }

    /// <summary>The one source kept, and what it was kept under.</summary>
    private sealed record Kept(IPipelineStep Read, IReadOnlyList<string> Paths, string Fingerprint, IRowSource Rows, KindProposal Proposal);
}

/// <summary>What a notebook's first block reads: the step, the paths of its files in its order, and how it reads their bytes.</summary>
/// <param name="Step">The block's step, which an entry is known by.</param>
/// <param name="Paths">The paths of the files as they were written, in the order the step reads them.</param>
/// <param name="Open">How the step reads the files' bytes, handed with the paths a refusal names them by.</param>
internal sealed record SourceFiles(IPipelineStep Step, IReadOnlyList<string> Paths, Func<IReadOnlyList<byte[]>, IReadOnlyList<string>, IRowSource> Open);
