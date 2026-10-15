// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace DeepSharp.Pipelines;

/// <summary>A file on its way into place: the name its bytes were written under, where they go, and whether they replace what stands there.</summary>
/// <param name="Own">The name of its own the bytes were written to.</param>
/// <param name="Path">Where the bytes go.</param>
/// <param name="Replaces">Whether they replace a file standing at the path.</param>
internal readonly record struct Placing(string Own, string Path, bool Replaces)
{
    /// <summary>Makes the move once.</summary>
    public void Done() => File.Move(Own, Path, Replaces);
}

/// <summary>
/// A file written whole: its bytes go to a name of its own beside it and are moved into place in one step, so whoever
/// looks finds nothing or the whole of it, and whatever stops a write leaves nothing of its own behind.
/// </summary>
/// <remarks>
/// <para>
/// The name is this write's alone, so nothing of this program holds it and nothing else ever reads it. The bytes are on
/// the disk, not only in the system's cache, before the move makes them the file. A file written once is moved without
/// replacing anything — a replacing move can fail a reader or silently replace what a racing writer had just put there —
/// and a refusal is read for what it says: the same bytes already standing are the same file, other bytes are somebody
/// else's and stay.
/// </para>
/// <para>
/// Windows refuses a move onto a place that is taken in one step. On Linux and macOS .NET looks at the place and then
/// renames, so two writers that race for one place can both be told it is free and the later replaces the earlier. The
/// writers of one place therefore take turns at it, in this process, which is where they race in practice. Between
/// processes the system's own refusal is all there is, and a file replaced all the same is caught by whoever kept the
/// fingerprint of what they wrote, as the record of a landing does.
/// </para>
/// </remarks>
internal static class WholeFile
{
    // Sixty-four turns, a place at the one its path's hash names: two places that share a turn wait for each other and
    // nothing worse, and nothing grows with the number of files written.
    private static readonly SemaphoreSlim[] Turns = [.. Enumerable.Range(0, 64).Select(_ => new SemaphoreSlim(1, 1))];

    private static SemaphoreSlim TurnAt(string path) => Turns[(uint)StringComparer.Ordinal.GetHashCode(Path.GetFullPath(path)) % (uint)Turns.Length];

    // A name of its own beside the file, hidden where the system hides what starts with a dot.
    private static string OwnNameBeside(string path)
    {
        var full = Path.GetFullPath(path);

        return Path.Join(Path.GetDirectoryName(full), $".{Path.GetFileName(full)}.{Guid.NewGuid():N}.tmp");
    }

    private static FileStream Create(string own, FileOptions options) =>
        new(own, FileMode.CreateNew, FileAccess.Write, FileShare.None, bufferSize: 4096, options);

    // What stopped a write is gone with it: a name that was never made, or one already moved, is nothing to remove.
    private static void Remove(string own) => File.Delete(own);

    // How long a move that is refused for a moment is made again before the refusal is the answer. What refuses it — a scanner, an
    // indexer, a sync client looking at the file — lets go within milliseconds (measured, on a notebook's saved columns: 1 write
    // in 470 under load was answered 'access denied' with no process holding the file, 117 of 118 went through at the next try),
    // and a person waits for a save no longer than this before being told.
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(3);

    // The longest pause between two tries.
    private const int LongestPause = 50;

    // Windows' ERROR_SHARING_VIOLATION and ERROR_LOCK_VIOLATION, as the HRESULT an IOException carries.
    private const int SharingViolation = unchecked((int)0x80070020);
    private const int LockViolation = unchecked((int)0x80070021);

    // A refusal that goes away by itself: 'access denied' or a sharing or lock violation. Not a folder where the file goes, nor a
    // file marked read-only — nothing lets go of those — and not 'the file is already there', which is an answer, nor a missing folder.
    private static bool Passes(Exception refusal, string path) =>
        refusal is UnauthorizedAccessException or IOException { HResult: SharingViolation or LockViolation } && !RefusedForGood(path);

    private static bool RefusedForGood(string path)
    {
        try
        {
            return (File.GetAttributes(path) & (FileAttributes.Directory | FileAttributes.ReadOnly)) != 0;
        }
        catch (Exception unseen) when (unseen is IOException or UnauthorizedAccessException)
        {
            // Nothing stands there to refuse it, or it cannot be looked at: no reason to give up yet.
            return false;
        }
    }

    /// <summary>Makes a move, and makes it again for a moment when the system refuses it for a reason that goes away.</summary>
    /// <param name="placing">The move.</param>
    /// <param name="patience">How long a refusal is waited out before it is the answer.</param>
    /// <param name="move">How the move is made.</param>
    /// <exception cref="UnauthorizedAccessException">The refusal that was still there when the patience was spent.</exception>
    /// <exception cref="IOException">As above, or a refusal that does not go away.</exception>
    internal static void MoveInto(Placing placing, TimeSpan patience, Action<Placing> move)
    {
        var waited = Stopwatch.StartNew();

        for (var pause = 1; ; pause = Math.Min(pause * 2, LongestPause))
        {
            try
            {
                move(placing);

                return;
            }
            catch (Exception refusal) when (waited.Elapsed < patience && Passes(refusal, placing.Path))
            {
                Thread.Sleep(pause);
            }
        }
    }

    /// <summary>Makes a move, and makes it again for a moment when the system refuses it for a reason that goes away.</summary>
    /// <param name="placing">The move.</param>
    /// <param name="patience">How long a refusal is waited out before it is the answer.</param>
    /// <param name="move">How the move is made.</param>
    /// <param name="cancellation">Ends the waiting between the tries.</param>
    /// <returns>When the move is made.</returns>
    /// <exception cref="OperationCanceledException">The waiting between two tries was cancelled.</exception>
    internal static async Task MoveIntoAsync(Placing placing, TimeSpan patience, Action<Placing> move, CancellationToken cancellation)
    {
        var waited = Stopwatch.StartNew();

        for (var pause = 1; ; pause = Math.Min(pause * 2, LongestPause))
        {
            try
            {
                move(placing);

                return;
            }
            catch (Exception refusal) when (waited.Elapsed < patience && Passes(refusal, placing.Path))
            {
                await Task.Delay(pause, cancellation);
            }
        }
    }

    /// <summary>Writes the bytes whole, replacing what stood at the path.</summary>
    /// <param name="bytes">The bytes.</param>
    /// <param name="path">The file.</param>
    public static void Replace(byte[] bytes, string path)
    {
        var own = OwnNameBeside(path);

        try
        {
            using (var stream = Create(own, FileOptions.None))
            {
                stream.Write(bytes);
                stream.Flush(flushToDisk: true);
            }

            MoveInto(new Placing(own, path, Replaces: true), Patience, static placing => placing.Done());
        }
        catch
        {
            Remove(own);

            throw;
        }
    }

    /// <summary>Writes the bytes whole, replacing what stood at the path.</summary>
    /// <param name="bytes">The bytes.</param>
    /// <param name="path">The file.</param>
    /// <param name="cancellation">Ends the write before the bytes are moved into place.</param>
    /// <returns>When the file is written.</returns>
    public static async Task ReplaceAsync(byte[] bytes, string path, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();

        var own = OwnNameBeside(path);

        try
        {
            await Fill(own, bytes, cancellation);

            await MoveIntoAsync(new Placing(own, path, Replaces: true), Patience, static placing => placing.Done(), cancellation);
        }
        catch
        {
            Remove(own);

            throw;
        }
    }

    /// <summary>Writes the bytes whole where nothing stands, and never replaces what does.</summary>
    /// <param name="bytes">The bytes.</param>
    /// <param name="path">The file.</param>
    /// <param name="cancellation">Ends the write before the bytes are moved into place.</param>
    /// <returns>When the file stands there: written by this call, or already holding the same bytes.</returns>
    /// <exception cref="IOException">Other bytes stand there, or the move was refused for another reason.</exception>
    public static async Task WriteOnceAsync(byte[] bytes, string path, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();

        var own = OwnNameBeside(path);

        try
        {
            await Fill(own, bytes, cancellation);

            var turn = TurnAt(path);

            await turn.WaitAsync(cancellation);

            try
            {
                try
                {
                    await MoveIntoAsync(new Placing(own, path, Replaces: false), Patience, static placing => placing.Done(), cancellation);
                }
                catch (IOException) when (File.Exists(path))
                {
                    var standing = await File.ReadAllBytesAsync(path, cancellation);

                    if (!standing.AsSpan().SequenceEqual(bytes))
                    {
                        throw new IOException($"'{path}' holds other bytes than the ones written, and a file written once is never replaced: the first stays.");
                    }
                }
            }
            finally
            {
                turn.Release();
            }
        }
        finally
        {
            Remove(own);
        }
    }

    private static async Task Fill(string own, byte[] bytes, CancellationToken cancellation)
    {
        await using var stream = Create(own, FileOptions.Asynchronous);

        await stream.WriteAsync(bytes, cancellation);
        await stream.FlushAsync(cancellation);

        stream.Flush(flushToDisk: true);
    }
}
