// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A file written whole: its bytes go to a name of its own beside it and are moved into place in one step, so whoever
/// looks finds nothing or the whole of it, and whatever stops a write leaves nothing of its own behind.
/// </summary>
/// <remarks>
/// The name is this write's alone, so nothing of this program holds it and nothing else ever reads it. The bytes are on
/// the disk, not only in the system's cache, before the move makes them the file. A file written once is moved without
/// replacing anything — the system refuses that move deterministically, where a replacing one can fail a reader or
/// silently replace what a racing writer had just put there — and a refusal is read for what it says: the same bytes
/// already standing are the same file, other bytes are somebody else's and stay.
/// </remarks>
internal static class WholeFile
{
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

            File.Move(own, path, overwrite: true);
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

            File.Move(own, path, overwrite: true);
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

            try
            {
                File.Move(own, path, overwrite: false);
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
