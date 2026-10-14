// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace DeepSharp.Tests;

/// <summary>A folder in the temp directory that one test writes into, lists, and removes with itself.</summary>
/// <remarks>
/// Programs that are not the test look into the folder too: a scanner that opens each new file leaves a copy of its own beside
/// it for a moment, and holds it open. A name that stands for a moment is not a name that stays, so the listing waits for the
/// names it expects and gives up only when the time is up; and a folder with a file still held is tried again until it is let go.
/// What stays held, or stays standing, still fails the test that has it.
/// </remarks>
public sealed class ScratchFolder : IDisposable
{
    /// <summary>Makes the folder.</summary>
    /// <param name="prefix">What its name begins with.</param>
    public ScratchFolder(string prefix) => Path = Directory.CreateTempSubdirectory(prefix).FullName;

    /// <summary>Where the folder is.</summary>
    public string Path { get; }

    /// <summary>A name in the folder, as a path.</summary>
    /// <param name="name">The file's name.</param>
    /// <returns>The path.</returns>
    public string At(string name) => System.IO.Path.Join(Path, name);

    /// <summary>The names in the folder, in ordinal order.</summary>
    /// <returns>What stands there now.</returns>
    public string[] Standing() => [.. Directory.GetFileSystemEntries(Path).Select(entry => System.IO.Path.GetFileName(entry)).Order(StringComparer.Ordinal)];

    /// <summary>The names in the folder once they are the names expected, or as they stand when the time is up.</summary>
    /// <param name="expected">The names the test expects, in ordinal order.</param>
    /// <param name="within">How long a name that is not expected may stand before it counts.</param>
    /// <returns>The names that stand.</returns>
    public string[] Standing(IReadOnlyCollection<string> expected, TimeSpan within)
    {
        var waited = Stopwatch.StartNew();
        var standing = Standing();

        while (!standing.SequenceEqual(expected) && waited.Elapsed < within)
        {
            Thread.Sleep(Pause);
            standing = Standing();
        }

        return standing;
    }

    /// <summary>Removes the folder with what is in it.</summary>
    public void Dispose() => Dispose(TimeSpan.FromSeconds(2));

    /// <summary>Removes the folder with what is in it, waiting for what is held by another program to be let go.</summary>
    /// <param name="patience">How long a file may stay held before the folder is said to be stuck.</param>
    /// <exception cref="IOException">A file stayed held for the whole of the time.</exception>
    /// <exception cref="UnauthorizedAccessException">A file stayed beyond reach for the whole of the time.</exception>
    public void Dispose(TimeSpan patience)
    {
        var waited = Stopwatch.StartNew();

        while (true)
        {
            try
            {
                Directory.Delete(Path, recursive: true);

                return;
            }
            catch (DirectoryNotFoundException)
            {
                return;
            }
            catch (Exception held) when (held is IOException or UnauthorizedAccessException && waited.Elapsed < patience)
            {
                // A file another program holds open, or has just deleted and the system has not yet let go of: both pass.
                Thread.Sleep(Pause);
            }
        }
    }

    private static readonly TimeSpan Pause = TimeSpan.FromMilliseconds(10);
}
