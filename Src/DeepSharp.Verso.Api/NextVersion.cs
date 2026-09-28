// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// The version of a notebook its turn publishes next, as whoever asked for it waits for it: everyone who asked before it
/// is made is told it, however many asked at once, and whoever asks while it is made is told the one after — so each is
/// told a version made after they asked.
/// </summary>
internal sealed class NextVersion
{
    // Whoever asked since the last version was begun, all waiting on one answer; nothing when nobody has.
    private TaskCompletionSource<NotebookVersion>? _asked;

    /// <summary>Waits for the next version made after this ask.</summary>
    /// <returns>The version.</returns>
    public Task<NotebookVersion> AskAsync()
    {
        var mine = new TaskCompletionSource<NotebookVersion>(TaskCreationOptions.RunContinuationsAsynchronously);

        // Whoever asked first since the last version was begun sets the answer everyone waits on, so no ask replaces another.
        return (Interlocked.CompareExchange(ref _asked, mine, null) ?? mine).Task;
    }

    /// <summary>Makes the next version, and tells it to everyone who asked before it was begun.</summary>
    /// <param name="make">Makes the version.</param>
    /// <returns>The version.</returns>
    public async Task<NotebookVersion> TellAsync(Func<Task<NotebookVersion>> make)
    {
        // Taken first, so whoever asks while it is made waits for the one after.
        var asked = Interlocked.Exchange(ref _asked, null);
        var version = await make();

        asked?.TrySetResult(version);

        return version;
    }
}
