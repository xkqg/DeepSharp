// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>One thing at a time, each after every one that came before it.</summary>
/// <remarks>
/// Each notebook's host owns one, for everything done to that notebook, and the process owns one for its C# runs,
/// because a C# kernel takes over the process's console. A thing handed on from inside another on the same lane would wait for the one it is
/// inside, so nothing is. Taking a place in the line is one exchange of its end, so the order is the order they came.
/// </remarks>
internal sealed class Lane
{
    // The end of the line: the next one waits for it.
    private Task _last = Task.CompletedTask;

    /// <summary>Does one thing once everything that came before it is done.</summary>
    /// <typeparam name="T">What it answers.</typeparam>
    /// <param name="work">The thing.</param>
    /// <returns>Its answer.</returns>
    public async Task<T> TakeTurnAsync<T>(Func<Task<T>> work)
    {
        var mine = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var before = Interlocked.Exchange(ref _last, mine.Task);

        await before;

        try
        {
            return await work();
        }
        finally
        {
            mine.SetResult();
        }
    }
}
