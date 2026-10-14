// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What is done with a task that nobody awaits.</summary>
internal static class TaskExtensions
{
    extension(Task task)
    {
        /// <summary>Watches a task that nobody awaits, so what fails in it is not left to the finalizer.</summary>
        /// <returns>The task that does the watching.</returns>
        internal Task Observed() =>
            task.ContinueWith(
                static watched =>
                {
                    // Read, so that it is seen: a failure nobody reads is handed on by the finalizer.
                    _ = watched.Exception;
                },
                CancellationToken.None,
                TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
    }
}
