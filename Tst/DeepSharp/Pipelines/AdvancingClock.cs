// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A task that waits on a clock a test holds, run to its end by moving that clock on.
/// </summary>
/// <remarks>
/// What waits on the clock is the code's own pacing and its retries, and a test that waited for them would wait for as long
/// as a person does. The clock is moved on a step at a time, with the work given real time to run in between, so each
/// moment a thing is waited for is passed in the order it would be.
/// </remarks>
internal static class AdvancingClock
{
    extension(FakeTimeProvider clock)
    {
        /// <summary>Moves the clock on until the task is done.</summary>
        /// <typeparam name="T">What the task gives.</typeparam>
        /// <param name="task">The task.</param>
        /// <param name="step">How far the clock moves each time; a quarter of a second when none is given.</param>
        /// <returns>What the task gave.</returns>
        public async Task<T> RunAsync<T>(Task<T> task, TimeSpan? step = null)
        {
            var by = step ?? TimeSpan.FromMilliseconds(250);

            for (var turns = 0; !task.IsCompleted; turns++)
            {
                Assert.True(turns < 400_000, "The task was still waiting after 400,000 steps of the clock.");

                await Task.WhenAny(task, Task.Delay(1, TestContext.Current.CancellationToken));

                if (!task.IsCompleted)
                {
                    clock.Advance(by);
                }
            }

            return await task;
        }
    }
}
