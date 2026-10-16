// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A task nobody awaits that fails is handed by the finalizer, long after, to whatever is running then — in a process that
/// runs C# cells, the output of a cell that had nothing to do with it. Watching it keeps that from happening.
/// </summary>
[Collection(FaultsLeftToTheFinalizer.Name)]
public sealed class TaskExtensionsTests
{
    // Made and dropped in a method of its own, so nothing here still holds the task when the collector looks.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Drop(Exception failure, bool watched)
    {
        var task = Task.FromException(failure);

        if (watched)
        {
            _ = task.Observed();
        }
    }

    // A full collection, asked for through the call that reports what is left after one: what finds the task dropped above.
    private static void LetTheCollectorRun() => _ = GC.GetTotalMemory(forceFullCollection: true);

    // How many times the finalizer handed this failure on, whatever else is running in the process.
    private static int Handed(Exception failure, bool watched)
    {
        var handed = 0;

        void Count(object? sender, UnobservedTaskExceptionEventArgs unobserved)
        {
            if (unobserved.Exception.InnerExceptions.Contains(failure))
            {
                Interlocked.Increment(ref handed);
            }
        }

        TaskScheduler.UnobservedTaskException += Count;

        try
        {
            Drop(failure, watched);

            LetTheCollectorRun();
            GC.WaitForPendingFinalizers();
            LetTheCollectorRun();

            return Volatile.Read(ref handed);
        }
        finally
        {
            TaskScheduler.UnobservedTaskException -= Count;
        }
    }

    [Fact]
    public void ATaskNobodyWatches_ThatFails_IsHandedToTheFinalizer()
    {
        // The control: what the test below counts is there to be counted.
        Assert.Equal(1, Handed(new InvalidOperationException("nobody watched this"), watched: false));
    }

    [Fact]
    public void ATaskThatIsWatched_ThatFails_IsNotLeftToTheFinalizer()
    {
        Assert.Equal(0, Handed(new InvalidOperationException("this was watched"), watched: true));
    }

    [Fact]
    public void TheFailureTheControlLeavesToTheFinalizer_IsHandedOnWhileNoOtherTestRunsACell()
    {
        // Verso's engine hands a failure nobody watched to the notebook running a cell when it surfaces, as that cell's output:
        // left to the finalizer beside a test that runs one, the control's failure printed into that test's cell.
        var collection = typeof(TaskExtensionsTests).GetCustomAttributes(typeof(CollectionAttribute), inherit: false).Cast<CollectionAttribute>().SingleOrDefault();
        var definition = typeof(TaskExtensionsTests).Assembly.GetTypes()
            .SelectMany(type => type.GetCustomAttributes(typeof(CollectionDefinitionAttribute), inherit: false).Cast<CollectionDefinitionAttribute>())
            .SingleOrDefault(each => each.Name == collection?.Name);

        Assert.True(definition?.DisableParallelization, "The tests that leave a failure to the finalizer run beside the others.");
    }
}
