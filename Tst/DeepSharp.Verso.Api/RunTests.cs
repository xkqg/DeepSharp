// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A run is one value — whether it has yet to start, runs, was stopped or ended by itself, what it runs now, and the
/// verbs on their way through it — and a stop is one step: unless the run already ended by itself, it takes the run's
/// end, marks the run, tells the notebook, and only then takes what runs now as the kernel to start afresh, handing the
/// run's own flow that decision whole. Each test is a fixed order of calls on one thread.
/// </summary>
public sealed class RunTests
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private static readonly Guid Cell = Guid.NewGuid();

    // A run whose stop tells the notebook by doing what is given.
    private static Run Telling(Action tell) => new(1, cell: null, takesTheCSharpTurn: true, new Told(tell));

    [Fact]
    public async Task AStopTakesWhatRunsAtTheStop_ThoughItEndsBeforeAnyoneLooks()
    {
        var run = Telling(() => { });

        Assert.True(run.Starts());
        run.Began(Cell, "pdd");
        Assert.True(run.Stop());
        run.Ended(Cell);

        var stop = await run.Stopped;

        Assert.False(stop.BeforeItRan);
        Assert.Equal(new Underway(Cell, "pdd", stop.Ran!.Value.Since), stop.Ran);
        Assert.Null(run.Now);
    }

    [Fact]
    public async Task AStopMarksTheRunAndTellsTheNotebookBeforeItDecides()
    {
        Run? run = null;
        var markedWhenTold = false;

        // What the engine begins while the notebook is told began before the decision, which takes it.
        run = Telling(() =>
        {
            markedWhenTold = run!.Token.IsCancellationRequested;
            run.Began(Cell, "csharp");
        });

        run.Starts();

        Assert.True(run.Stop());
        Assert.True(markedWhenTold);
        Assert.Equal(Cell, (await run.Stopped).Ran?.Cell);
    }

    [Fact]
    public void AStopAfterTheRunEndedByItself_StopsAndTellsNothing()
    {
        var told = 0;
        var run = Telling(() => told++);

        run.Starts();

        Assert.True(run.Ends());
        Assert.False(run.Stop());
        Assert.Equal(0, told);
        Assert.False(run.Token.IsCancellationRequested);
        Assert.False(run.Claimed);
    }

    [Fact]
    public void TheRunsOwnEndAfterAStopClaimedIt_IsTheStops()
    {
        Run? run = null;
        bool? endedByItself = null;

        run = Telling(() => endedByItself = run!.Ends());
        run.Starts();

        Assert.True(run.Stop());
        Assert.False(endedByItself);
        Assert.False(run.Ends());
        Assert.True(run.Claimed);
    }

    [Fact]
    public async Task AStopBeforeTheRunStarted_NeverLetsItStart()
    {
        var told = 0;
        var run = Telling(() => told++);

        Assert.True(run.Stop());
        Assert.False(run.Starts());
        Assert.True((await run.Stopped).BeforeItRan);
        Assert.Equal(1, told);
    }

    [Fact]
    public void ASecondStop_TellsNothing()
    {
        var told = 0;
        var run = Telling(() => told++);

        run.Starts();

        Assert.True(run.Stop());
        Assert.False(run.Stop());
        Assert.Equal(1, told);
    }

    [Fact]
    public async Task AVerbLetThroughBeforeAStop_KeepsTheStopDrainingUntilItEnds_AndOneAfterIsRefused()
    {
        var run = Telling(() => { });

        run.Starts();

        var admitted = run.Admit();

        Assert.True(run.Stop());

        var stop = await run.Stopped;

        Assert.False(stop.Drained.IsCompleted);
        Assert.Throws<OperationCanceledException>(() => run.Admit());

        admitted.Dispose();
        await stop.Drained.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task RunAllsReset_IsOnItsWayUntilTheFirstCellBegins()
    {
        var run = Telling(() => { });

        run.Starts();

        var resetting = run.AdmitUntilBegan();

        Assert.True(run.Stop());

        var stop = await run.Stopped;

        Assert.False(stop.Drained.IsCompleted);

        // The first cell begins once the reset ended; the verb's own end then takes nothing back twice.
        run.Began(Cell, "csharp");
        await stop.Drained.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        resetting.Dispose();
    }

    // The notebook of such a run: hears nothing of what it runs now, and is told of its stop by doing what is given.
    private sealed class Told(Action tell) : IRunListener
    {
        public void Moved()
        {
        }

        public Task StoppedAsync()
        {
            tell();

            return Task.CompletedTask;
        }
    }
}
