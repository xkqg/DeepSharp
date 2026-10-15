// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// VS Code sends a keystroke a quarter of a second after it lands, and nothing tells the host one is on its way. A
/// gesture that can change the blocks therefore waits longer than that before it reads them, so a change it writes is
/// never written over text still on its way; a pick or a view changes nothing, and does not wait. The wait is made on a
/// clock the test holds, so it ends when the test says and a machine under load decides nothing about it.
/// </summary>
public sealed class SettleTests : IDisposable
{
    // The quarter of a second VS Code holds a keystroke before it sends it.
    private static readonly TimeSpan HeldByVSCode = TimeSpan.FromMilliseconds(250);

    // How long a gesture is given to end on a clock nobody moves: a gesture that waited there would never end.
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-settle-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
    ];

    public SettleTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<Notebook> NotebookAsync()
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static CellModel SchemaBlock(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        SchemaBlock(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    // A gesture made through the host as an application of your own makes it, given its patience to end.
    private static Task<GestureResult> MadeAsync(Notebook notebook, string interaction, string payload = "") =>
        notebook.GestureAsync(SchemaBlock(notebook), interaction, payload).WaitAsync(Patience, TestContext.Current.CancellationToken);

    [Fact]
    public async Task AGestureThatCanChangeTheBlocks_ReadsThemOnlyOnceWhatWasOnItsWayHasLanded()
    {
        await using var notebook = await NotebookAsync();
        var clock = notebook.HoldTheSettleWait();

        await MadeAsync(notebook, StepRenderer.Columns);
        var tick = List(notebook).Row("sex").Included.Action;
        var typed = SchemaBlock(notebook).Source.Replace(
            "{\"name\": \"fare\"", "{\"name\": \"embarked\", \"kind\": \"text\", \"optional\": false}, {\"name\": \"fare\"", StringComparison.Ordinal);

        // The tick waits for what VS Code still holds, and holds the notebook while it does.
        var sending = notebook.GestureAsync(SchemaBlock(notebook), tick, "true");

        await notebook.UntilAGestureWaitsAsync();

        // What VS Code still holds lands while the tick waits — a column typed into the schema by hand — written into
        // the notebook the way VS Code writes it, whenever it lands and past any order the host keeps.
        SchemaBlock(notebook).Source = typed;
        clock.Advance(NotebookSession.SettleTime);

        var sent = await sending.WaitAsync(Patience, TestContext.Current.CancellationToken);

        Assert.False(sent.StateChanged);
        Assert.Equal(typed, SchemaBlock(notebook).Source);
        Assert.DoesNotContain("\"sex\"", SchemaBlock(notebook).Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task APickOrAView_ChangesNothing_AndDoesNotWait_WhileAChangeDoes()
    {
        await using var notebook = await NotebookAsync();
        var clock = notebook.HoldTheSettleWait();

        // The first view reads the source once; the ones after it draw what was read. The clock is never moved for any of
        // them: one that waited would not end.
        await MadeAsync(notebook, StepRenderer.Show);
        await MadeAsync(notebook, StepRenderer.Columns);
        await MadeAsync(notebook, StepRenderer.Show);
        await MadeAsync(notebook, StepRenderer.Columns);
        await MadeAsync(notebook, List(notebook).SelectOf(StepRenderer.ListType)!.Value.Action, "target.distribution");
        await MadeAsync(notebook, List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Action, "range");
        await MadeAsync(notebook, List(notebook).SelectOf(StepRenderer.ListRangeKind, "include")!.Value.Action, "integer");

        Assert.Equal(0, notebook.Session.GesturesWaited);

        // A change waits for the settle time on the clock and for no less. It is given real time to end early, which it
        // must not: a stall can only make this look at less, never fail it.
        var changing = notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row("pclass").Kind.Action, "category");

        await notebook.UntilAGestureWaitsAsync();
        clock.Advance(NotebookSession.SettleTime - TimeSpan.FromMilliseconds(1));
        await Task.Delay(HeldByVSCode * 1.5, TestContext.Current.CancellationToken);

        Assert.False(changing.IsCompleted, "a change to the blocks ended a millisecond before the settle time had passed");

        clock.Advance(TimeSpan.FromMilliseconds(1));

        Assert.True((await changing.WaitAsync(Patience, TestContext.Current.CancellationToken)).StateChanged);
        Assert.Equal(1, notebook.Session.GesturesWaited);
    }

    [Fact]
    public async Task ANotebookClosedWhileAGestureStillWaits_OnAClockNobodyMoves_IsClosedAllTheSame()
    {
        // A test that fails with a gesture still waiting on its held clock must end as a failed test, not as a host that
        // never closes the notebook.
        var notebook = await NotebookAsync();

        notebook.HoldTheSettleWait();
        await MadeAsync(notebook, StepRenderer.Columns);

        var waiting = notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row("sex").Included.Action, "true");

        await notebook.UntilAGestureWaitsAsync();
        await notebook.DisposeAsync().AsTask().WaitAsync(Patience, TestContext.Current.CancellationToken);

        // Whatever came of the gesture is read, so that nothing is left for the finalizer to hand on.
        await Task.WhenAny(waiting, Task.Delay(Patience, TestContext.Current.CancellationToken));

        Assert.True(waiting.IsCompleted);
        _ = waiting.Exception;
    }

    [Fact]
    public void TheWait_IsLongerThanTheQuarterSecondVSCodeHoldsAKeystroke() =>
        Assert.True(NotebookSession.SettleTime > HeldByVSCode, $"the wait is {NotebookSession.SettleTime.TotalMilliseconds} ms");
}
