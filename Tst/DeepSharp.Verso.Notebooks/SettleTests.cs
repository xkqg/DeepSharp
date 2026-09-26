// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// VS Code sends a keystroke a quarter of a second after it lands, and nothing tells the host one is on its way. A
/// gesture that can change the blocks therefore waits longer than that before it reads them, so a change it writes is
/// never written over text still on its way; a pick or a view changes nothing, and does not wait.
/// </summary>
public sealed class SettleTests : IDisposable
{
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

    // How long a gesture takes, made through the host as an application of your own makes it.
    private static async Task<TimeSpan> ClockedAsync(Notebook notebook, string interaction, string payload = "")
    {
        var clock = Stopwatch.StartNew();

        await notebook.GestureAsync(SchemaBlock(notebook), interaction, payload);

        return clock.Elapsed;
    }

    [Fact]
    public async Task AGestureThatCanChangeTheBlocks_ReadsThemOnlyOnceWhatWasOnItsWayHasLanded()
    {
        await using var notebook = await NotebookAsync();

        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);
        var tick = List(notebook).Row("sex").Included.Action;
        var typed = SchemaBlock(notebook).Source.Replace(
            "{\"name\": \"fare\"", "{\"name\": \"embarked\", \"kind\": \"text\", \"optional\": false}, {\"name\": \"fare\"", StringComparison.Ordinal);

        // What VS Code still holds lands while the tick waits — a column typed into the schema by hand — written into
        // the notebook the way VS Code writes it, whenever it lands and past any order the host keeps.
        var sending = notebook.GestureAsync(SchemaBlock(notebook), tick, "true");

        await Task.Delay(NotebookSession.SettleTime / 6, TestContext.Current.CancellationToken);
        SchemaBlock(notebook).Source = typed;

        var sent = await sending;

        Assert.False(sent.StateChanged);
        Assert.Equal(typed, SchemaBlock(notebook).Source);
        Assert.DoesNotContain("\"sex\"", SchemaBlock(notebook).Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task APickOrAView_ChangesNothing_AndDoesNotWait_WhileAChangeDoes()
    {
        await using var notebook = await NotebookAsync();

        // The first view reads the source once; the ones clocked after it draw what was read.
        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Show);
        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);

        Assert.All(
            [
                await ClockedAsync(notebook, StepRenderer.Show),
                await ClockedAsync(notebook, StepRenderer.Columns),
                await ClockedAsync(notebook, List(notebook).SelectOf(StepRenderer.ListType)!.Value.Action, "target.distribution"),
                await ClockedAsync(notebook, List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Action, "range"),
                await ClockedAsync(notebook, List(notebook).SelectOf(StepRenderer.ListRangeKind, "include")!.Value.Action, "integer"),
            ],
            took => Assert.True(took < NotebookSession.SettleTime, $"a pick or a view took {took.TotalMilliseconds} ms"));

        Assert.True(
            await ClockedAsync(notebook, List(notebook).Row("pclass").Kind.Action, "category") >= NotebookSession.SettleTime,
            "a change to the blocks did not wait");
    }

    [Fact]
    public void TheWait_IsLongerThanTheQuarterSecondVSCodeHoldsAKeystroke() =>
        Assert.True(NotebookSession.SettleTime > TimeSpan.FromMilliseconds(250), $"the wait is {NotebookSession.SettleTime.TotalMilliseconds} ms");
}
