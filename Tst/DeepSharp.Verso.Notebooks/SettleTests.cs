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

    private static NotebookSession Session(Notebook notebook) => notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

    private static CellModel SchemaBlock(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        SchemaBlock(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    [Fact]
    public async Task AGestureThatCanChangeTheBlocks_ReadsThemOnlyOnceWhatWasOnItsWayHasLanded()
    {
        await using var notebook = await NotebookAsync();

        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);
        var tick = List(notebook).Row("sex").Included.Action;
        var typed = SchemaBlock(notebook).Source.Replace(
            "{\"name\": \"fare\"", "{\"name\": \"embarked\", \"kind\": \"text\", \"optional\": false}, {\"name\": \"fare\"", StringComparison.Ordinal);

        // What VS Code still holds lands while the gesture waits: a column typed into the schema by hand.
        Session(notebook).Settle = () =>
        {
            SchemaBlock(notebook).Source = typed;

            return Task.CompletedTask;
        };

        var sent = await notebook.GestureAsync(SchemaBlock(notebook), tick, "true");

        Assert.False(sent.StateChanged);
        Assert.Equal(typed, SchemaBlock(notebook).Source);
        Assert.DoesNotContain("\"sex\"", SchemaBlock(notebook).Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task APickOrAView_ChangesNothing_AndDoesNotWait()
    {
        await using var notebook = await NotebookAsync();
        var waited = 0;

        Session(notebook).Settle = () =>
        {
            waited++;

            return Task.CompletedTask;
        };

        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Show);
        await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);
        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).SelectOf(StepRenderer.ListType)!.Value.Action, "target.distribution");
        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Action, "range");
        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).SelectOf(StepRenderer.ListRangeKind, "include")!.Value.Action, "integer");

        Assert.Equal(0, waited);

        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row("pclass").Kind.Action, "category");

        Assert.Equal(1, waited);
    }

    [Fact]
    public async Task TheWait_IsLongerThanTheQuarterSecondVSCodeHoldsAKeystroke()
    {
        var clock = Stopwatch.StartNew();

        await new NotebookSession().Settle();

        Assert.True(clock.Elapsed >= TimeSpan.FromMilliseconds(250), $"waited {clock.ElapsedMilliseconds} ms");
    }
}
