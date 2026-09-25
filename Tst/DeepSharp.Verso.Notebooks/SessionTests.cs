// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook's session owns what only holds between calls: which block shows what, and whether a view there
/// still says what the blocks make; that one gesture on a notebook runs at a time, so a request is taken by the run
/// its own gesture started; and that a fit is handed on only for the steps the blocks declare.
/// </summary>
public sealed class SessionTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-session-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public SessionTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    [Fact]
    public void ForgettingStaleViews_ForgetsExactlyThoseWhoseStepsOrOffersChanged_AndSaysWhich()
    {
        CellModel[] cells = [.. Titanic.Select(Block)];
        var before = NotebookPipeline.Of(cells);
        var session = new NotebookSession();
        string[] sourceColumns = ["survived", "pclass", "sex", "age", "fare"];

        session.Showing(cells[0].Id, before.ViewKeyOf(cells[0].Id)!, DataGrid.HeaderOf(before.Readable, sourceColumns));
        session.Showing(cells[4].Id, before.ViewKeyOf(cells[4].Id)!, DataGrid.HeaderOf(before.Readable, ["survived", "pclass", "age", "fare", "age_was_missing"]));

        // The fill below the split changes: the source's rows are the same, and so are the offers their grid made.
        cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";

        Assert.Equal([cells[4].Id], session.ForgetStale(NotebookPipeline.Of(cells), except: null));
        Assert.Equal([cells[0].Id], session.Shown.Keys);

        // Age is dropped at the end: the source's rows are the same, but its grid offered to exclude age.
        cells[4].Source = """{"step": "drop.columns", "columns": ["age"]}""";

        Assert.Equal([cells[0].Id], session.ForgetStale(NotebookPipeline.Of(cells), except: null));
        Assert.Empty(session.Shown);

        // The block a gesture was made on is left to that gesture.
        session.Showing(cells[1].Id, "an old key", DataGrid.HeaderOf(before.Readable, []));

        Assert.Empty(session.ForgetStale(NotebookPipeline.Of(cells), except: cells[1].Id));
    }

    [Fact]
    public async Task GesturesOnOneNotebook_RunOneAtATime_InTheOrderTheyCame()
    {
        var session = new NotebookSession();
        var mayFinish = new TaskCompletionSource();
        var order = new List<string>();

        var first = session.OneAtATimeAsync(async () =>
        {
            order.Add("first begins");
            await mayFinish.Task;
            order.Add("first ends");

            return (string?)"first";
        });
        var second = session.OneAtATimeAsync(() =>
        {
            order.Add("second");

            return Task.FromResult<string?>("second");
        });

        Assert.Equal(["first begins"], order);
        Assert.False(second.IsCompleted);

        mayFinish.SetResult();

        Assert.Equal(["first", "second"], await Task.WhenAll(first, second));
        Assert.Equal(["first begins", "first ends", "second"], order);
    }

    [Fact]
    public async Task AGestureMadeWhileAnotherRuns_WaitsForIt_BeforeTouchingTheNotebook()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var declare = notebook.Scaffold.Cells[1];
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var mayFinish = new TaskCompletionSource();
        var running = session.OneAtATimeAsync(async () =>
        {
            await mayFinish.Task;

            return (string?)null;
        });

        var waiting = notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.False(waiting.IsCompleted);
        Assert.Empty(declare.Outputs);

        mayFinish.SetResult();
        await running;
        await waiting;

        Assert.True(declare.Outputs[^1].Content.Heads("fare"));
    }

    [Fact]
    public async Task AFitOfStepsTheBlocksNoLongerDeclare_IsNeverHandedOver()
    {
        // A run asked for one declaration, finished after the blocks became another: what it learned is not theirs.
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var last = notebook.Scaffold.Cells[^1];
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var asked = NotebookPipeline.Of(notebook.Scaffold.Cells);

        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        session.Publish(NotebookPipeline.Of(notebook.Scaffold.Cells));
        session.Request(last.Id, asked.RequestFor(last.Id, ViewTrigger.Run, page: 0));

        await notebook.RunAsync(last);

        Assert.False(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out var handed) && handed!.Contains("\"fitted\"", StringComparison.Ordinal));
        Assert.Equal(0, session.RunsFitted);
    }
}
