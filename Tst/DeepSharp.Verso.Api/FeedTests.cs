// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Net;
using System.Text.RegularExpressions;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What a view of an open notebook is told: the notebook as it stood when the view began, and then each change after it,
/// numbered, cell by cell — whatever made the change, a click, a clear or a form, for none of which the engine says a word.
/// What a C# cell displays while it runs reaches every view before the run ends, with the cell that runs and since when.
/// A view that stops reading holds one change, the latest of each cell, and never holds the notebook up.
/// </summary>
public sealed partial class FeedTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-feed-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public FeedTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // The notebook's own layout, which lets a person do everything to its cells.
    private static readonly HostedLayout InTheNotebook = new("notebook", (LayoutAllows)255);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static string Fill(int round) => $$"""{"step": "fill.missing", "column": "age", "with": "{{(round % 2 == 0 ? "mean" : "median")}}", "round": {{round}}}""";

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, NotebookModel notebook)
    {
        var path = Path.Join(_folder, name);

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
    }

    private Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        return OpenAsync(notebooks, name, notebook);
    }

    // The next change a view is told of; a view told nothing for half a minute fails the test rather than hang it.
    private static async Task<NotebookChange> NextAsync(NotebookSubscription view)
    {
        using var patience = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        patience.CancelAfter(TimeSpan.FromSeconds(30));

        await using var changes = view.ReadAllAsync(patience.Token).GetAsyncEnumerator(patience.Token);

        Assert.True(await changes.MoveNextAsync());

        return changes.Current;
    }

    // What a control a block drew carries, read the way a browser reads the attribute.
    private static string ActionOf(HostedCell cell, Func<string, bool> which) =>
        cell.Outputs.SelectMany(output => Action().Matches(output.Content)).Select(match => WebUtility.HtmlDecode(match.Groups[1].Value)).First(which);

    [GeneratedRegex("data-action=\"([^\"]*)\"")]
    private static partial Regex Action();

    [Fact]
    public async Task AViewThatBegins_IsGivenTheNotebookAsItStands_ThenEachChangeAfterIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        using var view = host.Subscribe();

        Assert.Equal(host.Current, view.Snapshot);
        Assert.Equal(0, view.Snapshot.Version);
        Assert.Equal(Titanic, view.Snapshot.Cells.Select(cell => cell.Source));
        Assert.Null(view.Snapshot.Running);

        await host.EditAsync(host.Cells[3].Id, Fill(1));

        var change = await NextAsync(view);

        Assert.Equal(1, change.Version);
        Assert.Equal(host.Current.Version, change.Version);
        Assert.Null(change.Order);
        Assert.Equal(Fill(1), Assert.Single(change.Cells).Source);
        Assert.Null(change.Running);
    }

    [Fact]
    public async Task AClickThatRewritesABlock_PutsTheNewCellInTheOldOnesPlace_AndTheOldOneIsGone()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var read = host.Cells[0].Id;
        var declare = host.Cells[1].Id;
        var card = await host.RunAsync(read);

        await host.GestureAsync(new HostedGesture(read, StepRenderer.Id, ActionOf(card, action => action == "deepsharp.show"), ""));

        var grid = host.Cells[0];
        using var view = host.Subscribe();

        await host.GestureAsync(new HostedGesture(
            read,
            StepRenderer.Id,
            ActionOf(grid, action => action.StartsWith("deepsharp.include ", StringComparison.Ordinal) && action.Contains("\"deck\"", StringComparison.Ordinal)),
            "true"));

        var change = await NextAsync(view);

        Assert.NotNull(change.Order);
        Assert.Equal(host.Cells.Select(cell => cell.Id), change.Order);
        Assert.DoesNotContain(declare, change.Order);
        Assert.Contains(change.Cells, cell => cell.Id == change.Order[1] && cell.Source.Contains("\"deck\"", StringComparison.Ordinal));
        Assert.Contains(change.Cells, cell => cell.Id == read);
    }

    [Fact]
    public async Task AClear_IsTold_ThoughTheEngineSaysNothingOfIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block), CSharp("1 + 1")]);

        await host.RunToolbarAsync("verso.action.run-all");

        using var view = host.Subscribe();

        await host.RunToolbarAsync("verso.action.clear-outputs");

        var change = await NextAsync(view);

        Assert.Equal(host.Cells.Count, change.Cells.Count);
        Assert.All(change.Cells, cell => Assert.Empty(cell.Outputs));
    }

    [Fact]
    public async Task AFormChange_AndHowACellIsShown_AreTold()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var normalise = host.Cells[4].Id;

        using var view = host.Subscribe();

        await host.SetPropertyAsync(normalise, StepForm.Id, "scale", "minmax");

        var formed = Assert.Single((await NextAsync(view)).Cells);

        Assert.Equal(normalise, formed.Id);
        Assert.Contains("\"minmax\"", formed.Source, StringComparison.Ordinal);

        await host.SetPropertyAsync(normalise, "verso.propertyprovider.display", "outputVisibility", "hidden");

        var shown = Assert.Single((await NextAsync(view)).Cells);

        Assert.Equal(normalise, shown.Id);
        Assert.Equal("\"hidden\"", shown.Metadata["verso:ui.outputVisibility"]);
    }

    [Fact]
    public async Task WhatACSharpCellDisplaysWhileItRuns_ReachesAViewBeforeTheRunEnds_WithTheCellThatRunsAndSinceWhen()
    {
        var go = Path.Join(_folder, "go");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "display.verso",
            CSharp($$"""Verso.Abstractions.DisplayExtensions.Display("step 0"); while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); } Verso.Abstractions.DisplayExtensions.Display("step 1");"""));
        var cell = host.Cells[0].Id;

        using var view = host.Subscribe();

        var before = DateTimeOffset.UtcNow;
        var running = host.RunAsync(cell);
        NotebookChange seen;

        do
        {
            seen = await NextAsync(view);
        }
        while (!seen.Cells.Any(each => each.Id == cell && each.Outputs.Any(output => output.Content.Contains("step 0", StringComparison.Ordinal))));

        Assert.False(running.IsCompleted);
        Assert.Equal(cell, seen.Running?.Cell);

        // A view that begins while the cell runs is told which cell runs, and since when.
        using var late = host.Subscribe();
        var since = late.Snapshot.Running!.Value.Since;

        Assert.Equal(cell, late.Snapshot.Running.Value.Cell);
        Assert.InRange(since, before, DateTimeOffset.UtcNow);

        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);

        var ran = await running;

        Assert.Contains("step 1", string.Concat(ran.Outputs.Select(output => output.Content)), StringComparison.Ordinal);
        Assert.Null(host.Current.Running);
        Assert.Equal(1, ran.ExecutionCount);
        Assert.Equal("Success", ran.LastStatus);
        Assert.NotNull(ran.LastElapsed);
    }

    [Fact]
    public async Task WhatABackgroundTaskOfACellDisplaysAfterItsRun_IsToldToo()
    {
        var go = Path.Join(_folder, "go");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "later.verso",
            CSharp($$"""_ = System.Threading.Tasks.Task.Run(async () => { while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); } Verso.Abstractions.DisplayExtensions.Display("later"); });"""));
        var cell = host.Cells[0].Id;

        await host.RunAsync(cell);

        using var view = host.Subscribe();

        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);

        var change = await NextAsync(view);

        Assert.Contains("later", string.Concat(Assert.Single(change.Cells).Outputs.Select(output => output.Content)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AViewThatStopsReading_HoldsOneChange_TheLatestOfEachCell()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var fill = host.Cells[3].Id;
        var normalise = host.Cells[4].Id;

        using var view = host.Subscribe();

        for (var round = 0; round < 20; round++)
        {
            await host.EditAsync(fill, Fill(round));
        }

        await host.EditAsync(normalise, Titanic[4].Replace("standard", "minmax", StringComparison.Ordinal));

        var held = await NextAsync(view);

        Assert.Equal(21, held.Version);
        Assert.Equal(host.Current.Version, held.Version);
        Assert.Equal(2, held.Cells.Count);
        Assert.Equal(Fill(19), held.Cells.Single(cell => cell.Id == fill).Source);
        Assert.Contains("minmax", held.Cells.Single(cell => cell.Id == normalise).Source, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WhatIsRefused_OrOnlyLooks_IsNoNewVersion()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var before = host.Current;

        await Assert.ThrowsAsync<CellGoneException>(() => host.EditAsync(Guid.NewGuid(), Titanic[0]));
        await host.ToolbarAsync();
        await host.PropertiesAsync(host.Cells[4].Id);

        Assert.Equal(before, host.Current);
        Assert.Equal(0, host.Current.Version);
    }

    [Fact]
    public async Task ClosingTheNotebooks_EndsEveryView_AndBeginsNoMore()
    {
        var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var view = host.Subscribe();

        await notebooks.DisposeAsync();

        await foreach (var change in view.ReadAllAsync(TestContext.Current.CancellationToken))
        {
            Assert.Fail($"A closed notebook told a view of version {change.Version}.");
        }

        Assert.Throws<ObjectDisposedException>(host.Subscribe);
    }

    [Fact]
    public async Task AViewThatEnds_IsToldNothingMore_AndTheOthersGoOn()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var leaving = host.Subscribe();

        using var staying = host.Subscribe();

        leaving.Dispose();

        await host.EditAsync(host.Cells[3].Id, Fill(1));

        await foreach (var change in leaving.ReadAllAsync(TestContext.Current.CancellationToken))
        {
            Assert.Fail($"A view that ended was told of version {change.Version}.");
        }

        Assert.Equal(1, (await NextAsync(staying)).Version);
    }

    [Fact]
    public void ChangesTakenTogether_HoldTheLatestOfEachCell_OnlyCellsStillStanding_AndTheLatestRun()
    {
        var a = new HostedCell(Guid.NewGuid(), "code", "csharp", "a", [], new Dictionary<string, string>(), null, null, null);
        var b = a with { Id = Guid.NewGuid(), Source = "b" };
        var c = a with { Id = Guid.NewGuid(), Source = "c" };
        var run = new HostedRun(a.Id, DateTimeOffset.UnixEpoch);

        var both = new NotebookChange(1, [a.Id, b.Id], [a, b], null, InTheNotebook).Then(new NotebookChange(2, null, [a with { Source = "a2" }], run, InTheNotebook));

        Assert.Equal(new NotebookChange(2, [a.Id, b.Id], [b, a with { Source = "a2" }], run, InTheNotebook), both);

        var all = both.Then(new NotebookChange(3, [c.Id, a.Id], [c], null, InTheNotebook));

        Assert.Equal(3, all.Version);
        Assert.Equal([c.Id, a.Id], all.Order);
        Assert.Equal([a with { Source = "a2" }, c], all.Cells);
        Assert.Null(all.Running);

        // Where neither says the order changed, nothing is left out.
        Assert.Equal(new NotebookChange(5, null, [a, b], null, InTheNotebook), new NotebookChange(4, null, [a], null, InTheNotebook).Then(new NotebookChange(5, null, [b], null, InTheNotebook)));
    }

    [Fact]
    public void ACopyOfWhatACellShows_TakenWhileARunAddsToIt_IsWholeOrNone()
    {
        var outputs = new List<CellOutput>();
        var written = new CellOutput("text/plain", "x");
        var writing = true;

        // As a run that shows more and more does: the list is cleared, and grows its storage again as it fills.
        var writer = new Thread(() =>
        {
            while (Volatile.Read(ref writing))
            {
                outputs.Clear();
                outputs.TrimExcess();

                for (var i = 0; i < 64; i++)
                {
                    outputs.Add(written);
                }
            }
        });

        writer.Start();

        var whole = 0;
        var none = 0;
        var broken = 0;
        var clock = Stopwatch.StartNew();

        while (clock.Elapsed < TimeSpan.FromSeconds(10) && (clock.Elapsed < TimeSpan.FromMilliseconds(300) || whole == 0 || none == 0))
        {
            if (outputs.Shown() is { } shown)
            {
                whole++;
                broken += shown.Count(output => output.Content != "x");
            }
            else
            {
                none++;
            }
        }

        Volatile.Write(ref writing, false);
        writer.Join();

        Assert.True(whole > 0 && none > 0, $"{whole} whole, {none} none");
        Assert.Equal(0, broken);
    }

    [Fact]
    public async Task ACellCaughtHalfWritten_KeepsWhatItShowedBefore()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "sum.verso", CSharp("1 + 1"));
        var cell = host.Cells[0].Id;
        var shown = (await host.RunAsync(cell)).Outputs;

        // A place counted and not yet filled, as a run adding to the list leaves it for a moment.
        host.Scaffold.GetCell(cell)!.Outputs.Add(null!);

        var edited = await host.EditAsync(cell, "1 + 2");

        Assert.Equal("1 + 2", edited.Source);
        Assert.Equal(shown, edited.Outputs);
        Assert.NotEmpty(shown);
    }

    [Fact]
    public async Task WhatAPartKeepsWithACell_ThatJsonCannotWrite_IsToldAsItsText()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var fill = host.Cells[3].Id;

        // A part may keep anything with a cell; a type is one thing JSON refuses to write.
        host.Scaffold.GetCell(fill)!.Metadata["part.kept"] = typeof(int);

        var edited = await host.EditAsync(fill, Fill(1));

        Assert.Equal("System.Int32", edited.Metadata["part.kept"]);
    }

    [Fact]
    public void TwoLooksAtAVersionOrAChange_AreEqual_WhileTheyHoldTheSame()
    {
        var cell = new HostedCell(Guid.NewGuid(), "code", "csharp", "1 + 1", [new HostedOutput("text/plain", "2", IsError: false)], new Dictionary<string, string>(), 1, "Success", TimeSpan.FromMilliseconds(5));
        var run = new HostedRun(cell.Id, DateTimeOffset.UnixEpoch);
        var version = new NotebookVersion(3, [cell], run, InTheNotebook);
        var change = new NotebookChange(3, [cell.Id], [cell], run, InTheNotebook);

        Assert.Equal(version, version with { Cells = [cell with { Outputs = [new HostedOutput("text/plain", "2", IsError: false)] }] });
        Assert.Equal(version.GetHashCode(), (version with { Cells = [cell] }).GetHashCode());
        Assert.NotEqual(version, version with { Version = 4 });
        Assert.NotEqual(version, version with { Cells = [] });
        Assert.NotEqual(version, version with { Running = null });
        Assert.NotEqual(version, version with { Layout = new HostedLayout("dashboard", LayoutAllows.CellResize | LayoutAllows.CellExecute) });

        Assert.Equal(change, change with { Order = [cell.Id], Cells = [cell] });
        Assert.Equal(change.GetHashCode(), (change with { Order = [cell.Id] }).GetHashCode());
        Assert.Equal(change with { Order = null }, change with { Order = null });
        Assert.NotEqual(change, change with { Version = 4 });
        Assert.NotEqual(change, change with { Order = null });
        Assert.NotEqual(change with { Order = null }, change);
        Assert.NotEqual(change, change with { Order = [] });
        Assert.NotEqual(change, change with { Cells = [] });
        Assert.NotEqual(change, change with { Running = null });
        Assert.NotEqual(change, change with { Layout = new HostedLayout(null, (LayoutAllows)127) });
    }
}
