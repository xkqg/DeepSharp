// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.RegularExpressions;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What a person does to an open notebook reaches it through its host, one thing at a time and in the order it came:
/// typing a cell's text, running a cell, a click on a control a block drew, stopping a run. A click on a cell that a
/// change before it replaced is refused, since nothing it meant stands any more. What a part answers a click with is
/// shown by the cell the click was made on. A C# run that never ends is stopped the only way the engine can stop it,
/// by restarting its kernel. And C# runs take their turn one at a time across every notebook the application has
/// open, because a C# kernel takes over the process's console while it runs.
/// </summary>
public sealed partial class VerbTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-verbs-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public VerbTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

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

    // What a control a block drew carries, read the way a browser reads the attribute.
    private static string ActionOf(HostedCell cell, Func<string, bool> which) =>
        cell.Outputs.SelectMany(output => Action().Matches(output.Content)).Select(match => WebUtility.HtmlDecode(match.Groups[1].Value)).First(which);

    [GeneratedRegex("data-action=\"([^\"]*)\"")]
    private static partial Regex Action();

    private static string IncludeBoxOf(HostedCell cell, string column) =>
        ActionOf(cell, action => action.StartsWith("deepsharp.include ", StringComparison.Ordinal) && action.Contains($"\"{column}\"", StringComparison.Ordinal));

    private static DeclareStep Declared(NotebookHost host) =>
        host.Cells.Select(cell => StepCatalog.BuiltIn().ReadStep(cell.Source)).OfType<DeclareStep>().Single();

    // The grid under the source's block, as a person gets it: run the block, then press the card's button.
    private static async Task<HostedCell> ShownAsync(NotebookHost host, Guid read)
    {
        var card = await host.RunAsync(read);

        await host.GestureAsync(new HostedGesture(read, StepRenderer.Id, ActionOf(card, action => action == "deepsharp.show"), ""));

        return host.Cells.Single(cell => cell.Id == read);
    }

    [Fact]
    public async Task TypingACellsText_AndRunningIt_ShowWhatTheCellShows()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var fill = host.Cells[3];
        const string mean = """{"step": "fill.missing", "column": "age", "with": "mean"}""";

        var typed = await host.EditAsync(fill.Id, mean);

        Assert.Equal(mean, typed.Source);
        Assert.Equal(mean, host.Cells[3].Source);
        Assert.Empty(typed.Outputs);

        var ran = await host.RunAsync(fill.Id);

        Assert.Equal(fill.Id, ran.Id);
        Assert.False(Assert.Single(ran.Outputs).IsError);
    }

    [Fact]
    public async Task AClickOnABlocksControl_IsHandedToThePartItNames_AndSaysWhetherTheBlocksChanged()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var read = host.Cells[0].Id;

        var grid = await ShownAsync(host, read);
        var ticked = await host.GestureAsync(new HostedGesture(read, StepRenderer.Id, IncludeBoxOf(grid, "deck"), "true"));

        Assert.True(ticked.StateChanged);
        Assert.Null(ticked.Answer);
        Assert.Contains(Declared(host).Columns, column => column.Name == "deck" && !column.Excluded);
    }

    [Fact]
    public async Task WhatReachesANotebook_TakesItsTurn_InTheOrderItCame()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "mixed.verso", CSharp("await System.Threading.Tasks.Task.Delay(600);"), Block(Titanic[0]));
        var other = Titanic[0].Replace("titanic.csv", "other.csv", StringComparison.Ordinal);

        var running = host.RunAsync(host.Cells[0].Id);
        var typing = host.EditAsync(host.Cells[1].Id, other);

        Assert.False(typing.IsCompleted);

        await running;

        Assert.Equal(other, (await typing).Source);
    }

    [Fact]
    public async Task ACellAChangeBeforeItReplaced_IsGone_AndAVerbOnItChangesNothing()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var read = host.Cells[0].Id;
        var declare = host.Cells[1].Id;
        var grid = await ShownAsync(host, read);

        await host.GestureAsync(new HostedGesture(read, StepRenderer.Id, IncludeBoxOf(grid, "deck"), "true"));

        var standing = host.Cells.Select(cell => (cell.Id, cell.Source)).ToArray();
        var gone = await Assert.ThrowsAsync<CellGoneException>(() => host.GestureAsync(new HostedGesture(declare, StepRenderer.Id, "deepsharp.show", "")));

        Assert.Equal(declare, gone.Cell);
        Assert.Equal(host.Current.Version, gone.Version);
        Assert.DoesNotContain(host.Current.Cells, cell => cell.Id == declare);
        await Assert.ThrowsAsync<CellGoneException>(() => host.EditAsync(declare, Titanic[1]));
        await Assert.ThrowsAsync<CellGoneException>(() => host.RunAsync(declare));
        Assert.Equal(standing, host.Cells.Select(cell => (cell.Id, cell.Source)));
    }

    [Fact]
    public async Task WhatAPartAnswersAClickWith_IsShownByTheCellTheClickWasMadeOn()
    {
        var notebook = new NotebookModel
        {
            Parameters = new() { ["title"] = new NotebookParameterDefinition { Type = "string", Default = "Titanic" } },
        };

        notebook.Cells.Add(new CellModel { Type = "parameters", Source = "" });

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "parameters.verso", notebook);
        var parameters = host.Cells[0].Id;

        var answered = await host.GestureAsync(new HostedGesture(parameters, "verso.renderer.parameters", "parameter-update", """{"name": "title", "value": "Lusitania"}"""));

        Assert.True(answered.StateChanged);
        Assert.Contains("Lusitania", answered.Answer, StringComparison.Ordinal);
        Assert.Equal(answered.Answer, Assert.Single(host.Cells[0].Outputs).Content);
    }

    [Fact]
    public async Task ACSharpRunThatNeverEnds_IsStoppedByRestartingItsKernel_AndTheNotebookGoesOn()
    {
        var started = Path.Join(_folder, "started");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "runaway.verso",
            CSharp($$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }"""),
            CSharp("1 + 1"));
        var runaway = host.Cells[0].Id;

        var running = host.RunAsync(runaway);

        // The run is under way once it has written the file; nothing but a stop ends it.
        while (!File.Exists(started))
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        host.Stop();
        await running;

        Assert.Contains("2", string.Concat((await host.RunAsync(host.Cells[1].Id)).Outputs.Select(output => output.Content)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task StoppingWhenNothingRuns_ChangesNothing()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var before = host.Cells;

        host.Stop();

        Assert.Equal(before, host.Cells);
        Assert.False(Assert.Single((await host.RunAsync(host.Cells[0].Id)).Outputs).IsError);
    }

    [Fact]
    public async Task AClickNamingNoPartThatAnswersClicks_IsRefused()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(
            () => host.GestureAsync(new HostedGesture(host.Cells[0].Id, "no.such.part", "deepsharp.show", "")));

        Assert.Contains("no.such.part", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CSharpRunsInTwoNotebooks_TakeTheirTurnsOneAtATime_SoNeitherPrintsIntoTheOther()
    {
        const string printing = """for (var i = 0; i < 20; i++) { System.Console.Write("{0}"); await System.Threading.Tasks.Task.Delay(5); }""";

        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(printing.Replace("{0}", "A", StringComparison.Ordinal)));
        var b = await OpenAsync(notebooks, "b.verso", CSharp(printing.Replace("{0}", "B", StringComparison.Ordinal)));

        var ran = await Task.WhenAll(a.RunAsync(a.Cells[0].Id), b.RunAsync(b.Cells[0].Id));
        var printedByA = string.Concat(ran[0].Outputs.Select(output => output.Content));
        var printedByB = string.Concat(ran[1].Outputs.Select(output => output.Content));

        Assert.Equal(20, printedByA.Count(letter => letter == 'A'));
        Assert.DoesNotContain("B", printedByA, StringComparison.Ordinal);
        Assert.Equal(20, printedByB.Count(letter => letter == 'B'));
        Assert.DoesNotContain("A", printedByB, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CSharpRunsOfTwoSeparateNotebookHolders_TakeTheirTurnsToo_SinceTheConsoleIsTheWholeProcesss()
    {
        const string printing = """for (var i = 0; i < 20; i++) { System.Console.Write("{0}"); await System.Threading.Tasks.Task.Delay(5); }""";

        await using var these = new OpenNotebooks();
        await using var those = new OpenNotebooks();
        var a = await OpenAsync(these, "a.verso", CSharp(printing.Replace("{0}", "A", StringComparison.Ordinal)));
        var b = await OpenAsync(those, "b.verso", CSharp(printing.Replace("{0}", "B", StringComparison.Ordinal)));

        var ran = await Task.WhenAll(a.RunAsync(a.Cells[0].Id), b.RunAsync(b.Cells[0].Id));
        var printedByA = string.Concat(ran[0].Outputs.Select(output => output.Content));
        var printedByB = string.Concat(ran[1].Outputs.Select(output => output.Content));

        Assert.Equal(20, printedByA.Count(letter => letter == 'A'));
        Assert.DoesNotContain("B", printedByA, StringComparison.Ordinal);
        Assert.Equal(20, printedByB.Count(letter => letter == 'B'));
        Assert.DoesNotContain("A", printedByB, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ClosingTheNotebooks_LetsTheChangeUnderWayFinish_AndRefusesTheNext()
    {
        var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var read = host.Cells[0].Id;
        var grid = await ShownAsync(host, read);

        // The tick waits three tenths of a second before it reads the blocks; the close comes first.
        var ticking = host.GestureAsync(new HostedGesture(read, StepRenderer.Id, IncludeBoxOf(grid, "deck"), "true"));

        await notebooks.DisposeAsync();

        Assert.True((await ticking).StateChanged);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => host.RunAsync(read));
    }
}
