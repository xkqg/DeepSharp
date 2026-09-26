// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// The toolbar of an open notebook: every button the engine has — Verso's own and DeepSharp's — each saying whether
/// it can be pressed now, and a press that takes its turn like anything else done to the notebook. A file a button
/// hands over goes to whoever pressed it, and nothing is written beside the notebook. A button that may run C# cells
/// takes the application's C# turn, so it prints into no other notebook.
/// </summary>
public sealed class ToolbarTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-toolbar-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public ToolbarTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        var path = Path.Join(_folder, name);

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TheToolbar_OffersEveryButtonTheEngineHas_EachSayingWhetherItCanBePressedNow()
    {
        await using var notebooks = new OpenNotebooks();
        var whole = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var broken = await OpenAsync(notebooks, "broken.verso", Block(Titanic[0]), Block("""{"step": "declare"}"""));

        var buttons = await whole.ToolbarAsync();
        var export = buttons.Single(button => button.Id == ExportPipelineAction.Id);

        Assert.Equal(ToolbarPlace.ExportMenu, export.Place);
        Assert.Equal("Export the pipeline", export.Label);
        Assert.True(export.IsEnabled);
        Assert.Equal(ToolbarPlace.MainToolbar, buttons.Single(button => button.Id == RunPipelineAction.Id).Place);
        Assert.Contains(buttons, button => button.Id == "verso.action.run-all");
        Assert.False((await broken.ToolbarAsync()).Single(button => button.Id == ExportPipelineAction.Id).IsEnabled);
    }

    [Fact]
    public async Task AButtonOnACellsToolbar_SaysWhichCellsItCanBePressedFor_AskedForEachCellAsVersosEditorAsksIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "cells.verso", CSharp("1 + 1"), CSharp("2 + 2"));
        var (shown, silent) = (host.Cells[0].Id, host.Cells[1].Id);

        await host.RunAsync(shown);

        var buttons = await host.ToolbarAsync();

        // Running a cell asks only that the layout lets cells run; clearing one, that it shows something.
        Assert.Equal([shown, silent], buttons.Single(button => button.Id == "verso.action.run-cell").EnabledFor);
        Assert.Equal([shown], buttons.Single(button => button.Id == "verso.action.clear-cell-output").EnabledFor);
        Assert.Empty(buttons.Single(button => button.Id == "verso.action.run-all").EnabledFor);
    }

    [Fact]
    public void TwoLooksAtAButton_AreEqual_WhileTheyHoldTheSame()
    {
        var cell = Guid.NewGuid();
        var button = new HostedToolbarAction("id", "Label", null, IconOnly: false, IsPrimary: false, null, ToolbarPlace.CellToolbar, 1, IsEnabled: true, [cell]);

        Assert.Equal(button, button with { EnabledFor = [cell] });
        Assert.Equal(button.GetHashCode(), (button with { EnabledFor = [cell] }).GetHashCode());
        Assert.NotEqual(button, button with { EnabledFor = [] });
        Assert.NotEqual(button, button with { IsEnabled = false });
        Assert.NotEqual(button, button with { Label = "Other" });
    }

    [Fact]
    public async Task AFileAButtonHandsOver_GoesToWhoeverPressedIt_AndNothingIsWrittenBesideTheNotebook()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        var file = await host.RunToolbarAsync(ExportPipelineAction.Id);

        Assert.NotNull(file);
        Assert.Equal("titanic.pipeline.json", file.Value.Name);
        Assert.Equal("application/json", file.Value.ContentType);
        Assert.Contains("\"normalise\"", Encoding.UTF8.GetString(file.Value.Bytes), StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Join(_folder, "titanic.pipeline.json")));
    }

    [Fact]
    public async Task AButtonThatHandsNothingOver_DoesWhatItDoes_AndReturnsNoFile()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        Assert.Null(await host.RunToolbarAsync(RunPipelineAction.Id));
        Assert.NotEmpty(host.Cells[^1].Outputs);
    }

    [Fact]
    public async Task AButtonTheEngineDoesNotHave_IsRefused()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => host.RunToolbarAsync("no.such.button"));

        Assert.Contains("no.such.button", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AButtonThatRunsCSharpCells_TakesTheApplicationsCSharpTurn_SoNothingPrintsIntoAnotherNotebook()
    {
        const string printing = """for (var i = 0; i < 20; i++) { System.Console.Write("{0}"); await System.Threading.Tasks.Task.Delay(5); }""";

        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", CSharp(printing.Replace("{0}", "A", StringComparison.Ordinal)));
        var b = await OpenAsync(notebooks, "b.verso", CSharp(printing.Replace("{0}", "B", StringComparison.Ordinal)));

        await Task.WhenAll(a.RunAsync(a.Cells[0].Id), b.RunToolbarAsync("verso.action.run-all"));

        var printedByA = string.Concat(a.Cells[0].Outputs.Select(output => output.Content));
        var printedByB = string.Concat(b.Cells[0].Outputs.Select(output => output.Content));

        Assert.Equal(20, printedByA.Count(letter => letter == 'A'));
        Assert.DoesNotContain("B", printedByA, StringComparison.Ordinal);
        Assert.Equal(20, printedByB.Count(letter => letter == 'B'));
        Assert.DoesNotContain("A", printedByB, StringComparison.Ordinal);
    }

    [Fact]
    public async Task EveryButtonTheEngineHas_CanBePressed_AndDoesWhatItSays()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block), CSharp("1 + 1")]);
        var csharp = host.Cells[^1].Id;

        await host.RunToolbarAsync("verso.action.run-all");

        Assert.All(host.Cells, cell => Assert.NotEmpty(cell.Outputs));

        await host.RunToolbarAsync("verso.action.clear-cell-output", csharp);

        Assert.Empty(host.Cells[^1].Outputs);
        Assert.NotEmpty(host.Cells[0].Outputs);

        await host.RunToolbarAsync("verso.action.run-cell", csharp);

        Assert.Contains("2", string.Concat(host.Cells[^1].Outputs.Select(output => output.Content)), StringComparison.Ordinal);

        await host.RunToolbarAsync("verso.action.clear-outputs");

        Assert.All(host.Cells, cell => Assert.Empty(cell.Outputs));

        // Verso names its own exports: a page after the notebook's title, "notebook" for one that has none, and a copy
        // of the notebook after its file.
        Assert.Equal(("notebook.html", "text/html"), Named(await host.RunToolbarAsync("verso.action.export-html")));
        Assert.Equal(("notebook.md", "text/markdown"), Named(await host.RunToolbarAsync("verso.action.export-markdown")));
        Assert.Equal(("titanic.verso", "application/octet-stream"), Named(await host.RunToolbarAsync("verso.action.export-verso")));

        await host.RunToolbarAsync(RunPipelineAction.Id);

        Assert.True(host.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));

        await host.RunToolbarAsync("verso.action.restart-kernel");

        Assert.False(host.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));

        var layout = host.Scaffold.NotebookOps.ActiveLayoutId;
        var theme = host.Scaffold.NotebookOps.ActiveThemeId;

        await host.RunToolbarAsync("verso.switchLayout");
        await host.RunToolbarAsync("verso.switchTheme");

        Assert.NotEqual(layout, host.Scaffold.NotebookOps.ActiveLayoutId);
        Assert.NotEqual(theme, host.Scaffold.NotebookOps.ActiveThemeId);

        static (string, string) Named(HostedFile? file) => (file!.Value.Name, file.Value.ContentType);
    }

    [Fact]
    public async Task AButtonsContext_HasNoCellOfItsOwnToWriteInto_AndNothingToCancel()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var context = new ToolbarContext(host.Scaffold, []);

        await context.WriteOutputAsync(new CellOutput("text/plain", "written"));

        Assert.Equal(CancellationToken.None, context.CancellationToken);
        Assert.All(host.Cells, cell => Assert.Empty(cell.Outputs));
        Assert.Null(context.Handed);
    }

    [Fact]
    public async Task AButtonWhoseRunNeverEnds_IsStoppedLikeACellsRun()
    {
        var started = Path.Join(_folder, "started");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks,
            "runaway.verso",
            CSharp($$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }"""));

        var pressing = host.RunToolbarAsync("verso.action.run-all");

        while (!File.Exists(started))
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        host.Stop();

        Assert.Null(await pressing);
    }

    [Fact]
    public async Task AButtonsRefusal_ReachesWhoeverPressedIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "broken.verso", Block(Titanic[0]), Block("""{"step": "declare"}"""));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => host.RunToolbarAsync(ExportPipelineAction.Id));

        Assert.Contains("block 2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoLooksAtAFile_AreEqual_WhileTheyHoldTheSame()
    {
        var file = new HostedFile("titanic.pipeline.json", "application/json", [1, 2, 3]);

        Assert.Equal(file, file with { Bytes = [1, 2, 3] });
        Assert.Equal(file.GetHashCode(), (file with { Bytes = [1, 2, 3] }).GetHashCode());
        Assert.NotEqual(file, file with { Name = "other.json" });
        Assert.NotEqual(file, file with { ContentType = "text/plain" });
        Assert.NotEqual(file, file with { Bytes = [1, 2, 4] });
    }
}
