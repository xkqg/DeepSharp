// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook's two buttons. "Run the pipeline" fits every step on the training rows and replays it, the one run
/// that learns, and hands C# cells the declaration with what it learned; "Export the pipeline" saves the pipeline
/// file, with what the fit learned only while it is the fit of the steps the blocks declare now. A C# cell reads the
/// pipeline afresh every time it runs, under a key no C# variable can have, and so never sees one the blocks no
/// longer make — as far as a part is told: an edit in the editor that was never run, or a block deleted or moved,
/// reaches no part until the notebook is read again.
/// </summary>
public sealed class ToolbarTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-toolbar-").FullName;

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

    private string NotebookPath => Path.Join(_folder, "titanic.verso");

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(NotebookPath);

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static T Action<T>(Notebook notebook)
        where T : IToolbarAction => notebook.Host.GetToolbarActions().OfType<T>().Single();

    private static string? HandedOver(Notebook notebook) =>
        notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out var pipeline) ? pipeline : null;

    [Fact]
    public async Task TheButtons_ArePartsVersoLoads_WhereAPersonLooksForThem()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var run = Action<RunPipelineAction>(notebook);
        var export = Action<ExportPipelineAction>(notebook);

        Assert.Equal(ToolbarPlacement.MainToolbar, run.Placement);
        Assert.Equal(ToolbarPlacement.ExportMenu, export.Placement);
        Assert.Equal(RunPipelineAction.Id, run.ActionId);
        Assert.Equal(ExportPipelineAction.Id, export.ActionId);
        Assert.Equal(run.ActionId, run.ExtensionId);
        Assert.Equal(export.ActionId, export.ExtensionId);

        foreach (IToolbarAction action in new IToolbarAction[] { run, export })
        {
            Assert.False(string.IsNullOrWhiteSpace(action.DisplayName));
            Assert.StartsWith("<svg", action.Icon, StringComparison.Ordinal);
            Assert.False(action.IconOnly);
            Assert.False(action.IsPrimary);
            Assert.Null(action.ConfirmationPrompt);
            Assert.Equal(0, action.Order);
        }

        Assert.False(string.IsNullOrWhiteSpace(run.Name));
        Assert.False(string.IsNullOrWhiteSpace(run.Description));
        Assert.False(string.IsNullOrWhiteSpace(export.Name));
        Assert.False(string.IsNullOrWhiteSpace(export.Description));
    }

    [Fact]
    public async Task RunningThePipeline_FitsEveryStep_ShowsTheLastBlock_AndHandsOverWhatItLearned()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var run = Action<RunPipelineAction>(notebook);

        Assert.True(await run.IsEnabledAsync(gesture));

        await run.ExecuteAsync(gesture);

        var handed = PreparedData.FromJson(HandedOver(notebook)!, NotebookVerbs.Catalog());
        var fresh = new Pipeline(handed.Declaration, rows: null, SourceFolder.OfDocument(NotebookPath)).Run();

        Assert.Equal(fresh.ToJson(), HandedOver(notebook));
        Assert.True(notebook.Scaffold.Cells[^1].Outputs.Count >= 2);
        Assert.True(notebook.Scaffold.Cells[^1].Outputs[1].Content.Heads("fare"));
    }

    [Fact]
    public async Task RunningANotebookThatMakesNoPipeline_SaysWhyAtTheBlockThatStopsIt_AndHandsNothingOver()
    {
        await using var notebook = await NotebookAsync(Titanic[0], Titanic[1], """{"step": "split.stratified", "column": "survived"}""", Titanic[3]);
        var gesture = new ToolbarGesture(notebook, NotebookPath);

        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);

        var stopping = notebook.Scaffold.Cells[2];

        Assert.Contains(stopping.Outputs, output => output.IsError);
        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ARunButtonVersoDidNotLoad_RefusesToRun_HavingNoNotebookToRun()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new RunPipelineAction().ExecuteAsync(new ToolbarGesture(notebook, NotebookPath)));

        Assert.Contains("not loaded by Verso", refused.Message, StringComparison.Ordinal);
        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ANotebookWithNoBlocks_HasNothingToRunOrExport()
    {
        await using var notebook = await NotebookAsync();
        var gesture = new ToolbarGesture(notebook, NotebookPath);

        Assert.False(await Action<RunPipelineAction>(notebook).IsEnabledAsync(gesture));
        Assert.False(await Action<ExportPipelineAction>(notebook).IsEnabledAsync(gesture));

        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);

        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task AGesture_KeepsWhatARunLearned_WhileTheBlocksDeclareTheSameSteps()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);

        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);
        var fitted = HandedOver(notebook);

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(fitted, HandedOver(notebook));

        // Another step: what was learned belongs to steps the blocks no longer declare.
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.NotEqual(fitted, HandedOver(notebook));
        Assert.DoesNotContain("\"fitted\"", HandedOver(notebook)!, StringComparison.Ordinal);

        // Whatever a C# cell wrote under the key is not taken for a pipeline.
        notebook.Scaffold.Variables.Set(StepKernel.HandOver, "not a pipeline");
        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(NotebookPipelineText(notebook), HandedOver(notebook));
    }

    private static string NotebookPipelineText(Notebook notebook) =>
        new PipelineDeclaration([.. notebook.Scaffold.Cells.Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))]).ToJson();

    [Fact]
    public async Task ARunTheRowsRefuse_LeavesNoFitHandedOverOrExported()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);

        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);

        Assert.Contains("\"fitted\"", HandedOver(notebook)!, StringComparison.Ordinal);

        // The file is gone: what was learned from it is no longer what these steps would learn.
        File.Delete(Path.Join(_folder, "titanic.csv"));
        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);
        await Action<ExportPipelineAction>(notebook).ExecuteAsync(gesture);

        Assert.DoesNotContain("\"fitted\"", HandedOver(notebook) ?? string.Empty, StringComparison.Ordinal);
        Assert.DoesNotContain("\"fitted\"", Encoding.UTF8.GetString(gesture.Downloads[^1].Data), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AShowOverOtherBytes_HandsOverTheDeclarationAlone()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var file = Path.Join(_folder, "titanic.csv");

        await Action<RunPipelineAction>(notebook).ExecuteAsync(new ToolbarGesture(notebook, NotebookPath));

        // Half the rows: the same steps would learn other numbers from these bytes.
        var lines = File.ReadAllLines(file);
        File.WriteAllLines(file, lines.Take((lines.Length / 2) + 1));
        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(NotebookPipelineText(notebook), HandedOver(notebook));
    }

    [Fact]
    public async Task TheToolbarRunTwice_OverTheSameStepsAndBytes_FitsOnce()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var run = Action<RunPipelineAction>(notebook);
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

        await run.ExecuteAsync(gesture);
        var fitted = HandedOver(notebook);
        await run.ExecuteAsync(gesture);

        Assert.Equal(1, session.RunsFitted);
        Assert.Equal(fitted, HandedOver(notebook));

        File.AppendAllText(Path.Join(_folder, "titanic.csv"), "1,1,female,30.0,0,0,80.0,S,First,woman,False,B,Southampton,yes,True\n");
        await run.ExecuteAsync(gesture);

        Assert.Equal(2, session.RunsFitted);
    }

    [Fact]
    public async Task AFitNoRunOfThisNotebookMade_IsNeverKeptOrExported()
    {
        // A C# cell can write anything under the key, a genuine fit included; only a fit this notebook's own run
        // made of these steps over these bytes is handed on.
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var elsewhere = new Pipeline(PipelineDeclaration.FromJson(NotebookPipelineText(notebook), NotebookVerbs.Catalog()), rows: null, SourceFolder.OfDocument(NotebookPath)).Run().ToJson();

        notebook.Scaffold.Variables.Set(StepKernel.HandOver, elsewhere);
        await Action<ExportPipelineAction>(notebook).ExecuteAsync(gesture);

        Assert.Equal(NotebookPipelineText(notebook), Encoding.UTF8.GetString(gesture.Downloads[^1].Data));

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(NotebookPipelineText(notebook), HandedOver(notebook));
    }

    [Fact]
    public async Task ExportingThePipeline_SavesTheDeclaration_NamedAfterTheNotebook()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var export = Action<ExportPipelineAction>(notebook);

        Assert.True(await export.IsEnabledAsync(gesture));

        await export.ExecuteAsync(gesture);

        var file = Assert.Single(gesture.Downloads);

        Assert.Equal("titanic.pipeline.json", file.FileName);
        Assert.Equal("application/json", file.ContentType);
        Assert.Equal(NotebookPipelineText(notebook), Encoding.UTF8.GetString(file.Data));
    }

    [Fact]
    public async Task ExportingAfterARun_SavesWhatTheFitLearned_OnlyWhileItIsTheFitOfTheStepsDeclaredNow()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var export = Action<ExportPipelineAction>(notebook);

        await Action<RunPipelineAction>(notebook).ExecuteAsync(gesture);
        await export.ExecuteAsync(gesture);

        var saved = Encoding.UTF8.GetString(gesture.Downloads[^1].Data);

        Assert.Equal(HandedOver(notebook), saved);
        Assert.NotNull(PreparedData.FromJson(saved, NotebookVerbs.Catalog()).Fitted);

        // A block changed by hand: the fit belongs to steps the notebook no longer declares.
        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        await export.ExecuteAsync(gesture);

        Assert.Equal(NotebookPipelineText(notebook), Encoding.UTF8.GetString(gesture.Downloads[^1].Data));
    }

    [Fact]
    public async Task APipelineWhoseRowsAreHandedIn_ExportsItsDeclaration_ForCodeThatHandsRowsIn()
    {
        // A notebook shows no rows of it, since it hands none in; the file is what code that does hand rows in runs.
        await using var notebook = await NotebookAsync("""{"step": "read.rows", "description": "passengers"}""", Titanic[1]);
        var gesture = new ToolbarGesture(notebook, NotebookPath);

        await Action<ExportPipelineAction>(notebook).ExecuteAsync(gesture);

        var saved = Encoding.UTF8.GetString(Assert.Single(gesture.Downloads).Data);

        Assert.Equal(NotebookPipelineText(notebook), saved);
        Assert.IsType<ReadRowsStep>(PipelineDeclaration.FromJson(saved, NotebookVerbs.Catalog()).Steps[0]);
    }

    [Fact]
    public async Task ANotebookNeverSaved_ExportsAPipelineFileOfTheUsualName()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock(Titanic[0].Replace("titanic.csv", Path.Join(_folder, "titanic.csv").Replace("\\", "\\\\", StringComparison.Ordinal), StringComparison.Ordinal));
        var gesture = new ToolbarGesture(notebook, filePath: null);

        await Action<ExportPipelineAction>(notebook).ExecuteAsync(gesture);

        Assert.Equal("pipeline.json", Assert.Single(gesture.Downloads).FileName);
    }

    [Fact]
    public async Task ExportingANotebookThatMakesNoPipeline_IsRefused_SayingWhichBlockStopsIt()
    {
        await using var notebook = await NotebookAsync(Titanic[0], """{"step": "declare"}""");
        var gesture = new ToolbarGesture(notebook, NotebookPath);
        var export = Action<ExportPipelineAction>(notebook);

        Assert.False(await export.IsEnabledAsync(gesture));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => export.ExecuteAsync(gesture));

        Assert.Contains("block 2", refused.Message, StringComparison.Ordinal);
        Assert.Empty(gesture.Downloads);
    }

    [Fact]
    public async Task ACSharpCell_SeesTheDeclarationWrittenAfterItsFirstRun()
    {
        // Verso declares a C# variable for every value a cell can name, once; the hand-over's key is no C# name, so a
        // C# cell reads it afresh every time and never sees the pipeline as it was the first time.
        await using var notebook = await NotebookAsync(Titanic);
        var csharp = notebook.Scaffold.AddCell(
            "code", "csharp", """Variables.TryGet<string>("deepsharp.pipeline", out var pipeline) ? pipeline : "none" """);

        async Task<string> ReadAsync() => string.Concat((await notebook.RunAsync(csharp)).Select(output => output.Content));

        Assert.Contains("none", await ReadAsync(), StringComparison.Ordinal);

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Contains("median", await ReadAsync(), StringComparison.Ordinal);

        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";
        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        var third = await ReadAsync();

        Assert.Contains("mean", third, StringComparison.Ordinal);
        Assert.DoesNotContain("median", third, StringComparison.Ordinal);
    }
}
