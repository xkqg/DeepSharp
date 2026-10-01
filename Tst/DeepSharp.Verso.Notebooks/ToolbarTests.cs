// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook's two buttons. "Run the pipeline" fits every step on the training rows and replays it, the one run
/// that learns, and hands C# cells the declaration with what it learned; "Export the pipeline" saves the pipeline
/// file, with what the fit learned only while it is the fit of the steps the blocks declare now. A C# cell reads the
/// pipeline afresh every time it runs, under a key no C# variable can have, and so never sees one the blocks no
/// longer make: an edit in the editor that was never run, or a block added, deleted or moved, is caught up with at the
/// next gesture, and at once when the host changed the cells itself.
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

    // What a file a button handed over holds, as text.
    private static string Text(HostedFile? file) => Encoding.UTF8.GetString(file!.Value.Bytes);

    [Fact]
    public async Task TheButtons_ArePartsVersoLoads_WhereAPersonLooksForThem()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var run = Action<RunPipelineAction>(notebook);
        var export = Action<ExportPipelineAction>(notebook);
        var takeOver = Action<TakeOverAction>(notebook);
        var restore = Action<RestoreBlocksAction>(notebook);

        Assert.Equal(ToolbarPlacement.MainToolbar, run.Placement);
        Assert.Equal(ToolbarPlacement.ExportMenu, export.Placement);
        Assert.Equal(ToolbarPlacement.MainToolbar, takeOver.Placement);
        Assert.Equal(ToolbarPlacement.MainToolbar, restore.Placement);
        Assert.Equal(RunPipelineAction.Id, run.ActionId);
        Assert.Equal(ExportPipelineAction.Id, export.ActionId);
        Assert.Equal(TakeOverAction.Id, takeOver.ActionId);
        Assert.Equal(RestoreBlocksAction.Id, restore.ActionId);
        Assert.Equal(run.ActionId, run.ExtensionId);
        Assert.Equal(export.ActionId, export.ExtensionId);
        Assert.Equal(takeOver.ActionId, takeOver.ExtensionId);
        Assert.Equal(restore.ActionId, restore.ExtensionId);
        Assert.Equal("Take over the saved columns", takeOver.DisplayName);
        Assert.Equal("Make blocks of the steps", restore.DisplayName);

        // Run first, then the take-over beside it, then the way back for steps a format forgot.
        Assert.Equal([0, 0, 1, 2], new IToolbarAction[] { run, export, takeOver, restore }.Select(action => action.Order));

        foreach (NotebookExtension action in new NotebookExtension[] { run, export, takeOver, restore })
        {
            var button = (IToolbarAction)action;

            Assert.False(string.IsNullOrWhiteSpace(button.DisplayName));
            Assert.StartsWith("<svg", button.Icon, StringComparison.Ordinal);
            Assert.False(button.IconOnly);
            Assert.False(button.IsPrimary);
            Assert.Null(button.ConfirmationPrompt);
            Assert.False(string.IsNullOrWhiteSpace(action.Name));
            Assert.False(string.IsNullOrWhiteSpace(action.Description));
        }
    }

    [Fact]
    public async Task RunningThePipeline_FitsEveryStep_ShowsTheLastBlock_AndHandsOverWhatItLearned()
    {
        await using var notebook = await NotebookAsync(Titanic);

        Assert.True(await notebook.EnabledAsync(RunPipelineAction.Id));

        await notebook.PressAsync(RunPipelineAction.Id);

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

        await notebook.PressAsync(RunPipelineAction.Id);

        var stopping = notebook.Scaffold.Cells[2];

        Assert.Contains(stopping.Outputs, output => output.IsError);
        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ARunButtonVersoDidNotLoad_RefusesToRun_HavingNoNotebookToRun()
    {
        await using var notebook = await NotebookAsync(Titanic);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new RunPipelineAction().ExecuteAsync(notebook.ToolbarContext()));

        Assert.Contains("not loaded by Verso", refused.Message, StringComparison.Ordinal);
        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task ANotebookWithNoBlocks_HasNothingToRunOrExport()
    {
        await using var notebook = await NotebookAsync();

        Assert.False(await notebook.EnabledAsync(RunPipelineAction.Id));
        Assert.False(await notebook.EnabledAsync(ExportPipelineAction.Id));

        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Null(HandedOver(notebook));
    }

    [Fact]
    public async Task AGesture_KeepsWhatARunLearned_WhileTheBlocksDeclareTheSameSteps()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.PressAsync(RunPipelineAction.Id);
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

        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Contains("\"fitted\"", HandedOver(notebook)!, StringComparison.Ordinal);

        // The file is gone: what was learned from it is no longer what these steps would learn.
        File.Delete(Path.Join(_folder, "titanic.csv"));
        await notebook.PressAsync(RunPipelineAction.Id);

        var exported = Text(await notebook.PressAsync(ExportPipelineAction.Id));

        Assert.DoesNotContain("\"fitted\"", HandedOver(notebook) ?? string.Empty, StringComparison.Ordinal);
        Assert.DoesNotContain("\"fitted\"", exported, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AShowOverOtherBytes_HandsOverTheDeclarationAlone()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var file = Path.Join(_folder, "titanic.csv");

        await notebook.PressAsync(RunPipelineAction.Id);

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
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

        await notebook.PressAsync(RunPipelineAction.Id);
        var fitted = HandedOver(notebook);
        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Equal(1, session.RunsFitted);
        Assert.Equal(fitted, HandedOver(notebook));

        File.AppendAllText(Path.Join(_folder, "titanic.csv"), "1,1,female,30.0,0,0,80.0,S,First,woman,False,B,Southampton,yes,True\n");
        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Equal(2, session.RunsFitted);
    }

    [Fact]
    public async Task AFitNoRunOfThisNotebookMade_IsNeverKeptOrExported()
    {
        // A C# cell can write anything under the key, a genuine fit included; only a fit this notebook's own run
        // made of these steps over these bytes is handed on.
        await using var notebook = await NotebookAsync(Titanic);
        var elsewhere = new Pipeline(PipelineDeclaration.FromJson(NotebookPipelineText(notebook), NotebookVerbs.Catalog()), rows: null, SourceFolder.OfDocument(NotebookPath)).Run().ToJson();

        notebook.Scaffold.Variables.Set(StepKernel.HandOver, elsewhere);

        Assert.Equal(NotebookPipelineText(notebook), Text(await notebook.PressAsync(ExportPipelineAction.Id)));

        await notebook.GestureAsync(notebook.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(NotebookPipelineText(notebook), HandedOver(notebook));
    }

    [Fact]
    public async Task ExportingThePipeline_SavesTheDeclaration_NamedAfterTheNotebook()
    {
        await using var notebook = await NotebookAsync(Titanic);

        Assert.True(await notebook.EnabledAsync(ExportPipelineAction.Id));

        var file = (await notebook.PressAsync(ExportPipelineAction.Id))!.Value;

        Assert.Equal("titanic.pipeline.json", file.Name);
        Assert.Equal("application/json", file.ContentType);
        Assert.Equal(NotebookPipelineText(notebook), Encoding.UTF8.GetString(file.Bytes));
    }

    [Fact]
    public async Task ExportingAfterARun_SavesWhatTheFitLearned_OnlyWhileItIsTheFitOfTheStepsDeclaredNow()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.PressAsync(RunPipelineAction.Id);

        var saved = Text(await notebook.PressAsync(ExportPipelineAction.Id));

        Assert.Equal(HandedOver(notebook), saved);
        Assert.NotNull(PreparedData.FromJson(saved, NotebookVerbs.Catalog()).Fitted);

        // A block changed by hand: the fit belongs to steps the notebook no longer declares.
        notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";

        Assert.Equal(NotebookPipelineText(notebook), Text(await notebook.PressAsync(ExportPipelineAction.Id)));
    }

    [Fact]
    public async Task ExportingAfterARunOverAParquetFile_KeepsTheFitOnlyWhileTheFileHoldsTheBytesItLearnedFrom()
    {
        var file = Path.Join(_folder, "titanic.parquet");
        File.Copy(Repository.Fixture("titanic.parquet"), file);
        await using var notebook = await NotebookAsync(["""{"step": "read.parquet", "path": "titanic.parquet"}""", .. Titanic[1..]]);

        await notebook.PressAsync(RunPipelineAction.Id);

        var saved = Text(await notebook.PressAsync(ExportPipelineAction.Id));

        Assert.Equal(HandedOver(notebook), saved);
        Assert.Contains("\"fitted\"", saved, StringComparison.Ordinal);

        // One byte more: the file no longer holds the bytes the fit was learned from.
        using (var appended = File.Open(file, FileMode.Append))
        {
            appended.WriteByte(0);
        }

        Assert.Equal(NotebookPipelineText(notebook), Text(await notebook.PressAsync(ExportPipelineAction.Id)));
    }

    [Theory]
    [InlineData("titanic.parquet", """{"step": "read.parquet", "path": "titanic.parquet"}""")]
    [InlineData("titanic.xlsx", """{"step": "read.excel", "path": "titanic.xlsx"}""")]
    [InlineData("titanic.json", """{"step": "read.json", "path": "titanic.json"}""")]
    public async Task ThePassengerListRunFromAParquetFileAWorkbookOrJson_SplitsAsItsCommaSeparatedFileDoes(string file, string block)
    {
        // The same cells in another file are the same rows: the toolbar's run hands over the fit of the same split, learned
        // from the same training rows, as the notebook that reads the comma-separated file hands over.
        File.Copy(Repository.Fixture(file), Path.Join(_folder, file));
        await using var csv = await NotebookAsync(Titanic);
        await using var other = await Notebook.OpenAsync(Path.Join(_folder, "other.verso"));

        foreach (var each in (string[])[block, .. Titanic[1..]])
        {
            other.AddBlock(each);
        }

        await csv.PressAsync(RunPipelineAction.Id);
        await other.PressAsync(RunPipelineAction.Id);

        var handed = PreparedData.FromJson(HandedOver(other)!, NotebookVerbs.Catalog());
        var run = new Pipeline(handed.Declaration, rows: null, SourceFolder.Of(_folder)).Run();
        // The file's second line: the first row under its header.
        var second = run.Table.Identities.Select(identity => identity.ReadAt).ToList().IndexOf(0);

        Assert.Equal(run.ToJson(), HandedOver(other));
        Assert.Equal([623, 133, 135], new[] { Part.Train, Part.Validation, Part.Test }.Select(run.CountIn));
        Assert.StartsWith("7195ecd6", run.Table.Identities[second].Key.ToString(), StringComparison.Ordinal);
        Assert.Equal(Part.Train, run.Parts[second]);
        Assert.Equal(Learned(HandedOver(csv)!), Learned(HandedOver(other)!));
    }

    // What each step learned, as the handed-over text writes it, without the key of the steps above it.
    private static IReadOnlyList<string> Learned(string pipeline) =>
        [.. JsonNode.Parse(pipeline)!["fitted"]!.AsArray().Select(entry => entry!["learned"]!.ToJsonString())];

    [Fact]
    public async Task APipelineWhoseRowsAreHandedIn_ExportsItsDeclaration_ForCodeThatHandsRowsIn()
    {
        // A notebook shows no rows of it, since it hands none in; the file is what code that does hand rows in runs.
        await using var notebook = await NotebookAsync("""{"step": "read.rows", "description": "passengers"}""", Titanic[1]);

        var saved = Text(await notebook.PressAsync(ExportPipelineAction.Id));

        Assert.Equal(NotebookPipelineText(notebook), saved);
        Assert.IsType<ReadRowsStep>(PipelineDeclaration.FromJson(saved, NotebookVerbs.Catalog()).Steps[0]);
    }

    [Fact]
    public async Task ANotebookNeverSaved_ExportsAPipelineFileOfTheUsualName()
    {
        await using var notebook = await Notebook.OpenAsync();
        notebook.AddBlock(Titanic[0].Replace("titanic.csv", Path.Join(_folder, "titanic.csv").Replace("\\", "\\\\", StringComparison.Ordinal), StringComparison.Ordinal));

        Assert.Equal("pipeline.json", (await notebook.PressAsync(ExportPipelineAction.Id))!.Value.Name);
    }

    [Fact]
    public async Task ExportingANotebookThatMakesNoPipeline_IsRefused_SayingWhichBlockStopsIt()
    {
        await using var notebook = await NotebookAsync(Titanic[0], """{"step": "declare"}""");

        Assert.False(await notebook.EnabledAsync(ExportPipelineAction.Id));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => notebook.PressAsync(ExportPipelineAction.Id));

        Assert.Contains("block 2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExportPressedWhileAGestureHoldsTheNotebook_WaitsForIt_AndExportsTheBlocksItLeft()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;
        var mayFinish = new TaskCompletionSource();
        var holding = session.OneAtATimeAsync(CancellationToken.None, async _ =>
        {
            await mayFinish.Task;

            // What the gesture ahead writes lands only as it ends.
            notebook.Scaffold.Cells[3].Source = """{"step": "fill.missing", "column": "age", "with": "mean"}""";

            return true;
        });

        var exporting = notebook.PressAsync(ExportPipelineAction.Id);

        Assert.False(exporting.IsCompleted);

        mayFinish.SetResult();
        await holding;

        var saved = Text(await exporting);

        Assert.Equal(NotebookPipelineText(notebook), saved);
        Assert.Contains("mean", saved, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheReadmesCSharpCell_RunsThePipelineANotebookOfAParquetFileHandsOver()
    {
        // The README's C# cell as it stands on the page, run in the notebook after the toolbar's run over a Parquet file: its
        // catalog knows every verb a block can hold, so it reads the pipeline the blocks hand over and runs it over the file.
        // A package the page takes from NuGet is the one this suite was built with.
        File.Copy(Repository.Fixture("titanic.parquet"), Path.Join(_folder, "titanic.parquet"));
        await using var notebook = await NotebookAsync(["""{"step": "read.parquet", "path": "titanic.parquet"}""", .. Titanic[1..]]);
        var readme = File.ReadAllText(Path.Join(Repository.Root, "README.md")).ReplaceLineEndings("\n");
        var cell = Regex.Matches(readme, "```csharp\\n(?<code>.*?)```", RegexOptions.Singleline)
            .Select(match => match.Groups["code"].Value)
            .Single(code => code.Contains(StepKernel.HandOver, StringComparison.Ordinal));
        var local = Regex.Replace(
            cell, "#r \"nuget: (?<package>[^\"]+)\"", match => $"#r \"{Path.Join(AppContext.BaseDirectory, match.Groups["package"].Value + ".dll")}\"");
        var source = $"#r \"{Path.Join(AppContext.BaseDirectory, "DeepSharp.Pipelines.dll")}\"\n"
            + local.Replace(".Run();", ".Run();\n    Variables.Set(\"readme.rows\", prepared.Table.RowCount);", StringComparison.Ordinal);

        await notebook.PressAsync(RunPipelineAction.Id);

        // The engine calls a cell that met an exception a run that ended, so what the cell wrote is read too.
        var outputs = await notebook.RunAsync(notebook.Scaffold.AddCell("code", "csharp", source));

        Assert.NotNull(HandedOver(notebook));
        Assert.DoesNotContain(outputs, output => output.Content.Contains("Exception", StringComparison.Ordinal));
        Assert.Equal(891, notebook.Scaffold.Variables.Get<int>("readme.rows"));
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
