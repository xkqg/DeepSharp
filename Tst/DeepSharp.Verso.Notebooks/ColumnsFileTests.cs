// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// What a notebook decided about its columns is saved in a file of its own beside it, named after it: written after
/// every change the blocks accept and every run of the whole pipeline, as the blocks hold it, and never by a block
/// that is only shown. A file that cannot be read is not written over, and a file that cannot be written says so at
/// the block.
/// </summary>
public sealed class ColumnsFileTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-columns-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public ColumnsFileTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string NotebookPath => Path.Join(_folder, "titanic.verso");

    private string ColumnsFile => Path.Join(_folder, "titanic.columns.json");

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(NotebookPath);

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private static DeclareStep Declared(Notebook notebook) => (DeclareStep)StepCatalog.BuiltIn().ReadStep(notebook.Scaffold.Cells[1].Source);

    private PipelinePreset Saved() => PipelinePreset.FromJson(File.ReadAllText(ColumnsFile), StepCatalog.BuiltIn());

    private static Task RunAsync(Notebook notebook, string? path) =>
        notebook.Host.GetToolbarActions().OfType<RunPipelineAction>().Single().ExecuteAsync(new ToolbarGesture(notebook, path));

    [Fact]
    public async Task AChangeTheBlocksAccept_IsSavedBesideTheNotebook_AsTheBlocksHoldIt()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.True(File.Exists(ColumnsFile));
        Assert.Equal(Declared(notebook), Saved().Declare);
        Assert.True(Saved().Declare.Columns[1].Excluded);
        Assert.Null(Saved().Source);
    }

    [Fact]
    public async Task ShowingOrPagingTheData_SavesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Show);
        await notebook.GestureAsync(notebook.Scaffold.Cells[0], StepRenderer.Page, "1");

        Assert.False(File.Exists(ColumnsFile));
    }

    [Fact]
    public async Task AChangeTheRulesRefuse_SavesNothing()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Category, "fare", ticked: true);

        Assert.False(File.Exists(ColumnsFile));
    }

    [Fact]
    public async Task BlocksThatDoNotMakeAPipelineYet_SaveNothing_ThoughTheChangeIsMade()
    {
        await using var notebook = await NotebookAsync([.. Titanic, """{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}"""]);

        var gesture = await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.True(gesture.StateChanged);
        Assert.False(File.Exists(ColumnsFile));
    }

    [Fact]
    public async Task RunningTheWholePipeline_SavesIt()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await RunAsync(notebook, NotebookPath);

        Assert.Equal(Declared(notebook), Saved().Declare);
    }

    [Fact]
    public async Task TheSourceTheListLastShowed_IsKeptWhenTheFileIsWrittenAgain()
    {
        await using var notebook = await NotebookAsync(Titanic);
        string[] shown = ["survived", "pclass", "sex", "age", "fare"];

        await File.WriteAllTextAsync(ColumnsFile, new PipelinePreset(Declared(notebook), source: shown).ToJson(), TestContext.Current.CancellationToken);
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.Equal(shown, Saved().Source);
        Assert.True(Saved().Declare.Columns[1].Excluded);
    }

    [Fact]
    public async Task AFileThatCannotBeRead_IsNotWrittenOver_AndTheBlockSaysSo()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await File.WriteAllTextAsync(ColumnsFile, "not the saved columns", TestContext.Current.CancellationToken);
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.Equal("not the saved columns", await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && output.Content.Contains("cannot be read, so they are not written over", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFileThatCannotBeWritten_SaysSoAtTheBlock_AndLeavesNothingHalfWritten()
    {
        // A folder where the file goes refuses the file on every system; an open file refuses it only on some.
        await using var notebook = await NotebookAsync(Titanic);

        Directory.CreateDirectory(ColumnsFile);
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.True(Directory.Exists(ColumnsFile));
        Assert.Empty(Directory.GetFiles(_folder, "*.tmp"));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && output.Content.Contains("could not be written beside the notebook", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFileAnotherProgramHoldsOpen_CannotBeRead_IsNotWrittenOver_AndTheBlockSaysSo()
    {
        await using var notebook = await NotebookAsync(Titanic);

        await File.WriteAllTextAsync(ColumnsFile, new PipelinePreset(Declared(notebook)).ToJson(), TestContext.Current.CancellationToken);
        var before = await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken);

        await using (new FileStream(ColumnsFile, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
        {
            await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);
        }

        Assert.Equal(before, await File.ReadAllTextAsync(ColumnsFile, TestContext.Current.CancellationToken));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && output.Content.Contains("cannot be read, so they are not written over", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFileThatCannotBeWrittenEvenUnderANameOfItsOwn_SaysSoToo()
    {
        // Something already stands under the name the file is first written as, and cannot be taken away.
        await using var notebook = await NotebookAsync(Titanic);

        Directory.CreateDirectory($"{ColumnsFile}.{Environment.ProcessId}.tmp");
        await notebook.TickAsync(notebook.Scaffold.Cells[1], StepRenderer.Include, "pclass", ticked: false);

        Assert.False(File.Exists(ColumnsFile));
        Assert.Contains(notebook.Scaffold.Cells[1].Outputs, output => output.IsError
            && output.Content.Contains("could not be written beside the notebook", StringComparison.Ordinal));
    }

    [Fact]
    public async Task TheSameDecisionsAgain_LeaveTheFileAsItWas()
    {
        await using var notebook = await NotebookAsync(Titanic);
        var longAgo = new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        await RunAsync(notebook, NotebookPath);
        File.SetLastWriteTimeUtc(ColumnsFile, longAgo);
        await RunAsync(notebook, NotebookPath);

        Assert.Equal(longAgo, File.GetLastWriteTimeUtc(ColumnsFile));
    }

    [Fact]
    public async Task APipelineWithoutASchema_HasNoColumnsToSave()
    {
        await using var notebook = await NotebookAsync(Titanic[0]);

        await RunAsync(notebook, NotebookPath);

        Assert.False(File.Exists(ColumnsFile));
    }

    [Fact]
    public async Task TheFilesBesideANotebook_AreNamedAfterIt_AndANotebookNeverSavedHasNone()
    {
        await using var notebook = await NotebookAsync(Titanic);

        Assert.Equal(ColumnsFile, new ToolbarGesture(notebook, NotebookPath).ColumnsFilePath());
        Assert.Null(new ToolbarGesture(notebook, null).ColumnsFilePath());
        Assert.Equal("titanic.pipeline.json", new ToolbarGesture(notebook, NotebookPath).PipelineFileName());
        Assert.Equal("pipeline.json", new ToolbarGesture(notebook, null).PipelineFileName());
    }
}
