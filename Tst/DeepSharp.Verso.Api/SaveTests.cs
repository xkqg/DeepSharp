// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// Saving an open notebook the way Verso's own editors save it: through the serializer for its format, which leaves out
/// what a block shows and keeps what a C# cell printed, past the guards that run before writing, and written whole under
/// a name of its own before it takes the file's place. Saved under another name, the notebook is that file from then
/// on — what DeepSharp writes beside it follows — and a name another open notebook holds is refused.
/// </summary>
public sealed class SaveTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-save-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
    ];

    public SaveTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(At(name), TestContext.Current.CancellationToken);
    }

    private static async Task<NotebookModel> ReadAsync(string path) =>
        await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));

    [Fact]
    public async Task ANotebookSaved_LeavesOutWhatABlockShows_AndKeepsWhatACSharpCellPrinted()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(
            notebooks, "titanic.verso", [.. Titanic.Select(Block), new CellModel { Type = "code", Language = "csharp", Source = """System.Console.Write("printed");""" }]);

        await host.RunToolbarAsync("verso.action.run-all");
        await host.SaveAsync();

        var saved = await ReadAsync(At("titanic.verso"));

        Assert.Equal(host.Cells.Select(cell => cell.Source), saved.Cells.Select(cell => cell.Source));
        Assert.All(saved.Cells.Where(cell => cell.Type == StepCellType.StepType), cell => Assert.Empty(cell.Outputs));
        Assert.Contains("printed", string.Concat(saved.Cells[^1].Outputs.Select(output => output.Content)), StringComparison.Ordinal);
        Assert.Empty(Directory.GetFiles(_folder, "*.tmp"));
    }

    [Fact]
    public async Task SavedUnderAnotherName_TheNotebookIsThatFileFromThenOn_AndWhatItNamesAfterItselfFollows()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await notebooks.SaveAsAsync(host, At("renamed.verso"));

        Assert.Equal(At("renamed.verso"), host.FilePath);
        Assert.Same(host, await notebooks.OpenAsync(At("renamed.verso"), TestContext.Current.CancellationToken));
        Assert.Equal("renamed.pipeline.json", (await host.RunToolbarAsync(ExportPipelineAction.Id))!.Value.Name);
        Assert.Equal(host.Cells.Select(cell => cell.Source), (await ReadAsync(At("renamed.verso"))).Cells.Select(cell => cell.Source));

        // The old file is only a file now: opening it opens a notebook of its own.
        Assert.NotSame(host, await notebooks.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task SavedUnderItsOwnName_ItIsSimplySaved()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.EditAsync(host.Cells[0].Id, Titanic[0].Replace("titanic.csv", "other.csv", StringComparison.Ordinal));
        await notebooks.SaveAsAsync(host, At("titanic.verso"));

        Assert.Contains("other.csv", (await ReadAsync(At("titanic.verso"))).Cells[0].Source, StringComparison.Ordinal);
        Assert.Same(host, await notebooks.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ANameAnotherOpenNotebookHolds_IsRefused_AndNothingIsWritten()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var other = await OpenAsync(notebooks, "other.verso", Block(Titanic[0]));
        var before = await File.ReadAllTextAsync(At("other.verso"), TestContext.Current.CancellationToken);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => notebooks.SaveAsAsync(host, At("other.verso")));

        Assert.Contains("other.verso", refused.Message, StringComparison.Ordinal);
        Assert.Equal(before, await File.ReadAllTextAsync(At("other.verso"), TestContext.Current.CancellationToken));
        Assert.Equal(At("titanic.verso"), host.FilePath);
        Assert.Same(other, await notebooks.OpenAsync(At("other.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task SavingBlocksAsJupyter_IsRefusedByTheGuard_AndTheNotebookStaysWhereItWas()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await Assert.ThrowsAsync<InvalidOperationException>(() => notebooks.SaveAsAsync(host, At("titanic.ipynb")));

        Assert.False(File.Exists(At("titanic.ipynb")));
        Assert.Equal(At("titanic.verso"), host.FilePath);
        Assert.NotSame(host, await OpenOtherAsync(notebooks));
    }

    [Fact]
    public async Task AFormatNoSerializerWrites_IsRefused()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        var refused = await Assert.ThrowsAsync<NotSupportedException>(() => notebooks.SaveAsAsync(host, At("titanic.txt")));

        Assert.Contains("titanic.txt", refused.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(At("titanic.txt")));
        Assert.Equal(At("titanic.verso"), host.FilePath);
    }

    [Fact]
    public async Task SavingANotebookTheseNotebooksDoNotHold_IsRefused()
    {
        await using var these = new OpenNotebooks();
        await using var those = new OpenNotebooks();
        var host = await OpenAsync(those, "titanic.verso", [.. Titanic.Select(Block)]);

        await Assert.ThrowsAsync<InvalidOperationException>(() => these.SaveAsAsync(host, At("renamed.verso")));

        // Holding a notebook of the same file is not holding this one.
        await these.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken);

        await Assert.ThrowsAsync<InvalidOperationException>(() => these.SaveAsAsync(host, At("renamed.verso")));

        Assert.False(File.Exists(At("renamed.verso")));
    }

    // A notebook refused a new name leaves that name free: a Jupyter file saved there by someone else opens as its own.
    private async Task<NotebookHost> OpenOtherAsync(OpenNotebooks notebooks)
    {
        await File.WriteAllTextAsync(
            At("titanic.ipynb"),
            """{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}""",
            Encoding.UTF8,
            TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(At("titanic.ipynb"), TestContext.Current.CancellationToken);
    }
}
