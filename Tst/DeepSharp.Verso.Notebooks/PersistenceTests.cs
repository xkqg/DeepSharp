// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A pipeline notebook is saved as .verso, which keeps what kind of cell each one is. Jupyter's format keeps a cell's
/// text and loses its kind, so a block saved that way would come back as code: saving one is refused, and so is
/// opening a Jupyter file whose code cells turn out to be steps. Verso's own converter between formats runs no such
/// guard; that is written down, not trusted. What a block shows is left out of the file by the serializer each of
/// Verso's editors saves with.
/// </summary>
public sealed class PersistenceTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-persistence-").FullName;

    public PersistenceTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private static NotebookModel Holding(string type, string source)
    {
        var model = new NotebookModel();
        model.Cells.Add(new CellModel { Type = type, Language = type == StepCellType.StepType ? StepKernel.Language : "csharp", Source = source });

        return model;
    }

    private static string Ipynb(string source) =>
        $$"""{"cells": [{"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [{{JsonSerializer.Serialize(source)}}]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}""";

    [Fact]
    public async Task TheGuard_IsAPartVersoLoads_ForJupyterAlone()
    {
        await using var notebook = await Notebook.OpenAsync();
        var guard = notebook.Host.GetPostProcessors().OfType<JupyterGuard>().Single();

        Assert.Equal(JupyterGuard.Id, guard.ExtensionId);
        Assert.False(string.IsNullOrWhiteSpace(guard.Name));
        Assert.False(string.IsNullOrWhiteSpace(guard.Description));
        Assert.True(guard.CanProcess(null, "jupyter"));
        Assert.True(guard.CanProcess("old.IPYNB", "whatever"));
        Assert.False(guard.CanProcess("new.verso", "verso"));
        Assert.False(guard.CanProcess(null, "verso"));
        Assert.Equal(0, guard.Priority);
    }

    [Fact]
    public async Task APipelineNotebook_IsRefusedWhenSavedAsJupyter()
    {
        var guard = new JupyterGuard();

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => guard.PreSerializeAsync(Holding(StepCellType.StepType, Titanic[0]), null));

        Assert.Contains(".verso", refused.Message, StringComparison.Ordinal);

        var plain = Holding("code", "1 + 1");

        Assert.Same(plain, await guard.PreSerializeAsync(plain, "plain.ipynb"));
    }

    [Fact]
    public async Task AJupyterFileWhoseCodeCellsAreSteps_IsRefusedWhenOpened()
    {
        var guard = new JupyterGuard();
        var saved = await new JupyterSerializer().DeserializeAsync(Ipynb(Titanic[0]));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => guard.PostDeserializeAsync(saved, "old.ipynb"));

        Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);

        var plain = await new JupyterSerializer().DeserializeAsync(Ipynb("1 + 1"));

        Assert.Same(plain, await guard.PostDeserializeAsync(plain, "plain.ipynb"));
    }

    [Fact]
    public async Task ANotebookSavedAsVerso_ReadsBackAsTheSameDeclaration()
    {
        await using var notebook = await Notebook.OpenAsync();

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var serializer = new VersoSerializer();
        var back = await serializer.DeserializeAsync(await serializer.SerializeAsync(notebook.Scaffold.Notebook));

        Assert.All(back.Cells, cell => Assert.Equal(StepCellType.StepType, cell.Type));
        Assert.Equal(
            NotebookPipeline.Of(notebook.Scaffold.Notebook.Cells).Readable,
            NotebookPipeline.Of(back.Cells).Readable);
        Assert.Equal(Titanic.Length, NotebookPipeline.Of(back.Cells).Readable.Steps.Count);
    }

    [Fact]
    public async Task WhatABlockShows_IsLeftOutOfTheFile_ByTheSerializerEachEditorSavesWith()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var read = notebook.Scaffold.Cells[0];

        await notebook.GestureAsync(read, StepRenderer.Show);

        // The block shows its card and a page of the data, with the grid's boxes.
        Assert.Contains(read.Outputs, output => output.Content.Contains(StepRenderer.Include, StringComparison.Ordinal));

        // VS Code's host saves through the serializer the extension host holds for the format, and the browser editor
        // through one handed the host's cell types; both know what a block's type says of its outputs.
        INotebookSerializer[] editors =
        [
            notebook.Host.GetSerializers().Single(serializer => serializer.FormatId == "verso"),
            new VersoSerializer(notebook.Host.GetCellTypes()),
        ];

        foreach (var serializer in editors)
        {
            var back = await serializer.DeserializeAsync(await serializer.SerializeAsync(notebook.Scaffold.Notebook));

            Assert.All(back.Cells, cell => Assert.Empty(cell.Outputs));
            Assert.Equal(NotebookPipeline.Of(notebook.Scaffold.Notebook.Cells).Readable, NotebookPipeline.Of(back.Cells).Readable);
        }

        // A serializer that knows no cell type cannot tell a block from any other cell, and keeps what it shows.
        var blind = new VersoSerializer();
        var kept = await blind.DeserializeAsync(await blind.SerializeAsync(notebook.Scaffold.Notebook));

        Assert.NotEmpty(kept.Cells[0].Outputs);
    }
}
