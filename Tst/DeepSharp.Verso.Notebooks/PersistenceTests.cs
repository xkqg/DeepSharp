// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A pipeline notebook is saved as .verso, the one format of Verso's that keeps what kind of cell each one is. Every
/// other keeps a cell's text and loses its kind — Jupyter brings a block back as a raw cell, Markdown as text fencing
/// the block's text in, and a file another program wrote may hold it as code — so saving a block in any of them is
/// refused, and so is opening such a file whose cells turn out to be steps. Verso's own converter between formats runs
/// no such guard on the way out; that is written down, not trusted, and what it writes is refused on the way in. What a
/// block shows is left out of the file by the serializer each of Verso's editors saves with.
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
    public async Task OfTheEnginesFormats_OnlyVersoKeepsABlock_WrittenAndReadBack()
    {
        // What the guard stands on, measured on Verso's own serializers rather than assumed: a Verso whose other
        // formats learn to keep a cell's kind fails this, and the guard can let that format through then.
        await using var notebook = await Notebook.OpenAsync();
        var formats = new HashSet<string>();
        var keeping = new List<string>();
        var neverWritten = new List<string>();
        var cameBackAs = new Dictionary<string, string>();

        foreach (var serializer in notebook.Host.GetSerializers())
        {
            formats.Add(serializer.FormatId);
            string written;

            try
            {
                written = await serializer.SerializeAsync(Holding(StepCellType.StepType, Titanic[0]));
            }
            catch (NotSupportedException)
            {
                neverWritten.Add(serializer.FormatId);

                continue;
            }

            var back = await serializer.DeserializeAsync(written);

            if (back.Cells.Any(cell => cell.Type == StepCellType.StepType))
            {
                keeping.Add(serializer.FormatId);

                continue;
            }

            // What the format made of the block is refused on the way in, whatever kind of cell it came back as.
            cameBackAs[serializer.FormatId] = Assert.Single(back.Cells).Type;
            var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => new FormatGuard().PostDeserializeAsync(back, $"old.{serializer.FormatId}"));

            Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);
        }

        Assert.Superset(new HashSet<string> { "verso", "jupyter", "dib", "markdown" }, formats);
        Assert.Equal(["verso"], keeping);
        Assert.Equal(new Dictionary<string, string> { ["jupyter"] = "raw", ["markdown"] = "markdown" }, cameBackAs);

        // .dib is read and never written by this Verso: a file of it can hold blocks only as something else.
        Assert.Equal(["dib"], neverWritten);
    }

    [Fact]
    public async Task ANotebookOfTextBlocksAndCode_WrittenInAnyOtherFormat_IsRefusedWhenReadBack()
    {
        // A notebook as people write them: text, then the blocks, then more text and a C# cell. Markdown reads the text
        // and the blocks' fences back as one cell of text and the C# cell as code, so a block is found inside a cell of
        // text as well as as one.
        await using var opened = await Notebook.OpenAsync();
        var notebook = new NotebookModel();

        notebook.Cells.Add(new CellModel { Type = "markdown", Source = "# The passenger list\n\nEach block below is one step." });

        foreach (var block in Titanic)
        {
            notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = block });
        }

        notebook.Cells.Add(new CellModel { Type = "markdown", Source = "The cell below trains a network." });
        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "var answer = 42;" });

        var cameBackAs = new Dictionary<string, string>();

        foreach (var serializer in opened.Host.GetSerializers().Where(each => each.FormatId is not "verso" and not "dib"))
        {
            var back = await serializer.DeserializeAsync(await serializer.SerializeAsync(notebook));
            var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => new FormatGuard().PostDeserializeAsync(back, $"old.{serializer.FormatId}"));

            cameBackAs[serializer.FormatId] = string.Join(" ", back.Cells.Select(cell => cell.Type));
            Assert.Contains("'read.csv'", refused.Message, StringComparison.Ordinal);
        }

        Assert.Equal(
            new Dictionary<string, string> { ["jupyter"] = "markdown raw raw raw raw markdown code", ["markdown"] = "markdown code" },
            cameBackAs);
    }

    [Theory]
    [InlineData("verso")]
    [InlineData("verso-native")]
    public async Task ANotebookOfBlocks_SavedInVersosOwnFormat_UnderEitherOfItsNames_IsWrittenWithEveryBlock(string format)
    {
        // Verso names its own format twice: its writer says "verso", and the post-processors' contract says
        // "verso-native", the name Verso's VS Code host hands them when it saves a notebook as .verso. Saved as that
        // host saves — every post-processor that takes the format, then the writer — every block is written.
        await using var notebook = await Notebook.OpenAsync();

        foreach (var block in Titanic)
        {
            notebook.AddBlock(block);
        }

        var model = notebook.Scaffold.Notebook;

        foreach (var processor in notebook.Host.GetPostProcessors().Where(each => each.CanProcess(null, format)).OrderBy(each => each.Priority))
        {
            model = await processor.PreSerializeAsync(model, null);
        }

        var written = await new VersoSerializer().DeserializeAsync(await new VersoSerializer().SerializeAsync(model));

        Assert.Equal(Titanic.Length, written.Cells.Count(cell => cell.Type == StepCellType.StepType));
    }

    [Fact]
    public async Task TheGuard_IsAPartVersoLoads_ForEveryFormatButVersosOwn()
    {
        await using var notebook = await Notebook.OpenAsync();
        var guard = notebook.Host.GetPostProcessors().OfType<FormatGuard>().Single();

        Assert.Equal(FormatGuard.Id, guard.ExtensionId);
        Assert.False(string.IsNullOrWhiteSpace(guard.Name));
        Assert.False(string.IsNullOrWhiteSpace(guard.Description));
        Assert.All(notebook.Host.GetSerializers(), serializer => Assert.Equal(serializer.FormatId != "verso", guard.CanProcess(null, serializer.FormatId)));
        Assert.True(guard.CanProcess("old.IPYNB", "whatever"));
        Assert.False(guard.CanProcess("new.verso", "verso"));
        Assert.Equal(0, guard.Priority);
    }

    [Fact]
    public async Task APipelineNotebook_IsRefusedWhenSavedInAFormatThatCannotKeepABlock()
    {
        var guard = new FormatGuard();

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => guard.PreSerializeAsync(Holding(StepCellType.StepType, Titanic[0]), null));

        Assert.Contains(".verso", refused.Message, StringComparison.Ordinal);

        var plain = Holding("code", "1 + 1");

        Assert.Same(plain, await guard.PreSerializeAsync(plain, "plain.ipynb"));
    }

    [Fact]
    public async Task AJupyterFileWhoseCodeCellsAreSteps_IsRefusedWhenOpened()
    {
        var guard = new FormatGuard();
        var saved = await new JupyterSerializer().DeserializeAsync(Ipynb(Titanic[0]));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => guard.PostDeserializeAsync(saved, "old.ipynb"));

        Assert.Contains("read.csv", refused.Message, StringComparison.Ordinal);

        var plain = await new JupyterSerializer().DeserializeAsync(Ipynb("1 + 1"));

        Assert.Same(plain, await guard.PostDeserializeAsync(plain, "plain.ipynb"));
    }

    [Theory]
    [InlineData("""{"step": "read.parquet", "path": "titanic.parquet"}""", "read.parquet")]
    [InlineData("""{"step": "read.excel", "path": "titanic.xlsx", "sheet": "passengers"}""", "read.excel")]
    [InlineData("""{"step": "read.json", "path": "titanic.json"}""", "read.json")]
    public async Task AJupyterFileWhoseCodeCellsAreParquetExcelOrJsonSteps_IsRefusedWhenOpened(string block, string verb)
    {
        // A block reading any file a first block may read is recognised by its text as well as one reading a
        // comma-separated file, so a notebook of them saved in a format that forgets blocks is not opened as code.
        var saved = await new JupyterSerializer().DeserializeAsync(Ipynb(block));

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => new FormatGuard().PostDeserializeAsync(saved, "old.ipynb"));

        Assert.Contains($"'{verb}'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("pdd", "\n")]
    [InlineData("pdd", "\r\n")]
    [InlineData("deepsharp.step", "\n")]
    public async Task ABlockMarkdownFencedIn_IsRefusedWhenOpened_UnderItsLanguageOrItsKind_WhateverEndsItsLines(string fence, string line)
    {
        var notebook = new NotebookModel();
        notebook.Cells.Add(new CellModel { Type = "markdown", Source = $"```{fence}{line}{Titanic[0]}{line}```{line}" });

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => new FormatGuard().PostDeserializeAsync(notebook, "old.md"));

        Assert.Contains("'read.csv'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("markdown", "# The passenger list\n\nRead as `read.csv`.")]
    [InlineData("markdown", "```csharp\n{\"step\": \"read.csv\", \"path\": \"titanic.csv\"}\n```")]
    [InlineData("markdown", "```pdd\nnot a step\n```")]
    [InlineData("raw", "not a step")]
    [InlineData("html", "{\"step\": \"read.csv\", \"path\": \"titanic.csv\"}")]
    public async Task ACellHoldingNoBlocksText_IsOpened_WhateverItsKind(string type, string source)
    {
        // Text, another language's fence, a fence of the block's language around what is no step, and a kind of cell
        // no format makes of a block: none of them is a block a format forgot.
        var notebook = new NotebookModel();
        notebook.Cells.Add(new CellModel { Type = type, Source = source });

        Assert.Same(notebook, await new FormatGuard().PostDeserializeAsync(notebook, "plain.ipynb"));
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
