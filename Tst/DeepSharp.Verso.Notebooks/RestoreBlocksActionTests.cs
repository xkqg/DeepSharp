// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A format that forgets what kind of cell a block was keeps its text: Jupyter as a raw cell, Markdown as a fence of the
/// block's language inside a cell of text, and a file another program wrote as code. Where the guard against such a file
/// is not asked — Verso's browser editor asks it nothing — the notebook opens with its steps as text, and saving it there
/// writes them into a .verso as text. The toolbar's button makes blocks of them again, in their places, wherever the
/// notebook is open; text that is no step stays as it was.
/// </summary>
public sealed class RestoreBlocksActionTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-restore-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
    ];

    public RestoreBlocksActionTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static IPipelineStep Step(string text) => NotebookVerbs.Catalog().ReadStep(text);

    private static CellModel Cell(string type, string source, string? language = null) => new() { Type = type, Language = language, Source = source };

    // A .verso file holding these cells, opened in the host an application of your own uses — as Verso's browser editor
    // leaves one behind when it saves a notebook it opened from a Jupyter copy.
    private async Task<Notebook> OpenedAsync(params CellModel[] cells)
    {
        var model = new NotebookModel();

        foreach (var cell in cells)
        {
            model.Cells.Add(cell);
        }

        var path = Path.Join(_folder, "titanic.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(model), TestContext.Current.CancellationToken);

        return await Notebook.OpenAsync(path);
    }

    private static IEnumerable<(string Type, string Text)> Kinds(Notebook notebook) =>
        notebook.Scaffold.Cells.Select(cell => (cell.Type, cell.Type == StepCellType.StepType ? Step(cell.Source).AsBlockText() : cell.Source));

    [Fact]
    public async Task RawCellsHoldingSteps_AreMadeBlocksAgain_InTheirPlaces_AndTheRestStaysAsItWas()
    {
        await using var notebook = await OpenedAsync(
            Cell("markdown", "# The passenger list"),
            Cell("raw", Titanic[0]),
            Cell("raw", Titanic[2]),
            Cell("raw", "not a step"),
            Cell("code", "var answer = 42;", "csharp"));

        Assert.True(await notebook.EnabledAsync(RestoreBlocksAction.Id));

        await notebook.PressAsync(RestoreBlocksAction.Id);

        Assert.Equal(
            [
                ("markdown", "# The passenger list"),
                (StepCellType.StepType, Step(Titanic[0]).AsBlockText()),
                (StepCellType.StepType, Step(Titanic[2]).AsBlockText()),
                ("raw", "not a step"),
                ("code", "var answer = 42;"),
            ],
            Kinds(notebook));
        Assert.All(notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType), cell => Assert.Equal(StepKernel.Language, cell.Language));
        Assert.False(await notebook.EnabledAsync(RestoreBlocksAction.Id));
    }

    [Fact]
    public async Task AMarkdownCellFoldingTextAndFencedSteps_IsSplitIntoItsTextAndItsBlocks_InTheirOrder()
    {
        // Verso's Markdown reader brings the fences of several blocks back as one cell of text, with the text between them.
        var folded = $"# The passenger list\n\nRead and declared:\n\n```pdd\n{Titanic[0]}\n```\n\n```deepsharp.step\n{Titanic[1]}\n```\n\nThen split.\n\n```csharp\n{Titanic[2]}\n```\n";

        await using var notebook = await OpenedAsync(Cell("markdown", folded.ReplaceLineEndings("\r\n")));

        await notebook.PressAsync(RestoreBlocksAction.Id);

        Assert.Equal(
            [
                ("markdown", "# The passenger list\n\nRead and declared:"),
                (StepCellType.StepType, Step(Titanic[0]).AsBlockText()),
                (StepCellType.StepType, Step(Titanic[1]).AsBlockText()),
                ("markdown", $"Then split.\n\n```csharp\n{Titanic[2]}\n```"),
            ],
            Kinds(notebook));
    }

    [Fact]
    public async Task ACodeCellHoldingAStep_AsAFileAnotherProgramWroteHoldsOne_IsMadeABlock()
    {
        await using var notebook = await OpenedAsync(Cell("code", Titanic[0], "csharp"));

        await notebook.PressAsync(RestoreBlocksAction.Id);

        Assert.Equal([(StepCellType.StepType, Step(Titanic[0]).AsBlockText())], Kinds(notebook));
    }

    [Fact]
    public async Task ANotebookWhoseCellsHoldNoForgottenStep_HasTheButtonOff_AndPressedAnywayItChangesNothing()
    {
        await using var notebook = await OpenedAsync(
            Cell(StepCellType.StepType, Titanic[0], StepKernel.Language),
            Cell("markdown", "A block reads the file with `read.csv`."),
            Cell("markdown", "```pdd\nnot a step\n```"),
            Cell("html", Titanic[2]),
            Cell("code", "var answer = 42;", "csharp"));
        var before = Kinds(notebook).ToArray();

        Assert.False(await notebook.EnabledAsync(RestoreBlocksAction.Id));

        await notebook.PressAsync(RestoreBlocksAction.Id);

        Assert.Equal(before, Kinds(notebook));
    }

    [Fact]
    public async Task ANotebookSavedOverWithItsBlocksAsRawCells_IsWholeAgain_AndItsPipelineRuns()
    {
        // As Verso's browser editor writes it: the blocks of the passenger list's notebook, each come back from Jupyter as
        // a raw cell, saved as .verso.
        await using var notebook = await OpenedAsync([.. Titanic.Select(text => Cell("raw", text))]);

        Assert.False(await notebook.EnabledAsync(RunPipelineAction.Id));

        await notebook.PressAsync(RestoreBlocksAction.Id);

        Assert.Equal(Titanic.Select(Step), notebook.Scaffold.Cells.Select(cell => Step(cell.Source)));
        Assert.True(await notebook.EnabledAsync(RunPipelineAction.Id));

        await notebook.PressAsync(RunPipelineAction.Id);

        Assert.Contains("891 rows", notebook.Scaffold.Cells[^1].Outputs[^1].Content, StringComparison.Ordinal);
    }
}
