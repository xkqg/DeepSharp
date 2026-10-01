// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A C# block the wiki marks as a notebook's cell is run as one, in Verso's own engine: after the blocks above it have
/// handed their pipeline over, with the packages it names as the repository built them. The library's suite compiles every
/// other block of the wiki; this runs these.
/// </summary>
public sealed class WikiCellTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-wiki-cell-").FullName;

    public WikiCellTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // The passengers' class, fare and whether they survived, split and fitted: blocks that hand a pipeline over.
    private static readonly string[] Blocks =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "normalise", "column": "fare", "scale": "midrange", "outOfRange": "pass"}""",
        """{"step": "target", "column": "survived"}""",
    ];

    [Fact]
    public async Task EveryNotebookCellOfTheWiki_RunsInVersosEngine_OnThePipelineTheBlocksHandOver()
    {
        var cells = Wiki.Pages().SelectMany(page => page.Cells).ToArray();

        Assert.NotEmpty(cells);

        foreach (var code in cells.Select(cell => cell.Code))
        {
            await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, $"{Guid.NewGuid():N}.verso"));
            var blocks = Blocks.Select(notebook.AddBlock).ToArray();

            await notebook.GestureAsync(blocks[^1], StepRenderer.Show);

            Assert.True(notebook.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _), "The blocks handed nothing over for the cell to read.");

            await notebook.RunAsync(notebook.Scaffold.AddCell("code", "csharp", code.WithPackagesAsBuilt()));
        }
    }
}
