// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Keeps a notebook of pipeline steps in .verso, the one format of Verso's that keeps a block a block, on the way out
/// and on the way in.
/// </summary>
/// <remarks>
/// Every other format keeps a cell's text and loses what kind of cell it is: Jupyter and .dib bring a block back as
/// code, and Markdown as text. Saving a notebook that holds a block in any of them is refused, and so is opening such a
/// file whose code cells turn out to be steps — recognised by their text, the one thing the format kept. The guard
/// takes every format but Verso's own, so a format it does not know is refused a block rather than trusted with one;
/// the save is recognised by its format, since Verso names no file when it saves. Verso's own command-line converter
/// between formats runs no such guard, and that is written down in the architecture and on the notebook's page rather
/// than trusted.
/// </remarks>
[VersoExtension]
public sealed class FormatGuard : NotebookExtension, INotebookPostProcessor
{
    /// <summary>The guard's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.format-guard";

    private const string Verso = "verso";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline notebooks stay .verso";

    /// <inheritdoc />
    public override string Description =>
        "Refuses to save a notebook of pipeline steps in any format but .verso, which alone keeps a block a block, and to open a file of another format whose code cells are steps.";

    /// <inheritdoc />
    public int Priority => 0;

    /// <inheritdoc />
    /// <remarks>Every format but Verso's own.</remarks>
    public bool CanProcess(string? filePath, string formatId) => formatId != Verso;

    /// <inheritdoc />
    public Task<NotebookModel> PreSerializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        return notebook.Cells.Any(cell => cell.Type == StepCellType.StepType)
            ? throw new InvalidOperationException(
                "A notebook of pipeline steps is saved as .verso: any other format keeps a cell's text and loses what kind "
                + "of cell it is, so every block would come back as something else. Save it as .verso, or export the pipeline from the export menu.")
            : Task.FromResult(notebook);
    }

    /// <inheritdoc />
    public Task<NotebookModel> PostDeserializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        var catalog = NotebookVerbs.Catalog();
        var step = notebook.Cells.Where(cell => cell.Type == "code").Select(cell => catalog.TryReadStep(cell.Source)).FirstOrDefault(each => each is not null);

        return step is null
            ? Task.FromResult(notebook)
            : throw new InvalidOperationException(
                $"This file holds pipeline steps saved as code cells — '{step.Verb}' among them. Its format kept their text "
                + "and lost that each was a block, so it is not opened as though they were code: open the .verso notebook they came from.");
    }
}
