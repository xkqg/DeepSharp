// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Keeps a notebook of pipeline steps out of Jupyter's format, on the way out and on the way in.
/// </summary>
/// <remarks>
/// Jupyter's format keeps a cell's text and loses what kind of cell it is, so every block saved there would come
/// back as code. Saving a notebook that holds a block as Jupyter is refused, and so is opening a Jupyter file whose
/// code cells turn out to be steps — recognised by their text, the one thing Jupyter kept. The save is recognised by
/// its format, since Verso names no file when it saves. Verso's own command-line converter between formats runs no
/// such guard, and that is written down in the package's notes rather than trusted.
/// </remarks>
[VersoExtension]
public sealed class JupyterGuard : NotebookExtension, INotebookPostProcessor
{
    /// <summary>The guard's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.jupyter-guard";

    private const string Jupyter = "jupyter";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline notebooks stay .verso";

    /// <inheritdoc />
    public override string Description =>
        "Refuses to save a notebook of pipeline steps as Jupyter, which would turn every block into code, and to open a Jupyter file whose code cells are steps.";

    /// <inheritdoc />
    public int Priority => 0;

    /// <inheritdoc />
    public bool CanProcess(string? filePath, string formatId) =>
        formatId == Jupyter || (filePath?.EndsWith(".ipynb", StringComparison.OrdinalIgnoreCase) ?? false);

    /// <inheritdoc />
    public Task<NotebookModel> PreSerializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        return notebook.Cells.Any(cell => cell.Type == StepCellType.StepType)
            ? throw new InvalidOperationException(
                "A notebook of pipeline steps is saved as .verso: Jupyter keeps a cell's text and loses what kind of cell it "
                + "is, so every block would come back as code. Save it as .verso, or export the pipeline from the export menu.")
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
                $"This Jupyter file holds pipeline steps saved as code cells — '{step.Verb}' among them. Jupyter kept their "
                + "text and lost that each was a block, so it is not opened as though they were code: open the .verso notebook they came from.");
    }
}
