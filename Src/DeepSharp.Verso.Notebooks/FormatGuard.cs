// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Keeps a notebook of pipeline steps in .verso, the one format whose file keeps a block a block wherever it is opened, on
/// the way out and on the way in.
/// </summary>
/// <remarks>
/// Every other format keeps a cell's text and may lose what kind of cell it is. Markdown brings a block back as text that
/// fences the block's text in, and a .dib file holds it as code. Jupyter keeps the kind beside the cell, where Verso reads
/// it back, but a Jupyter file is opened by programs that know nothing of it and show each block as a raw cell, and a
/// Jupyter file such a program wrote holds the steps as code. Saving a notebook that holds a block in any of them is
/// refused, and so is opening such a file whose cells turn out to be steps held as some other kind of cell — recognised by
/// their text, the one thing every format keeps; a Jupyter file Verso wrote, whose blocks come back as blocks, opens. The
/// guard takes every format but Verso's own, so a format it does not know is refused a block rather than trusted with one;
/// the save is recognised by its format, since Verso names no file when it saves. Every host of Verso's asks it, before it
/// writes a file and after it reads one — the VS Code host, the browser editor <c>verso serve</c> starts, the command
/// line's conversion, and DeepSharp's own server — and a file that holds its steps as text anyway is made blocks of again
/// by <see cref="RestoreBlocksAction"/>, which reads a cell by the same rule as the guard.
/// </remarks>
[VersoExtension]
public sealed class FormatGuard : NotebookExtension, INotebookPostProcessor
{
    /// <summary>The guard's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.format-guard";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline notebooks stay .verso";

    /// <inheritdoc />
    public override string Description =>
        "Refuses to save a notebook of pipeline steps in any format but .verso, whose file alone keeps a block a block wherever it is opened, and to open a file of another format whose cells are steps held as some other kind of cell.";

    /// <inheritdoc />
    public int Priority => 0;

    /// <inheritdoc />
    /// <remarks>Every format but Verso's own, under either of the names Verso gives it.</remarks>
    public bool CanProcess(string? filePath, string formatId) => !formatId.IsVersosOwn();

    /// <inheritdoc />
    public Task<NotebookModel> PreSerializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        return notebook.Cells.Any(cell => cell.Type == StepCellType.StepType)
            ? throw new InvalidOperationException(
                "A notebook of pipeline steps is saved as .verso: any other format keeps a cell's text and may lose what kind "
                + "of cell it is — Jupyter keeps it only for Verso, and every other program shows each block as a raw cell. Save it as .verso, or export the pipeline from the export menu.")
            : Task.FromResult(notebook);
    }

    /// <inheritdoc />
    public Task<NotebookModel> PostDeserializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        var catalog = NotebookVerbs.Catalog();
        var step = notebook.Cells.SelectMany(cell => cell.Parts(catalog)).Select(part => part.Step).FirstOrDefault(each => each is not null);

        return step is null
            ? Task.FromResult(notebook)
            : throw new InvalidOperationException(
                $"This file holds pipeline steps saved as cells of another kind — '{step.Verb}' among them. Its format kept "
                + "their text and lost that each was a block, so it is not opened as though they were something else: open the .verso notebook they came from.");
    }
}
