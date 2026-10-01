// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Keeps a notebook of pipeline steps in .verso, the one format of Verso's that keeps a block a block, on the way out
/// and on the way in.
/// </summary>
/// <remarks>
/// Every other format keeps a cell's text and loses what kind of cell it is: Jupyter brings a block back as a raw cell,
/// Markdown as text that fences the block's text in, and a Jupyter or .dib file another program wrote may hold it as
/// code. Saving a notebook that holds a block in any of them is refused, and so is opening such a file whose cells turn
/// out to be steps in any of those forms — recognised by their text, the one thing the format kept. The guard takes
/// every format but Verso's own, so a format it does not know is refused a block rather than trusted with one; the save
/// is recognised by its format, since Verso names no file when it saves. The guard runs where the host asks it — Verso's
/// VS Code host and DeepSharp's own server do, before they write a file and after they read one. Verso's command-line
/// converter asks nothing on the way out, and the browser editor <c>verso serve</c> starts asks nothing either way; both
/// are written down in the architecture and on the notebook's page rather than trusted.
/// </remarks>
[VersoExtension]
public sealed partial class FormatGuard : NotebookExtension, INotebookPostProcessor
{
    /// <summary>The guard's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.format-guard";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline notebooks stay .verso";

    /// <inheritdoc />
    public override string Description =>
        "Refuses to save a notebook of pipeline steps in any format but .verso, which alone keeps a block a block, and to open a file of another format whose cells are steps.";

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
                "A notebook of pipeline steps is saved as .verso: any other format keeps a cell's text and loses what kind "
                + "of cell it is, so every block would come back as something else. Save it as .verso, or export the pipeline from the export menu.")
            : Task.FromResult(notebook);
    }

    /// <inheritdoc />
    public Task<NotebookModel> PostDeserializeAsync(NotebookModel notebook, string? filePath)
    {
        ArgumentNullException.ThrowIfNull(notebook);

        var catalog = NotebookVerbs.Catalog();
        var step = notebook.Cells.SelectMany(BlockTexts).Select(catalog.TryReadStep).FirstOrDefault(each => each is not null);

        return step is null
            ? Task.FromResult(notebook)
            : throw new InvalidOperationException(
                $"This file holds pipeline steps saved as cells of another kind — '{step.Verb}' among them. Its format kept "
                + "their text and lost that each was a block, so it is not opened as though they were something else: open the .verso notebook they came from.");
    }

    // The texts a cell holds where a format kept a block's text and forgot the block: all of a raw or a code cell's, and
    // each fence Markdown wrote a block in, under the block's language or its kind — Markdown reads the fences of several
    // blocks back as one cell of text with whatever text stood between them. A cell of any other kind holds no block's text.
    private static IEnumerable<string> BlockTexts(CellModel cell) => cell.Type switch
    {
        "raw" or "code" => [cell.Source],
        "markdown" => Fenced().Matches(cell.Source.ReplaceLineEndings("\n")).Select(fence => fence.Groups["text"].Value),
        _ => [],
    };

    // A fence opened on a line of its own with the block's language or its kind, and closed on a line of its own, as
    // Markdown writes a block.
    [GeneratedRegex(@"^```(?:pdd|deepsharp\.step)[ \t]*\n(?<text>[\s\S]*?)\n?^```[ \t]*$", RegexOptions.Multiline)]
    private static partial Regex Fenced();
}
