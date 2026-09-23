// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// A block of a pipeline: one step, written as the step's own JSON.
/// </summary>
/// <remarks>
/// The notebook is the declaration, block by block: the blocks in the order they stand are the steps in the order
/// they run, each block's text is exactly what the step writes itself as, and saving the notebook saves the steps.
/// What a block shows is never saved — it is worked out from the data each time it is asked for, and the data is
/// somebody's own. Edited as text in the editor, or field by field in Verso's properties panel; both write the
/// same text.
/// </remarks>
[VersoExtension]
public sealed class StepCellType : NotebookExtension, ICellType
{
    /// <summary>The cell type's id as an extension.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.cell";

    /// <summary>The type a block's cell is saved under.</summary>
    public const string StepType = "deepsharp.step";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp pipeline step";

    /// <inheritdoc />
    public override string Description => "A block of a DeepSharp pipeline: one step, written as its own JSON.";

    /// <inheritdoc />
    public string CellTypeId => StepType;

    /// <inheritdoc />
    public string DisplayName => "Pipeline step";

    /// <inheritdoc />
    /// <remarks>Three boxes joined in order: the steps of a pipeline.</remarks>
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M2 3h8v5H2zM14 9.5h8v5h-8zM2 16h8v5H2zM10 5h6v4.5h-1.5V6.5H10zM16 14.5V19h-6v-1.5h4.5v-3z\"/></svg>";

    /// <inheritdoc />
    /// <remarks>
    /// The kernel Verso runs this type's blocks through, which reaches the session this type keeps directly. The
    /// kernel Verso loads as a part of its own serves the editor, and runs the blocks only while this type is
    /// switched off.
    /// </remarks>
    public ILanguageKernel Kernel => _kernel ??= new StepKernel(this);

    private StepKernel? _kernel;

    /// <summary>What the notebook's gestures leave for its blocks, and what they show: the notebook's session.</summary>
    /// <remarks>
    /// Kept here because the block type is the one object every part reaches: the kernel it carries directly, every
    /// other part through the host that loaded it, this type switched off included. It lives as long as this type
    /// and is shared with nothing else; nothing reaches for it of its own accord.
    /// </remarks>
    internal NotebookSession Session { get; } = new();

    /// <inheritdoc />
    public ICellRenderer Renderer { get; } = new StepRenderer();

    /// <inheritdoc />
    public bool IsEditable => true;

    /// <inheritdoc />
    public bool PersistsOutputs => false;

    /// <inheritdoc />
    /// <remarks>A new block starts as the step that reads a CSV file, written the way the step writes itself.</remarks>
    public string GetDefaultContent()
    {
        var catalog = NotebookVerbs.Catalog();

        return catalog.ReadStep(catalog.Describe(ReadCsvStep.Name).Template).AsBlockText();
    }
}
