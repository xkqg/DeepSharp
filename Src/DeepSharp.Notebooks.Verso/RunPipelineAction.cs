// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// The button that runs the whole pipeline the blocks declare: the one run that fits.
/// </summary>
/// <remarks>
/// Every step that learns is fitted on the training rows and replayed unchanged on the others, and what it learned is
/// handed to C# cells beside the declaration. The run is made by the last block's own kernel, which shows the rows
/// there; a notebook whose blocks make no pipeline says why at the block that stops it, and hands nothing over.
/// </remarks>
[VersoExtension]
public sealed class RunPipelineAction : NotebookExtension, IToolbarAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.run";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp run the pipeline";

    /// <inheritdoc />
    public override string Description => "Runs the whole pipeline the blocks declare, fitting every step on the training rows, and hands what it learned to C# cells.";

    /// <inheritdoc />
    public string ActionId => Id;

    /// <inheritdoc />
    public string DisplayName => "Run the pipeline";

    /// <inheritdoc />
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M2 3h8v5H2zM2 16h8v5H2zM14 5l8 7-8 7z\"/></svg>";

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    /// <inheritdoc />
    public int Order => 0;

    /// <inheritdoc />
    /// <remarks>Whenever there is a block: a notebook that makes no pipeline is told why where it stops.</remarks>
    public Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        return Task.FromResult(context.NotebookCells.Any(cell => cell.Type == StepCellType.StepType));
    }

    /// <inheritdoc />
    public async Task ExecuteAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;

        await session.OneAtATimeAsync(async () =>
        {
            var assembled = NotebookPipeline.Of(context.NotebookCells);

            if (assembled.Blocks.Count == 0)
            {
                return false;
            }

            session.Publish(assembled);
            session.HandOver(context.Variables, assembled);

            // The whole pipeline runs at its last block; one the blocks do not make is refused at the block that stops it.
            var at = assembled.Whole ? assembled.Blocks[^1].Cell : assembled.Blocks.First(block => block.Faults.Count > 0).Cell;

            session.Request(at, assembled.RequestFor(at, page: 0, run: true));
            await context.Notebook.ExecuteCellAsync(at);

            return true;
        });
    }
}
