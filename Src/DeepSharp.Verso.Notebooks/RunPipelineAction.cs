// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The button that runs the whole pipeline the blocks declare: the one run that fits.
/// </summary>
/// <remarks>
/// Every step that learns is fitted on the training rows and replayed unchanged on the others, and what it learned is
/// handed to C# cells beside the declaration. The run is made by the last block's own kernel, which shows the rows
/// there; a notebook whose blocks make no pipeline says why at the block that stops it, and hands nothing over.
/// </remarks>
[VersoExtension]
public sealed class RunPipelineAction : ToolbarButton
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
    public override string DisplayName => "Run the pipeline";

    /// <inheritdoc />
    public override int Order => 0;

    /// <inheritdoc />
    private protected override ToolbarIcon Drawing => new("M2 3h8v5H2zM2 16h8v5H2zM14 5l8 7-8 7z");

    /// <inheritdoc />
    /// <remarks>Whenever there is a block: a notebook that makes no pipeline is told why where it stops.</remarks>
    public override Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        return Task.FromResult(context.NotebookCells.Any(cell => cell.Type == StepCellType.StepType));
    }

    /// <inheritdoc />
    public override Task ExecuteAsync(IToolbarActionContext context) =>
        RunOnSessionAsync(context, async (session, turn) =>
        {
            var assembled = NotebookPipeline.Of(context.NotebookCells);

            if (assembled.Blocks.Count == 0)
            {
                return false;
            }

            // What changed since the last gesture without one seeing it is caught up with first — each grid the blocks no
            // longer make cleared before it is forgotten — and the pipeline the blocks make is handed over, as one write.
            await session.LetThroughAsync(turn, async () =>
            {
                foreach (var stale in session.StaleIn(assembled, except: null))
                {
                    await context.Notebook.ClearOutputAsync(stale);
                }

                session.CaughtUp(assembled, context.Variables, except: null);
                session.HandOver(context.Variables, assembled);
            });

            // The whole pipeline runs at its last block; one the blocks do not make is refused at the block that stops it.
            var at = assembled.Whole ? assembled.Blocks[^1].Cell : assembled.Blocks.First(block => block.Faults.Count > 0).Cell;

            await session.AskAsync(at, assembled.RequestFor(at, ViewTrigger.Run, page: 0), turn, context.Notebook);

            return true;
        });
}
