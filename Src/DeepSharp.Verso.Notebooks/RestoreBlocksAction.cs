// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The button that makes blocks again of the steps a format kept the text of and forgot were blocks.
/// </summary>
/// <remarks>
/// A notebook of blocks saved in another format than .verso comes back with its steps as text — a raw cell from Jupyter,
/// a fence in a cell of text from Markdown, a code cell from a file another program wrote. Where the guard is asked, such
/// a file is refused when it is opened; Verso's browser editor asks it nothing, opens the file, and saves its steps into a
/// .verso as text. The button is on wherever a cell holds such a step, in every editor, and pressing it puts a block in
/// each step's place, the text around them staying text, so the notebook is whole again before or after it was saved.
/// What the blocks no longer make is taken back, as after any change to them; nothing runs.
/// </remarks>
[VersoExtension]
public sealed class RestoreBlocksAction : NotebookExtension, IToolbarAction
{
    /// <summary>The button's id.</summary>
    public const string Id = "io.github.xkqg.deepsharp.notebooks.blocks";

    /// <inheritdoc />
    public override string ExtensionId => Id;

    /// <inheritdoc />
    public override string Name => "DeepSharp make blocks of the steps";

    /// <inheritdoc />
    public override string Description =>
        "Makes blocks again of the pipeline steps a notebook holds as text, where a format other than .verso forgot each was a block.";

    /// <inheritdoc />
    public string ActionId => Id;

    /// <inheritdoc />
    public string DisplayName => "Make blocks of the steps";

    /// <inheritdoc />
    public string Icon =>
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\" width=\"16\" height=\"16\" fill=\"currentColor\">"
        + "<path d=\"M3 4h8v6H3zM13 4h8v6h-8zM3 14h8v6H3zM13 17h3v-3l5 4.5-5 4.5v-3h-3z\"/></svg>";

    /// <inheritdoc />
    public bool IconOnly => false;

    /// <inheritdoc />
    public bool IsPrimary => false;

    /// <inheritdoc />
    public string? ConfirmationPrompt => null;

    /// <inheritdoc />
    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    /// <inheritdoc />
    public int Order => 2;

    /// <inheritdoc />
    /// <remarks>When a cell holds a step a format forgot was a block.</remarks>
    public Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var catalog = NotebookVerbs.Catalog();

        return Task.FromResult(context.NotebookCells.Any(cell => cell.HoldsForgottenSteps(catalog)));
    }

    /// <inheritdoc />
    public async Task ExecuteAsync(IToolbarActionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var session = RequiredSession;
        var catalog = NotebookVerbs.Catalog();

        await session.OneAtATimeAsync(context.CancellationToken, async turn =>
        {
            Forgotten[] holding = [.. context.NotebookCells.Select(cell => new Forgotten(cell, cell.Parts(catalog))).Where(each => each.Parts.Any(part => part.Step is not null))];

            if (holding.Length == 0)
            {
                return false;
            }

            // Every cell holding steps gives way to its parts in its place, as one write; what the blocks no longer make is
            // then taken back, each grid they no longer make cleared before it is forgotten.
            await session.LetThroughAsync(turn, async () =>
            {
                foreach (var forgotten in holding)
                {
                    await RestoreAsync(context, forgotten);
                }

                var now = NotebookPipeline.Of(context.NotebookCells);

                foreach (var stale in session.StaleIn(now, except: null))
                {
                    await context.Notebook.ClearOutputAsync(stale);
                }

                session.CaughtUp(now, context.Variables, except: null);
            });

            return true;
        });
    }

    // A cell's parts in its place — a block for each step, a cell of its own kind for the text around them — and the cell gone.
    private static async Task RestoreAsync(IToolbarActionContext context, Forgotten forgotten)
    {
        var at = context.IndexOf(forgotten.Cell.Id);

        foreach (var part in forgotten.Parts)
        {
            var made = part.Step is { } step
                ? await context.InsertedAsync(at, new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = step.AsBlockText() })
                : await context.InsertedAsync(at, new CellModel { Type = forgotten.Cell.Type, Language = forgotten.Cell.Language, Source = part.Text });

            at = context.IndexOf(made) + 1;
        }

        await context.Notebook.RemoveCellAsync(forgotten.Cell.Id);
    }

    /// <summary>A cell holding steps a format forgot were blocks, and the parts its text falls into.</summary>
    /// <param name="Cell">The cell.</param>
    /// <param name="Parts">Its parts, in their order.</param>
    private readonly record struct Forgotten(CellModel Cell, IReadOnlyList<CellPart> Parts);
}
