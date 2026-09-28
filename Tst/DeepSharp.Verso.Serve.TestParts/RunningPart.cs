// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A part that answers a click on a control naming it by running the cell after the one clicked — as any part may, through
/// the notebook it is handed.
/// </summary>
[VersoExtension]
public sealed class RunningPart : IExtension, ICellInteractionHandler
{
    /// <summary>The part's name, which a control names to reach it.</summary>
    public const string Id = "deepsharp.tests.running-part";

    /// <inheritdoc />
    public string ExtensionId => Id;

    /// <inheritdoc />
    public string Name => "A part that runs the next cell on a click";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public async Task<string?> OnCellInteractionAsync(CellInteractionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        var next = context.NotebookModel!.Cells.SkipWhile(cell => cell.Id != context.CellId).Skip(1).First();

        await context.Notebook!.ExecuteCellAsync(next.Id);

        // Nothing to answer: the clicked cell keeps what it shows.
        return null;
    }
}
