// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A layout a part brings, as any extension may: it draws a slot for every cell and a button that hands over a file, and
/// lets cells be added, taken away, moved and written while it shows the notebook.
/// </summary>
[VersoExtension]
public sealed class SlottingLayout : ILayoutEngine, ILayoutInteractionHandler
{
    /// <summary>The part's id, which a notebook names with the layout's.</summary>
    public const string Part = "deepsharp.tests.slotting-layout";

    /// <summary>The layout's id.</summary>
    public const string Id = "slots";

    /// <summary>The name of the file its button hands over.</summary>
    public const string Handed = "handed.txt";

    /// <inheritdoc />
    public string ExtensionId => Part;

    /// <inheritdoc />
    public string Name => "A layout of slots";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public string LayoutId => Id;

    /// <inheritdoc />
    public string DisplayName => "Slots";

    /// <inheritdoc />
    public string? Icon => null;

    /// <inheritdoc />
    public LayoutCapabilities Capabilities =>
        LayoutCapabilities.CellInsert | LayoutCapabilities.CellDelete | LayoutCapabilities.CellReorder | LayoutCapabilities.CellEdit | LayoutCapabilities.CellExecute;

    /// <inheritdoc />
    public bool RequiresCustomRenderer => true;

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public Task<RenderResult> RenderLayoutAsync(IReadOnlyList<CellModel> cells, IVersoContext context) =>
        Task.FromResult(new RenderResult(
            "text/html",
            $"""<div class="slots"><button type="button" data-action="hand" data-payload="the payload" data-target-id="the target">Hand over</button>{string.Concat(cells.Select(cell => $"<div class=\"slot\" data-cell-slot=\"{cell.Id}\"></div>"))}</div>"""));

    /// <inheritdoc />
    public Task<CellContainerInfo> GetCellContainerAsync(Guid cellId, IVersoContext context) => Task.FromResult(new CellContainerInfo(cellId, 0, 0, 0, 0));

    /// <inheritdoc />
    public Task OnCellAddedAsync(Guid cellId, int index, IVersoContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnCellRemovedAsync(Guid cellId, IVersoContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnCellMovedAsync(Guid cellId, int newIndex, IVersoContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Dictionary<string, object> GetLayoutMetadata() => [];

    /// <inheritdoc />
    public Task ApplyLayoutMetadata(Dictionary<string, object> metadata, IVersoContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnLayoutInteractionAsync(LayoutInteractionContext context) =>
        context.InteractionType == "hand"
            ? context.Verso.RequestFileDownloadAsync(Handed, "text/plain", Encoding.UTF8.GetBytes($"{context.Payload} / {context.TargetId}"))
            : Task.CompletedTask;
}
