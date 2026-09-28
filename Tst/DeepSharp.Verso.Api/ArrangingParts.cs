// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A layout a part brings, as any extension may: it draws what it is given to draw, in the way it is given, and does what it
/// is given to do when a person acts on what it drew.
/// </summary>
/// <param name="id">Its id.</param>
/// <param name="draw">What it draws for the cells.</param>
/// <param name="acted">What it does when a person acts on what it drew; nothing when none is given.</param>
/// <param name="custom">Whether it draws an arrangement of its own rather than the notebook's list.</param>
/// <param name="isolation">Whether it draws inline or in a frame of its own.</param>
internal sealed class ArrangingLayout(
    string id,
    Func<IReadOnlyList<CellModel>, RenderResult> draw,
    Func<LayoutInteractionContext, Task>? acted = null,
    bool custom = true,
    LayoutRendererIsolation isolation = LayoutRendererIsolation.Inline) : ILayoutEngine, ILayoutInteractionHandler
{
    public string ExtensionId => $"deepsharp.tests.layout.{id}";

    public string Name => "A layout that draws what it is given to";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LayoutId => id;

    public string DisplayName => id;

    public string? Icon => null;

    public LayoutCapabilities Capabilities =>
        LayoutCapabilities.CellInsert | LayoutCapabilities.CellDelete | LayoutCapabilities.CellReorder | LayoutCapabilities.CellEdit | LayoutCapabilities.CellExecute;

    public bool RequiresCustomRenderer => custom;

    public LayoutRendererIsolation RendererIsolation => isolation;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<RenderResult> RenderLayoutAsync(IReadOnlyList<CellModel> cells, IVersoContext context) => Task.FromResult(draw(cells));

    public Task<CellContainerInfo> GetCellContainerAsync(Guid cellId, IVersoContext context) => Task.FromResult(new CellContainerInfo(cellId, 0, 0, 0, 0));

    public Task OnCellAddedAsync(Guid cellId, int index, IVersoContext context) => Task.CompletedTask;

    public Task OnCellRemovedAsync(Guid cellId, IVersoContext context) => Task.CompletedTask;

    public Task OnCellMovedAsync(Guid cellId, int newIndex, IVersoContext context) => Task.CompletedTask;

    public Dictionary<string, object> GetLayoutMetadata() => [];

    public Task ApplyLayoutMetadata(Dictionary<string, object> metadata, IVersoContext context) => Task.CompletedTask;

    public Task OnLayoutInteractionAsync(LayoutInteractionContext context) => acted?.Invoke(context) ?? Task.CompletedTask;
}
