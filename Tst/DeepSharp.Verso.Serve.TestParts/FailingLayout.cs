// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A layout whose part fails to draw its arrangement, as a part may: a page shows the notebook's list instead, and says why.
/// </summary>
[VersoExtension]
public sealed class FailingLayout : ILayoutEngine
{
    /// <summary>The part's id, which a notebook names with the layout's.</summary>
    public const string Part = "deepsharp.tests.failing-layout";

    /// <summary>The layout's id.</summary>
    public const string Id = "fails-to-draw";

    /// <summary>What it says when it fails.</summary>
    public const string Why = "It could not draw.";

    /// <inheritdoc />
    public string ExtensionId => Part;

    /// <inheritdoc />
    public string Name => "A layout that fails to draw";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public string LayoutId => Id;

    /// <inheritdoc />
    public string DisplayName => "Fails to draw";

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
    public Task<RenderResult> RenderLayoutAsync(IReadOnlyList<CellModel> cells, IVersoContext context) => throw new InvalidOperationException(Why);

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
}
