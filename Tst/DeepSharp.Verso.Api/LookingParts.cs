// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>A toolbar button that, asked whether it can be pressed, runs the notebook's first cell — as any part may try.</summary>
internal sealed class LookingButton : IToolbarAction
{
    public string ExtensionId => "deepsharp.tests.looking-button";

    public string Name => "A button that runs a cell when it is looked at";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public const string Id = "deepsharp.tests.looking";

    public string ActionId => Id;

    public string DisplayName => "Looking";

    public string? Icon => null;

    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public async Task<bool> IsEnabledAsync(IToolbarActionContext context)
    {
        await context.Notebook.ExecuteCellAsync(context.NotebookCells[0].Id);

        return true;
    }

    public Task ExecuteAsync(IToolbarActionContext context) => Task.CompletedTask;
}

/// <summary>A button on a cell's toolbar that cannot tell, for any cell, whether it can be pressed — as a part may fail.</summary>
internal sealed class FailingCellButton : IToolbarAction
{
    public const string Id = "deepsharp.tests.failing-cell";

    public const string Why = "It cannot tell for a cell.";

    public string ExtensionId => "deepsharp.tests.failing-cell-button";

    public string Name => "A cell's button that fails when asked for a cell";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string ActionId => Id;

    public string DisplayName => "Failing";

    public string? Icon => null;

    public ToolbarPlacement Placement => ToolbarPlacement.CellToolbar;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    // Asked for the notebook as a whole it says no; asked for a cell it fails.
    public Task<bool> IsEnabledAsync(IToolbarActionContext context) =>
        context.SelectedCellIds.Count > 0 ? throw new InvalidOperationException(Why) : Task.FromResult(false);

    public Task ExecuteAsync(IToolbarActionContext context) => Task.CompletedTask;
}

/// <summary>A properties part that, asked to draw its section, runs the cell it is drawn for — as any part may try.</summary>
internal sealed class LookingPanel : ICellPropertyProvider
{
    public string ExtensionId => "deepsharp.tests.looking-panel";

    public string Name => "A panel that runs its cell when it is drawn";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public bool AppliesTo(CellModel cell, ICellRenderContext context) => true;

    public async Task<PropertySection> GetPropertiesSectionAsync(CellModel cell, ICellRenderContext context)
    {
        await context.Notebook.ExecuteCellAsync(cell.Id);

        return new PropertySection("Looking", null, []);
    }

    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context) => Task.CompletedTask;
}

/// <summary>A properties part that, asked to draw its section, takes until it is let go — as a slow part does.</summary>
/// <param name="until">When it is let go.</param>
internal sealed class WaitingPanel(Task until) : ICellPropertyProvider
{
    public string ExtensionId => "deepsharp.tests.waiting-panel";

    public string Name => "A panel that takes until it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public bool AppliesTo(CellModel cell, ICellRenderContext context) => true;

    public async Task<PropertySection> GetPropertiesSectionAsync(CellModel cell, ICellRenderContext context)
    {
        await until;

        return new PropertySection("Waiting", null, []);
    }

    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context) => Task.CompletedTask;
}
