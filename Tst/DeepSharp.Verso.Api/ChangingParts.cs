// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>A part that answers a click on a control its cell drew by doing what it is given to do — as any part may.</summary>
/// <param name="clicked">What it does with a click, and what it answers.</param>
internal sealed class ClickingPart(Func<CellInteractionContext, Task<string?>> clicked) : IExtension, ICellInteractionHandler
{
    public const string Id = "deepsharp.tests.clicking-part";

    public string ExtensionId => Id;

    public string Name => "A part that acts on a click";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<string?> OnCellInteractionAsync(CellInteractionContext context) => clicked(context);
}

/// <summary>A properties part that, when a field of its section changes, does what it is given to do — as any part may.</summary>
/// <param name="changed">What it does when a field changes.</param>
internal sealed class ChangingPanel(Func<CellModel, ICellRenderContext, Task> changed) : ICellPropertyProvider
{
    public const string Id = "deepsharp.tests.changing-panel";

    public string ExtensionId => Id;

    public string Name => "A panel that acts when a field changes";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public bool AppliesTo(CellModel cell, ICellRenderContext context) => true;

    public Task<PropertySection> GetPropertiesSectionAsync(CellModel cell, ICellRenderContext context) =>
        Task.FromResult(new PropertySection("Changing", null, []));

    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context) => changed(cell, context);
}
