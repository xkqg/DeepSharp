// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A toolbar button whose press is stopped while it works — as a person pressing Stop at that moment does — and which then
/// hands a file over, in the same breath, before anything could look at the stop.
/// </summary>
/// <param name="stop">Stops the press under way.</param>
internal sealed class HandingButton(Action stop) : IToolbarAction
{
    public const string Id = "deepsharp.tests.hand-over";

    public string ExtensionId => "deepsharp.tests.handing-button";

    public string Name => "A button stopped while it hands a file over";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string ActionId => Id;

    public string DisplayName => "Hand over";

    public string? Icon => null;

    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<bool> IsEnabledAsync(IToolbarActionContext context) => Task.FromResult(true);

    public Task ExecuteAsync(IToolbarActionContext context)
    {
        stop();

        return context.RequestFileDownloadAsync("stopped.txt", "text/plain", [1, 2, 3]);
    }
}
