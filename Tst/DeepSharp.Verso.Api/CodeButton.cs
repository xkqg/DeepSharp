// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A toolbar button that runs code of its own, in no cell, through the notebook it is handed — as any part may. Verso
/// loads it as it loads every extension, so the notebook's own machinery runs it.
/// </summary>
/// <param name="code">The code it runs.</param>
/// <param name="then">What it does after the code, if anything.</param>
/// <param name="language">The language it names; none for the notebook's default kernel.</param>
internal sealed class CodeButton(string code, Func<Task>? then = null, string? language = "csharp") : IToolbarAction
{
    public const string Id = "deepsharp.tests.run-code";

    public string ExtensionId => "deepsharp.tests.code-button";

    public string Name => "A button that runs code";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string ActionId => Id;

    public string DisplayName => "Run code";

    public string? Icon => null;

    public ToolbarPlacement Placement => ToolbarPlacement.MainToolbar;

    public int Order => 0;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<bool> IsEnabledAsync(IToolbarActionContext context) => Task.FromResult(true);

    public async Task ExecuteAsync(IToolbarActionContext context)
    {
        await context.Notebook.ExecuteCodeAsync(code, language, context.CancellationToken);

        if (then is not null)
        {
            await then();
        }
    }
}
