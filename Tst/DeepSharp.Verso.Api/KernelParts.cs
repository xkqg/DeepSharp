// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A cell type that brings a kernel of its own and registers it nowhere else, as a part may: the engine finds that kernel
/// by its language for any cell that names it.
/// </summary>
/// <param name="began">The file its kernel writes as a run begins.</param>
/// <param name="go">The file its kernel waits for before the run ends.</param>
internal sealed class KernelledType(string began, string go) : ICellType
{
    public string ExtensionId => "deepsharp.tests.kernelled-type";

    public string Name => "A cell type with a kernel of its own";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string CellTypeId => "deepsharp.tests.kernelled";

    public string DisplayName => "Kernelled";

    public string? Icon => null;

    public ICellRenderer Renderer { get; } = new DrawingPart();

    public ILanguageKernel? Kernel { get; } = new HeldKernel(began, go);

    public bool IsEditable => true;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public string GetDefaultContent() => string.Empty;
}

/// <summary>A kernel whose runs go on until they are let go.</summary>
/// <param name="began">The file it writes as a run begins.</param>
/// <param name="go">The file it waits for before the run ends.</param>
internal sealed class HeldKernel(string began, string go) : ILanguageKernel
{
    public const string Language = "held";

    public string ExtensionId => "deepsharp.tests.held-kernel";

    public string Name => "A kernel that runs until it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Held";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Task.CompletedTask;

    public async Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context)
    {
        await File.WriteAllTextAsync(began, "on", TestContext.Current.CancellationToken);

        while (!File.Exists(go))
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        return [];
    }

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
