// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A part that brings a renderer and nothing else, as a part may: the engine draws a cell of its type, and runs no kernel
/// for it.
/// </summary>
internal sealed class DrawingPart : ICellRenderer
{
    public const string Type = "deepsharp.tests.drawn";

    public string ExtensionId => "deepsharp.tests.drawing-part";

    public string Name => "A part that only draws";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string CellTypeId => Type;

    public string DisplayName => "Drawn";

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task<RenderResult> RenderInputAsync(string source, ICellRenderContext context) => Task.FromResult(new RenderResult("text/plain", source));

    public Task<RenderResult> RenderOutputAsync(CellOutput output, ICellRenderContext context) => Task.FromResult(new RenderResult(output.MimeType, output.Content));

    public string? GetEditorLanguage() => null;
}

/// <summary>
/// A cell type with no kernel whose drawing waits until it is let go: while it draws, the cell runs, and no kernel does.
/// </summary>
/// <param name="began">The file its drawing writes as it begins.</param>
/// <param name="go">The file its drawing waits for.</param>
internal sealed class HeldDrawing(string began, string go) : ICellType
{
    public const string Type = "deepsharp.tests.held";

    public string ExtensionId => "deepsharp.tests.held-drawing";

    public string Name => "A cell type that draws until it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string CellTypeId => Type;

    public string DisplayName => "Held";

    public string? Icon => null;

    public ICellRenderer Renderer { get; } = new Drawing(began, go);

    public ILanguageKernel? Kernel => null;

    public bool IsEditable => true;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public string GetDefaultContent() => string.Empty;

    private sealed class Drawing(string began, string go) : ICellRenderer
    {
        public string ExtensionId => "deepsharp.tests.held-drawing.renderer";

        public string Name => "The held drawing";

        public string Version => "1.0.0";

        public string? Author => null;

        public string? Description => null;

        public string CellTypeId => Type;

        public string DisplayName => "Held";

        public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

        public Task OnUnloadedAsync() => Task.CompletedTask;

        public async Task<RenderResult> RenderInputAsync(string source, ICellRenderContext context)
        {
            await File.WriteAllTextAsync(began, "on", TestContext.Current.CancellationToken);

            while (!File.Exists(go))
            {
                await Task.Delay(10, TestContext.Current.CancellationToken);
            }

            return new RenderResult("text/plain", source);
        }

        public Task<RenderResult> RenderOutputAsync(CellOutput output, ICellRenderContext context) => Task.FromResult(new RenderResult(output.MimeType, output.Content));

        public string? GetEditorLanguage() => null;
    }
}
