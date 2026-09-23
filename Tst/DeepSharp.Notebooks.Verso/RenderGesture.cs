// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// What Verso hands the properties panel with a cell: the notebook's variables and the cell it is about. Everything
/// else a render context carries throws, so a part that reaches for more than it should fails here rather than
/// working by accident.
/// </summary>
internal sealed class RenderGesture(Notebook notebook, CellModel cell) : ICellRenderContext
{
    public Guid CellId => cell.Id;

    public IVariableStore Variables => notebook.Scaffold.Variables;

    public CancellationToken CancellationToken => CancellationToken.None;

    public IReadOnlyDictionary<string, object> CellMetadata => throw Unused();

    public (double Width, double Height) Dimensions => throw Unused();

    public bool IsSelected => throw Unused();

    public IThemeContext Theme => throw Unused();

    public LayoutCapabilities LayoutCapabilities => throw Unused();

    public IExtensionHostContext ExtensionHost => throw Unused();

    public INotebookMetadata NotebookMetadata => throw Unused();

    public INotebookOperations Notebook => throw Unused();

    public string? ActiveLayoutId => throw Unused();

    public Task WriteOutputAsync(CellOutput output) => throw Unused();

    private static NotSupportedException Unused() => new("A properties panel has no need of this.");
}
