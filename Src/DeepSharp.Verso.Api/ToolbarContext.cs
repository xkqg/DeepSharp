// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a toolbar button is handed when it is pressed, the way Verso's own editor hands it: the notebook's cells as they
/// stand at each look, the cells it is pressed for, its variables, what can be done to it, and a way to hand over a file,
/// which is kept for whoever pressed the button.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <param name="selected">The cells the button is pressed for.</param>
internal sealed class ToolbarContext(Scaffold scaffold, IReadOnlyList<Guid> selected) : IToolbarActionContext
{
    /// <summary>The file the button handed over last; nothing when it handed none.</summary>
    public HostedFile? Handed { get; private set; }

    public IReadOnlyList<Guid> SelectedCellIds => selected;

    public IReadOnlyList<CellModel> NotebookCells => scaffold.Cells;

    public string? ActiveKernelId => scaffold.DefaultKernelId;

    public IVariableStore Variables => scaffold.Variables;

    public CancellationToken CancellationToken => CancellationToken.None;

    public IThemeContext Theme => scaffold.ThemeContext;

    public LayoutCapabilities LayoutCapabilities => scaffold.LayoutCapabilities;

    public IExtensionHostContext ExtensionHost => scaffold.ExtensionHostContext;

    public INotebookMetadata NotebookMetadata => scaffold.Metadata;

    public INotebookOperations Notebook => scaffold.NotebookOps;

    public string? ActiveLayoutId => scaffold.NotebookOps.ActiveLayoutId;

    // A button has no cell of its own to write into.
    public Task WriteOutputAsync(CellOutput output) => Task.CompletedTask;

    public Task RequestFileDownloadAsync(string fileName, string contentType, byte[] data)
    {
        Handed = new HostedFile(fileName, contentType, data);

        return Task.CompletedTask;
    }
}
