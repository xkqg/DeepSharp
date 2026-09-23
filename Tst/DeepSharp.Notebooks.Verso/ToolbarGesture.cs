// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// A click on a toolbar button, as the host hands it to the action: the notebook's cells, its variables, what can be
/// done to it, and where it is saved. Everything else a toolbar context carries throws, so an action that reaches
/// for more than it should fails here rather than working by accident.
/// </summary>
internal sealed class ToolbarGesture(Notebook notebook, string? filePath) : IToolbarActionContext, INotebookMetadata
{
    /// <summary>Every file the action handed the person to save: its name, its type and its bytes.</summary>
    public List<Download> Downloads { get; } = [];

    public IReadOnlyList<Guid> SelectedCellIds => [];

    public IReadOnlyList<CellModel> NotebookCells => notebook.Scaffold.Cells;

    public string? ActiveKernelId => null;

    public IVariableStore Variables => notebook.Scaffold.Variables;

    public CancellationToken CancellationToken => CancellationToken.None;

    public INotebookOperations Notebook => notebook.Scaffold.NotebookOps;

    public INotebookMetadata NotebookMetadata => this;

    public string? Title => null;

    public string? DefaultKernelId => null;

    public string? FilePath => filePath;

    public Dictionary<string, NotebookParameterDefinition>? Parameters => throw Unused();

    public Task RequestFileDownloadAsync(string fileName, string contentType, byte[] data)
    {
        Downloads.Add(new Download(fileName, contentType, data));

        return Task.CompletedTask;
    }

    public Task WriteOutputAsync(CellOutput output) => throw Unused();

    public IThemeContext Theme => throw Unused();

    public LayoutCapabilities LayoutCapabilities => throw Unused();

    public IExtensionHostContext ExtensionHost => throw Unused();

    public string? ActiveLayoutId => throw Unused();

    private static NotSupportedException Unused() => new("A toolbar action has no need of this.");

    /// <summary>A file handed over to be saved.</summary>
    /// <param name="FileName">What it is called.</param>
    /// <param name="ContentType">What it holds.</param>
    /// <param name="Data">Its bytes.</param>
    internal sealed record Download(string FileName, string ContentType, byte[] Data);
}
