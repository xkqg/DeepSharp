// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a part is handed about the notebook it acts on, the way Verso's own editor hands it: its variables, what can be
/// done to it, its file and theme, and a way to hand over a file, which is kept for whoever asked the part to act.
/// </summary>
/// <param name="scaffold">The notebook.</param>
internal abstract class NotebookContext(Scaffold scaffold) : IVersoContext
{
    /// <summary>The file the part handed over last; nothing when it handed none.</summary>
    public HostedFile? Handed { get; private set; }

    /// <summary>The notebook.</summary>
    protected Scaffold Scaffold => scaffold;

    public IVariableStore Variables => scaffold.Variables;

    public CancellationToken CancellationToken => CancellationToken.None;

    public IThemeContext Theme => scaffold.ThemeContext;

    public LayoutCapabilities LayoutCapabilities => scaffold.LayoutCapabilities;

    public IExtensionHostContext ExtensionHost => scaffold.ExtensionHostContext;

    public INotebookMetadata NotebookMetadata => scaffold.Metadata;

    public INotebookOperations Notebook => scaffold.NotebookOps;

    public string? ActiveLayoutId => scaffold.NotebookOps.ActiveLayoutId;

    // A part acting on the notebook as a whole, or on a cell's properties, has no output of its own to write into.
    public Task WriteOutputAsync(CellOutput output) => Task.CompletedTask;

    public Task RequestFileDownloadAsync(string fileName, string contentType, byte[] data)
    {
        Handed = new HostedFile(fileName, contentType, data);

        return Task.CompletedTask;
    }
}
