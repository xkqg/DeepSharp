// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What the host hands a part to act on the notebook with — the only operations any part the host asks is given. What a
/// part may do through it is the host's to say, by what it asks the part for: a press is a run, a click or a change of a
/// field is a change, and a look does nothing to the notebook.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <remarks>
/// Whatever the part is asked for, it reads which layout and theme the notebook is shown in; every other verb is let
/// through, or refused, as the port says, and one that runs code is run as the port runs it.
/// </remarks>
internal abstract class NotebookPort(Scaffold scaffold) : INotebookOperations
{
    /// <summary>Marked when the run the part acts in is stopped; none while it acts in no run.</summary>
    public virtual CancellationToken Token => CancellationToken.None;

    public string? ActiveLayoutId => Engine.ActiveLayoutId;

    public string? ActiveThemeId => Engine.ActiveThemeId;

    /// <summary>The notebook.</summary>
    protected Scaffold Scaffold => scaffold;

    /// <summary>The notebook's own operations, which a verb the port lets through goes to.</summary>
    protected INotebookOperations Engine => scaffold.NotebookOps;

    public abstract Task ExecuteCellAsync(Guid cellId);

    public abstract Task ExecuteAllAsync();

    public abstract Task ExecuteFromAsync(Guid cellId);

    public abstract Task ExecuteCodeAsync(string code, string? language = null, CancellationToken ct = default);

    public abstract Task<IReadOnlyList<CellOutput>> ExecuteCodeCaptureOutputsAsync(string code, string? language = null, CancellationToken ct = default);

    public async Task ClearOutputAsync(Guid cellId)
    {
        using var admitted = Admit();

        await Engine.ClearOutputAsync(cellId);
    }

    public async Task ClearAllOutputsAsync()
    {
        using var admitted = Admit();

        await Engine.ClearAllOutputsAsync();
    }

    public async Task RestartKernelAsync(string? kernelId = null)
    {
        using var admitted = Admit();

        await Engine.RestartKernelAsync(kernelId);
    }

    public async Task<string> InsertCellAsync(int index, string type, string? language = null)
    {
        using var admitted = Admit();

        return await Engine.InsertCellAsync(index, type, language);
    }

    public async Task RemoveCellAsync(Guid cellId)
    {
        using var admitted = Admit();

        await Engine.RemoveCellAsync(cellId);
    }

    public async Task MoveCellAsync(Guid cellId, int newIndex)
    {
        using var admitted = Admit();

        await Engine.MoveCellAsync(cellId, newIndex);
    }

    public void SetActiveLayout(string layoutId)
    {
        using var admitted = Admit();

        Engine.SetActiveLayout(layoutId);
    }

    public void SetActiveTheme(string themeId)
    {
        using var admitted = Admit();

        Engine.SetActiveTheme(themeId);
    }

    /// <summary>
    /// Lets a verb through, or refuses it with why, before it touches the notebook; one let through is held until it has
    /// landed, so a stop of the run it acts in waits for it before a kernel starts afresh.
    /// </summary>
    /// <returns>What is held while the verb is on its way.</returns>
    protected abstract Admission Admit();
}
