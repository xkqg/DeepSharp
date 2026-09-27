// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// The notebook as one run hands it to what the run asks of it — a toolbar button's action — so a stop reaches every cell
/// the action would still run: once the run is stopped nothing more is done, and a cell not yet begun never begins.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <param name="run">The run.</param>
/// <remarks>
/// Verso's engine counts a cell as begun before it looks at whether its run was stopped, so each cell is refused here
/// first. The engine's own loop over every cell looks between cells, as Verso's browser editor has it look when its Stop
/// ends a Run All; everything else goes to the notebook's own operations.
/// </remarks>
internal sealed class RunOperations(Scaffold scaffold, Run run) : INotebookOperations
{
    private INotebookOperations Notebook => scaffold.NotebookOps;

    public string? ActiveLayoutId => Notebook.ActiveLayoutId;

    public string? ActiveThemeId => Notebook.ActiveThemeId;

    public async Task ExecuteCellAsync(Guid cellId)
    {
        run.Token.ThrowIfCancellationRequested();
        await scaffold.ExecuteCellAsync(cellId, run.Token);
    }

    public async Task ExecuteAllAsync()
    {
        run.Token.ThrowIfCancellationRequested();
        await scaffold.ExecuteAllAsync(run.Token);
    }

    // As Verso's own operations run from a cell on, and refused before each cell once the run is stopped.
    public async Task ExecuteFromAsync(Guid cellId)
    {
        run.Token.ThrowIfCancellationRequested();

        var cells = scaffold.Cells;
        var from = cells.TakeWhile(cell => cell.Id != cellId).Count();

        if (from == cells.Count)
        {
            throw new InvalidOperationException($"Cell {cellId} not found.");
        }

        foreach (var cell in cells.Skip(from))
        {
            run.Token.ThrowIfCancellationRequested();
            await scaffold.ExecuteCellAsync(cell.Id, run.Token);
        }
    }

    public Task ClearOutputAsync(Guid cellId)
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.ClearOutputAsync(cellId);
    }

    public Task ClearAllOutputsAsync()
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.ClearAllOutputsAsync();
    }

    public Task RestartKernelAsync(string? kernelId = null)
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.RestartKernelAsync(kernelId);
    }

    public Task<string> InsertCellAsync(int index, string type, string? language = null)
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.InsertCellAsync(index, type, language);
    }

    public Task RemoveCellAsync(Guid cellId)
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.RemoveCellAsync(cellId);
    }

    public Task MoveCellAsync(Guid cellId, int newIndex)
    {
        run.Token.ThrowIfCancellationRequested();

        return Notebook.MoveCellAsync(cellId, newIndex);
    }

    public async Task ExecuteCodeAsync(string code, string? language = null, CancellationToken ct = default)
    {
        run.Token.ThrowIfCancellationRequested();

        using var either = CancellationTokenSource.CreateLinkedTokenSource(run.Token, ct);

        // Code with no cell begins no cell the engine tells of, so the run is told here what runs, for a stop to start its
        // kernel afresh: the language it was asked in, none being the notebook's default kernel, which is where the engine
        // runs it and what a restart naming no kernel starts.
        run.Began(null, language);

        try
        {
            await scaffold.ExecuteCodeAsync(code, language, either.Token);
        }
        finally
        {
            run.Ended(null);
        }
    }

    // Forwarded as the engine has it: the interface's own default runs the code and hands back no outputs.
    public async Task<IReadOnlyList<CellOutput>> ExecuteCodeCaptureOutputsAsync(string code, string? language = null, CancellationToken ct = default)
    {
        run.Token.ThrowIfCancellationRequested();

        using var either = CancellationTokenSource.CreateLinkedTokenSource(run.Token, ct);

        run.Began(null, language);

        try
        {
            return await scaffold.ExecuteCodeCaptureOutputsAsync(code, language, either.Token);
        }
        finally
        {
            run.Ended(null);
        }
    }

    public void SetActiveLayout(string layoutId)
    {
        run.Token.ThrowIfCancellationRequested();
        Notebook.SetActiveLayout(layoutId);
    }

    public void SetActiveTheme(string themeId)
    {
        run.Token.ThrowIfCancellationRequested();
        Notebook.SetActiveTheme(themeId);
    }
}
