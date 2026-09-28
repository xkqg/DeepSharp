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
/// ends a Run All; everything else goes to the notebook's own operations, let through on the run and held until it has
/// landed. What runs cells or code is not held: a stop ends it by starting a kernel afresh, never by waiting for it.
/// </remarks>
internal sealed class RunPort(Scaffold scaffold, Run run) : NotebookPort(scaffold)
{
    public override CancellationToken Token => run.Token;

    public override async Task ExecuteCellAsync(Guid cellId)
    {
        run.ThrowIfStopped();
        await Scaffold.ExecuteCellAsync(cellId, run.Token);
    }

    // Run All puts every kernel away and clears the notebook's variables before its first cell begins: on its way until
    // then, so a stop during that reset waits for it.
    public override async Task ExecuteAllAsync()
    {
        using var resetting = run.AdmitUntilBegan();

        await Scaffold.ExecuteAllAsync(run.Token);
    }

    // As Verso's own operations run from a cell on, and refused before each cell once the run is stopped.
    public override async Task ExecuteFromAsync(Guid cellId)
    {
        run.ThrowIfStopped();

        var cells = Scaffold.Cells;
        var from = cells.TakeWhile(cell => cell.Id != cellId).Count();

        if (from == cells.Count)
        {
            throw new InvalidOperationException($"Cell {cellId} not found.");
        }

        foreach (var cell in cells.Skip(from))
        {
            run.ThrowIfStopped();
            await Scaffold.ExecuteCellAsync(cell.Id, run.Token);
        }
    }

    public override Task ExecuteCodeAsync(string code, string? language = null, CancellationToken ct = default) =>
        CodeAsync(language, ct, token => Scaffold.ExecuteCodeAsync(code, language, token));

    // Forwarded as the engine has it: the interface's own default runs the code and hands back no outputs.
    public override Task<IReadOnlyList<CellOutput>> ExecuteCodeCaptureOutputsAsync(string code, string? language = null, CancellationToken ct = default) =>
        CodeAsync(language, ct, token => Scaffold.ExecuteCodeCaptureOutputsAsync(code, language, token));

    // Let through on the run until a stop takes it, and held until it has landed.
    protected override Admission Admit() => run.Admit();

    // Code with no cell begins no cell the engine tells of, so the run is told here what runs — the kernel the engine runs
    // it in, the language it was asked in, else the notebook's default kernel — before it is let run: a stop before that
    // refuses it, and one after takes that kernel as the one to start afresh. It is handed the run's own token, marked in
    // the very step that stops the run, unless it was asked with another that can end it too.
    private async Task<T> CodeAsync<T>(string? language, CancellationToken ct, Func<CancellationToken, Task<T>> runs)
    {
        run.Began(null, language ?? Scaffold.DefaultKernelId);

        try
        {
            run.ThrowIfStopped();

            if (!ct.CanBeCanceled || ct == run.Token)
            {
                return await runs(run.Token);
            }

            using var either = CancellationTokenSource.CreateLinkedTokenSource(run.Token, ct);

            return await runs(either.Token);
        }
        finally
        {
            run.Ended(null);
        }
    }
}
