// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a toolbar button is handed when it is pressed, or asked whether it can be: the notebook, its cells as they stand
/// at each look, and the cells the button is pressed for.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <param name="selected">The cells the button is pressed for.</param>
/// <param name="notebook">What the button may do to the notebook: a press's run's operations, or the notebook's own.</param>
/// <param name="stopped">Marked when the press's run is stopped; none for a button asked whether it can be pressed.</param>
internal sealed class ToolbarContext(Scaffold scaffold, IReadOnlyList<Guid> selected, INotebookOperations notebook, CancellationToken stopped)
    : NotebookContext(scaffold, notebook, stopped), IToolbarActionContext
{
    public IReadOnlyList<Guid> SelectedCellIds => selected;

    public IReadOnlyList<CellModel> NotebookCells => Scaffold.Cells;

    public string? ActiveKernelId => Scaffold.DefaultKernelId;
}
