// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a toolbar button is handed when it is pressed: the notebook, its cells as they stand at each look, and the
/// cells the button is pressed for.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <param name="selected">The cells the button is pressed for.</param>
internal sealed class ToolbarContext(Scaffold scaffold, IReadOnlyList<Guid> selected) : NotebookContext(scaffold), IToolbarActionContext
{
    public IReadOnlyList<Guid> SelectedCellIds => selected;

    public IReadOnlyList<CellModel> NotebookCells => Scaffold.Cells;

    public string? ActiveKernelId => Scaffold.DefaultKernelId;
}
