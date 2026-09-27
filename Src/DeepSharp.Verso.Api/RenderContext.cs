// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a part drawing or changing a cell's properties is handed: the notebook, and the cell — the one a panel is shown
/// for, which is the one selected. A host of notebooks draws no pixels, so the cell has no size.
/// </summary>
/// <param name="scaffold">The notebook.</param>
/// <param name="cell">The cell.</param>
internal sealed class RenderContext(Scaffold scaffold, CellModel cell)
    : NotebookContext(scaffold, scaffold.NotebookOps, CancellationToken.None), ICellRenderContext
{
    public Guid CellId => cell.Id;

    public IReadOnlyDictionary<string, object> CellMetadata => cell.Metadata;

    public (double Width, double Height) Dimensions => (0, 0);

    public bool IsSelected => true;
}
