// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What the layout a notebook is shown in lets a person do to its cells, named as Verso names it.</summary>
[Flags]
public enum LayoutAllows
{
    /// <summary>Nothing.</summary>
    None = 0,

    /// <summary>Adding a cell.</summary>
    CellInsert = 1,

    /// <summary>Taking a cell away.</summary>
    CellDelete = 2,

    /// <summary>Moving a cell.</summary>
    CellReorder = 4,

    /// <summary>Changing a cell's text or kind.</summary>
    CellEdit = 8,

    /// <summary>Resizing a cell.</summary>
    CellResize = 16,

    /// <summary>Running a cell.</summary>
    CellExecute = 32,

    /// <summary>Choosing several cells at once.</summary>
    MultiSelect = 64,

    /// <summary>The layout hears of the notebook's changes.</summary>
    NotebookEvents = 128,
}
