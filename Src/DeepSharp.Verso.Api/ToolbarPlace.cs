// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>Where a toolbar button belongs.</summary>
public enum ToolbarPlace
{
    /// <summary>The notebook's own toolbar.</summary>
    MainToolbar = 0,

    /// <summary>A cell's toolbar: the button acts on the cells it is pressed for.</summary>
    CellToolbar = 1,

    /// <summary>A menu opened on a cell.</summary>
    ContextMenu = 2,

    /// <summary>The menu of ways to export the notebook.</summary>
    ExportMenu = 3,
}
