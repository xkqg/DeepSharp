// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// The layout a notebook is shown in, what it lets a person do to the notebook's cells, and whether it has a cell's
/// properties panel.
/// </summary>
/// <param name="Id">
/// The layout's id. A notebook that names no layout, or one the engine does not have, is shown in the notebook's own
/// layout, as Verso's editors show it, and keeps the name it had; nothing only when the engine has no layout at all, which
/// then allows every cell action.
/// </param>
/// <param name="Allows">What it lets a person do to the cells: a page offers only these, as Verso's editors do.</param>
/// <param name="HasPropertiesPanel">
/// Whether it has a cell's properties panel, as Verso's editors offer the panel only in a layout that has one — the
/// notebook's own layout, not the dashboard or the presentation; a layout the engine does not have counts as having one,
/// as it allows every cell action.
/// </param>
public readonly record struct HostedLayout(string? Id, LayoutAllows Allows, bool HasPropertiesPanel);
