// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>The layout a notebook is shown in, and what it lets a person do to the notebook's cells.</summary>
/// <param name="Id">The layout's id; nothing when the notebook names a layout the engine does not have, which then allows every cell action.</param>
/// <param name="Allows">What it lets a person do to the cells: a page offers only these, as Verso's editors do.</param>
public readonly record struct HostedLayout(string? Id, LayoutAllows Allows);
