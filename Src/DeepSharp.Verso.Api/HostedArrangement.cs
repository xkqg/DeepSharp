// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// What the layout a notebook is shown in draws of its own, as the engine draws it: Verso's dashboard its grid of tiles, its
/// presentation its column of what the cells show — each with a slot for every cell it shows, which names the cell
/// (<c>data-cell-slot</c>), for a view to place the cell in. Nothing for a layout that draws no arrangement of its own: the
/// notebook's own layout is its list of cells, which a version already says. Why, when the layout's part failed to draw.
/// </summary>
/// <param name="Html">The arrangement; nothing when the layout draws none of its own, or failed to draw.</param>
/// <param name="Fault">Why the layout's part failed to draw; nothing when it did not fail.</param>
public readonly record struct HostedArrangement(string? Html, string? Fault);
