// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// What a person does to the arrangement a layout drew — a tile moved, resized or run — as Verso's editors hand it to the
/// layout's own part: the act as the arrangement names it on its control (<c>data-action</c>), what the act carries, and
/// what it is aimed at (<c>data-target-id</c>).
/// </summary>
/// <param name="Layout">The layout that drew what was acted on; an act on a layout the notebook is no longer shown in is refused.</param>
/// <param name="Action">The act, as the arrangement names it.</param>
/// <param name="Payload">What the act carries: for a tile moved or resized, its place as JSON.</param>
/// <param name="Target">What the act is aimed at: for a tile, its cell; nothing when it names nothing.</param>
public readonly record struct HostedLayoutInteraction(string Layout, string Action, string Payload, string? Target);
