// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A theme the engine can draw a notebook in, as Verso's View panel lists it.</summary>
/// <param name="Id">The theme's id, which a switch names and a notebook saves as its choice.</param>
/// <param name="Name">The name a person knows it by.</param>
/// <param name="Tone">What kind of theme it is.</param>
/// <param name="Css">
/// Its colours, fonts and spacing as the custom properties every Verso surface styles itself from — the one block Verso's
/// engine writes for a theme — for a view to draw the notebook in.
/// </param>
public readonly record struct HostedTheme(string Id, string Name, ThemeTone Tone, string Css);
