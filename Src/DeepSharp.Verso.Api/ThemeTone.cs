// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What kind of theme a theme is, as Verso's engine says it.</summary>
public enum ThemeTone
{
    /// <summary>Dark text on a light ground.</summary>
    Light,

    /// <summary>Light text on a dark ground.</summary>
    Dark,

    /// <summary>The strongest contrast, for readers who need it.</summary>
    HighContrast,
}
