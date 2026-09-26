// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What came of a click.</summary>
/// <param name="StateChanged">Whether it changed the notebook's cells, so a page showing them draws them again.</param>
/// <param name="Answer">What the part answered, which the cell now shows; nothing when it answered nothing.</param>
public readonly record struct GestureResult(bool StateChanged, string? Answer);
