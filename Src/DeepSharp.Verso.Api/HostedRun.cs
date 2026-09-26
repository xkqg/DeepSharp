// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A run under way: the cell that runs, and since when, so a view that begins while it runs can offer to stop it.</summary>
/// <param name="Cell">The cell that runs.</param>
/// <param name="Since">When it began.</param>
public readonly record struct HostedRun(Guid Cell, DateTimeOffset Since);
