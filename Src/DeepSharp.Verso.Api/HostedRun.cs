// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// A run under way — a cell's, or a toolbar button's — so a view can say what runs, or waits to, and offer to stop it.
/// </summary>
/// <param name="Number">Which run it is, counted from one for each open notebook: what a stop names.</param>
/// <param name="Cell">The cell that runs, or waits to; nothing for a button's run while none of its cells runs.</param>
/// <param name="Since">When what runs now began, or when the run was asked while nothing of it runs.</param>
/// <param name="Waits">
/// Whether it waits for a C# run of another notebook to end, since C# runs take their turn across the process.
/// </param>
public readonly record struct HostedRun(long Number, Guid? Cell, DateTimeOffset Since, bool Waits);
