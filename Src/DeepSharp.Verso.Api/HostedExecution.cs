// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// Something the engine runs that no run owns, so a view shows it running with nothing to stop: a block a change runs —
/// a click, a changed field — or what a stopped run left running.
/// </summary>
/// <param name="Cell">The cell that runs; nothing for code a button ran in no cell.</param>
/// <param name="Since">When it began.</param>
/// <param name="LeftBehind">
/// Whether a stop left it behind: the run it was part of was stopped while it ran, its kernel was started afresh and the
/// C# turn handed on, and the code goes on in the background until it ends by itself — code that never ends, until the
/// application does.
/// </param>
public readonly record struct HostedExecution(Guid? Cell, DateTimeOffset Since, bool LeftBehind);
