// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A cell of an open notebook as it stands: what it is, what it holds, and what it shows.</summary>
/// <param name="Id">The cell's id; a change that rewrites a block replaces it.</param>
/// <param name="Type">The cell's type: a DeepSharp block, C# code, markdown and so on.</param>
/// <param name="Language">The language its text is written in, when it has one.</param>
/// <param name="Source">Its text.</param>
/// <param name="Outputs">What it shows, in order.</param>
public readonly record struct HostedCell(Guid Id, string Type, string? Language, string Source, IReadOnlyList<HostedOutput> Outputs);
