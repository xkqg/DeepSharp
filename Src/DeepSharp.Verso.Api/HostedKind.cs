// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A kind a cell can be: its type, the language its text is written in, and the name a person knows it by.</summary>
/// <param name="Type">The cell's type: code, Markdown, a DeepSharp block and so on.</param>
/// <param name="Language">The language its text is written in, when the kind has one: C# for code, the blocks' own for a block.</param>
/// <param name="Name">The name a person knows it by, as Verso's editors show it.</param>
public readonly record struct HostedKind(string Type, string? Language, string Name);
