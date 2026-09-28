// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// A kind a cell can be: its type, the language its text is written in, the name a person knows it by, whether a person
/// writes its text, and whether its cell is shown rendered once it has run.
/// </summary>
/// <param name="Type">The cell's type: code, Markdown, a DeepSharp block and so on.</param>
/// <param name="Language">The language its text is written in, when the kind has one: C# for code, the blocks' own for a block.</param>
/// <param name="Name">The name a person knows it by, as Verso's editors show it.</param>
/// <param name="Editable">
/// Whether a person writes its text; a kind whose cell is drawn from what the notebook holds, such as the parameters
/// form, is not written, and Verso's editors show no text for it and run it once when it shows nothing.
/// </param>
/// <param name="Rendered">
/// Whether a cell of this kind, once it has run and while it is not selected, is shown rendered in place of its text, as
/// Verso's editors fold Markdown, HTML and Mermaid; while selected, its text stands in for its rendering, and running it
/// leaves the selection so the rendering shows. It is the engine's own word on the kind, from the part that draws it.
/// </param>
/// <remarks>A verb that takes a kind finds it among the notebook's kinds by its type and language alone.</remarks>
public readonly record struct HostedKind(string Type, string? Language, string Name, bool Editable, bool Rendered);
