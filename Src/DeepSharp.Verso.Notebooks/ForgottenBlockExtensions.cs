// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>One part of a cell's text: a step a format kept the text of and forgot was a block, or text that is none.</summary>
/// <param name="Text">The part's text.</param>
/// <param name="Step">The step its text reads as; nothing for text that is no step.</param>
internal readonly record struct CellPart(string Text, IPipelineStep? Step);

/// <summary>
/// Where a cell holds steps a format forgot were blocks: every format but Verso's own keeps a block's text and loses what
/// kind of cell it was.
/// </summary>
/// <remarks>
/// Jupyter brings a block back as a raw cell, and a Jupyter or .dib file another program wrote may hold one as code; such
/// a cell is a step when the whole of its text reads as one. Markdown writes a block as a fence of the block's language or
/// its kind, and reads the fences of several blocks back as one cell of text with the text between them; each such fence
/// whose text reads as a step is one, and the text around them stays text. A cell of any other kind holds no step. The
/// guard that refuses such a file and the button that makes blocks of its steps again read a cell by this one rule.
/// </remarks>
internal static partial class ForgottenBlockExtensions
{
    extension(CellModel cell)
    {
        /// <summary>The parts a cell's text falls into, in their order: the steps it holds, and the text around them.</summary>
        /// <param name="catalog">The steps a text may read as.</param>
        /// <returns>The parts; one part, no step, for a cell that holds none.</returns>
        public IReadOnlyList<CellPart> Parts(StepCatalog catalog) => cell.Type switch
        {
            "raw" or "code" => [new CellPart(cell.Source, catalog.TryReadStep(cell.Source))],
            "markdown" => Fences(cell.Source.ReplaceLineEndings("\n"), catalog),
            _ => [new CellPart(cell.Source, null)],
        };

        /// <summary>Whether a cell holds a step a format forgot was a block.</summary>
        /// <param name="catalog">The steps a text may read as.</param>
        /// <returns>Whether any of its parts is a step.</returns>
        public bool HoldsForgottenSteps(StepCatalog catalog) => cell.Parts(catalog).Any(part => part.Step is not null);
    }

    // A cell of text cut at each fence that holds a step: the text before it, the step, and so on, the text after the last
    // one. A fence of the block's language whose text is no step stays in the text around it.
    private static IReadOnlyList<CellPart> Fences(string text, StepCatalog catalog)
    {
        var parts = new List<CellPart>();
        var from = 0;

        foreach (Match fence in Fenced().Matches(text))
        {
            if (catalog.TryReadStep(fence.Groups["text"].Value) is not { } step)
            {
                continue;
            }

            Text(parts, text[from..fence.Index]);
            parts.Add(new CellPart(fence.Groups["text"].Value, step));
            from = fence.Index + fence.Length;
        }

        if (parts.Count == 0)
        {
            return [new CellPart(text, null)];
        }

        Text(parts, text[from..]);

        return parts;
    }

    // Text between steps, without the blank lines that only parted it from them; nothing when it is only blank.
    private static void Text(List<CellPart> parts, string text)
    {
        if (!string.IsNullOrWhiteSpace(text))
        {
            parts.Add(new CellPart(text.Trim(), null));
        }
    }

    // A fence opened on a line of its own with the block's language or its kind, and closed on a line of its own, as
    // Markdown writes a block.
    [GeneratedRegex(@"^```(?:pdd|deepsharp\.step)[ \t]*\n(?<text>[\s\S]*?)\n?^```[ \t]*$", RegexOptions.Multiline)]
    private static partial Regex Fenced();
}
