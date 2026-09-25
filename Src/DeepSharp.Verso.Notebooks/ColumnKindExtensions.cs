// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>A column's kind as a notebook writes it, and back: the one word a control carries and a card shows.</summary>
internal static class ColumnKindExtensions
{
    /// <summary>The word a kind is written as.</summary>
    /// <param name="kind">The kind.</param>
    /// <returns>Its name, in lower case, as a pipeline file writes it.</returns>
    public static string Word(this ColumnKind kind) => kind.ToString().ToLowerInvariant();

    /// <summary>The kind a word names.</summary>
    /// <param name="word">The word, as a control carries it.</param>
    /// <returns>The kind; nothing for a word that names none.</returns>
    public static ColumnKind? AsKind(this string? word) =>
        Enum.TryParse<ColumnKind>(word, ignoreCase: true, out var kind) && Enum.IsDefined(kind) ? kind : null;
}
