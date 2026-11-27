// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>A word a control carries, read back as a column's kind; the word a kind is written as is its <c>Word()</c>, the pipeline file's own.</summary>
internal static class ColumnKindExtensions
{
    extension(string? word)
    {
        /// <summary>The kind a word names.</summary>
        /// <returns>
        /// The kind whose word it is, in whatever case; nothing for any other word — a number, or several words joined by a
        /// comma, which the runtime's own parser reads as a kind of its own.
        /// </returns>
        public ColumnKind? AsKind() =>
            Enum.GetValues<ColumnKind>().Where(kind => string.Equals(kind.Word(), word, StringComparison.OrdinalIgnoreCase)).Cast<ColumnKind?>().FirstOrDefault();
    }
}
