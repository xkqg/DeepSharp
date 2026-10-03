// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The columns a key can name, read the same way by the form and by the editor.</summary>
internal static class KnownColumnExtensions
{
    extension(IEnumerable<KnownColumn> columns)
    {
        /// <summary>The names of the columns of a kind a parameter works on, each once, in the order they come.</summary>
        /// <param name="accepts">The kinds the parameter works on.</param>
        /// <returns>Their names.</returns>
        public IReadOnlyList<string> NamesOf(IReadOnlyList<ColumnKind> accepts) =>
            [.. columns.Where(column => accepts.Contains(column.Kind)).Select(column => column.Name).Distinct(StringComparer.Ordinal)];
    }
}
