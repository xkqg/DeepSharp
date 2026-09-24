// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>The columns a key can name, read the same way by the form and by the editor.</summary>
internal static class KnownColumnExtensions
{
    /// <summary>The names of the columns of a kind a parameter works on, each once, in the order they come.</summary>
    /// <param name="columns">The columns known.</param>
    /// <param name="accepts">The kinds the parameter works on.</param>
    /// <returns>Their names.</returns>
    public static IReadOnlyList<string> NamesOf(this IEnumerable<KnownColumn> columns, IReadOnlyList<ColumnKind> accepts) =>
        [.. columns.Where(column => accepts.Contains(column.Kind)).Select(column => column.Name).Distinct(StringComparer.Ordinal)];
}
