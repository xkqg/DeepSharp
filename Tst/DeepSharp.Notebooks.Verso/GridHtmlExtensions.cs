// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Notebooks;

/// <summary>What a test reads off a grid's page.</summary>
internal static class GridHtmlExtensions
{
    /// <summary>Whether the grid has a column of that name: its header, then its buttons or the end of the cell.</summary>
    /// <param name="grid">The grid's page.</param>
    /// <param name="column">The column's name.</param>
    /// <returns><see langword="true"/> when the column is a header of the grid.</returns>
    public static bool Heads(this string grid, string column) =>
        grid.Contains($"<th>{column}</th>", StringComparison.Ordinal)
        || grid.Contains($"<th>{column} <button", StringComparison.Ordinal);
}
