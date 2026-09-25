// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Notebooks;

/// <summary>Where a column stands among a source's columns.</summary>
internal static class HeaderExtensions
{
    /// <summary>The place of a column in a source's header.</summary>
    /// <param name="header">The source's columns, in their order.</param>
    /// <param name="column">The column.</param>
    /// <returns>Its place, counting from nought; minus one for a column the source does not have.</returns>
    public static int PlaceOf(this IReadOnlyList<string> header, string column)
    {
        for (var at = 0; at < header.Count; at++)
        {
            if (header[at] == column)
            {
                return at;
            }
        }

        return -1;
    }
}
