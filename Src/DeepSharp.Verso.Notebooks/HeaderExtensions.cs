// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Notebooks;

/// <summary>Where a column stands among a source's columns.</summary>
internal static class HeaderExtensions
{
    extension(IReadOnlyList<string> header)
    {
        /// <summary>The place of a column in a source's header.</summary>
        /// <param name="column">The column.</param>
        /// <returns>Its place, counting from nought; minus one for a column the source does not have.</returns>
        public int PlaceOf(string column)
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

        /// <summary>The columns of a source from one to another, whichever stands first, both included.</summary>
        /// <param name="from">Where a range was started.</param>
        /// <param name="to">Where it was ended.</param>
        /// <returns>The columns in the source's order; the one it was ended on alone when the source lacks either.</returns>
        public IReadOnlyList<string> Between(string from, string to)
        {
            var start = header.PlaceOf(from);
            var end = header.PlaceOf(to);

            return start < 0 || end < 0 ? [to] : [.. header.Skip(Math.Min(start, end)).Take(Math.Abs(end - start) + 1)];
        }
    }
}
