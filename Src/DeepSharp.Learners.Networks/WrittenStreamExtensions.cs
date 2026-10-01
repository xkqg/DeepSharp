// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;

namespace DeepSharp.Learners.Networks;

/// <summary>What a JSON writer wrote into a stream, as the text of a file that is the same file on every system.</summary>
internal static class WrittenStreamExtensions
{
    extension(MemoryStream stream)
    {
        /// <summary>
        /// The text the stream holds, each line ended with a line feed alone: every carriage return the writer ended a line
        /// with is left out of its bytes before the text is made.
        /// </summary>
        /// <returns>The text, made once, at the length of the file itself.</returns>
        /// <remarks>
        /// A writer ends its lines the way the machine does on some runtimes. Making the text first and ending its lines
        /// afterwards made a text a character longer for every line, on those machines alone — and .NET makes no text longer
        /// than 1,073,741,791 characters, so the largest network such a machine could write was smaller than the file it
        /// would have written. In JSON a writer writes, a carriage return inside a string is escaped, so the only ones among
        /// its bytes end its lines; no byte of a character written in more than one byte is one.
        /// </remarks>
        public string TextWithLineFeeds()
        {
            var written = stream.GetBuffer().AsSpan(0, (int)stream.Length);
            var kept = 0;

            foreach (var letter in written)
            {
                if (letter != (byte)'\r')
                {
                    written[kept++] = letter;
                }
            }

            return Encoding.UTF8.GetString(written[..kept]);
        }
    }
}
