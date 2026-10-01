// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>What a file holds, shown in a message about the file.</summary>
/// <remarks>
/// A file is written by whoever wrote it, so its text is shown in a message as words and nothing else: a key that holds a
/// line break cannot start a line of its own in a log, one that holds a bidirectional override cannot turn round what
/// follows it, and one of a million letters cannot make a message of a million letters. Every reader of a pipeline file
/// that echoes the file's text in a refusal shows it this way. The same rule as <c>DeepSharp.Networks.FileTextExtensions</c>
/// — kept here rather than referenced there, since a pipeline never carries the deep-learning core as a dependency.
/// </remarks>
internal static class FileTextExtensions
{
    // How many characters of a file's text a message shows, its escapes included.
    private const int Shown = 200;

    private const char Backslash = (char)92;

    extension(string text)
    {
        /// <summary>
        /// The text as a message shows what a file holds: each character that would break the message's line, hide what
        /// follows it or turn it round — a control character, a format character such as a bidirectional override, a line
        /// or paragraph separator, half of a surrogate pair standing alone — written as its escape: <c>\r</c>, <c>\n</c> and
        /// <c>\t</c> by name, any other control character as <c>\x</c> and two hexadecimal digits, and the rest as <c>\u</c>
        /// and four. The backslash itself is written <c>\\</c>, so an escape is never the file's own text. Past 200
        /// characters, escapes included, the text is cut short and says how long it was: <c>mmm… (1000000 characters)</c>.
        /// Quote marks are the message's to add.
        /// </summary>
        /// <returns>The text as a message shows it: the text itself when nothing in it needs showing otherwise.</returns>
        /// <exception cref="ArgumentNullException">There is no text.</exception>
        internal string Quoted()
        {
            ArgumentNullException.ThrowIfNull(text);

            var shown = new StringBuilder();

            for (var at = 0; at < text.Length; at++)
            {
                var pair = char.IsHighSurrogate(text[at]) && at + 1 < text.Length && char.IsLowSurrogate(text[at + 1]);
                var piece = pair ? text.Substring(at, 2) : Escaped(text[at]);

                if (shown.Length + piece.Length > Shown)
                {
                    return string.Create(CultureInfo.InvariantCulture, $"{shown}… ({text.Length} characters)");
                }

                shown.Append(piece);
                at += pair ? 1 : 0;
            }

            return shown.Equals(text) ? text : shown.ToString();
        }
    }

    // One character as a message shows it.
    private static string Escaped(char character) => character switch
    {
        Backslash => $"{Backslash}{Backslash}",
        '\r' => $"{Backslash}r",
        '\n' => $"{Backslash}n",
        '\t' => $"{Backslash}t",
        _ when char.IsControl(character) => string.Create(CultureInfo.InvariantCulture, $"{Backslash}x{(int)character:x2}"),
        _ when char.IsSurrogate(character)
               || char.GetUnicodeCategory(character) is UnicodeCategory.Format or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator
            => string.Create(CultureInfo.InvariantCulture, $"{Backslash}u{(int)character:x4}"),
        _ => character.ToString(),
    };
}
