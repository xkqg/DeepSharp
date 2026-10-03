// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a file's bytes hold, once they are in hand.
/// </summary>
/// <remarks>
/// For bytes read once to be fingerprinted and then parsed as they are, so the rows are the ones the fingerprint names:
/// they are read here as the file itself would be read, never by a rule of their own.
/// </remarks>
public static class FileBytesExtensions
{
    extension(byte[] bytes)
    {
        /// <summary>
        /// The text a file's bytes hold, read as a file is read as text: in the encoding its byte-order mark names, UTF-8 when
        /// it names none.
        /// </summary>
        /// <returns>The text, without its byte-order mark.</returns>
        /// <exception cref="ArgumentNullException">No bytes are handed in.</exception>
        /// <remarks>The text <c>File.ReadAllText</c> reads from a file holding the same bytes.</remarks>
        public string AsText()
        {
            ArgumentNullException.ThrowIfNull(bytes);

            using var reader = new StreamReader(new MemoryStream(bytes, writable: false), Encoding.UTF8, detectEncodingFromByteOrderMarks: true);

            return reader.ReadToEnd();
        }
    }
}
