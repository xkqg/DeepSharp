// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Security.Cryptography;
using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a file's bytes hold, once they are in hand.
/// </summary>
/// <remarks>
/// For bytes read once to be fingerprinted and then parsed as they are, so the rows are the ones the fingerprint names:
/// they are read here as the file itself would be read, never by a rule of their own. The same bytes are also written
/// whole, so a file is never seen half written, and fingerprinted the one way every part of the library fingerprints.
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

        /// <summary>The fingerprint of a file's bytes: their SHA-256, as lower-case hexadecimal.</summary>
        /// <returns>Sixty-four characters, the spelling <c>sha256sum</c> writes.</returns>
        /// <exception cref="ArgumentNullException">No bytes are handed in.</exception>
        /// <remarks>
        /// What names the bytes a pipeline read and a landing holds, wherever it is asked: the same bytes give the same
        /// fingerprint on every machine, and any other bytes give another.
        /// </remarks>
        public string Fingerprint()
        {
            ArgumentNullException.ThrowIfNull(bytes);

            return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        }

        /// <summary>Writes the bytes as a file, whole, replacing what stands at the path.</summary>
        /// <param name="path">Where the file is.</param>
        /// <exception cref="ArgumentNullException">No bytes are handed in, or no path.</exception>
        /// <exception cref="IOException">The file could not be written or moved into place.</exception>
        /// <remarks>
        /// The bytes go to a name of their own beside the file, are on the disk, and are moved into place in one step: whoever
        /// looks finds the old file or the new one, never half of either. A write or a move that fails leaves nothing of its
        /// own behind.
        /// </remarks>
        public void WriteWhole(string path)
        {
            ArgumentNullException.ThrowIfNull(bytes);
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            WholeFile.Replace(bytes, path);
        }

        /// <summary>Writes the bytes as a file, whole, replacing what stands at the path.</summary>
        /// <param name="path">Where the file is.</param>
        /// <param name="cancellation">Ends the write before the bytes are moved into place.</param>
        /// <returns>When the file is written.</returns>
        /// <exception cref="ArgumentNullException">No bytes are handed in, or no path.</exception>
        /// <exception cref="OperationCanceledException">The write was cancelled before the file was moved into place.</exception>
        /// <exception cref="IOException">The file could not be written or moved into place.</exception>
        /// <remarks>As <see cref="WriteWhole"/>, without holding a thread while the bytes are written.</remarks>
        public Task WriteWholeAsync(string path, CancellationToken cancellation = default)
        {
            ArgumentNullException.ThrowIfNull(bytes);
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            return WholeFile.ReplaceAsync(bytes, path, cancellation);
        }

        /// <summary>Writes the bytes as a file, whole, where nothing stands — and never replaces what does.</summary>
        /// <param name="path">Where the file is.</param>
        /// <param name="cancellation">Ends the write before the bytes are moved into place.</param>
        /// <returns>When the file stands there: written by this call, or already holding the same bytes.</returns>
        /// <exception cref="ArgumentNullException">No bytes are handed in, or no path.</exception>
        /// <exception cref="OperationCanceledException">The write was cancelled before the file was moved into place.</exception>
        /// <exception cref="IOException">Other bytes already stand there, or the move was refused for another reason.</exception>
        /// <remarks>
        /// For a file that is written once and then read for what it was. The move never replaces: of many writers racing
        /// for one place exactly one file stands, the same bytes already there are the same file, and other bytes are
        /// refused with the first left as it was. A move refused while nothing stands there is not swallowed.
        /// </remarks>
        public Task WriteWholeOnceAsync(string path, CancellationToken cancellation = default)
        {
            ArgumentNullException.ThrowIfNull(bytes);
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            return WholeFile.WriteOnceAsync(bytes, path, cancellation);
        }
    }
}
