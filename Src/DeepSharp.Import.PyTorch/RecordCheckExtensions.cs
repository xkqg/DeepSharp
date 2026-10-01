// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Import.PyTorch;

/// <summary>The check a zip archive keeps of each of its records: the CRC-32 of the record's bytes.</summary>
internal static class RecordCheckExtensions
{
    // The remainder of each byte, as the reflected polynomial 0xEDB88320 of zip's CRC-32 leaves it.
    private static readonly uint[] Remainders = [.. Enumerable.Range(0, 256).Select(value => Remainder((uint)value))];

    extension(ReadOnlySpan<byte> bytes)
    {
        /// <summary>The CRC-32 of the bytes, as a zip archive keeps it of a record.</summary>
        internal uint Crc32()
        {
            var check = uint.MaxValue;

            foreach (var value in bytes)
            {
                check = Remainders[(check ^ value) & 0xFF] ^ (check >> 8);
            }

            return ~check;
        }
    }

    private static uint Remainder(uint value)
    {
        for (var bit = 0; bit < 8; bit++)
        {
            value = (value & 1) != 0 ? 0xEDB88320 ^ (value >> 1) : value >> 1;
        }

        return value;
    }
}
