// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using Onnxify.Safetensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>A tensor as a file PyTorch wrote holds it: its name, and its numbers in the kind and the layout PyTorch saved them in.</summary>
/// <param name="Name">Its name in the file: the path of the slot it is for, as PyTorch names its state.</param>
/// <param name="Numbers">
/// Its numbers, its kind of number and its shape, as the file holds them: read out of the file only when asked for, so the
/// numbers of a tensor no slot takes are never read.
/// </param>
internal readonly record struct StoredTensor(string Name, Func<TensorView> Numbers);

/// <summary>A file's numbers as the 32-bit floats a slot holds.</summary>
internal static class StoredNumbersExtensions
{
    // Reads one number from the bytes that hold it.
    private delegate float Reading(ReadOnlySpan<byte> bytes);

    extension(TensorView stored)
    {
        /// <summary>
        /// The numbers as 32-bit floats, in the order they are stored: 16-bit floats and bfloat16 widened exactly, 64-bit
        /// floats rounded to the nearest; nothing for numbers that are not floats.
        /// </summary>
        internal float[]? Singles() => stored.DataType switch
        {
            DataType.F32 => Each(stored.Data.Span, sizeof(float), static bytes => BinaryPrimitives.ReadSingleLittleEndian(bytes)),
            DataType.F16 => Each(stored.Data.Span, sizeof(ushort), static bytes => (float)BinaryPrimitives.ReadHalfLittleEndian(bytes)),
            DataType.Bf16 => Each(stored.Data.Span, sizeof(ushort), static bytes => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(bytes) << 16)),
            DataType.F64 => Each(stored.Data.Span, sizeof(double), static bytes => (float)BinaryPrimitives.ReadDoubleLittleEndian(bytes)),
            _ => null,
        };
    }

    // Every number the bytes hold, each read from as many bytes as one takes, little-endian as safetensors stores them.
    private static float[] Each(ReadOnlySpan<byte> bytes, int size, Reading read)
    {
        var values = new float[bytes.Length / size];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = read(bytes.Slice(at * size, size));
        }

        return values;
    }
}
