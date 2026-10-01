// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A pickle written instruction by instruction, protocol 2 as torch.save writes it, and the archive torch.save would put it
/// in: what a test needs of a file nobody's writer would write.
/// </summary>
internal sealed class HandWrittenPickle
{
    private readonly List<byte> _bytes = [0x80, 0x02];

    /// <summary>Where the next instruction goes.</summary>
    public int At => _bytes.Count;

    /// <summary>Where the last instruction written stands: the byte a refusal of it names.</summary>
    public int Last { get; private set; }

    /// <summary>The pickle as written so far, with no STOP after it.</summary>
    public byte[] Unfinished => [.. _bytes];

    /// <summary>An instruction, by its byte, with the bytes it reads after it.</summary>
    public HandWrittenPickle Op(char instruction, params byte[] arguments) => Op((byte)instruction, arguments);

    /// <summary>An instruction, by its byte, with the bytes it reads after it.</summary>
    public HandWrittenPickle Op(byte instruction, params byte[] arguments)
    {
        Last = _bytes.Count;
        _bytes.Add(instruction);
        _bytes.AddRange(arguments);

        return this;
    }

    /// <summary>GLOBAL: a module and a name, each on a line.</summary>
    public HandWrittenPickle Global(string module, string name) => Op('c', Encoding.UTF8.GetBytes($"{module}\n{name}\n"));

    /// <summary>BINUNICODE: text, its length first.</summary>
    public HandWrittenPickle Text(string text) => Op('X', [.. Length(Encoding.UTF8.GetByteCount(text)), .. Encoding.UTF8.GetBytes(text)]);

    /// <summary>A whole number: BININT1 below 256, BININT otherwise.</summary>
    public HandWrittenPickle Number(int value) => value is >= 0 and < 256 ? Op('K', (byte)value) : Op('J', Length(value));

    /// <summary>MARK.</summary>
    public HandWrittenPickle Mark() => Op('(');

    /// <summary>TUPLE: every value since the last mark.</summary>
    public HandWrittenPickle Tuple() => Op('t');

    /// <summary>A tuple of whole numbers.</summary>
    public HandWrittenPickle Numbers(params int[] values)
    {
        Mark();

        foreach (var value in values)
        {
            Number(value);
        }

        return Tuple();
    }

    /// <summary>A storage, loaded by its persistent id as torch.save writes one: ('storage', its kind, its key, 'cpu', its count).</summary>
    public HandWrittenPickle Storage(string key, int count, string kind = "FloatStorage") =>
        Mark().Text("storage").Global("torch", kind).Text(key).Text("cpu").Number(count).Tuple().Op('Q');

    /// <summary>A tensor as torch.save writes one: torch._utils._rebuild_tensor_v2 called on its storage, offset, lengths and strides.</summary>
    public HandWrittenPickle Tensor(string key, int count, int[] lengths, int[] strides, int offset = 0) =>
        Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage(key, count).Number(offset).Numbers(lengths).Numbers(strides)
            .Op(0x89).Global("collections", "OrderedDict").Op(')').Op('R').Tuple().Op('R');

    /// <summary>The pickle, its STOP written.</summary>
    public byte[] Stop() => [.. _bytes, (byte)'.'];

    /// <summary>The archive torch.save writes, holding a pickle and the storages it names, and the files it keeps beside them.</summary>
    /// <param name="pickle">data.pkl.</param>
    /// <param name="storages">Each storage's bytes, by its key.</param>
    public static MemoryStream Archive(byte[] pickle, IReadOnlyDictionary<string, byte[]>? storages = null) =>
        Zip(
        [
            new("archive/data.pkl", pickle),
            new("archive/byteorder", "little"u8.ToArray()),
            new("archive/version", "3\n"u8.ToArray()),
            .. (storages ?? new Dictionary<string, byte[]>()).Select(storage => new KeyValuePair<string, byte[]>($"archive/data/{storage.Key}", storage.Value)),
        ]);

    /// <summary>A zip archive holding the given files, in that order, stored as torch.save stores them.</summary>
    public static MemoryStream Zip(IEnumerable<KeyValuePair<string, byte[]>> files, CompressionLevel level = CompressionLevel.NoCompression)
    {
        var zip = new MemoryStream();

        using (var archive = new ZipArchive(zip, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, bytes) in files)
            {
                using var entry = archive.CreateEntry(name, level).Open();
                entry.Write(bytes);
            }
        }

        zip.Position = 0;

        return zip;
    }

    /// <summary>Numbers as the 32-bit floats a storage keeps, little-endian.</summary>
    public static byte[] Floats(params float[] values) => [.. values.SelectMany(value => Length(BitConverter.SingleToInt32Bits(value)))];

    private static byte[] Length(int value)
    {
        var bytes = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);

        return bytes;
    }
}
