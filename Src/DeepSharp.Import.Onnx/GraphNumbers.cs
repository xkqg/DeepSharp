// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Globalization;
using DeepSharp.Networks;
using Onnx;
using static Onnx.TensorProto.Types;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// The values a graph holds rather than works out — its initializers, its constants, and what an identity hands on of
/// them — by name, read as numbers wherever the graph keeps them: in itself, or in a file beside it.
/// </summary>
/// <param name="folder">
/// The folder the graph's own file stands in, where its exporter keeps the numbers it keeps beside it; nothing for a graph
/// read from anything else.
/// </param>
internal sealed class GraphNumbers(string? folder)
{
    private readonly Dictionary<string, Kept> _kept = new(StringComparer.Ordinal);

    /// <summary>Whether the graph holds a value of the name, rather than working it out.</summary>
    public bool Holds(string name) => _kept.ContainsKey(name);

    /// <summary>Keeps a value the graph holds; false when it holds one of that name already.</summary>
    /// <param name="name">Its name in the graph.</param>
    /// <param name="tensor">The value.</param>
    /// <param name="source">Where the graph holds it, as a fault names it: <c>initializer '0.weight'</c>.</param>
    public bool Keep(string name, TensorProto tensor, string source) => _kept.TryAdd(name, new Kept(tensor, source));

    /// <summary>Keeps a name for a value the graph holds under another: what an identity hands on.</summary>
    public void Alias(string name, string of) => _kept[name] = _kept[of];

    /// <summary>
    /// A value as single-precision numbers: as written, widened exactly from half precision or bfloat16, or rounded from
    /// double precision; what is wrong with it otherwise, in the words that follow the name of the slot it was read for.
    /// </summary>
    public HeldTensor Floats(string name)
    {
        if (!_kept.TryGetValue(name, out var kept))
        {
            return new HeldTensor($"'{name}'", []) { Fault = "is no value the graph holds." };
        }

        if (kept.Lengths is not { } lengths)
        {
            return new HeldTensor(kept.Source, []) { Fault = kept.Unheld };
        }

        var type = (DataType)kept.Tensor.DataType;
        var width = type switch
        {
            DataType.Float => 4,
            DataType.Double => 8,
            DataType.Float16 or DataType.Bfloat16 => 2,
            _ => 0,
        };

        if (width == 0)
        {
            return new HeldTensor(kept.Source, lengths) { Fault = $"holds single-precision numbers here, and is written as {TypeName(kept.Tensor.DataType)}." };
        }

        var count = lengths.Aggregate(1, (total, length) => total * length);

        if (kept.InBytes)
        {
            var bytes = kept.Bytes((long)count * width, folder);

            return bytes.Fault is { } fault
                ? new HeldTensor(kept.Source, lengths) { Fault = fault }
                : new HeldTensor(kept.Source, lengths) { Values = Decoded(type, bytes.Bytes.Span, count) };
        }

        var typed = Typed(type, kept.Tensor);

        return typed.Length == count
            ? new HeldTensor(kept.Source, lengths) { Values = typed }
            : new HeldTensor(kept.Source, lengths) { Fault = kept.Short(typed.Length, count, "numbers") };
    }

    /// <summary>A value as whole numbers, as ONNX writes a reshape's target; what is wrong with it otherwise.</summary>
    public HeldWholes Wholes(string name)
    {
        if (!_kept.TryGetValue(name, out var kept))
        {
            return new HeldWholes(null, "is no value the graph holds.");
        }

        if (kept.Tensor.DataType != (int)DataType.Int64)
        {
            return new HeldWholes(null, $"is written as {TypeName(kept.Tensor.DataType)}, and ONNX writes one as INT64.");
        }

        if (kept.Lengths is not { } lengths)
        {
            return new HeldWholes(null, kept.Unheld);
        }

        var count = lengths.Aggregate(1, (total, length) => total * length);

        if (kept.InBytes)
        {
            var bytes = kept.Bytes(count * 8L, folder);

            return bytes.Fault is { } fault
                ? new HeldWholes(null, fault)
                : new HeldWholes([.. Enumerable.Range(0, (int)count).Select(at => BinaryPrimitives.ReadInt64LittleEndian(bytes.Bytes.Span[(at * 8)..]))], null);
        }

        return kept.Tensor.Int64Data.Count == count
            ? new HeldWholes([.. kept.Tensor.Int64Data], null)
            : new HeldWholes(null, kept.Short(kept.Tensor.Int64Data.Count, count, "numbers"));
    }

    /// <summary>ONNX's name for a type of number: <c>FLOAT</c>, <c>INT64</c>, <c>BFLOAT16</c>.</summary>
    public static string TypeName(int type) => ((DataType)type).ToString().ToUpperInvariant();

    // Numbers written as little-endian bytes, as single-precision ones.
    private static float[] Decoded(DataType type, ReadOnlySpan<byte> bytes, int count)
    {
        var values = new float[count];

        for (var at = 0; at < count; at++)
        {
            values[at] = type switch
            {
                DataType.Float => BinaryPrimitives.ReadSingleLittleEndian(bytes[(at * 4)..]),
                DataType.Double => (float)BinaryPrimitives.ReadDoubleLittleEndian(bytes[(at * 8)..]),
                DataType.Float16 => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes[(at * 2)..])),
                _ => Brain(BinaryPrimitives.ReadUInt16LittleEndian(bytes[(at * 2)..])),
            };
        }

        return values;
    }

    // Numbers written in the field ONNX keeps their type in — half precision and bfloat16 as the low bits of whole numbers —
    // as single-precision ones.
    private static float[] Typed(DataType type, TensorProto tensor) => type switch
    {
        DataType.Float => [.. tensor.FloatData],
        DataType.Double => [.. tensor.DoubleData.Select(value => (float)value)],
        DataType.Float16 => [.. tensor.Int32Data.Select(bits => (float)BitConverter.UInt16BitsToHalf((ushort)bits))],
        _ => [.. tensor.Int32Data.Select(bits => Brain((ushort)bits))],
    };

    // A bfloat16: the upper half of a single-precision number, the rest nothing.
    private static float Brain(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);
}

/// <summary>A value a graph holds, and where it holds it, as a fault names it.</summary>
/// <param name="Tensor">The value, as the graph writes it.</param>
/// <param name="Source">Where: <c>initializer '0.weight'</c>, or the constant node that holds it.</param>
internal readonly record struct Kept(TensorProto Tensor, string Source)
{
    /// <summary>The length of each of its axes; nothing when one is below nothing, or they hold more than a tensor can.</summary>
    public int[]? Lengths
    {
        get
        {
            // Counted no further than one past what a tensor holds, so a product of many long axes cannot wrap round.
            const long Beyond = int.MaxValue + 1L;
            var count = 1L;

            foreach (var length in Tensor.Dims)
            {
                if (length < 0)
                {
                    return null;
                }

                count = Math.Min(count * Math.Min(length, Beyond), Beyond);
            }

            return count < Beyond ? [.. Tensor.Dims.Select(length => (int)length)] : null;
        }
    }

    /// <summary>What is wrong with a value whose lengths no tensor holds.</summary>
    public string Unheld => $"is written with the lengths [{string.Join(", ", Tensor.Dims)}], which no tensor here holds.";

    /// <summary>Whether its numbers are written as bytes — in the graph, or in a file beside it — rather than in a typed field.</summary>
    public bool InBytes => Tensor.DataLocation == DataLocation.External || Tensor.RawData.Length > 0;

    /// <summary>
    /// Its bytes, when there are as many as it takes; what is wrong otherwise. Bytes kept beside the graph are read from the
    /// file its record names, inside the folder the graph stands in, from the byte and for the length the record says —
    /// to the file's end when it says none.
    /// </summary>
    public HeldBytes Bytes(long expected, string? folder)
    {
        if (Tensor.DataLocation != DataLocation.External)
        {
            return Tensor.RawData.Length == expected ? new HeldBytes(Tensor.RawData.Memory, null) : new HeldBytes(default, Short(Tensor.RawData.Length, expected, "bytes"));
        }

        var record = Tensor.ExternalData.GroupBy(entry => entry.Key).ToDictionary(entries => entries.Key, entries => entries.Last().Value, StringComparer.Ordinal);

        if (!record.TryGetValue("location", out var location))
        {
            return new HeldBytes(default, "is kept in another file, and the graph does not say which.");
        }

        if (folder is null)
        {
            return new HeldBytes(
                default,
                $"is kept in '{location.Quoted()}', a file beside the graph, and a graph read from a stream that is not its own file has no folder to find it in: read the graph from its file, or export it with external_data=False.");
        }

        var root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder)) + Path.DirectorySeparatorChar;
        var path = Path.GetFullPath(Path.Combine(root, location));

        if (!path.StartsWith(root, StringComparison.Ordinal))
        {
            return new HeldBytes(default, $"is said to be kept in '{location.Quoted()}', outside the folder the graph stands in, and numbers are read from beside the graph alone.");
        }

        if (!File.Exists(path))
        {
            return new HeldBytes(default, $"is kept in '{location.Quoted()}', and no such file stands beside the graph.");
        }

        return Beside(new SideFile(path, location, record), expected);
    }

    /// <summary>What is wrong with numbers written in another count than their lengths take.</summary>
    public string Short(long held, long takes, string what) =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"is written as a {(Tensor.Dims.Count == 0 ? "scalar" : string.Join('x', Tensor.Dims))} tensor of {GraphNumbers.TypeName(Tensor.DataType)}, and the graph holds {held} {what} for it, where it takes {takes}.");

    // The bytes a file beside the graph holds for this value, once the record's offset and length are read and held to the
    // file, and their count to what the value takes before a byte is read.
    private HeldBytes Beside(SideFile file, long expected)
    {
        var offset = file.Count("offset");
        var length = file.Count("length");

        if ((offset.Fault ?? length.Fault) is { } fault)
        {
            return new HeldBytes(default, fault);
        }

        using var data = File.OpenRead(file.Path);
        var from = offset.Value ?? 0;

        if (from > data.Length || length.Value > data.Length - from)
        {
            return new HeldBytes(
                default,
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"is kept in '{file.Location.Quoted()}' from byte {from}{(length.Value is { } stated ? $" for {stated} bytes" : string.Empty)}, and the file holds {data.Length}."));
        }

        var take = length.Value ?? (data.Length - from);

        if (take != expected)
        {
            return new HeldBytes(default, Short(take, expected, "bytes"));
        }

        var bytes = new byte[take];
        data.Position = from;
        data.ReadExactly(bytes);

        return new HeldBytes(bytes, null);
    }
}

/// <summary>A file beside a graph that holds a value's bytes, and the record the graph keeps of where in it they stand.</summary>
/// <param name="Path">The file, found inside the folder the graph stands in.</param>
/// <param name="Location">The file as the graph names it.</param>
/// <param name="Record">The record: its location, and the offset and the length of the bytes, when it states them.</param>
internal readonly record struct SideFile(string Path, string Location, IReadOnlyDictionary<string, string> Record)
{
    /// <summary>A count of bytes the record states: nothing when it states none; its fault when it is not a count.</summary>
    public StatedCount Count(string key)
    {
        if (!Record.TryGetValue(key, out var text))
        {
            return new StatedCount(null, null);
        }

        return long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var count)
            ? new StatedCount(count, null)
            : new StatedCount(null, $"is kept in '{Location.Quoted()}' with its {key} written as '{text.Quoted()}', which is not a count of bytes.");
    }
}

/// <summary>A count of bytes a record states, or what is wrong with it.</summary>
/// <param name="Value">The count; nothing when the record states none, or it is not one.</param>
/// <param name="Fault">What is wrong with it; nothing when it is a count, or is not stated.</param>
internal readonly record struct StatedCount(long? Value, string? Fault);

/// <summary>A value's bytes, or what is wrong with them.</summary>
/// <param name="Bytes">The bytes, little-endian; empty when there is a fault.</param>
/// <param name="Fault">What is wrong, in the words that follow the value's name; nothing when the bytes are as many as it takes.</param>
internal readonly record struct HeldBytes(ReadOnlyMemory<byte> Bytes, string? Fault);

/// <summary>A value read as whole numbers, or what is wrong with it.</summary>
/// <param name="Values">The numbers; nothing when there is a fault.</param>
/// <param name="Fault">What is wrong, in the words that follow the value's name.</param>
internal readonly record struct HeldWholes(long[]? Values, string? Fault);

/// <summary>How a number a node holds is laid out for the slot it goes into, as the node declares its layout.</summary>
internal enum Laying
{
    /// <summary>As the graph writes it: a bias, a normalisation's numbers, a weight matrix written inputs by outputs.</summary>
    AsWritten,

    /// <summary>A weight matrix a Gemm declares written outputs by inputs, turned round to inputs by outputs.</summary>
    Transposed,

    /// <summary>A convolution's kernel, written channels out by channels in by rows by columns, laid out as the window's places times channels in by channels out.</summary>
    Kernel,
}

/// <summary>An image's shape as a flatten found it: so many rows, columns and channels.</summary>
/// <param name="Rows">Its rows.</param>
/// <param name="Columns">Its columns.</param>
/// <param name="Channels">Its channels.</param>
internal readonly record struct ImageRow(int Rows, int Columns, int Channels)
{
    /// <summary>How many values the image holds, and so the row it is flattened into.</summary>
    public int Count => Rows * Columns * Channels;
}

/// <summary>A value a graph holds as single-precision numbers, and where; or what is wrong with it.</summary>
/// <param name="Source">Where the graph holds it, as a fault names it.</param>
/// <param name="Lengths">The length of each of its axes, as the graph writes them.</param>
internal readonly record struct HeldTensor(string Source, int[] Lengths)
{
    /// <summary>Its values, row-major; nothing when it cannot be read.</summary>
    public float[]? Values { get; init; }

    /// <summary>What is wrong with it, in the words that follow the name of the slot it was read for.</summary>
    public string? Fault { get; init; }

    /// <summary>
    /// The value laid out for its slot, as its node declares it, and along the row an image was flattened into — which ONNX
    /// flattens channel by channel and a network here place by place — turned from the one order to the other.
    /// </summary>
    /// <param name="laying">How its node declares it laid out.</param>
    /// <param name="flattened">The image a flatten before it found, when the value lies along the row that image became.</param>
    /// <remarks>Only for a value that could be read. A value of other lengths than the laying takes goes on as it is, for the load to refuse it.</remarks>
    public HeldTensor LaidOut(Laying laying, ImageRow? flattened)
    {
        var laid = laying switch
        {
            Laying.Transposed => Transposed(),
            Laying.Kernel => Kernel(),
            _ => this,
        };

        return flattened is { } image && laid.Lengths is [var along, ..] && along == image.Count ? laid.Turned(image) : laid;
    }

    // A matrix turned round: rows by columns to columns by rows.
    private HeldTensor Transposed()
    {
        var rows = Lengths[0];
        var columns = Lengths[1];
        var turned = new float[Values!.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                turned[(column * rows) + row] = Values[(row * columns) + column];
            }
        }

        return this with { Lengths = [columns, rows], Values = turned };
    }

    // A kernel channels out by channels in by rows by columns, as rows by columns by channels in by channels out, the first
    // three one axis: the window's places times the channels in.
    private HeldTensor Kernel()
    {
        var outputs = Lengths[0];
        var inputs = Lengths[1];
        var rows = Lengths[2];
        var columns = Lengths[3];
        var laid = new float[Values!.Length];

        for (var output = 0; output < outputs; output++)
        {
            for (var input = 0; input < inputs; input++)
            {
                for (var row = 0; row < rows; row++)
                {
                    for (var column = 0; column < columns; column++)
                    {
                        laid[(((((row * columns) + column) * inputs) + input) * outputs) + output] = Values[(((((output * inputs) + input) * rows) + row) * columns) + column];
                    }
                }
            }
        }

        return this with { Lengths = [rows * columns * inputs, outputs], Values = laid };
    }

    // The value's first axis, a flattened image's row, from ONNX's order — channel, row, column — to this one's: row,
    // column, channel; whatever follows that axis moves with it.
    private HeldTensor Turned(ImageRow image)
    {
        var each = Values!.Length / image.Count;
        var turned = new float[Values.Length];

        for (var row = 0; row < image.Rows; row++)
        {
            for (var column = 0; column < image.Columns; column++)
            {
                for (var channel = 0; channel < image.Channels; channel++)
                {
                    var here = (((row * image.Columns) + column) * image.Channels) + channel;
                    var there = (((channel * image.Rows) + row) * image.Columns) + column;
                    Values.AsSpan(there * each, each).CopyTo(turned.AsSpan(here * each, each));
                }
            }
        }

        return this with { Values = turned };
    }
}
