// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;

namespace DeepSharp.Pipelines;

/// <summary>
/// One image, one sound or one run of token ids as a cell of a source's text: a fixed shape and the numbers that fill it, written as
/// one value the source hands over like any other.
/// </summary>
/// <remarks>
/// <para>
/// The text is a kind, the shape and the payload: <c>u8[28x28x1]:…</c> for bytes, <c>s16[16000]:…</c> for sixteen-bit samples,
/// <c>f32[…]:…</c> for floats and <c>ids[…]:…</c> for whole numbers, the payload base64. A reader writes it and a declared media
/// column reads it, so the source's contract is the one it always had and a row is known by what it holds — never by the file
/// it was read from.
/// </para>
/// <para>
/// A byte lands between minus one and one as <c>v / 127.5 − 1</c> and a sample as <c>v / 32768</c>, by one rule wherever it stands:
/// nothing here learns a scale from the rows, so the same pixel is the same number in every part and on every day it is served. Ids
/// are whole numbers, kept as they are, and a column of them says so.
/// </para>
/// </remarks>
public sealed class MediaCell
{
    private const string BytesKind = "u8";
    private const string SamplesKind = "s16";
    private const string FloatsKind = "f32";
    private const string IdsKind = "ids";

    private MediaCell(int[] shape, float[] values, bool ids)
    {
        Shape = shape;
        Values = values;
        HoldsIds = ids;
    }

    /// <summary>The shape of the array: rows, columns and channels of an image, the length of a sound, the length of a run of ids.</summary>
    public IReadOnlyList<int> Shape { get; }

    /// <summary>The numbers, laid out one axis after another, the last axis fastest.</summary>
    public float[] Values { get; }

    /// <summary>Whether the numbers are ids — whole numbers, kept as they are — rather than values between minus one and one.</summary>
    public bool HoldsIds { get; }

    /// <summary>A cell of bytes: the pixels of an image, one byte each.</summary>
    /// <param name="shape">The shape: every axis one or more.</param>
    /// <param name="bytes">The bytes, as many as the shape holds.</param>
    /// <returns>The cell as text.</returns>
    /// <exception cref="ArgumentException">The shape is empty or has an axis below one, or the bytes are not as many as it holds.</exception>
    public static string Pixels(IReadOnlyList<int> shape, ReadOnlySpan<byte> bytes) =>
        Written(BytesKind, shape, bytes.Length, bytes);

    /// <summary>A cell of sixteen-bit samples: a sound, one signed sample each.</summary>
    /// <param name="shape">The shape: every axis one or more.</param>
    /// <param name="samples">The samples, as many as the shape holds.</param>
    /// <returns>The cell as text.</returns>
    /// <exception cref="ArgumentException">The shape is empty or has an axis below one, or the samples are not as many as it holds.</exception>
    public static string Samples(IReadOnlyList<int> shape, ReadOnlySpan<short> samples)
    {
        var bytes = new byte[samples.Length * 2];

        for (var at = 0; at < samples.Length; at++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(at * 2), samples[at]);
        }

        return Written(SamplesKind, shape, samples.Length, bytes);
    }

    /// <summary>A cell of numbers kept as they are.</summary>
    /// <param name="shape">The shape: every axis one or more.</param>
    /// <param name="values">The numbers, as many as the shape holds.</param>
    /// <returns>The cell as text.</returns>
    /// <exception cref="ArgumentException">The shape is empty or has an axis below one, or the numbers are not as many as it holds.</exception>
    public static string Floats(IReadOnlyList<int> shape, ReadOnlySpan<float> values)
    {
        var bytes = new byte[values.Length * 4];

        for (var at = 0; at < values.Length; at++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(at * 4), values[at]);
        }

        return Written(FloatsKind, shape, values.Length, bytes);
    }

    /// <summary>A cell of ids: whole numbers of nought or more, such as the ids of a text's tokens.</summary>
    /// <param name="shape">The shape: every axis one or more.</param>
    /// <param name="ids">The ids, as many as the shape holds.</param>
    /// <returns>The cell as text.</returns>
    /// <exception cref="ArgumentException">The shape is empty or has an axis below one, the ids are not as many as it holds, or one is below nought.</exception>
    public static string Ids(IReadOnlyList<int> shape, ReadOnlySpan<int> ids)
    {
        var bytes = new byte[ids.Length * 4];

        for (var at = 0; at < ids.Length; at++)
        {
            if (ids[at] < 0)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"An id is a whole number of nought or more, and the {at + 1}th is {ids[at]}."), nameof(ids));
            }

            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(at * 4), ids[at]);
        }

        return Written(IdsKind, shape, ids.Length, bytes);
    }

    /// <summary>Reads a cell back from its text.</summary>
    /// <param name="text">The text a reader wrote.</param>
    /// <returns>The cell.</returns>
    /// <exception cref="FormatException">The text is no media cell: its kind is none of the four, its shape or payload is not what it says, or the payload holds a number it cannot.</exception>
    public static MediaCell Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var open = text.IndexOf('[', StringComparison.Ordinal);
        var close = text.IndexOf("]:", StringComparison.Ordinal);

        if (open < 1 || close < open)
        {
            throw new FormatException("A media cell is a kind, a shape in square brackets and a payload after a colon, such as u8[28x28x1]:…");
        }

        var kind = text[..open];
        var shape = ShapeOf(text[(open + 1)..close]);
        byte[] payload;

        try
        {
            payload = Convert.FromBase64String(text[(close + 2)..]);
        }
        catch (FormatException)
        {
            throw new FormatException("A media cell's payload is base64, and this is not.");
        }

        var count = shape.Aggregate(1L, (all, axis) => all * axis);
        var width = kind switch
        {
            BytesKind => 1,
            SamplesKind => 2,
            FloatsKind or IdsKind => 4,
            _ => throw new FormatException($"'{kind}' is no kind of media cell: a media cell is {BytesKind}, {SamplesKind}, {FloatsKind} or {IdsKind}."),
        };

        if (payload.Length != count * width)
        {
            throw new FormatException(
                string.Create(CultureInfo.InvariantCulture, $"A {kind} cell of shape {string.Join("x", shape)} holds {count * width} bytes, and this one holds {payload.Length}."));
        }

        var values = new float[count];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = kind switch
            {
                BytesKind => (payload[at] / 127.5f) - 1f,
                SamplesKind => BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(at * 2)) / 32768f,
                FloatsKind => BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan(at * 4)),
                _ => BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(at * 4)),
            };

            if (kind == IdsKind && values[at] < 0)
            {
                throw new FormatException("An id is a whole number of nought or more.");
            }

            if (kind == FloatsKind && !float.IsFinite(values[at]))
            {
                throw new FormatException("A number of a media cell is a finite number.");
            }
        }

        return new MediaCell(shape, values, kind == IdsKind);
    }

    private static int[] ShapeOf(string axes)
    {
        var parts = axes.Split('x');
        var shape = new int[parts.Length];

        for (var at = 0; at < parts.Length; at++)
        {
            if (!int.TryParse(parts[at], NumberStyles.None, CultureInfo.InvariantCulture, out shape[at]) || shape[at] < 1)
            {
                throw new FormatException($"A media cell's shape is whole numbers of one or more joined by x, and '{axes}' is not.");
            }
        }

        return shape;
    }

    private static string Written(string kind, IReadOnlyList<int> shape, int count, ReadOnlySpan<byte> bytes)
    {
        ArgumentNullException.ThrowIfNull(shape);

        if (shape.Count == 0 || shape.Any(axis => axis < 1))
        {
            throw new ArgumentException("A shape has at least one axis, and every axis holds one or more.", nameof(shape));
        }

        var expected = shape.Aggregate(1L, (all, axis) => all * axis);

        if (count != expected)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"A shape of {string.Join("x", shape)} holds {expected} numbers, and {count} were given."),
                nameof(shape));
        }

        return string.Create(CultureInfo.InvariantCulture, $"{kind}[{string.Join("x", shape)}]:{Convert.ToBase64String(bytes)}");
    }
}

/// <summary>
/// A column of arrays of one shape: the pixels of images, the samples of sounds, the ids of tokens — one array a row, kept whole.
/// </summary>
/// <remarks>
/// Not a cell of text for each number: sixty thousand images of twenty-eight by twenty-eight are forty-seven million of those, and
/// as arrays they are one for each image. Every array in a column has the shape of the first — one that has another is refused,
/// naming it, and nothing is resized — and a row is known by what it holds, a digest of its numbers, not by the file it was read
/// from.
/// </remarks>
public sealed class MediaColumn : IColumn
{
    private readonly float[]?[] _values;
    private readonly string?[] _digests;

    /// <summary>A column of these arrays, in this order.</summary>
    /// <param name="name">The column's name.</param>
    /// <param name="cells">The arrays; nothing where a cell is a gap.</param>
    /// <exception cref="ArgumentException">No cell holds an array, so there is no shape; one has another shape or means than the first.</exception>
    public MediaColumn(string name, IEnumerable<MediaCell?> cells)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(cells);

        MediaCell?[] held = [.. cells];
        var first = held.FirstOrDefault(cell => cell is not null)
            ?? throw new ArgumentException($"Column '{name}' holds no array, so it has no shape: a column of media needs one at the least.", nameof(cells));

        Name = name;
        Shape = first.Shape;
        Ids = first.HoldsIds;
        _values = new float[]?[held.Length];

        for (var row = 0; row < held.Length; row++)
        {
            if (held[row] is not { } cell)
            {
                continue;
            }

            if (!cell.Shape.SequenceEqual(Shape) || cell.HoldsIds != Ids)
            {
                throw new ArgumentException(
                    $"Column '{name}', row {row + 1}: {Describe(cell)} is not {Describe(first)}, the first. A column holds arrays of one shape, and nothing is resized.",
                    nameof(cells));
            }

            _values[row] = cell.Values;
        }

        _digests = new string?[held.Length];
    }

    // The same column holding some of its rows, in the order given: the arrays are shared, and so are the digests worked out.
    private MediaColumn(MediaColumn source, IReadOnlyList<int> rows)
    {
        Name = source.Name;
        Shape = source.Shape;
        Ids = source.Ids;
        _values = [.. rows.Select(row => source._values[row])];
        _digests = [.. rows.Select(row => source._digests[row])];
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public ColumnKind Kind => ColumnKind.Media;

    /// <inheritdoc />
    public int Count => _values.Length;

    /// <summary>The shape of one array.</summary>
    public IReadOnlyList<int> Shape { get; }

    /// <summary>Whether the arrays are ids, whole numbers kept as they are, rather than values between minus one and one.</summary>
    public bool Ids { get; }

    /// <summary>How many numbers one array holds.</summary>
    public int Width => Shape.Aggregate(1, (all, axis) => all * axis);

    /// <summary>The numbers of one row's array.</summary>
    /// <param name="row">The row.</param>
    /// <returns>The numbers, or nothing where the cell is a gap.</returns>
    public float[]? ValuesAt(int row) => _values[row];

    /// <inheritdoc />
    public bool IsMissing(int row) => _values[row] is null;

    /// <inheritdoc />
    /// <remarks>A digest of the numbers and the shape, so two rows that hold the same array are the same text.</remarks>
    public string? TextAt(int row)
    {
        if (_values[row] is not { } values)
        {
            return null;
        }

        return _digests[row] ??= Digest(values);
    }

    /// <inheritdoc />
    public IColumn Rows(IReadOnlyList<int> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        return new MediaColumn(this, rows);
    }

    /// <inheritdoc />
    public int LeadingGaps() => _values.TakeWhile(value => value is null).Count();

    private static string Describe(MediaCell cell) => $"{(cell.HoldsIds ? "ids" : "values")} of {string.Join("x", cell.Shape)}";

    private string Digest(float[] values)
    {
        var bytes = new byte[Shape.Count * 4 + values.Length * 4];

        for (var axis = 0; axis < Shape.Count; axis++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(axis * 4), Shape[axis]);
        }

        for (var at = 0; at < values.Length; at++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((Shape.Count * 4) + (at * 4)), values[at]);
        }

        return "media:" + Convert.ToHexStringLower(SHA256.HashData(bytes));
    }
}
