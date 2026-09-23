// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace DeepSharp.Pipelines;

/// <summary>
/// Who a row is: where it was when it was read, and what it says.
/// </summary>
/// <param name="ReadAt">The row's place among the rows as they were read, counting from nought.</param>
/// <param name="Key">A digest of the record the row was read from, the same whatever order the rows arrive in.</param>
/// <remarks>
/// Owned by the table, and carried with the row by every act that moves rows: keeping some of them, putting
/// them in order. The place is how a run finds a row again within one run — the way back for a target, which
/// handed-in row a served one is — and the key is how the same row is known across runs, when the same file
/// arrives in another order.
/// </remarks>
public readonly record struct RowIdentity(int ReadAt, RowKey Key);

/// <summary>
/// A digest of what a record says: every cell with the name of its column, whatever order the columns came in.
/// </summary>
/// <remarks>
/// SHA-256 over a written-down encoding, never a hash code, so a row has the same key on every machine and in
/// every process: the cells are taken in the order of their column names, compared character by character;
/// each is written as its name, then either a mark saying it is a gap or its text with the spaces around it
/// taken off, each name and text preceded by its length in UTF-8 bytes. A gap and an empty cell are different
/// records. Changing any of that changes every key, and with it every split that ranks rows by their keys.
/// <para>
/// The key identifies a row and nothing more: no value reaches a fit or a model through it, so a column the
/// schema left out can still take part in the key without taking part in anything else.
/// </para>
/// </remarks>
public readonly record struct RowKey : IComparable<RowKey>
{
    private readonly ulong _first;
    private readonly ulong _second;
    private readonly ulong _third;
    private readonly ulong _fourth;

    private RowKey(ReadOnlySpan<byte> digest)
    {
        _first = BinaryPrimitives.ReadUInt64BigEndian(digest);
        _second = BinaryPrimitives.ReadUInt64BigEndian(digest[8..]);
        _third = BinaryPrimitives.ReadUInt64BigEndian(digest[16..]);
        _fourth = BinaryPrimitives.ReadUInt64BigEndian(digest[24..]);
    }

    /// <summary>The key of one record.</summary>
    /// <param name="names">The column names, in the order the cells arrive.</param>
    /// <param name="cells">The cells, as text; nothing where a cell is a gap.</param>
    /// <returns>The record's key.</returns>
    /// <exception cref="ArgumentException">There are not as many cells as names.</exception>
    public static RowKey Of(IReadOnlyList<string> names, IReadOnlyList<string?> cells)
    {
        ArgumentNullException.ThrowIfNull(names);
        ArgumentNullException.ThrowIfNull(cells);

        return cells.Count == names.Count
            ? new RecordDigest(names).Of(cells)
            : throw new ArgumentException(
                $"A record of {names.Count} columns has {cells.Count} cells, and each cell belongs to one column.", nameof(cells));
    }

    /// <summary>Made from the thirty-two bytes of a digest.</summary>
    /// <param name="digest">The digest.</param>
    /// <returns>The key.</returns>
    internal static RowKey FromDigest(ReadOnlySpan<byte> digest) => new(digest);

    /// <summary>This key as a split ranks it under a seed: a digest of the seed and the key.</summary>
    /// <param name="seed">The split's seed.</param>
    /// <returns>The rank, which orders like any key.</returns>
    /// <remarks>The seed as four bytes, least significant first, followed by the key's thirty-two.</remarks>
    internal RowKey Ranked(int seed)
    {
        Span<byte> written = stackalloc byte[36];
        BinaryPrimitives.WriteInt32LittleEndian(written, seed);
        WriteTo(written[4..]);

        Span<byte> digest = stackalloc byte[32];
        SHA256.HashData(written, digest);

        return new RowKey(digest);
    }

    /// <summary>Writes the thirty-two bytes of this key.</summary>
    /// <param name="destination">Where they go.</param>
    internal void WriteTo(Span<byte> destination)
    {
        BinaryPrimitives.WriteUInt64BigEndian(destination, _first);
        BinaryPrimitives.WriteUInt64BigEndian(destination[8..], _second);
        BinaryPrimitives.WriteUInt64BigEndian(destination[16..], _third);
        BinaryPrimitives.WriteUInt64BigEndian(destination[24..], _fourth);
    }

    /// <summary>Orders keys by their bytes, the same way on every machine.</summary>
    /// <param name="other">The key to compare with.</param>
    /// <returns>Below nought when this key comes first, nought when they are equal, above nought otherwise.</returns>
    public int CompareTo(RowKey other)
    {
        ReadOnlySpan<ulong> mine = [_first, _second, _third, _fourth];
        ReadOnlySpan<ulong> theirs = [other._first, other._second, other._third, other._fourth];

        return mine.SequenceCompareTo(theirs);
    }

    /// <summary>The key as sixty-four hexadecimal digits.</summary>
    /// <returns>The digits.</returns>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"{_first:x16}{_second:x16}{_third:x16}{_fourth:x16}");
}

/// <summary>
/// The keys of the records of one source: the order of its column names worked out once, then each record
/// digested without allocating.
/// </summary>
/// <param name="names">The column names, in the order the cells arrive.</param>
internal sealed class RecordDigest(IReadOnlyList<string> names)
{
    private const byte Gap = 0;
    private const byte Text = 1;

    // Positions in the order of their names; the sort is stable, so two columns of one name keep theirs.
    private readonly int[] _order = [.. Enumerable.Range(0, names.Count).OrderBy(at => names[at], StringComparer.Ordinal)];

    private readonly byte[][] _names = [.. names.Select(Encoding.UTF8.GetBytes)];

    /// <summary>The key of one record; a cell the record is short of is a gap.</summary>
    /// <param name="cells">The cells, as text; nothing where a cell is a gap.</param>
    /// <returns>The record's key.</returns>
    public RowKey Of(IReadOnlyList<string?> cells)
    {
        var size = 0;

        foreach (var at in _order)
        {
            size += 4 + _names[at].Length + 1 + 4 + (at < cells.Count && cells[at] is { } text ? Encoding.UTF8.GetMaxByteCount(text.Length) : 0);
        }

        var rented = ArrayPool<byte>.Shared.Rent(size);

        try
        {
            var written = 0;

            foreach (var at in _order)
            {
                written += Length(rented.AsSpan(written), _names[at].Length);
                _names[at].CopyTo(rented.AsSpan(written));
                written += _names[at].Length;

                if (at >= cells.Count || cells[at] is not { } text)
                {
                    rented[written++] = Gap;
                    continue;
                }

                rented[written++] = Text;

                var bytes = Encoding.UTF8.GetBytes(text.AsSpan().Trim(), rented.AsSpan(written + 4));
                written += Length(rented.AsSpan(written), bytes) + bytes;
            }

            Span<byte> digest = stackalloc byte[32];
            SHA256.HashData(rented.AsSpan(0, written), digest);

            return RowKey.FromDigest(digest);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static int Length(Span<byte> destination, int length)
    {
        BinaryPrimitives.WriteInt32LittleEndian(destination, length);

        return 4;
    }
}
