// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.IO.Compression;
using System.Text;
using DeepSharp.Networks;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// The archive torch.save writes, as PyTorch reads it: every file in one folder, its version, the byte order its numbers
/// are in, the pickle of the state, <c>data.pkl</c>, and one record of bytes for each storage the pickle loads.
/// </summary>
/// <remarks>
/// <para>
/// It is refused as PyTorch refuses it: a file that does not begin as a zip archive's first record does, which PyTorch looks
/// for before anything else — so bytes before an archive, or an archive after something else, are no archive; an archive
/// of no files, files in no folder, no version or one past PyTorch's 1 to 10, a byte order that is neither little nor big,
/// no <c>data.pkl</c>, and a TorchScript archive — a program, which PyTorch's weights-only reader refuses too. The format
/// PyTorch wrote before 1.6, pickles one after another, is named and refused.
/// </para>
/// <para>
/// PyTorch finds a record by its name whatever the case of its ASCII letters, so this reader does too; and a name the
/// archive holds twice, in the same case or another, is refused by name rather than read as one of the two.
/// </para>
/// <para>
/// The records it reads hold together no more bytes than the file itself does, as the records torch.save writes, stored as
/// they are, never do; a record saying it holds more than is left of that is refused before a byte of it is inflated, and
/// one holding more or fewer bytes than it says is refused at the first byte past what it says or at its end. A storage
/// saying it holds other bytes than its numbers take is refused before it is read at all.
/// </para>
/// </remarks>
internal sealed class TorchArchive : IDisposable
{
    // PyTorch's versions of the archive it reads, caffe2/serialize/versions.h.
    private const ulong Oldest = 1;
    private const ulong Newest = 10;

    // How the format before 1.6 begins: a pickle, protocol 2, of the number torch.save wrote first, 0x1950A86A20F9469CFC6C.
    private static readonly byte[] BeforeArchives = [0x80, 0x02, 0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19];

    // How an archive begins, as PyTorch tells one from the format before it: the signature of a zip archive's first record.
    private static readonly byte[] ArchiveBegins = [0x50, 0x4B, 0x03, 0x04];

    private readonly ZipArchive _zip;
    private readonly Dictionary<string, ZipArchiveEntry> _entries = new(AsciiCaseless.Names);

    // The file's own length, and how many of its bytes the records read so far leave for those still to be read.
    private readonly long _length;
    private long _left;
    private readonly string _folder;
    private readonly bool _bigEndian;
    private readonly Dictionary<string, byte[]> _records = new(StringComparer.Ordinal);

    /// <summary>The archive a file holds.</summary>
    /// <param name="bytes">The file, read whole into memory.</param>
    /// <exception cref="FormatException">The file is no archive torch.save writes.</exception>
    public TorchArchive(MemoryStream bytes)
    {
        _length = _left = bytes.Length;

        var begins = bytes.GetBuffer().AsSpan(0, (int)bytes.Length);

        if (!begins.StartsWith(ArchiveBegins))
        {
            throw new FormatException(begins.StartsWith(BeforeArchives)
                ? "The file is written in the format PyTorch wrote before 1.6, one pickle after another, which this reader does not read: load it with PyTorch and save it again with torch.save, which has written an archive since."
                : "The file is no archive torch.save writes: a zip archive begins with the header of its first record, and PyTorch takes nothing else for one.");
        }

        try
        {
            _zip = new ZipArchive(bytes, ZipArchiveMode.Read);
            Name(_zip.Entries);
        }
        catch (Exception refused) when (refused is InvalidDataException or IOException)
        {
            // A zip archive in memory fails no read of its own; .NET 8 reports some of its corruption as an IOException.
            throw new FormatException($"The file is no archive torch.save writes: {refused.Message.Quoted()}", refused);
        }

        if (_entries.Count == 0)
        {
            throw new FormatException("The file is an archive holding nothing, where torch.save writes data.pkl and the storages it names.");
        }

        var first = _zip.Entries[0].FullName;
        var folder = first.IndexOf('/', StringComparison.Ordinal);

        if (folder < 0)
        {
            throw new FormatException($"The file is an archive whose file '{first.Quoted()}' is in no folder, where torch.save writes every file of an archive into one.");
        }

        _folder = first[..(folder + 1)];

        CheckVersion();

        if (Entry("constants.pkl") is not null)
        {
            throw new FormatException(
                "The file is a TorchScript archive, a program PyTorch runs, which PyTorch's weights-only reader refuses too: save the network's state with torch.save(model.state_dict(), file).");
        }

        _bigEndian = Entry("byteorder") is { } order && Text(order) switch
        {
            "little" => false,
            "big" => true,
            var other => throw new FormatException($"The file says its numbers are in '{other.Quoted()}' byte order, and PyTorch writes them in little or big."),
        };
    }

    /// <summary>
    /// The state dictionary the archive holds: each tensor under its name, in the order the pickle set them, its numbers
    /// gathered out of its storage's record.
    /// </summary>
    /// <exception cref="FormatException">
    /// data.pkl is missing or refused; it holds anything but a dict of tensors by name; or a storage it loads is missing or
    /// holds other bytes than its numbers take.
    /// </exception>
    public IReadOnlyList<StoredTensor> StateDictionary()
    {
        var pickle = Entry("data.pkl") ?? throw new FormatException("The file is an archive without data.pkl, the pickle torch.save writes the state into.");
        var state = new WeightsOnlyPickle(Bytes(pickle)).Load();

        if (state is not PickledDict dictionary)
        {
            throw new FormatException($"The file holds {WeightsOnlyPickle.Described(state)}, where a state dictionary maps each slot's name to a tensor.");
        }

        List<StoredTensor> tensors = [];
        var bigEndian = _bigEndian;

        foreach (var (key, value) in dictionary.Entries)
        {
            if (key is not string name)
            {
                throw new FormatException($"The file's state dictionary holds {WeightsOnlyPickle.Described(key)} as a name, where each name is text.");
            }

            if (value is not PickledTensor tensor)
            {
                throw new FormatException(
                    $"The file's entry '{name.Quoted()}' is {WeightsOnlyPickle.Described(value)}, and a state dictionary holds a tensor under each name: a checkpoint keeping the state under a key of its own is read once the state is saved alone, torch.save(model.state_dict(), file).");
            }

            // The storage's record is read now, and refused now when it is missing or short; its numbers are gathered only
            // for a slot that takes them.
            var stored = Record(tensor.Storage);

            tensors.Add(new StoredTensor(name, () => tensor.Numbers(stored, bigEndian)));
        }

        return tensors;
    }

    /// <inheritdoc />
    public void Dispose() => _zip.Dispose();

    // The version the archive says it is, held to PyTorch's: the digits it begins with, after any space.
    private void CheckVersion()
    {
        var said = Text(Entry(".data/version") ?? Entry("version") ?? throw new FormatException("The file is an archive without a version, which torch.save writes into every archive.")).Trim();
        var digits = string.Concat(said.TakeWhile(char.IsAsciiDigit));

        if (digits.Length == 0)
        {
            throw new FormatException($"The file is an archive whose version, '{said.Quoted()}', is no number.");
        }

        if (!ulong.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out var version) || version is < Oldest or > Newest)
        {
            throw new FormatException($"The file is an archive of version {digits.Quoted()}, and PyTorch reads versions {Oldest} to {Newest}.");
        }
    }

    // A storage's numbers: its record, read once, holding exactly the bytes its numbers take.
    private byte[] Record(PickledStorage storage)
    {
        if (_records.TryGetValue(storage.Key, out var known))
        {
            return known;
        }

        var entry = Entry($"data/{storage.Key}") ?? throw new FormatException($"The file holds no storage '{storage.Key.Quoted()}', which data.pkl loads at byte {storage.At}.");

        if ((Int128)storage.Count * storage.Kind.Size != entry.Length)
        {
            throw new FormatException($"The file's storage '{storage.Key.Quoted()}' holds {entry.Length} bytes, where data.pkl says {storage.Count} numbers of {storage.Kind.Size} bytes.");
        }

        return _records[storage.Key] = Bytes(entry);
    }

    private ZipArchiveEntry? Entry(string name) => _entries.GetValueOrDefault(_folder + name);

    // Every record by its name, as PyTorch finds it: whatever the case of its ASCII letters; one name held twice is refused.
    private void Name(IEnumerable<ZipArchiveEntry> entries)
    {
        foreach (var entry in entries)
        {
            if (_entries.TryGetValue(entry.FullName, out var held))
            {
                throw new FormatException(held.FullName == entry.FullName
                    ? $"The file's archive names '{entry.FullName.Quoted()}' twice, and PyTorch would read one of the two: which is meant is not guessed."
                    : $"The file's archive names '{held.FullName.Quoted()}' twice, the second time as '{entry.FullName.Quoted()}', which PyTorch takes for the same name: which is meant is not guessed.");
            }

            _entries.Add(entry.FullName, entry);
        }
    }

    private string Text(ZipArchiveEntry entry) => Encoding.UTF8.GetString(Bytes(entry));

    // Names equal when their ASCII letters are, whatever their case, as the zip reader PyTorch uses compares them; every
    // other character as it is.
    private sealed class AsciiCaseless : IEqualityComparer<string>
    {
        public static readonly AsciiCaseless Names = new();

        public bool Equals(string? x, string? y) => string.Equals(Folded(x!), Folded(y!), StringComparison.Ordinal);

        public int GetHashCode(string name) => Folded(name).GetHashCode(StringComparison.Ordinal);

        private static string Folded(string name) =>
            string.Create(name.Length, name, static (folded, held) =>
            {
                for (var at = 0; at < held.Length; at++)
                {
                    folded[at] = held[at] is >= 'A' and <= 'Z' ? (char)(held[at] + 32) : held[at];
                }
            });
    }

    // Every byte a record holds: as many as it says, when the file has that many left for it, and not one more.
    private byte[] Bytes(ZipArchiveEntry entry)
    {
        var name = entry.FullName.Quoted();

        if (entry.Length > _left)
        {
            throw new FormatException(
                $"The file's '{name}' holds {entry.Length} bytes, and the records read of a file hold together no more bytes than the file's own {_length}, of which {_left} are left: torch.save stores every record as it is, so its files never hold more.");
        }

        var bytes = new byte[entry.Length];

        try
        {
            using var stream = entry.Open();
            stream.ReadExactly(bytes);

            if (stream.ReadByte() >= 0)
            {
                throw new FormatException($"The file's '{name}' holds more bytes than the {entry.Length} it says.");
            }
        }
        catch (EndOfStreamException)
        {
            throw new FormatException($"The file's '{name}' holds fewer bytes than the {entry.Length} it says.");
        }
        catch (InvalidDataException refused)
        {
            throw new FormatException($"The file's '{name}' cannot be read: {refused.Message.Quoted()}", refused);
        }

        // An inflater stops at the length the headers say, so a record inflating to more is caught by its check instead.
        if (((ReadOnlySpan<byte>)bytes).Crc32() is var check && check != entry.Crc32)
        {
            throw new FormatException(
                string.Create(CultureInfo.InvariantCulture, $"The file's '{name}' holds other bytes than its check says: their CRC-32 is {check:x8}, and the archive's {entry.Crc32:x8}."));
        }

        _left -= entry.Length;

        return bytes;
    }
}
