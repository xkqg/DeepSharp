// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using PureHDF;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The files Keras saved that the Keras reader's tests read (Fixtures/keras-fixtures.py made them), what Keras answered with
/// each, and saves made from them the way a file comes to say something else: its description edited, its numbers written
/// by hand.
/// </summary>
internal static class KerasFixtures
{
    /// <summary>The folder the fixtures stand in.</summary>
    public static string Folder { get; } = Path.Join(Repository.Root, "Tst", "DeepSharp", "Import", "Fixtures");

    /// <summary>What Keras answered, and the rows and images it answered for.</summary>
    public static JsonElement Answers { get; } = JsonDocument.Parse(File.ReadAllText(Path.Join(Folder, "keras-fixtures.json"))).RootElement;

    /// <summary>A fixture, as a file read from where it stands.</summary>
    public static MemoryStream Open(string file) => new(File.ReadAllBytes(Path.Join(Folder, file)));

    /// <summary>
    /// A <c>.keras</c> fixture whose description says things otherwise, each edit in turn: <c>path=json</c> sets the value at
    /// a path of its <c>config.json</c> — names and places from its root, dotted: <c>config.layers.1.config.units</c> — to
    /// the JSON after the first <c>=</c>, and <c>path=-</c> takes it out. Its numbers are the fixture's.
    /// </summary>
    public static MemoryStream Edited(string file, params string[] edits)
    {
        var entries = Entries(file);
        var config = JsonNode.Parse(entries["config.json"])!;

        foreach (var edit in edits)
        {
            var path = edit[..edit.IndexOf('=', StringComparison.Ordinal)].Split('.');
            var json = edit[(edit.IndexOf('=', StringComparison.Ordinal) + 1)..];
            var parent = path[..^1].Aggregate(config, Step);

            if (json == "-")
            {
                _ = parent is JsonArray list ? RemovedFrom(list, int.Parse(path[^1], CultureInfo.InvariantCulture)) : parent.AsObject().Remove(path[^1]);
            }
            else if (parent is JsonArray list)
            {
                list[int.Parse(path[^1], CultureInfo.InvariantCulture)] = JsonNode.Parse(json);
            }
            else
            {
                parent[path[^1]] = JsonNode.Parse(json);
            }
        }

        entries["config.json"] = Encoding.UTF8.GetBytes(config.ToJsonString());

        return Archive(entries);
    }

    /// <summary>A <c>.keras</c> fixture holding other numbers: its description, and a weights file written by hand.</summary>
    public static MemoryStream WithWeights(string file, H5File weights)
    {
        var entries = Entries(file);
        entries["model.weights.h5"] = Written(weights);

        return Archive(entries);
    }

    /// <summary>A <c>.keras</c> archive holding what it is given, each entry stored as Keras stores it.</summary>
    public static MemoryStream Archive(IReadOnlyDictionary<string, byte[]> entries)
    {
        var archive = new MemoryStream();

        using (var zip = new ZipArchive(archive, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (var (name, bytes) in entries)
            {
                using var entry = zip.CreateEntry(name, CompressionLevel.NoCompression).Open();
                entry.Write(bytes);
            }
        }

        archive.Position = 0;

        return archive;
    }

    /// <summary>The entries of a <c>.keras</c> fixture, by name.</summary>
    public static Dictionary<string, byte[]> Entries(string file)
    {
        using var zip = new ZipArchive(Open(file), ZipArchiveMode.Read);

        return zip.Entries.ToDictionary(entry => entry.FullName, entry =>
        {
            using var read = new MemoryStream();
            using (var opened = entry.Open())
            {
                opened.CopyTo(read);
            }

            return read.ToArray();
        });
    }

    /// <summary>An HDF5 file written by hand, as its bytes.</summary>
    public static byte[] Written(H5File file)
    {
        using var written = new MemoryStream();
        file.Write(written);

        return written.ToArray();
    }

    /// <summary>A dataset of single-precision numbers laid out along the given axes, every one of the given value.</summary>
    public static H5Dataset<float[]> Floats(ulong[] axes, float value) =>
        new([.. Enumerable.Repeat(value, (int)axes.Aggregate(1UL, (count, length) => count * length))], fileDims: axes);

    private static JsonNode Step(JsonNode node, string name) =>
        node is JsonArray list ? list[int.Parse(name, CultureInfo.InvariantCulture)]! : node[name]!;

    private static bool RemovedFrom(JsonArray list, int at)
    {
        list.RemoveAt(at);

        return true;
    }
}
