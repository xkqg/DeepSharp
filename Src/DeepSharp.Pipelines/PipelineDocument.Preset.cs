// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

// The preset's part of the file: the columns saved beside a notebook, written and read through the same door as a pipeline
// and a course. It is a file of its own — a schema, the columns to drop, an output, the source's names — with its own keys.
internal sealed partial class PipelineDocument
{
    private const string SourceKey = "source";
    private const string DeclareKey = "declare";
    private const string DropKey = "drop";
    private const string OutputKey = "output";

    private static readonly string[] PresetKeys = [VersionKey, SourceKey, DeclareKey, DropKey, OutputKey];

    // The steps a file of saved columns holds, by key, and what each is to the person who reads a refusal.
    private static readonly Dictionary<string, string> PresetSteps = new(StringComparer.Ordinal)
    {
        [DeclareKey] = "its schema",
        [OutputKey] = "its output",
    };

    /// <summary>Writes a preset: the version, then what it holds; a part it does not hold is not written.</summary>
    /// <param name="preset">The preset.</param>
    /// <returns>The file, indented, with a line feed between lines on every system.</returns>
    public static string Write(PipelinePreset preset)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, PipelineDeclaration.Version);
            WriteNames(writer, SourceKey, preset.Source);
            writer.WritePropertyName(DeclareKey);
            ((IPipelineStep)preset.Declare).WriteTo(writer);
            WriteNames(writer, DropKey, preset.Drop.Count > 0 ? preset.Drop : null);

            if (preset.Output is { } output)
            {
                writer.WritePropertyName(OutputKey);
                output.WriteTo(writer);
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
    }

    /// <summary>Reads a preset, its schema and output through the catalog.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs its schema and output may use.</param>
    /// <returns>The preset.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static PipelinePreset ReadPreset(string json, StepCatalog catalog) => Whole(json).Preset(catalog);

    private static void WriteNames(Utf8JsonWriter writer, string key, IReadOnlyList<string>? names)
    {
        if (names is null)
        {
            return;
        }

        writer.WriteStartArray(key);

        foreach (var name in names)
        {
            writer.WriteStringValue(name);
        }

        writer.WriteEndArray();
    }

    private PipelinePreset Preset(StepCatalog catalog)
    {
        using var document = JsonDocument.Parse(_text.Bytes);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            _text.Fault(_text.Root, "A file of saved columns is one JSON object, holding its schema under 'declare' and whatever else was decided.");

            throw _text.Refused();
        }

        // No preset was written before the second version, and its output may be a word only the second has: one that
        // names no version is not read as the first, as a pipeline file is.
        var version = root.TryGetProperty(VersionKey, out _) ? VersionOf(root) : NoVersion("A file of saved columns");

        FaultKeysNotHeld(root, PresetKeys, "A file of saved columns");

        var source = NamesUnder(root, SourceKey);
        var drop = NamesUnder(root, DropKey);
        var declare = StepUnder(root, DeclareKey, version, catalog);
        var output = StepUnder(root, OutputKey, version, catalog);

        if (!root.TryGetProperty(DeclareKey, out _))
        {
            _text.Fault(_text.Root, $"A file of saved columns holds its schema under '{DeclareKey}'.");
        }

        if (declare is not (null or DeclareStep))
        {
            _text.Fault(_text.Of(DeclareKey), $"'{DeclareKey}' holds a '{declare.Verb}', and a schema is a 'declare'.");
        }

        if (output is not (null or INamesTheAnswer))
        {
            _text.Fault(_text.Of(OutputKey), $"'{OutputKey}' holds a '{output.Verb}', which names no answer.");
        }

        if (drop is not null)
        {
            try
            {
                PipelinePreset.NamedOnce(drop);
            }
            catch (ArgumentException twice)
            {
                _text.Fault(_text.Of(DropKey), $"'{DropKey}': {StepCatalog.InTheFilesWords(twice)}");
            }
        }

        _text.ThrowIfFaulty();

        return new PipelinePreset((DeclareStep)declare!, drop, output as INamesTheAnswer, source);
    }

    /// <summary>A list of column names under a key, or nothing when the key is not there or holds something else.</summary>
    private string[]? NamesUnder(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var list))
        {
            return null;
        }

        if (list.ValueKind != JsonValueKind.Array
            || list.EnumerateArray().Any(each => each.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(each.GetString())))
        {
            _text.Fault(_text.Of(key), $"'{key}' is a list of column names.");

            return null;
        }

        return [.. list.EnumerateArray().Select(each => each.GetString()!)];
    }

    /// <summary>A step written under a key, read through the catalog; nothing when the key is not there or it cannot be read.</summary>
    private IPipelineStep? StepUnder(JsonElement root, string key, int version, StepCatalog catalog)
    {
        if (!root.TryGetProperty(key, out var written))
        {
            return null;
        }

        if (written.ValueKind != JsonValueKind.Object)
        {
            _text.Fault(_text.Of(key), $"A file of saved columns holds {PresetSteps[key]} under '{key}', as the step's own JSON object.");

            return null;
        }

        try
        {
            return catalog.Read(written, version);
        }
        catch (Exception fault) when (fault is FormatException or NotSupportedException)
        {
            _text.Fault(_text.Of(key), $"'{key}': {fault.Message}");

            return null;
        }
    }
}
