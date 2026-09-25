// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a saved pipeline holds once it is read back: its steps, and what each step learned.
/// </summary>
/// <param name="Declaration">The steps, in the order they were written.</param>
/// <param name="Fitted">What each step that learned learned, by its position; empty for a declaration alone.</param>
internal readonly record struct SavedPipeline(PipelineDeclaration Declaration, IReadOnlyDictionary<int, FittedStepValues> Fitted);

/// <summary>
/// The file a pipeline is saved as, written and read in this one place — and the file of saved columns beside a
/// notebook, <c>{"version": 2, "source": [...], "declare": {...}, "drop": [...], "output": {...}}</c>, with the same care.
/// </summary>
/// <remarks>
/// <code>{"version": 2, "declaration": [ ... ], "fitted": [ {"step": ..., "prefix": ..., "learned": { ... }} ]}</code>
/// The version is the one the file was written against, so a step whose meaning changed since is refused by
/// name rather than read as something it never meant. The declaration is what a person wrote. Each fitted
/// entry names the step that learned it and the key of the steps it was learned behind
/// (<see cref="PipelineDeclaration.KeyAt"/>), so a fit is only ever served under the steps it was fitted
/// behind: an edit above a step that learned leaves its entry matching nothing, and an edit below it leaves
/// the entry be.
/// <para>
/// Reading collects every fault before it refuses, each at its line and column. The text is walked token by
/// token first: that is where the places come from, and where a key written twice is caught, because the
/// document the values are then read from keeps neither the places nor the second of two keys.
/// </para>
/// </remarks>
internal sealed class PipelineDocument
{
    private const string VersionKey = "version";
    private const string DeclarationKey = "declaration";
    private const string FittedKey = "fitted";
    private const string SourceKey = "source";
    private const string DeclareKey = "declare";
    private const string DropKey = "drop";
    private const string OutputKey = "output";

    private static readonly string[] RootKeys = [VersionKey, DeclarationKey, FittedKey];
    private static readonly string[] PresetKeys = [VersionKey, SourceKey, DeclareKey, DropKey, OutputKey];
    private static readonly IReadOnlyDictionary<int, FittedStepValues> NothingFitted = new Dictionary<int, FittedStepValues>();

    private readonly byte[] _text;
    private readonly int[] _lineStarts;
    private readonly List<Fault> _faults = [];

    private PipelineDocument(string json)
    {
        _text = Encoding.UTF8.GetBytes(json);
        _lineStarts = LineStarts(_text);
    }

    /// <summary>Writes a pipeline: the version, the steps, and — when there is one — what the fit learned.</summary>
    /// <param name="declaration">The steps.</param>
    /// <param name="fitted">What each step learned, by its position; nothing for a declaration alone.</param>
    /// <returns>The file, indented, with a line feed between lines on every system.</returns>
    public static string Write(PipelineDeclaration declaration, IReadOnlyDictionary<int, FittedStepValues>? fitted)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, PipelineDeclaration.Version);
            writer.WriteStartArray(DeclarationKey);

            foreach (var step in declaration.Steps)
            {
                step.WriteTo(writer);
            }

            writer.WriteEndArray();

            if (fitted is not null)
            {
                writer.WriteStartArray(FittedKey);

                foreach (var (at, learned) in fitted.OrderBy(each => each.Key))
                {
                    writer.WriteStartObject();
                    writer.WriteString(Entry.StepKey, declaration.Steps[at].Verb);
                    writer.WriteString(Entry.PrefixKey, declaration.KeyAt(at));
                    writer.WritePropertyName(Entry.LearnedKey);
                    learned.WriteTo(writer);
                    writer.WriteEndObject();
                }

                writer.WriteEndArray();
            }

            writer.WriteEndObject();
        }

        // A writer ends its lines the way the machine does on some runtimes; a file is the same file everywhere.
        return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
    }

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
    public static PipelinePreset ReadPreset(string json, StepCatalog catalog) => new PipelineDocument(json).Preset(catalog);

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
        var places = Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            Add(places.Root, "A file of saved columns is one JSON object, holding its schema under 'declare' and whatever else was decided.");

            throw Refused();
        }

        // No preset was written before the second version, and its output may be a word only the second has: one that
        // names no version is not read as the first, as a pipeline file is.
        var version = root.TryGetProperty(VersionKey, out _) ? VersionOf(root, places) : NoVersion(places);

        foreach (var property in root.EnumerateObject().Where(property => !PresetKeys.Contains(property.Name)))
        {
            Add(places.Keys[property.Name], $"A file of saved columns has no '{property.Name}'. It holds: {string.Join(", ", PresetKeys)}.");
        }

        var source = NamesUnder(root, SourceKey, places);
        var drop = NamesUnder(root, DropKey, places);
        var declare = StepUnder(root, DeclareKey, "its schema", version, catalog, places);
        var output = StepUnder(root, OutputKey, "its output", version, catalog, places);

        if (!root.TryGetProperty(DeclareKey, out _))
        {
            Add(places.Root, $"A file of saved columns holds its schema under '{DeclareKey}'.");
        }

        if (declare is not (null or DeclareStep))
        {
            Add(places.Keys[DeclareKey], $"'{DeclareKey}' holds a '{declare.Verb}', and a schema is a 'declare'.");
        }

        if (output is not (null or INamesTheAnswer))
        {
            Add(places.Keys[OutputKey], $"'{OutputKey}' holds a '{output.Verb}', which names no answer.");
        }

        if (drop is not null)
        {
            try
            {
                PipelinePreset.NamedOnce(drop);
            }
            catch (ArgumentException twice)
            {
                Add(places.Keys[DropKey], $"'{DropKey}': {StepCatalog.InTheFilesWords(twice)}");
            }
        }

        ThrowIfFaulty();

        return new PipelinePreset((DeclareStep)declare!, drop, output as INamesTheAnswer, source);
    }

    private int NoVersion(Places places)
    {
        Add(places.Root, $"A file of saved columns names the version of the pipeline file it was written against, under '{VersionKey}'.");

        return PipelineDeclaration.Version;
    }

    /// <summary>A list of column names under a key, or nothing when the key is not there or holds something else.</summary>
    private string[]? NamesUnder(JsonElement root, string key, Places places)
    {
        if (!root.TryGetProperty(key, out var list))
        {
            return null;
        }

        if (list.ValueKind != JsonValueKind.Array
            || list.EnumerateArray().Any(each => each.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(each.GetString())))
        {
            Add(places.Keys[key], $"'{key}' is a list of column names.");

            return null;
        }

        return [.. list.EnumerateArray().Select(each => each.GetString()!)];
    }

    /// <summary>A step written under a key, read through the catalog; nothing when the key is not there or it cannot be read.</summary>
    private IPipelineStep? StepUnder(JsonElement root, string key, string what, int version, StepCatalog catalog, Places places)
    {
        if (!root.TryGetProperty(key, out var written))
        {
            return null;
        }

        if (written.ValueKind != JsonValueKind.Object)
        {
            Add(places.Keys[key], $"A file of saved columns holds {what} under '{key}', as the step's own JSON object.");

            return null;
        }

        try
        {
            return catalog.Read(written, version);
        }
        catch (Exception fault) when (fault is FormatException or NotSupportedException)
        {
            Add(places.Keys[key], $"'{key}': {fault.Message}");

            return null;
        }
    }

    /// <summary>Reads the steps a file declares, leaving what a fit learned to the fit.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs the file may use.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static PipelineDeclaration ReadDeclaration(string json, StepCatalog catalog) =>
        new PipelineDocument(json).Read(catalog, withFit: false).Declaration;

    /// <summary>Reads a whole pipeline: its steps, and what each step learned where it was learned.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs the file may use.</param>
    /// <returns>The steps and what they learned.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static SavedPipeline ReadPipeline(string json, StepCatalog catalog) =>
        new PipelineDocument(json).Read(catalog, withFit: true);

    /// <summary>Reads one step written on its own: a notebook block, say.</summary>
    /// <param name="json">The step, as the JSON object it was written as.</param>
    /// <param name="catalog">The verbs it may use.</param>
    /// <param name="writtenAgainst">The version of the pipeline file it was written against.</param>
    /// <returns>The step.</returns>
    /// <exception cref="PipelineFileException">The text is not one step the catalog reads; every fault is named.</exception>
    public static IPipelineStep ReadStep(string json, StepCatalog catalog, int writtenAgainst) =>
        new PipelineDocument(json).Step(catalog, writtenAgainst);

    private IPipelineStep Step(StepCatalog catalog, int writtenAgainst)
    {
        var places = Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);

        try
        {
            return catalog.Read(document.RootElement, writtenAgainst);
        }
        catch (Exception fault) when (fault is FormatException or NotSupportedException)
        {
            Add(places.Root, fault.Message);

            throw Refused();
        }
    }

    private SavedPipeline Read(StepCatalog catalog, bool withFit)
    {
        var places = Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            Add(places.Root, "A pipeline file is one JSON object, holding its 'declaration' and, once it is fitted, what the fit learned.");

            throw Refused();
        }

        var version = VersionOf(root, places);

        foreach (var property in root.EnumerateObject().Where(property => !RootKeys.Contains(property.Name)))
        {
            Add(places.Keys[property.Name], $"A pipeline file has no '{property.Name}'. It holds: {string.Join(", ", RootKeys)}.");
        }

        var fit = withFit && root.TryGetProperty(FittedKey, out var entries) ? entries : default;

        if (fit.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Array))
        {
            Add(places.Keys[FittedKey], fit.ValueKind == JsonValueKind.Object
                ? "The fitted half is filed by the position of each step, as DeepSharp 0.2 wrote it, and a position says "
                  + "nothing about which steps a fit was learned behind. Fit the pipeline again."
                : "The fitted half is a list, with an entry for each step that learned something.");
        }

        var declaration = DeclarationOf(root, version, catalog, places);

        // A file without a fitted half is read as one that fitted nothing, so a step that learned is missed
        // there too: a declaration alone, loaded as a fitted pipeline, used to serve every value unfitted.
        var fitted = withFit && declaration is not null && fit.ValueKind is JsonValueKind.Undefined or JsonValueKind.Array
            ? FittedOf(fit.ValueKind == JsonValueKind.Array ? [.. fit.EnumerateArray()] : [], declaration, places)
            : NothingFitted;

        ThrowIfFaulty();

        return new SavedPipeline(declaration!, fitted);
    }

    /// <summary>The version the file was written against: the first, when it names none.</summary>
    private int VersionOf(JsonElement root, Places places)
    {
        if (!root.TryGetProperty(VersionKey, out var written))
        {
            return 1;
        }

        if (!written.TryGetWholeNumber(out var version) || version < 1)
        {
            Add(places.Keys[VersionKey], $"The version is the whole number of the file format a pipeline was written against, counting from 1, and {written.GetRawText()} is not one.");

            return PipelineDeclaration.Version;
        }

        if (version > PipelineDeclaration.Version)
        {
            // A newer file may use words this version never had, so none of it is read: the one thing to say
            // is that it is newer, rather than a list of faults that are faults only to an older reader.
            Add(places.Keys[VersionKey], string.Create(
                CultureInfo.InvariantCulture,
                $"This file was written against version {version} of the pipeline file, by a newer DeepSharp than this one, which reads up to version {PipelineDeclaration.Version}. Nothing in it is read: read it with that DeepSharp."));

            throw Refused();
        }

        return version;
    }

    /// <summary>The steps, every one read, and the rules every declaration keeps; nothing when any of that fails.</summary>
    private PipelineDeclaration? DeclarationOf(JsonElement root, int version, StepCatalog catalog, Places places)
    {
        if (!root.TryGetProperty(DeclarationKey, out var written) || written.ValueKind != JsonValueKind.Array)
        {
            Add(
                written.ValueKind == JsonValueKind.Undefined ? places.Root : places.Keys[DeclarationKey],
                "A pipeline file holds its steps as a list, under 'declaration'.");

            return null;
        }

        var steps = new List<IPipelineStep>();
        var at = 0;

        foreach (var element in written.EnumerateArray())
        {
            try
            {
                steps.Add(catalog.Read(element, version));
            }
            catch (Exception fault) when (fault is FormatException or NotSupportedException)
            {
                Add(places.Steps[at], $"Step {at + 1}: {fault.Message}");
            }

            at++;
        }

        // The rules are about the whole list, so they are asked only of a list with every step in it: one step
        // missing would move every other one and make faults of steps that have none.
        if (steps.Count < at)
        {
            return null;
        }

        var broken = PipelineDeclaration.FaultsIn(steps);

        foreach (var fault in broken)
        {
            Add(places.Steps[fault.At], fault.ToString());
        }

        return broken.Count == 0 ? new PipelineDeclaration(steps) : null;
    }

    /// <summary>What each step learned, filed by the key of the steps it was learned behind.</summary>
    private Dictionary<int, FittedStepValues> FittedOf(IReadOnlyList<JsonElement> entries, PipelineDeclaration declaration, Places places)
    {
        var fitted = new Dictionary<int, FittedStepValues>();
        var positions = Enumerable.Range(0, declaration.Steps.Count).ToDictionary(declaration.KeyAt, StringComparer.Ordinal);
        var every = true;
        var at = 0;

        foreach (var element in entries)
        {
            try
            {
                Bind(Entry.Read(element), declaration, positions, fitted);
            }
            catch (FormatException fault)
            {
                Add(places.Entries[at], fault.Message);
                every = false;
            }

            at++;
        }

        // A step without its entry is asked about only when every entry found its step, so one entry that is
        // wrong is one fault, not that and a missing fit besides.
        if (every)
        {
            foreach (var missing in PreparedData.FitsMissing(declaration, fitted))
            {
                Add(places.Steps[missing.At], missing.ToString());
            }
        }

        return fitted;
    }

    /// <summary>Files an entry under the step it was learned by, or refuses it.</summary>
    private static void Bind(
        Entry entry,
        PipelineDeclaration declaration,
        Dictionary<string, int> positions,
        Dictionary<int, FittedStepValues> fitted)
    {
        if (!positions.TryGetValue(entry.Prefix, out var at))
        {
            throw new FormatException(
                $"What '{entry.Verb}' learned was fitted behind steps this declaration does not have: a step above it "
                + "changed after the fit. Fit the pipeline again.");
        }

        var step = declaration.Steps[at];

        if (step.Verb != entry.Verb)
        {
            throw new FormatException(
                $"This entry says '{entry.Verb}' learned it, and the step it was learned at is step {at + 1}, '{step.Verb}'.");
        }

        if (!PreparedData.WritesAnEntry(step))
        {
            throw new FormatException($"Step {at + 1}, '{step.Verb}', learns nothing, so nothing it learned can be here.");
        }

        if (!fitted.TryAdd(at, entry.Learned))
        {
            throw new FormatException(
                $"This is a second entry for step {at + 1}, '{step.Verb}', which is fitted once: the file cannot say which "
                + "of the two it learned.");
        }
    }

    /// <summary>Walks the text token by token: where everything stands, and every key written twice.</summary>
    private Places Survey()
    {
        var places = new Places();
        var reader = new Utf8JsonReader(_text);
        var names = new Stack<HashSet<string>>();
        string? under = null;

        try
        {
            while (reader.Read())
            {
                var at = (int)reader.TokenStartIndex;

                if (reader.CurrentDepth == 0 && reader.TokenType is not (JsonTokenType.EndObject or JsonTokenType.EndArray))
                {
                    places.Root = at;
                }

                // An element of one of the two lists at the top: a step, or a fitted entry.
                if (reader.CurrentDepth == 2
                    && reader.TokenType is not (JsonTokenType.PropertyName or JsonTokenType.EndObject or JsonTokenType.EndArray))
                {
                    (under switch { DeclarationKey => places.Steps, FittedKey => places.Entries, _ => null })?.Add(at);
                }

                switch (reader.TokenType)
                {
                    case JsonTokenType.StartObject:
                        names.Push(new HashSet<string>(StringComparer.Ordinal));
                        break;

                    case JsonTokenType.EndObject:
                        names.Pop();
                        break;

                    case JsonTokenType.PropertyName:
                        var name = reader.GetString()!;

                        if (!names.Peek().Add(name))
                        {
                            Add(at, $"'{name}' is written twice here, and only one of the two would be read.");
                        }

                        if (reader.CurrentDepth == 1)
                        {
                            under = name;
                            places.Keys.TryAdd(name, at);
                        }

                        break;
                }
            }
        }
        catch (JsonException fault)
        {
            // Nothing after this point can be read, so it is the last fault there is.
            Add(OffsetOf(fault), $"The text stops being JSON here: {WithoutItsPlace(fault.Message)}");

            throw Refused();
        }

        return places;
    }

    /// <summary>Where a reader's fault is, from the line and the byte in that line it names, both from nought.</summary>
    private int OffsetOf(JsonException fault)
    {
        var line = (int)Math.Min(fault.LineNumber.GetValueOrDefault(), _lineStarts.Length - 1);

        return (int)Math.Min(_lineStarts[line] + fault.BytePositionInLine.GetValueOrDefault(), _text.Length);
    }

    // The reader ends its message with where it stopped, counted from nought; the fault carries the place
    // counted from one, as an editor shows it, and saying it twice in two ways would only confuse.
    private static string WithoutItsPlace(string message) => message.Split(" LineNumber:")[0];

    private void Add(int offset, string message) => _faults.Add(new Fault(offset, message));

    private void ThrowIfFaulty()
    {
        if (_faults.Count > 0)
        {
            throw Refused();
        }
    }

    private PipelineFileException Refused() => new([.. _faults.OrderBy(fault => fault.Offset).Select(Placed)]);

    /// <summary>A fault at its line and column, both from one; the column in characters rather than bytes.</summary>
    private PipelineFileFault Placed(Fault fault)
    {
        var line = Array.BinarySearch(_lineStarts, fault.Offset);

        if (line < 0)
        {
            line = ~line - 1;
        }

        var column = Encoding.UTF8.GetCharCount(_text, _lineStarts[line], fault.Offset - _lineStarts[line]) + 1;

        return new PipelineFileFault(line + 1, column, fault.Message);
    }

    private static int[] LineStarts(byte[] text)
    {
        var starts = new List<int> { 0 };

        for (var at = 0; at < text.Length; at++)
        {
            if (text[at] == (byte)'\n')
            {
                starts.Add(at + 1);
            }
        }

        return [.. starts];
    }

    /// <summary>A fault, and the offset in the text of the thing it is about.</summary>
    private readonly record struct Fault(int Offset, string Message);

    /// <summary>Where the parts of a file stand in its text, as offsets of their first byte.</summary>
    private sealed class Places
    {
        /// <summary>The value the file is.</summary>
        public int Root { get; set; }

        /// <summary>Each key at the top of the file.</summary>
        public Dictionary<string, int> Keys { get; } = new(StringComparer.Ordinal);

        /// <summary>Each step of the declaration.</summary>
        public List<int> Steps { get; } = [];

        /// <summary>Each entry of the fitted half.</summary>
        public List<int> Entries { get; } = [];
    }

    /// <summary>One entry of the fitted half: the step that learned something, where, and what.</summary>
    /// <param name="Verb">The verb of the step that learned it.</param>
    /// <param name="Prefix">The key of the steps it was learned behind.</param>
    /// <param name="Learned">What it learned.</param>
    private readonly record struct Entry(string Verb, string Prefix, FittedStepValues Learned)
    {
        public const string StepKey = StepCatalog.StepKey;
        public const string PrefixKey = "prefix";
        public const string LearnedKey = "learned";

        private static readonly string[] Keys = [StepKey, PrefixKey, LearnedKey];

        public static Entry Read(JsonElement entry)
        {
            if (entry.ValueKind != JsonValueKind.Object)
            {
                throw new FormatException(
                    "An entry of the fitted half is an object: the step that learned something, the key of the steps it "
                    + "learned behind, and what it learned.");
            }

            if (entry.EnumerateObject().Select(property => property.Name).FirstOrDefault(name => !Keys.Contains(name)) is { } unknown)
            {
                throw new FormatException($"An entry of the fitted half has no '{unknown}'. It holds: {string.Join(", ", Keys)}.");
            }

            if (!entry.TryGetProperty(StepKey, out var verb) || verb.ValueKind != JsonValueKind.String)
            {
                throw new FormatException($"An entry of the fitted half names the step that learned it, as text under '{StepKey}'.");
            }

            if (!entry.TryGetProperty(PrefixKey, out var prefix) || prefix.ValueKind != JsonValueKind.String)
            {
                throw new FormatException(
                    $"An entry of the fitted half is filed under the key of the steps it was learned behind, as text under '{PrefixKey}'.");
            }

            if (!entry.TryGetProperty(LearnedKey, out var learned) || learned.ValueKind != JsonValueKind.Object)
            {
                throw new FormatException($"What a step learned is an object of names and values, under '{LearnedKey}'.");
            }

            return new Entry(verb.GetString()!, prefix.GetString()!, Values(learned));
        }

        private static FittedStepValues Values(JsonElement learned)
        {
            var values = new FittedStepValues();

            foreach (var value in learned.EnumerateObject())
            {
                var written = value.Value;

                switch (written.ValueKind)
                {
                    case JsonValueKind.Number:
                        values.Learned(value.Name, Held(written, value.Name));
                        break;

                    case JsonValueKind.String:
                        values.Learned(value.Name, written.GetString()!);
                        break;

                    // An empty list of words and an empty run of numbers are written alike, and either reads it.
                    case JsonValueKind.Array when written.GetArrayLength() == 0:
                        values.Learned(value.Name, Array.Empty<string>());
                        break;

                    case JsonValueKind.Array when written.EnumerateArray().All(each => each.ValueKind == JsonValueKind.Number):
                        values.Learned(value.Name, written.EnumerateArray().Select(each => Held(each, value.Name)).ToArray());
                        break;

                    case JsonValueKind.Array when written.EnumerateArray().All(each => each.ValueKind == JsonValueKind.String):
                        values.Learned(value.Name, written.EnumerateArray().Select(each => each.GetString()!).ToArray());
                        break;

                    default:
                        throw new FormatException(
                            $"A fit learns numbers, words, and lists of one or the other, and '{value.Name}' holds none of those.");
                }
            }

            return values;
        }

        private static double Held(JsonElement number, string name) =>
            number.TryGetDouble(out var value) && double.IsFinite(value)
                ? value
                : throw new FormatException($"'{name}' holds {number.GetRawText()}, a number too large to hold.");
    }
}
