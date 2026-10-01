// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a saved pipeline holds once it is read back: its steps, what each step learned, and which its run left out.
/// </summary>
/// <param name="Declaration">The steps, in the order they were written.</param>
/// <param name="Fitted">What each step that learned learned, by its position; empty for a declaration alone.</param>
/// <param name="Skipped">The places of the steps its run left out for a learner, in ascending order; empty for a run of every step.</param>
internal readonly record struct SavedPipeline(
    PipelineDeclaration Declaration, IReadOnlyDictionary<int, FittedStepValues> Fitted, IReadOnlyList<int> Skipped);

/// <summary>
/// The file a pipeline is saved as, written and read in this one place — and the file of saved columns beside a
/// notebook, <c>{"version": 4, "source": [...], "declare": {...}, "drop": [...], "output": {...}}</c>, with the same care.
/// </summary>
/// <remarks>
/// <code>{"version": 4, "declaration": [ ... ], "fitted": [ {"step": ..., "prefix": ..., "learned": { ... }} ], "skipped": [ {"step": ..., "prefix": ...} ]}</code>
/// The version is the one the file was written against, so a step whose meaning changed since is refused by
/// name rather than read as something it never meant. The declaration is what a person wrote. Each fitted
/// entry names the step that learned it and the key of the steps it was learned behind
/// (<see cref="PipelineDeclaration.KeyAt"/>), so a fit is only ever served under the steps it was fitted
/// behind: an edit above a step that learned leaves its entry matching nothing, and an edit below it leaves
/// the entry be. A run for a learner that left steps out writes each under <c>skipped</c>, by its verb and the same
/// key, so a replay leaves out the same steps; a run that left nothing out writes no such block, and its file is the
/// file a run of every step always wrote.
/// <para>
/// Reading collects every fault before it refuses, each at its line and column. The text is walked token by
/// token first: that is where the places come from, and where a key written twice is caught, because the
/// document the values are then read from keeps neither the places nor the second of two keys.
/// </para>
/// <para>
/// A pipeline can also stand inside a larger file, as the value of a key at its top — a trained network beside the
/// pipeline it was trained behind. It is read the same way, and its faults are placed in the larger file.
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
    private const string SkippedKey = "skipped";

    /// <summary>The first version of the pipeline file in which a run writes the steps it left out for its learner.</summary>
    internal const int SkippedSince = 4;

    private static readonly string[] RootKeys = [VersionKey, DeclarationKey, FittedKey, SkippedKey];
    private static readonly string[] PresetKeys = [VersionKey, SourceKey, DeclareKey, DropKey, OutputKey];

    // The steps a file of saved columns holds, by key, and what each is to the person who reads a refusal.
    private static readonly Dictionary<string, string> PresetSteps = new(StringComparer.Ordinal)
    {
        [DeclareKey] = "its schema",
        [OutputKey] = "its output",
    };
    private static readonly IReadOnlyDictionary<int, FittedStepValues> NothingFitted = new Dictionary<int, FittedStepValues>();

    // How many bytes of a larger file are made at a time while it is walked for the pipeline it holds: the walk holds no
    // more of the file than this and the token it has reached.
    private const int Piece = 1 << 16;

    private readonly string _json;
    private readonly List<Fault> _faults = [];

    // Where the parts of the text stand, as the survey that opens every read found them.
    private Places _places = new();

    // The UTF-8 of the part of the text the document is, and where that part starts in the whole text: all of the text,
    // unless the pipeline stands inside a larger file as the value of one of its keys, when it is that value alone. Every
    // place is a place in the whole text, so a fault is at the line and column of the file a person opens.
    private byte[] _text;
    private long _start;

    private PipelineDocument(string json, byte[] text)
    {
        _json = json;
        _text = text;
    }

    // A pipeline's own file: all of its text is the document.
    private static PipelineDocument Whole(string json) => new(json, Encoding.UTF8.GetBytes(json));

    /// <summary>
    /// Writes a pipeline: the version, the steps, what the fit learned when there is a fit, and the steps its run left out
    /// when it left any out.
    /// </summary>
    /// <param name="course">The steps as the run took them, each written as it was declared.</param>
    /// <param name="fitted">What each step learned, by its position; nothing for a declaration alone.</param>
    /// <returns>The file, indented, with a line feed between lines on every system.</returns>
    public static string Write(Course course, IReadOnlyDictionary<int, FittedStepValues>? fitted)
    {
        var buffer = new ArrayBufferWriter<byte>();
        var declaration = course.Declaration;

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, PipelineDeclaration.Version);
            writer.WriteStartArray(DeclarationKey);

            // What stands in a step's place is written as that step, so the steps come out as they were declared.
            foreach (var step in course.Steps)
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

            // Only a run that left steps out says so: a run of every step writes the file it always wrote, and is the fit
            // it always was.
            if (course.Skipped.Count > 0)
            {
                writer.WriteStartArray(SkippedKey);

                foreach (var at in course.Skipped)
                {
                    writer.WriteStartObject();
                    writer.WriteString(Entry.StepKey, course.Steps[at].Verb);
                    writer.WriteString(Entry.PrefixKey, declaration.KeyAt(at));
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
        Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            Add(_places.Root, "A file of saved columns is one JSON object, holding its schema under 'declare' and whatever else was decided.");

            throw Refused();
        }

        // No preset was written before the second version, and its output may be a word only the second has: one that
        // names no version is not read as the first, as a pipeline file is.
        var version = root.TryGetProperty(VersionKey, out _) ? VersionOf(root) : NoVersion();

        foreach (var property in root.EnumerateObject().Where(property => !PresetKeys.Contains(property.Name)))
        {
            Add(_places.Keys[property.Name], $"A file of saved columns has no '{property.Name.Quoted()}'. It holds: {string.Join(", ", PresetKeys)}.");
        }

        var source = NamesUnder(root, SourceKey);
        var drop = NamesUnder(root, DropKey);
        var declare = StepUnder(root, DeclareKey, version, catalog);
        var output = StepUnder(root, OutputKey, version, catalog);

        if (!root.TryGetProperty(DeclareKey, out _))
        {
            Add(_places.Root, $"A file of saved columns holds its schema under '{DeclareKey}'.");
        }

        if (declare is not (null or DeclareStep))
        {
            Add(_places.Keys[DeclareKey], $"'{DeclareKey}' holds a '{declare.Verb}', and a schema is a 'declare'.");
        }

        if (output is not (null or INamesTheAnswer))
        {
            Add(_places.Keys[OutputKey], $"'{OutputKey}' holds a '{output.Verb}', which names no answer.");
        }

        if (drop is not null)
        {
            try
            {
                PipelinePreset.NamedOnce(drop);
            }
            catch (ArgumentException twice)
            {
                Add(_places.Keys[DropKey], $"'{DropKey}': {StepCatalog.InTheFilesWords(twice)}");
            }
        }

        ThrowIfFaulty();

        return new PipelinePreset((DeclareStep)declare!, drop, output as INamesTheAnswer, source);
    }

    private int NoVersion()
    {
        Add(_places.Root, $"A file of saved columns names the version of the pipeline file it was written against, under '{VersionKey}'.");

        return PipelineDeclaration.Version;
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
            Add(_places.Keys[key], $"'{key}' is a list of column names.");

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
            Add(_places.Keys[key], $"A file of saved columns holds {PresetSteps[key]} under '{key}', as the step's own JSON object.");

            return null;
        }

        try
        {
            return catalog.Read(written, version);
        }
        catch (Exception fault) when (fault is FormatException or NotSupportedException)
        {
            Add(_places.Keys[key], $"'{key}': {fault.Message}");

            return null;
        }
    }

    /// <summary>Reads the steps a file declares, leaving what a fit learned to the fit.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs the file may use.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static PipelineDeclaration ReadDeclaration(string json, StepCatalog catalog) =>
        Whole(json).Read(catalog, withFit: false).Declaration;

    /// <summary>Reads a whole pipeline: its steps, and what each step learned where it was learned.</summary>
    /// <param name="json">The file.</param>
    /// <param name="catalog">The verbs the file may use.</param>
    /// <returns>The steps and what they learned.</returns>
    /// <exception cref="PipelineFileException">Anything in the file is wrong; every fault is named.</exception>
    public static SavedPipeline ReadPipeline(string json, StepCatalog catalog) =>
        Whole(json).Read(catalog, withFit: true);

    /// <summary>Reads a whole pipeline that stands inside a larger file, as the value of a key at its top.</summary>
    /// <param name="json">The larger file.</param>
    /// <param name="catalog">The verbs the pipeline may use.</param>
    /// <param name="property">The key the pipeline stands under.</param>
    /// <returns>The steps and what they learned.</returns>
    /// <exception cref="PipelineFileException">
    /// The larger file is not JSON, not one object, or holds the key not once; or anything in the pipeline is wrong. Every
    /// fault is named at its line and column in the larger file.
    /// </exception>
    public static SavedPipeline ReadPipelineIn(string json, StepCatalog catalog, string property) =>
        Within(json, property).Read(catalog, withFit: true);

    /// <summary>The document a pipeline standing inside a larger file is: the value of a key at the top of that file.</summary>
    /// <remarks>
    /// The whole file is read first, so text that stops being JSON anywhere in it is said where it stops; the value is then
    /// read as a pipeline's own file is, its places kept as places in the whole file. The file is read a piece of its UTF-8
    /// at a time, and of all of it only the pipeline's own bytes are kept: the file around a pipeline may be a network of
    /// millions of numbers, and a copy of every byte of it would cost more than the pipeline to read past them.
    /// </remarks>
    private static PipelineDocument Within(string json, string property)
    {
        var document = new PipelineDocument(json, []);
        var pieces = new Pieces(json);
        var pipeline = new ArrayBufferWriter<byte>();
        var root = -1L;
        var key = -1L;
        var start = -1L;
        var taking = false;

        try
        {
            do
            {
                var reader = pieces.Next();
                var from = 0;

                while (reader.Read())
                {
                    var at = pieces.Passed + reader.TokenStartIndex;

                    if (root < 0)
                    {
                        root = at;

                        if (reader.TokenType != JsonTokenType.StartObject)
                        {
                            document.Add(root, $"A file that holds a pipeline in place is one JSON object, with the pipeline under '{property}'.");

                            throw document.Refused();
                        }
                    }
                    else if (key >= 0)
                    {
                        // The value of the key the pipeline stands under: the first is the pipeline, taken as it passes.
                        if (start >= 0)
                        {
                            document.Add(key, $"'{property}' is written twice here, and only one of the two would be read.");
                        }
                        else
                        {
                            start = at;
                            from = (int)reader.TokenStartIndex;
                            taking = reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray;

                            if (!taking)
                            {
                                pipeline.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                            }
                        }

                        key = -1;
                    }
                    else if (taking && reader.CurrentDepth == 1 && reader.TokenType is JsonTokenType.EndObject or JsonTokenType.EndArray)
                    {
                        pipeline.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                        taking = false;
                    }
                    else if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1 && reader.ValueTextEquals(property))
                    {
                        key = at;
                    }
                }

                // A pipeline that goes on into the next piece: what of it this piece held.
                if (taking)
                {
                    pipeline.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                }

                pieces.Taken(ref reader);
            }
            while (!pieces.Last);
        }
        catch (JsonException fault)
        {
            // Nothing after this point can be read, so it is the last fault there is.
            document.Add(document.OffsetOf(fault), $"The text stops being JSON here: {WithoutItsPlace(fault.Message).Quoted()}");

            throw document.Refused();
        }

        if (start < 0)
        {
            document.Add(root, $"This file holds no '{property}', which is where the pipeline in it stands.");
        }

        document.ThrowIfFaulty();
        document._text = pipeline.WrittenSpan.ToArray();
        document._start = start;

        return document;
    }

    /// <summary>Reads one step written on its own: a notebook block, say.</summary>
    /// <param name="json">The step, as the JSON object it was written as.</param>
    /// <param name="catalog">The verbs it may use.</param>
    /// <param name="writtenAgainst">The version of the pipeline file it was written against.</param>
    /// <returns>The step.</returns>
    /// <exception cref="PipelineFileException">The text is not one step the catalog reads; every fault is named.</exception>
    public static IPipelineStep ReadStep(string json, StepCatalog catalog, int writtenAgainst) =>
        Whole(json).Step(catalog, writtenAgainst);

    private IPipelineStep Step(StepCatalog catalog, int writtenAgainst)
    {
        Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);

        try
        {
            return catalog.Read(document.RootElement, writtenAgainst);
        }
        catch (Exception fault) when (fault is FormatException or NotSupportedException)
        {
            Add(_places.Root, fault.Message);

            throw Refused();
        }
    }

    private SavedPipeline Read(StepCatalog catalog, bool withFit)
    {
        Survey();

        ThrowIfFaulty();

        using var document = JsonDocument.Parse(_text);
        var root = document.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
        {
            Add(_places.Root, "A pipeline file is one JSON object, holding its 'declaration' and, once it is fitted, what the fit learned.");

            throw Refused();
        }

        var version = VersionOf(root);

        foreach (var property in root.EnumerateObject().Where(property => !RootKeys.Contains(property.Name)))
        {
            Add(_places.Keys[property.Name], $"A pipeline file has no '{property.Name.Quoted()}'. It holds: {string.Join(", ", RootKeys)}.");
        }

        // What a run left out is written from the version in which a run for a learner first left steps out; an older file
        // that says it is refused where it says it, rather than read as something no reader of its version would read.
        var said = root.TryGetProperty(SkippedKey, out var skips);

        if (said && version < SkippedSince)
        {
            Add(_places.Keys[SkippedKey], string.Create(
                CultureInfo.InvariantCulture,
                $"What a run left out for its learner is written from version {SkippedSince} of the pipeline file, and this file was written against version {version}, which has no '{SkippedKey}'."));
        }

        var fit = withFit && root.TryGetProperty(FittedKey, out var entries) ? entries : default;

        if (fit.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Array))
        {
            Add(_places.Keys[FittedKey], fit.ValueKind == JsonValueKind.Object
                ? "The fitted half is filed by the position of each step, as DeepSharp 0.2 wrote it, and a position says "
                  + "nothing about which steps a fit was learned behind. Fit the pipeline again."
                : "The fitted half is a list, with an entry for each step that learned something.");
        }

        var declaration = DeclarationOf(root, version, catalog);

        if (!withFit || declaration is null || fit.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Array))
        {
            ThrowIfFaulty();

            return new SavedPipeline(declaration!, NothingFitted, []);
        }

        // What the run left out is read before what it learned, since a step it left out learned nothing. A record of it
        // that cannot be read leaves the steps the run took unknown, and a missing entry is then not asked about.
        var course = !said ? Course.Whole(declaration) : version < SkippedSince ? null : CourseOf(skips, declaration);

        // A file without a fitted half is read as one that fitted nothing, so a step that learned is missed
        // there too: a declaration alone, loaded as a fitted pipeline, used to serve every value unfitted.
        var fitted = FittedOf(fit.ValueKind == JsonValueKind.Array ? [.. fit.EnumerateArray()] : [], declaration, course);

        ThrowIfFaulty();

        return new SavedPipeline(declaration, fitted, course!.Skipped);
    }

    /// <summary>The steps a file's run left out, each bound to its step; nothing when anything in that record is wrong.</summary>
    private Course? CourseOf(JsonElement skips, PipelineDeclaration declaration)
    {
        if (skips.ValueKind != JsonValueKind.Array)
        {
            Add(_places.Keys[SkippedKey], "What a run left out is a list, with an entry for each step it left out.");

            return null;
        }

        var positions = Positions(declaration);
        var entryOf = new Dictionary<int, int>();
        var every = true;
        var index = 0;

        foreach (var element in skips.EnumerateArray())
        {
            try
            {
                var at = Bound(Skip.Read(element), declaration, positions);

                if (!entryOf.TryAdd(at, index))
                {
                    throw new FormatException($"This is a second entry for step {at + 1}, '{declaration.Steps[at].Verb}', which a run leaves out once.");
                }
            }
            catch (FormatException fault)
            {
                Add(_places.Skips[index], fault.Message);
                every = false;
            }

            index++;
        }

        if (!every)
        {
            return null;
        }

        var faults = Course.FaultsIn(declaration, [.. entryOf.Keys]);

        foreach (var fault in faults)
        {
            Add(_places.Skips[entryOf[fault.At]], fault.ToString());
        }

        return faults.Count == 0 ? Course.Of(declaration, [.. entryOf.Keys]) : null;
    }

    /// <summary>The place of the step an entry of what a run left out names, or why it names none.</summary>
    private static int Bound(Skip skip, PipelineDeclaration declaration, Dictionary<string, int> positions)
    {
        if (!positions.TryGetValue(skip.Prefix, out var at))
        {
            throw new FormatException(
                $"This entry says '{skip.Verb.Quoted()}' was left out behind steps this declaration does not have: a step above it changed "
                + "after the run. Run the pipeline again.");
        }

        return declaration.Steps[at].Verb == skip.Verb
            ? at
            : throw new FormatException(
                $"This entry says '{skip.Verb.Quoted()}' was left out, and the step at that place is step {at + 1}, '{declaration.Steps[at].Verb}'.");
    }

    // Every step's place, by the key of the steps up to it: what an entry of the file is filed under.
    private static Dictionary<string, int> Positions(PipelineDeclaration declaration) =>
        Enumerable.Range(0, declaration.Steps.Count).ToDictionary(declaration.KeyAt, StringComparer.Ordinal);

    /// <summary>The version the file was written against: the first, when it names none.</summary>
    private int VersionOf(JsonElement root)
    {
        if (!root.TryGetProperty(VersionKey, out var written))
        {
            return 1;
        }

        if (!written.TryGetWholeNumber(out var version) || version < 1)
        {
            Add(_places.Keys[VersionKey], $"The version is the whole number of the file format a pipeline was written against, counting from 1, and {written.GetRawText().Quoted()} is not one.");

            return PipelineDeclaration.Version;
        }

        if (version > PipelineDeclaration.Version)
        {
            // A newer file may use words this version never had, so none of it is read: the one thing to say
            // is that it is newer, rather than a list of faults that are faults only to an older reader.
            Add(_places.Keys[VersionKey], string.Create(
                CultureInfo.InvariantCulture,
                $"This file was written against version {version} of the pipeline file, by a newer DeepSharp than this one, which reads up to version {PipelineDeclaration.Version}. Nothing in it is read: read it with that DeepSharp."));

            throw Refused();
        }

        return version;
    }

    /// <summary>The steps, every one read, and the rules every declaration keeps; nothing when any of that fails.</summary>
    private PipelineDeclaration? DeclarationOf(JsonElement root, int version, StepCatalog catalog)
    {
        if (!root.TryGetProperty(DeclarationKey, out var written) || written.ValueKind != JsonValueKind.Array)
        {
            Add(
                written.ValueKind == JsonValueKind.Undefined ? _places.Root : _places.Keys[DeclarationKey],
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
                Add(_places.Steps[at], $"Step {at + 1}: {fault.Message}");
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
            Add(_places.Steps[fault.At], fault.ToString());
        }

        return broken.Count == 0 ? new PipelineDeclaration(steps) : null;
    }

    /// <summary>What each step learned, filed by the key of the steps it was learned behind.</summary>
    /// <remarks>Held to the steps as the run took them; when those are unknown, to the steps as declared, and nothing is missed.</remarks>
    private Dictionary<int, FittedStepValues> FittedOf(IReadOnlyList<JsonElement> entries, PipelineDeclaration declaration, Course? course)
    {
        var fitted = new Dictionary<int, FittedStepValues>();
        var taken = course ?? Course.Whole(declaration);
        var positions = Positions(declaration);
        var every = true;
        var at = 0;

        foreach (var element in entries)
        {
            try
            {
                Bind(Entry.Read(element), taken, positions, fitted);
            }
            catch (FormatException fault)
            {
                Add(_places.Entries[at], fault.Message);
                every = false;
            }

            at++;
        }

        // A step without its entry is asked about only when every entry found its step, so one entry that is
        // wrong is one fault, not that and a missing fit besides; and only when the steps the run took are known.
        if (every && course is not null)
        {
            foreach (var missing in PreparedData.FitsMissing(course, fitted))
            {
                Add(_places.Steps[missing.At], missing.ToString());
            }
        }

        return fitted;
    }

    /// <summary>Files an entry under the step it was learned by, or refuses it.</summary>
    private static void Bind(
        Entry entry,
        Course course,
        Dictionary<string, int> positions,
        Dictionary<int, FittedStepValues> fitted)
    {
        if (!positions.TryGetValue(entry.Prefix, out var at))
        {
            throw new FormatException(
                $"What '{entry.Verb.Quoted()}' learned was fitted behind steps this declaration does not have: a step above it "
                + "changed after the fit. Fit the pipeline again.");
        }

        var step = course.Declaration.Steps[at];

        if (step.Verb != entry.Verb)
        {
            throw new FormatException(
                $"This entry says '{entry.Verb.Quoted()}' learned it, and the step it was learned at is step {at + 1}, '{step.Verb}'.");
        }

        if (!PreparedData.WritesAnEntry(course.Steps[at]))
        {
            throw new FormatException(course.Skipped.Contains(at)
                ? $"Step {at + 1}, '{step.Verb}', was left out of the run this file was written from, so nothing it learned can be here."
                : $"Step {at + 1}, '{step.Verb}', learns nothing, so nothing it learned can be here.");
        }

        if (!fitted.TryAdd(at, entry.Learned))
        {
            throw new FormatException(
                $"This is a second entry for step {at + 1}, '{step.Verb}', which is fitted once: the file cannot say which "
                + "of the two it learned.");
        }
    }

    /// <summary>Walks the text token by token: where everything stands, kept as this document's places, and every key written twice.</summary>
    /// <remarks>
    /// Of the part of the text the document is, each place counted from the start of the whole text. A part inside a larger
    /// file is only ever set once the whole file has been read as JSON, so text that stops being JSON is met here in a
    /// document that is the whole text.
    /// </remarks>
    private void Survey()
    {
        var places = new Places();
        var reader = new Utf8JsonReader(_text);
        var names = new Stack<HashSet<string>>();
        string? under = null;

        try
        {
            while (reader.Read())
            {
                var at = _start + reader.TokenStartIndex;

                if (reader.CurrentDepth == 0 && reader.TokenType is not (JsonTokenType.EndObject or JsonTokenType.EndArray))
                {
                    places.Root = at;
                }

                // An element of one of the lists at the top: a step, a fitted entry, or a step a run left out.
                if (reader.CurrentDepth == 2
                    && reader.TokenType is not (JsonTokenType.PropertyName or JsonTokenType.EndObject or JsonTokenType.EndArray))
                {
                    (under switch { DeclarationKey => places.Steps, FittedKey => places.Entries, SkippedKey => places.Skips, _ => null })?.Add(at);
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
                            Add(at, $"'{name.Quoted()}' is written twice here, and only one of the two would be read.");
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
            Add(OffsetOf(fault), $"The text stops being JSON here: {WithoutItsPlace(fault.Message).Quoted()}");

            throw Refused();
        }

        _places = places;
    }

    /// <summary>Where a reader's fault is, from the line and the byte in that line it names, both from nought.</summary>
    /// <remarks>
    /// Counted in the whole text's UTF-8 by walking the text itself, up to its end: the reader was handed that UTF-8, so the
    /// line it names is one the text has.
    /// </remarks>
    private long OffsetOf(JsonException fault)
    {
        var at = 0;
        var bytes = 0L;

        for (var lines = fault.LineNumber.GetValueOrDefault(); lines > 0; lines--)
        {
            var end = _json.IndexOf('\n', at);

            bytes += Encoding.UTF8.GetByteCount(_json.AsSpan(at, end + 1 - at));
            at = end + 1;
        }

        return Math.Min(bytes + fault.BytePositionInLine.GetValueOrDefault(), bytes + Encoding.UTF8.GetByteCount(_json.AsSpan(at)));
    }

    // The reader ends its message with where it stopped, counted from nought; the fault carries the place
    // counted from one, as an editor shows it, and saying it twice in two ways would only confuse.
    private static string WithoutItsPlace(string message) => message.Split(" LineNumber:")[0];

    private void Add(long offset, string message) => _faults.Add(new Fault(offset, message));

    private void ThrowIfFaulty()
    {
        if (_faults.Count > 0)
        {
            throw Refused();
        }
    }

    private PipelineFileException Refused() => new([.. Placed([.. _faults.OrderBy(fault => fault.Offset)])]);

    /// <summary>
    /// Each fault at its line and column, both from one, the column in characters rather than bytes: the text walked once, a
    /// character at a time, each counted as the bytes of UTF-8 it is written in, up to each fault's offset in turn.
    /// </summary>
    /// <param name="faults">The faults, in the order they stand; each offset one within the text or at its end.</param>
    private IEnumerable<PipelineFileFault> Placed(Fault[] faults)
    {
        var bytes = 0L;
        var line = 1;
        var lineStart = 0;
        var at = 0;

        foreach (var fault in faults)
        {
            while (bytes < fault.Offset)
            {
                Rune.DecodeFromUtf16(_json.AsSpan(at), out var letter, out var read);
                bytes += letter.Utf8SequenceLength;
                at += read;

                if (_json[at - 1] == '\n')
                {
                    line++;
                    lineStart = at;
                }
            }

            yield return new PipelineFileFault(line, at - lineStart + 1, fault.Message);
        }
    }

    // A text read as JSON a piece of its UTF-8 at a time: each piece handed to a reader behind what the reader left unread of
    // the one before, so no more of the text is ever held as bytes than a piece and the token it ends in.
    private sealed class Pieces(string text)
    {
        // The most bytes of UTF-8 one character of the text is written in.
        private const int WidestLetter = 4;

        private readonly Encoder _encoder = Encoding.UTF8.GetEncoder();
        private byte[] _buffer = new byte[Piece];
        private JsonReaderState _state;
        private int _made;
        private int _held;

        // How many bytes of the text come before the piece a reader was last handed.
        public long Passed { get; private set; }

        // Whether the piece a reader was last handed ends the text.
        public bool Last { get; private set; }

        // The next piece, behind what the reader left unread of the one before: the buffer grows for a token longer than it.
        public Utf8JsonReader Next()
        {
            if (_buffer.Length - _held < WidestLetter)
            {
                Array.Resize(ref _buffer, _buffer.Length * 2);
            }

            _encoder.Convert(text.AsSpan(_made), _buffer.AsSpan(_held), flush: true, out var letters, out var bytes, out _);
            _made += letters;
            _held += bytes;
            Last = _made == text.Length;

            return new Utf8JsonReader(_buffer.AsSpan(0, _held), Last, _state);
        }

        // The bytes of the piece a reader was last handed, from one place in it to another.
        public ReadOnlySpan<byte> Bytes(int from, int to) => _buffer.AsSpan(from, to - from);

        // What a reader took of its piece: what it left is kept, in front of the next.
        public void Taken(ref Utf8JsonReader reader)
        {
            var taken = (int)reader.BytesConsumed;

            _state = reader.CurrentState;
            _buffer.AsSpan(taken, _held - taken).CopyTo(_buffer);
            _held -= taken;
            Passed += taken;
        }
    }

    /// <summary>A fault, and the offset in the text of the thing it is about.</summary>
    private readonly record struct Fault(long Offset, string Message);

    /// <summary>Where the parts of a file stand in its text, as offsets of their first byte.</summary>
    private sealed class Places
    {
        /// <summary>The value the file is.</summary>
        public long Root { get; set; }

        /// <summary>Each key at the top of the file.</summary>
        public Dictionary<string, long> Keys { get; } = new(StringComparer.Ordinal);

        /// <summary>Each step of the declaration.</summary>
        public List<long> Steps { get; } = [];

        /// <summary>Each entry of the fitted half.</summary>
        public List<long> Entries { get; } = [];

        /// <summary>Each entry of what a run left out.</summary>
        public List<long> Skips { get; } = [];
    }

    /// <summary>One entry of what a run left out: the step, and the key of the steps up to it.</summary>
    /// <param name="Verb">The verb of the step left out.</param>
    /// <param name="Prefix">The key of the steps up to it.</param>
    private readonly record struct Skip(string Verb, string Prefix)
    {
        private static readonly string[] Keys = [Entry.StepKey, Entry.PrefixKey];

        public static Skip Read(JsonElement entry)
        {
            if (entry.ValueKind != JsonValueKind.Object)
            {
                throw new FormatException("An entry of what a run left out is an object: the step it left out, and the key of the steps up to it.");
            }

            if (entry.EnumerateObject().Select(property => property.Name).FirstOrDefault(name => !Keys.Contains(name)) is { } unknown)
            {
                throw new FormatException($"An entry of what a run left out has no '{unknown.Quoted()}'. It holds: {string.Join(", ", Keys)}.");
            }

            if (!entry.TryGetProperty(Entry.StepKey, out var verb) || verb.ValueKind != JsonValueKind.String)
            {
                throw new FormatException($"An entry of what a run left out names the step it left out, as text under '{Entry.StepKey}'.");
            }

            return entry.TryGetProperty(Entry.PrefixKey, out var prefix) && prefix.ValueKind == JsonValueKind.String
                ? new Skip(verb.GetString()!, prefix.GetString()!)
                : throw new FormatException(
                    $"An entry of what a run left out is filed under the key of the steps up to it, as text under '{Entry.PrefixKey}'.");
        }
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
                throw new FormatException($"An entry of the fitted half has no '{unknown.Quoted()}'. It holds: {string.Join(", ", Keys)}.");
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
                            $"A fit learns numbers, words, and lists of one or the other, and '{value.Name.Quoted()}' holds none of those.");
                }
            }

            return values;
        }

        private static double Held(JsonElement number, string name) =>
            number.TryGetDouble(out var value) && double.IsFinite(value)
                ? value
                : throw new FormatException($"'{name.Quoted()}' holds {number.GetRawText().Quoted()}, a number too large to hold.");
    }
}
