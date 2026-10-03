// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>What a model predicted for one part a report measures, as the text of the predictions holds it.</summary>
/// <param name="Part">The part.</param>
/// <param name="Keys">The key of each row, in the order the part hands its rows over.</param>
/// <param name="Predictions">What was predicted for each row, in the units the answers were handed over in.</param>
/// <param name="Unfamiliar">
/// For each row, the features the model said it was handed a value of that it learned nothing about; nothing when it said
/// nothing of them.
/// </param>
internal readonly record struct WrittenPart(
    Part Part, IReadOnlyList<RowKey> Keys, IReadOnlyList<double[]> Predictions, IReadOnlyList<IReadOnlyList<string>>? Unfamiliar)
{
    /// <summary>What a model predicted for a part, as the text keeps it: the keys of its rows, and none of their numbers.</summary>
    /// <param name="part">The part.</param>
    /// <param name="given">The predictions measured for it, made for a batch the pipeline handed over.</param>
    /// <returns>The part as the text holds it.</returns>
    public static WrittenPart Of(Part part, PartPredictions given) => new(part, given.Batch.Keys!, given.Predictions, given.Unfamiliar);

    /// <summary>The predictions as measuring takes them: made for a batch that is the part and the keys of its rows.</summary>
    /// <returns>The predictions, whose batch holds no numbers, since the text carries none.</returns>
    public PartPredictions ToPartPredictions() =>
        new(new Batch([], [], null) { Part = Part, Keys = Keys }, Predictions) { Unfamiliar = Unfamiliar };
}

/// <summary>
/// The fit a model's predictions were made behind, as the text of the predictions names it: as two pipelines are compared, by
/// the one writer's text with the version left out, the version that text names, and the steps its run left out.
/// </summary>
/// <param name="Fit">The <see cref="PipelineText.FitDigest"/> of the pipeline they were made behind.</param>
/// <param name="Version">The version of the pipeline file that pipeline names.</param>
/// <param name="Skipped">The key of the steps up to each step that pipeline's run left out for its learner; none for a run of every step.</param>
internal readonly record struct WrittenFit(string Fit, int Version, IReadOnlyList<string> Skipped)
{
    /// <summary>The fit a pipeline is, as the text names it.</summary>
    /// <param name="prepared">The pipeline, run.</param>
    /// <returns>Its fit.</returns>
    public static WrittenFit Of(PreparedData prepared)
    {
        var text = PipelineText.Of(prepared);

        return new WrittenFit(text.FitDigest, text.Version, [.. prepared.Skipped.Select(prepared.Declaration.KeyAt)]);
    }

    /// <summary>
    /// Refuses predictions made behind another fit than a pipeline's, by the rule a network's file is held to its pipeline
    /// by: the version the fit names is one in which every step means what it means now, and the two are one fit — this
    /// run's text as a run that left out the same steps would have written it.
    /// </summary>
    /// <param name="here">The pipeline that would measure them.</param>
    /// <exception cref="ArgumentException">
    /// They were made behind another fit, behind a pipeline whose version no comparison knows, or behind a run that left out
    /// steps this pipeline does not have, that no run of it leaves out together, or that took a step this run left out.
    /// </exception>
    /// <remarks>
    /// A run for a learner shares with the run of every step the split, what was filled and the categories learned, so
    /// predictions made behind it are measured again on a run of every step: its text is written again as the run that left
    /// those steps out, which a learner of another kind was trained behind.
    /// </remarks>
    public void ThrowIfNotBehind(PreparedData here)
    {
        if (!PipelineText.IsComparable(Version))
        {
            throw new ArgumentException(
                Version > PipelineDeclaration.Version
                    ? string.Create(
                        CultureInfo.InvariantCulture,
                        $"These predictions were made behind a pipeline written against version {Version} of the pipeline file, by a newer DeepSharp than this one, which reads up to version {PipelineDeclaration.Version}: measure them with that DeepSharp, or make them again behind a pipeline this one writes.")
                    : string.Create(
                        CultureInfo.InvariantCulture,
                        $"These predictions name version {Version} of the pipeline file for the pipeline they were made behind, and no DeepSharp writes a pipeline under it, which is from {PipelineText.FirstComparable} to {PipelineDeclaration.Version}: make them again behind this pipeline."),
                "predictions");
        }

        var positions = Enumerable.Range(0, here.Declaration.Steps.Count).ToDictionary(here.Declaration.KeyAt, StringComparer.Ordinal);

        if (Skipped.Any(prefix => !positions.ContainsKey(prefix)))
        {
            throw new ArgumentException(
                "These predictions say the run they were made behind left out a step behind steps this pipeline does not have: they "
                + "were made behind another pipeline. Make them again behind this one.",
                "predictions");
        }

        int[] left = [.. Skipped.Select(prefix => positions[prefix])];

        // A run that left out a step the predictions' run took has nothing that step learned to measure them behind.
        if (here.Skipped.Except(left).ToArray() is [_, ..] taken)
        {
            throw new ArgumentException(
                $"These predictions were made behind a run that took steps this run left out: {string.Join("; ", taken.Select(at => $"step {at + 1}, '{here.Declaration.Steps[at].Verb}'"))}. "
                + "Measure them on a run of every step, Run(), or on the run they were made behind.",
                "predictions");
        }

        if (Course.FaultsIn(here.Declaration, left) is [var fault, ..])
        {
            throw new ArgumentException(
                $"These predictions say the run they were made behind left out steps no run of this pipeline leaves out so. {fault}",
                "predictions");
        }

        if (!string.Equals(PipelineText.Of(Encoding.UTF8.GetBytes(here.TextWithout(left))).FitDigest, Fit, StringComparison.Ordinal))
        {
            throw new ArgumentException(
                "These predictions were made behind another fit of the pipeline than this one — other steps, or other numbers learned "
                + "from other rows — so each would be measured through a way back, and against rows, it was not made behind. Make them "
                + "again behind this fit, training on it once more, and measure those.",
                "predictions");
        }
    }
}

/// <summary>What a model predicted for the parts a report measures, as their text holds it.</summary>
/// <param name="Behind">The fit the predictions were made behind.</param>
/// <param name="Answers">What each number of a row's predictions stands for, in the order the output names them.</param>
/// <param name="Parts">Each part, in the order the report names them.</param>
internal readonly record struct WrittenPredictions(WrittenFit Behind, IReadOnlyList<string> Answers, IReadOnlyList<WrittenPart> Parts)
{
    /// <summary>What a model predicted, as its text.</summary>
    /// <returns>The text, as one JSON object.</returns>
    public string ToJson() => PredictionsDocument.Write(this);

    /// <summary>What a model predicted, read back from its text.</summary>
    /// <param name="json">The text.</param>
    /// <returns>What it says.</returns>
    /// <exception cref="ArgumentNullException">There is no text.</exception>
    /// <exception cref="FormatException">The text is not what <see cref="ToJson"/> writes, said with where it is not; or a newer DeepSharp wrote it.</exception>
    public static WrittenPredictions FromJson(string json) => PredictionsDocument.Read(json);
}

/// <summary>
/// The text of what a model predicted for the parts a pipeline's report measures, which crosses to wherever the same pipeline
/// runs — a notebook's report block, whose types are not a C# cell's even when their names are — and is measured there again.
/// </summary>
/// <remarks>
/// One JSON object: the version of the text; the fit the predictions were made behind, as two pipelines are compared — the
/// <see cref="PipelineText.FitDigest"/> of its pipeline, the version of the pipeline file that names, and, for a run that left
/// steps out for its learner, the key of the steps up to each of them; the answers a row's numbers stand for; and for each
/// part, once, in the order the
/// report names them, the part as a pipeline's file writes it, the key of each row in the order the part hands them over, as
/// <see cref="RowKey.ToString"/> writes it, what was predicted for each row in the units the answers were handed over in, and —
/// when the model said it — the features of each row it learned nothing about. A number is written as the shortest text that
/// reads back as the same number, so the predictions come back to the last bit. Nothing else crosses: what the predictions are
/// measured against, and the way back to the answers' units, belong to the run that reads them.
/// </remarks>
internal static class PredictionsDocument
{
    /// <summary>The version of the text this library writes and reads up to.</summary>
    public const int Version = 1;

    private const string VersionKey = "version";
    private const string PipelineKey = "pipeline";
    private const string FitKey = "fit";
    private const string SkippedKey = "skipped";
    private const string AnswersKey = "answers";
    private const string PartsKey = "parts";
    private const string PartKey = "part";
    private const string KeysKey = "keys";
    private const string PredictionsKey = "predictions";
    private const string UnfamiliarKey = "unfamiliar";

    /// <summary>Writes what a model predicted as its text.</summary>
    /// <param name="predicted">The fit they were made behind, the answers, and each part the report measures.</param>
    /// <returns>The text, as one JSON object.</returns>
    public static string Write(WrittenPredictions predicted)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, Version);
            writer.WriteStartObject(PipelineKey);
            writer.WriteString(FitKey, predicted.Behind.Fit);
            writer.WriteNumber(VersionKey, predicted.Behind.Version);

            // Only predictions made behind a run that left steps out say which: those of a run of every step are the text
            // they always were.
            if (predicted.Behind.Skipped.Count > 0)
            {
                writer.WriteStartArray(SkippedKey);

                foreach (var prefix in predicted.Behind.Skipped)
                {
                    writer.WriteStringValue(prefix);
                }

                writer.WriteEndArray();
            }

            writer.WriteEndObject();
            writer.WriteStartArray(AnswersKey);

            foreach (var answer in predicted.Answers)
            {
                writer.WriteStringValue(answer);
            }

            writer.WriteEndArray();
            writer.WriteStartArray(PartsKey);

            foreach (var part in predicted.Parts)
            {
                Write(writer, part);
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    /// <summary>Reads what a model predicted back from its text.</summary>
    /// <param name="json">The text.</param>
    /// <returns>The fit they were made behind, the answers, and each part.</returns>
    /// <exception cref="ArgumentException">
    /// The text is not what <see cref="Write(WrittenPredictions)"/> writes, said with where it is not; or a newer DeepSharp wrote it.
    /// </exception>
    public static WrittenPredictions Read(string json)
    {
        JsonDocument document;

        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException)
        {
            throw NotTheText("it is not JSON");
        }

        using (document)
        {
            var root = document.RootElement;

            if (root.ValueKind != JsonValueKind.Object)
            {
                throw NotTheText("it is not one JSON object");
            }

            var version = root.TryGetProperty(VersionKey, out var named) && named.ValueKind == JsonValueKind.Number && named.TryGetInt32(out var number)
                ? number
                : 0;

            if (version > Version)
            {
                throw new ArgumentException(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"These predictions were written as version {version} of their text, by a newer DeepSharp, and this one reads up to version {Version}: measure them with that DeepSharp, or have them written again by this one."),
                    "predictions");
            }

            if (version < 1)
            {
                throw NotTheText("it names no version of it");
            }

            var answers = Names(List(root, AnswersKey, "it"), $"'{AnswersKey}'");
            WrittenPart[] parts = [.. List(root, PartsKey, "it").EnumerateArray().Select((part, at) => PartOf(part, at + 1))];

            return new WrittenPredictions(FitOf(root), answers, parts);
        }
    }

    // One part: which, the key of each row, what was predicted for each, and what the model said of each when it said it.
    private static void Write(Utf8JsonWriter writer, WrittenPart part)
    {
        writer.WriteStartObject();
        writer.WriteString(PartKey, part.Part.Word());
        writer.WriteStartArray(KeysKey);

        foreach (var key in part.Keys)
        {
            writer.WriteStringValue(key.ToString());
        }

        writer.WriteEndArray();
        writer.WriteStartArray(PredictionsKey);

        foreach (var row in part.Predictions)
        {
            writer.WriteStartArray();

            foreach (var value in row)
            {
                writer.WriteNumberValue(value);
            }

            writer.WriteEndArray();
        }

        writer.WriteEndArray();

        if (part.Unfamiliar is { } unfamiliar)
        {
            writer.WriteStartArray(UnfamiliarKey);

            foreach (var names in unfamiliar)
            {
                writer.WriteStartArray();

                foreach (var name in names)
                {
                    writer.WriteStringValue(name);
                }

                writer.WriteEndArray();
            }

            writer.WriteEndArray();
        }

        writer.WriteEndObject();
    }

    private static WrittenPart PartOf(JsonElement written, int number)
    {
        var where = string.Create(CultureInfo.InvariantCulture, $"part {number}");

        if (written.ValueKind != JsonValueKind.Object)
        {
            throw NotTheText($"{where} is not an object");
        }

        var part = written.TryGetProperty(PartKey, out var word) && word.ValueKind == JsonValueKind.String
            ? PartNamed(word.GetString()!, where)
            : throw NotTheText($"{where} names no part");

        return new WrittenPart(
            part,
            [.. Names(List(written, KeysKey, where), $"{where}'s '{KeysKey}'").Select(key => Key(key, where))],
            [.. List(written, PredictionsKey, where).EnumerateArray().Select(row => Numbers(row, where))],
            written.TryGetProperty(UnfamiliarKey, out var said) && said.ValueKind != JsonValueKind.Null ? Unfamiliar(said, where) : null);
    }

    // The fit the predictions were made behind: its digest, and the version of the pipeline file its pipeline names.
    private static WrittenFit FitOf(JsonElement root)
    {
        if (!root.TryGetProperty(PipelineKey, out var pipeline) || pipeline.ValueKind != JsonValueKind.Object)
        {
            throw NotTheText($"it names no pipeline they were made behind, as an object under '{PipelineKey}'");
        }

        var fit = pipeline.TryGetProperty(FitKey, out var digest) && digest.ValueKind == JsonValueKind.String
            ? digest.GetString()!
            : throw NotTheText($"its '{PipelineKey}' names no fit, as the text '{FitKey}'");

        var version = pipeline.TryGetProperty(VersionKey, out var named) && named.ValueKind == JsonValueKind.Number && named.TryGetInt32(out var whole)
            ? whole
            : throw NotTheText($"its '{PipelineKey}' names no version of the pipeline file, as the whole number '{VersionKey}'");

        IReadOnlyList<string> skipped = !pipeline.TryGetProperty(SkippedKey, out var left)
            ? []
            : left.ValueKind == JsonValueKind.Array && left.EnumerateArray().All(each => each.ValueKind == JsonValueKind.String)
                ? [.. left.EnumerateArray().Select(each => each.GetString()!)]
                : throw NotTheText($"its '{PipelineKey}' names what its run left out, under '{SkippedKey}', as something other than a list of the keys of steps");

        return new WrittenFit(fit, version, skipped);
    }

    // A part as a pipeline's file writes it, in whatever case it was typed.
    private static Part PartNamed(string word, string where) =>
        Vocabulary<Part>.Words.Contains(word, StringComparer.OrdinalIgnoreCase)
            ? Vocabulary<Part>.Read(word, PartKey)
            : throw NotTheText($"{where} names '{word}', which is none of the parts: {string.Join(", ", Vocabulary<Part>.Words)}");

    // The key of a row as RowKey writes it: sixty-four hexadecimal digits.
    private static RowKey Key(string written, string where) =>
        written.Length == 64 && written.All(char.IsAsciiHexDigit)
            ? RowKey.FromDigest(Convert.FromHexString(written))
            : throw NotTheText($"{where}'s key '{written}' is no row's key, which is sixty-four hexadecimal digits");

    // What was predicted for one row: a list of numbers. One too large for a double reads as an infinity, which measuring
    // refuses as it refuses every prediction that is not a finite number, naming its row.
    private static double[] Numbers(JsonElement row, string where)
    {
        var said = $"a row of {where}'s '{PredictionsKey}' is not a list of numbers";

        return row.ValueKind == JsonValueKind.Array
            ? [.. row.EnumerateArray().Select(value => value.ValueKind == JsonValueKind.Number ? value.GetDouble() : throw NotTheText(said))]
            : throw NotTheText(said);
    }

    // For each row, the names of the features the model said it learned nothing about.
    private static IReadOnlyList<string>[] Unfamiliar(JsonElement said, string where)
    {
        var rows = $"a row of {where}'s '{UnfamiliarKey}'";

        return said.ValueKind == JsonValueKind.Array
            ? [.. said.EnumerateArray().Select(row => row.ValueKind == JsonValueKind.Array ? (IReadOnlyList<string>)Names(row, rows) : throw NotTheText($"{rows} is not a list of names"))]
            : throw NotTheText($"{where}'s '{UnfamiliarKey}' is not a list of names for each row");
    }

    // A list a JSON object holds under a key.
    private static JsonElement List(JsonElement holder, string key, string holderIs) =>
        holder.TryGetProperty(key, out var list) && list.ValueKind == JsonValueKind.Array
            ? list
            : throw NotTheText($"{holderIs} has no '{key}' list");

    // A list of names, each one text.
    private static string[] Names(JsonElement list, string where) =>
        [.. list.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String ? item.GetString()! : throw NotTheText($"{where} holds {item.GetRawText()}, which is not text"))];

    private static ArgumentException NotTheText(string what) =>
        new($"These predictions are not the text Measures.PredictionsToJson writes: {what}.", "predictions");
}
