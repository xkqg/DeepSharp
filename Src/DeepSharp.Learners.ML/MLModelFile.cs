// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.ML;

/// <summary>
/// A trained model and the pipeline it was trained behind, as one file.
/// </summary>
/// <remarks>
/// <para>
/// The same promise a network's file makes: the pipeline's text is carried exactly as it was written, the model is tied
/// to that text by its digest, and a model read beside another fit of those steps is refused rather than quietly
/// answering from numbers that were learned somewhere else.
/// </para>
/// <para>
/// What a network's file does not have to say, this one does. The model inside it is ML.NET's own archive, which only
/// ML.NET can open, so the file names the version of ML.NET that wrote it and the processor it was written on — without
/// them a reader that cannot open it has nothing to tell anybody except the innermost words of somebody else's
/// exception. The archive's bytes are never digested: ML.NET stamps it with the clock it was saved at, so two saves of
/// one model differ while the model does not.
/// </para>
/// </remarks>
public sealed record MLModelFile
{
    private const string VersionKey = "version";
    private const string TrainerKey = "trainer";
    private const string SeedKey = "seed";
    private const string LibraryKey = "mlnet";
    private const string ProcessorKey = "processor";
    private const string AnswerKey = "answer";
    private const string ClassesKey = "classes";
    private const string FeaturesKey = "features";
    private const string RowsKey = "rows";
    private const string ModelKey = "model";
    private const string PipelineKey = "pipeline";

    /// <summary>The version of the model file this library writes.</summary>
    public const int Version = 1;

    private MLModelFile(PipelineText carried, PartDeclaration trainer, byte[] model)
    {
        Carried = carried;
        Trainer = trainer;
        Model = model;
    }

    /// <summary>The pipeline this model was trained behind, exactly as its own file wrote it.</summary>
    public PipelineText Carried { get; }

    /// <summary>The trainer as the declaration named it.</summary>
    public PartDeclaration Trainer { get; }

    /// <summary>The number the trainer's random draws were worked out from.</summary>
    public int Seed { get; init; }

    /// <summary>The version of ML.NET that wrote the model inside this file.</summary>
    public string Library { get; init; } = string.Empty;

    /// <summary>The processor the model was written on.</summary>
    public string Processor { get; init; } = string.Empty;

    /// <summary>The column the model was asked to predict.</summary>
    public string Answer { get; init; } = string.Empty;

    /// <summary>Whether that answer was one of two classes rather than a number.</summary>
    public bool Classes { get; init; }

    /// <summary>The features the model was handed, in their order.</summary>
    public IReadOnlyList<string> Features { get; init; } = [];

    /// <summary>How many training rows it learned from.</summary>
    public int Rows { get; init; }

    /// <summary>ML.NET's own archive of the model, as it saved it.</summary>
    public byte[] Model { get; }

    /// <summary>This file as its text.</summary>
    /// <returns>The text, a line feed between lines.</returns>
    public string ToJson()
    {
        using var written = new MemoryStream();
        using (var writer = new Utf8JsonWriter(written, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, Version);
            writer.WriteString(LibraryKey, Library);
            writer.WriteString(ProcessorKey, Processor);
            writer.WritePropertyName(TrainerKey);
            WriteTrainer(writer);
            writer.WriteNumber(SeedKey, Seed);
            writer.WriteString(AnswerKey, Answer);
            writer.WriteBoolean(ClassesKey, Classes);
            writer.WriteNumber(RowsKey, Rows);
            writer.WritePropertyName(FeaturesKey);
            writer.WriteStartArray();

            foreach (var feature in Features)
            {
                writer.WriteStringValue(feature);
            }

            writer.WriteEndArray();
            writer.WriteBase64String(ModelKey, Model);
            writer.WritePropertyName(PipelineKey);
            writer.WriteRawValue(Carried.Text);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(written.ToArray()).ReplaceLineEndings("\n");
    }

    /// <summary>A model file as it was trained.</summary>
    /// <param name="carried">The pipeline it was trained behind.</param>
    /// <param name="trainer">The trainer as the declaration named it.</param>
    /// <param name="model">ML.NET's own archive of the model.</param>
    /// <returns>The file, for the caller to say the rest of.</returns>
    /// <exception cref="ArgumentNullException">Anything it is made of is missing.</exception>
    public static MLModelFile Of(PipelineText carried, PartDeclaration trainer, byte[] model)
    {
        ArgumentNullException.ThrowIfNull(carried);
        ArgumentNullException.ThrowIfNull(model);

        return new MLModelFile(carried, trainer, model);
    }

    /// <summary>A model file read back from its text.</summary>
    /// <param name="json">The text.</param>
    /// <returns>What the file holds.</returns>
    /// <exception cref="ArgumentException">There is no text.</exception>
    /// <exception cref="PipelineFileException">The file is not one, or a key is missing or is not what it should be, each fault at its line and column.</exception>
    public static MLModelFile FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        var surveyed = SurveyedText.Whole(json);
        var read = JsonDocument.Parse(json);
        var root = read.RootElement;

        if (Number(surveyed, root, VersionKey) is var version && version > Version)
        {
            surveyed.Fault(surveyed.Of(VersionKey), $"This model file names version {version}, and this library reads version {Version}: it was written by a newer DeepSharp.");

            throw surveyed.Refused();
        }

        var pipeline = root.TryGetProperty(PipelineKey, out var carried)
            ? PipelineText.Of(Encoding.UTF8.GetBytes(carried.GetRawText()))
            : throw Refused(surveyed, $"A model file carries the pipeline it was trained behind, under '{PipelineKey}'.");

        return new MLModelFile(pipeline, ReadTrainer(surveyed, root), Bytes(surveyed, root))
        {
            Seed = (int)Number(surveyed, root, SeedKey),
            Library = Text(surveyed, root, LibraryKey),
            Processor = Text(surveyed, root, ProcessorKey),
            Answer = Text(surveyed, root, AnswerKey),
            Classes = root.TryGetProperty(ClassesKey, out var classes) && classes.ValueKind is JsonValueKind.True,
            Rows = (int)Number(surveyed, root, RowsKey),
            Features = root.TryGetProperty(FeaturesKey, out var features) && features.ValueKind is JsonValueKind.Array
                ? [.. features.EnumerateArray().Select(each => each.GetString() ?? string.Empty)]
                : [],
        };
    }

    private void WriteTrainer(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("kind", Trainer.Kind);

        foreach (var setting in Trainer.Settings)
        {
            writer.WriteNumber(setting.Key, setting.Value.Number);
        }

        writer.WriteEndObject();
    }

    private static PartDeclaration ReadTrainer(SurveyedText surveyed, JsonElement root)
    {
        if (!root.TryGetProperty(TrainerKey, out var trainer) || trainer.ValueKind is not JsonValueKind.Object)
        {
            throw Refused(surveyed, $"A model file says which trainer wrote it, under '{TrainerKey}'.");
        }

        var kind = trainer.TryGetProperty("kind", out var named) ? named.GetString() ?? string.Empty : string.Empty;

        return new PartDeclaration(
            kind,
            [.. trainer.EnumerateObject()
                .Where(setting => setting.Name != "kind")
                .Select(setting => new PartSetting(setting.Name, PartValue.Of(setting.Value.GetDouble())))]);
    }

    private static byte[] Bytes(SurveyedText surveyed, JsonElement root)
    {
        if (!root.TryGetProperty(ModelKey, out var model) || model.ValueKind is not JsonValueKind.String)
        {
            throw Refused(surveyed, $"A model file carries the trained model itself, under '{ModelKey}'.");
        }

        return model.GetBytesFromBase64();
    }

    private static double Number(SurveyedText surveyed, JsonElement root, string key) =>
        root.TryGetProperty(key, out var value) && value.ValueKind is JsonValueKind.Number
            ? value.GetDouble()
            : throw Refused(surveyed, $"A model file says '{key}', and it is a number.");

    private static string Text(SurveyedText surveyed, JsonElement root, string key) =>
        root.TryGetProperty(key, out var value) && value.ValueKind is JsonValueKind.String
            ? value.GetString()!
            : throw Refused(surveyed, $"A model file says '{key}', and it is a word.");

    // The fault placed at the file's own root, as the one reader of every file here places them, handed back so the
    // caller throws it: nothing comes after a refusal, and nothing pretends to.
    private static Exception Refused(SurveyedText surveyed, string message)
    {
        surveyed.Fault(surveyed.Root, message);

        return surveyed.Refused();
    }
}
