// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The frame every trained model's file is written in: one indented JSON object that opens with its version, holds the parts
/// only the kind of model knows, then the pipeline the model was trained behind exactly as its own file wrote it, and — when the
/// model has one — the interval; as text with a line feed between lines on every system.
/// </summary>
/// <param name="Version">The version of the file, written first.</param>
/// <param name="Pipeline">The pipeline the model was trained behind, written as it stands.</param>
/// <param name="Interval">The width the model's answers are worth; nothing when it has none, and then nothing is written for it.</param>
/// <remarks>
/// A network's file, an ensemble's, a boosted pair's and a tree's all say the same four things in the same order around what is
/// theirs alone, so a kind names its own parts and leaves the rest to this. What a kind writes is written between the version and
/// the pipeline, or, for the one part that stands after the pipeline — a checkpoint's training — between the pipeline and the
/// interval.
/// </remarks>
internal readonly record struct ModelFileFrame(int Version, PipelineText Pipeline, ConformalInterval? Interval)
{
    /// <summary>The key the version stands under.</summary>
    internal const string VersionKey = "version";

    /// <summary>The key the pipeline stands under.</summary>
    internal const string PipelineKey = "pipeline";

    /// <summary>The key the interval stands under.</summary>
    internal const string IntervalKey = "interval";

    /// <summary>What is asked of the length of the text, in bytes, once it is written and before it is made; nothing is asked unless it is set.</summary>
    public Action<long>? LengthChecked { get; init; }

    /// <summary>The file, with the parts of the kind of model it holds.</summary>
    /// <param name="parts">Writes the parts of the kind, each as a property of the object, between the version and the pipeline.</param>
    /// <param name="afterPipeline">Writes the part that stands after the pipeline and before the interval; nothing when the kind has none.</param>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    public string Written(Action<Utf8JsonWriter> parts, Action<Utf8JsonWriter>? afterPipeline = null)
    {
        ArgumentNullException.ThrowIfNull(parts);

        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber(VersionKey, Version);
            parts(writer);
            writer.WritePropertyName(PipelineKey);
            writer.WriteRawValue(Pipeline.Text);
            afterPipeline?.Invoke(writer);

            if (Interval is { } interval)
            {
                writer.WritePropertyName(IntervalKey);
                interval.WriteTo(writer);
            }

            writer.WriteEndObject();
        }

        LengthChecked?.Invoke(stream.Length);

        return stream.TextWithLineFeeds();
    }
}
