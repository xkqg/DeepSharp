// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A pipeline's own file as one writer writes it — indented by two spaces, a line feed between lines, every string escaped
/// as System.Text.Json escapes it by default, the keys in the order they stand and each number as it was written — and what
/// that text says of the fit it holds: its digest, and whether it is the same fit as another pipeline's.
/// </summary>
/// <remarks>
/// That is the layout a pipeline writes its own file in, so the text of a pipeline DeepSharp wrote comes out byte for byte as
/// it was written — in a file from 0.4.0 as in one from now — and one another writer spaced, ended its lines or escaped
/// otherwise comes out the same. Keys in another order, or a number spelled another way, are another text: only the writer
/// that chose them could say them back.
/// <para>
/// Whether two pipelines are one fit is judged on this text, never on a pipeline written again by whichever DeepSharp reads
/// it: that names the reader's own version of the pipeline file, and the next version would make every pipeline written
/// before it another fit. So two texts are compared with the version each names left out — <see cref="FitDigest"/> — and
/// only between versions in which every step means what it means now (<see cref="IsComparable"/>). One rule, whoever asks:
/// a network's file is held to the pipeline it carries by it, a checkpoint to the pipeline it goes on behind, and
/// predictions handed over as text (<see cref="Measures.PredictionsToJson"/>) to the run that measures them.
/// </para>
/// </remarks>
public sealed class PipelineText
{
    /// <summary>
    /// The first version of the pipeline file two pipelines' texts are compared from, fit for fit: the one 0.4.0 wrote, which
    /// the first network's files carry.
    /// </summary>
    /// <remarks>
    /// Every step this library ships means in it what it means now, and a test holds every one to that: a step that comes to
    /// mean something else from a later version cannot ship before the comparison knows its old meaning.
    /// </remarks>
    public const int FirstComparable = 3;

    private const string VersionKey = "version";

    private readonly string _unversioned;

    private PipelineText(JsonElement pipeline)
    {
        Text = Written(pipeline, withVersion: true);
        _unversioned = Written(pipeline, withVersion: false);
        Version = pipeline.TryGetProperty(VersionKey, out var named)
                  && int.TryParse(named.GetRawText(), NumberStyles.None, CultureInfo.InvariantCulture, out var version)
            ? version
            : 0;
    }

    /// <summary>The text.</summary>
    public string Text { get; }

    /// <summary>The version of the pipeline file the text names; nought when it names none that is a whole number.</summary>
    public int Version { get; }

    /// <summary>
    /// The SHA-256 of the text, in lowercase hexadecimal: the digest a network records of the fit it was trained behind, which
    /// holds the network to the pipeline its own file carries.
    /// </summary>
    public string Digest => Hex(Text);

    /// <summary>
    /// The SHA-256, in lowercase hexadecimal, of the text with the version of the pipeline file it names left out: what two
    /// pipelines are compared by, fit for fit, wherever the texts themselves cannot travel.
    /// </summary>
    /// <remarks>
    /// The same fit has the same one written under any version from <see cref="FirstComparable"/> to the one this library
    /// writes, and another fit — other steps, or other numbers learned from other rows — has another.
    /// </remarks>
    public string FitDigest => Hex(_unversioned);

    /// <summary>A pipeline as this library writes it.</summary>
    /// <param name="prepared">The pipeline, run or read back.</param>
    /// <returns>Its text.</returns>
    public static PipelineText Of(PreparedData prepared)
    {
        ArgumentNullException.ThrowIfNull(prepared);

        return Of(Encoding.UTF8.GetBytes(prepared.ToJson()));
    }

    /// <summary>A pipeline as a file carries it.</summary>
    /// <param name="json">The JSON object the pipeline stands as, in UTF-8, as the file holds it.</param>
    /// <returns>Its text.</returns>
    /// <exception cref="JsonException">The bytes are not JSON.</exception>
    /// <exception cref="ArgumentException">They are JSON, and not one object, as a pipeline's file is.</exception>
    public static PipelineText Of(ReadOnlyMemory<byte> json)
    {
        using var document = JsonDocument.Parse(json);

        return document.RootElement.ValueKind == JsonValueKind.Object
            ? new PipelineText(document.RootElement)
            : throw new ArgumentException("A pipeline's text is one JSON object: its version, its declaration and what its fit learned.", nameof(json));
    }

    /// <summary>
    /// Whether a text that names this version of the pipeline file can be held to one this library writes, fit for fit:
    /// whether every step means in it what it means now.
    /// </summary>
    /// <param name="version">The version the text names.</param>
    /// <returns>
    /// <see langword="true"/> from <see cref="FirstComparable"/> to the version this library writes; a newer version may give
    /// a step's words another meaning, and one before the first names a file no network's was written with.
    /// </returns>
    /// <remarks>A pipeline compared is not read, so this is asked before any two are held to one fit.</remarks>
    public static bool IsComparable(int version) => version >= FirstComparable && version <= PipelineDeclaration.Version;

    /// <summary>Whether this is the same fit as another pipeline: the one writer writes them alike, but for the version of the pipeline file each names.</summary>
    /// <param name="other">The other pipeline.</param>
    /// <returns><see langword="true"/> when they declare the same steps and learned the same numbers: their <see cref="FitDigest"/> is one.</returns>
    /// <remarks>
    /// The version says which meaning a reader gives a step's words, and a pipeline compared here is not read: whoever asks
    /// holds both to versions in which every step still means what it says now (<see cref="IsComparable"/>).
    /// </remarks>
    public bool IsTheFitOf(PipelineText other)
    {
        ArgumentNullException.ThrowIfNull(other);

        return string.Equals(FitDigest, other.FitDigest, StringComparison.Ordinal);
    }

    private static string Hex(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();

    // The one writer: the pipeline's members as they stand — all of them, or all but its version — as a pipeline file is laid out.
    private static string Written(JsonElement pipeline, bool withVersion)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();

            foreach (var member in pipeline.EnumerateObject().Where(member => withVersion || member.Name != VersionKey))
            {
                member.WriteTo(writer);
            }

            writer.WriteEndObject();
        }

        // A writer ends its lines the way the machine does on some runtimes; the text is the same text everywhere.
        return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n");
    }
}
