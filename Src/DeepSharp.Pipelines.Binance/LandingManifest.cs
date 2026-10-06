// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>One page of the venue's answer, as it came.</summary>
/// <param name="From">The start moment the page was asked for.</param>
/// <param name="Rows">How many candles it held.</param>
/// <param name="Sha256">The SHA-256 of the page's body, as lower-case hexadecimal.</param>
/// <param name="Date">The venue's <c>Date</c> header, as it was written; nothing when the page came without one.</param>
/// <param name="Uuid">The venue's <c>x-mbx-uuid</c> header; nothing when the page came without one.</param>
internal readonly record struct PageReceipt(DateTime From, int Rows, string Sha256, string? Date, string? Uuid);

/// <summary>
/// What a landing says of itself, beside its file: what was asked, when and of whom, under whose terms, what came back
/// page by page, and the fingerprint of the bytes that landed.
/// </summary>
/// <remarks>
/// <para>
/// A document of its own, written and read as the library's others are: a version first, so a newer one is refused whole
/// rather than half understood, and every key a reader needs named when it is missing. It says nothing of being complete:
/// a flag written next to the work is a promise nobody checked, and the file's own move is what completes a landing. The
/// fields that change from one fetch to the next — when, the venue's clock, the receipts — live here and never in the file
/// the pipeline reads, so the file's fingerprint names the candles alone.
/// </para>
/// <para>
/// The terms are the pointer the documentation of the venue's public data gives. Whether they reach a particular endpoint is
/// for whoever lands the data to read; the manifest records where, so the landing says what it was fetched under.
/// </para>
/// </remarks>
/// <param name="Host">The venue's address the pages were fetched from.</param>
/// <param name="Symbol">The market, as the venue names it.</param>
/// <param name="Interval">The candle length, as the venue spells it.</param>
/// <param name="From">The first moment of the window asked for.</param>
/// <param name="To">The moment the window ends before.</param>
/// <param name="AskedAt">When the landing began, by the clock it was handed.</param>
/// <param name="ServerTime">What the venue's own clock said, asked before the first page.</param>
/// <param name="SettleAfterCloseSeconds">How long after a candle closes it was left to settle before it was taken.</param>
/// <param name="PagesEstimated">How many pages the window was expected to take.</param>
/// <param name="Pages">Each page as it came.</param>
/// <param name="Rows">How many candles the file holds.</param>
/// <param name="Gaps">How many candles of the window the venue did not answer with: counted, never filled.</param>
/// <param name="First">When the first candle opens.</param>
/// <param name="Last">When the last candle opens.</param>
/// <param name="File">The file's name, in the folder the manifest stands in.</param>
/// <param name="Fingerprint">The SHA-256 of the file's bytes, as lower-case hexadecimal.</param>
internal sealed record LandingManifest(
    string Host,
    string Symbol,
    string Interval,
    DateTime From,
    DateTime To,
    DateTime AskedAt,
    DateTime ServerTime,
    int SettleAfterCloseSeconds,
    int PagesEstimated,
    IReadOnlyList<PageReceipt> Pages,
    int Rows,
    int Gaps,
    DateTime First,
    DateTime Last,
    string File,
    string Fingerprint)
{
    /// <summary>The version of the text this library writes and reads up to.</summary>
    public const int Version = 1;

    /// <summary>Where the terms of Binance's public data are written.</summary>
    public const string Terms = "https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf";

    private const string Endpoint = "/api/v3/klines";
    private const string Seconds = "yyyy-MM-dd'T'HH:mm:ss'Z'";
    private const string Milliseconds = "yyyy-MM-dd'T'HH:mm:ss.fff'Z'";

    /// <summary>The manifest as its text: one JSON object, two spaces to a level, a line feed between lines and at the end.</summary>
    /// <returns>The text.</returns>
    public string ToJson()
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteNumber("version", Version);
            writer.WriteString("venue", "Binance");
            writer.WriteString("host", Host);
            writer.WriteString("endpoint", Endpoint);
            writer.WriteString("terms", Terms);
            writer.WriteStartObject("asked");
            writer.WriteString("symbol", Symbol);
            writer.WriteString("interval", Interval);
            writer.WriteString("from", Spelled(From, Seconds));
            writer.WriteString("to", Spelled(To, Seconds));
            writer.WriteEndObject();
            writer.WriteString("askedAt", Spelled(AskedAt, Milliseconds));
            writer.WriteString("serverTime", Spelled(ServerTime, Milliseconds));
            writer.WriteNumber("settleAfterCloseSeconds", SettleAfterCloseSeconds);
            writer.WriteNumber("pagesEstimated", PagesEstimated);
            writer.WriteStartArray("pages");

            foreach (var page in Pages)
            {
                writer.WriteStartObject();
                writer.WriteString("from", Spelled(page.From, Seconds));
                writer.WriteNumber("rows", page.Rows);
                writer.WriteString("sha256", page.Sha256);

                if (page.Date is { } date)
                {
                    writer.WriteString("date", date);
                }

                if (page.Uuid is { } uuid)
                {
                    writer.WriteString("uuid", uuid);
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteNumber("rows", Rows);
            writer.WriteNumber("gaps", Gaps);
            writer.WriteString("first", Spelled(First, Seconds));
            writer.WriteString("last", Spelled(Last, Seconds));
            writer.WriteString("file", File);
            writer.WriteString("fingerprint", Fingerprint);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan).ReplaceLineEndings("\n") + "\n";
    }

    /// <summary>Reads a manifest back from its text.</summary>
    /// <param name="json">The text.</param>
    /// <returns>What it says.</returns>
    /// <exception cref="ArgumentNullException">There is no text.</exception>
    /// <exception cref="FormatException">
    /// The text is not what <see cref="ToJson"/> writes, said with what it lacks; or a newer version of this package wrote it.
    /// </exception>
    public static LandingManifest FromJson(string json)
    {
        ArgumentNullException.ThrowIfNull(json);

        JsonDocument document;

        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException)
        {
            throw NotAManifest("it is not JSON");
        }

        using (document)
        {
            var root = new Scope(document.RootElement, string.Empty);

            if (root.Element.ValueKind != JsonValueKind.Object)
            {
                throw NotAManifest("it is not one JSON object");
            }

            var version = root.Element.TryGetProperty("version", out var named) && named.ValueKind == JsonValueKind.Number && named.TryGetInt32(out var number) ? number : 0;

            if (version > Version)
            {
                throw new FormatException(string.Create(
                    CultureInfo.InvariantCulture,
                    $"This landing's manifest is version {version}, written by a newer DeepSharp.Pipelines.Binance, and this one reads up to version {Version}: read it with that one, or land the window again with this one."));
            }

            if (version < 1)
            {
                throw NotAManifest("it names no version of it");
            }

            var asked = root.Part("asked", JsonValueKind.Object, "the object that says what was asked");
            var file = root.Text("file", "the name of the landing's file");
            var fingerprint = root.Text("fingerprint", "the fingerprint of the landing's bytes");

            if (file != Path.GetFileName(file) || file.Contains('\\', StringComparison.Ordinal))
            {
                throw NotAManifest($"'file' names '{file}', which is not a file in the folder the manifest stands in");
            }

            if (!IsFingerprint(fingerprint))
            {
                throw NotAManifest("'fingerprint' is not the SHA-256 of the bytes as sixty-four lower-case hexadecimal characters");
            }

            return new LandingManifest(
                root.Text("host", "the venue's address"),
                asked.Text("symbol", "the market asked for"),
                asked.Text("interval", "the candle length asked for"),
                asked.Moment("from", "the first moment asked for"),
                asked.Moment("to", "the moment the window ends before"),
                root.Moment("askedAt", "when the landing began"),
                root.Moment("serverTime", "the venue's own clock"),
                root.Count("settleAfterCloseSeconds", "how long a candle was left to settle"),
                root.Count("pagesEstimated", "how many pages were expected"),
                [.. root.Part("pages", JsonValueKind.Array, "the pages as they came").Element.EnumerateArray().Select(page => Receipt(new Scope(page, "pages[]."))) ],
                root.Count("rows", "how many candles the file holds"),
                root.Count("gaps", "how many candles of the window the venue did not answer with"),
                root.Moment("first", "when the first candle opens"),
                root.Moment("last", "when the last candle opens"),
                file,
                fingerprint);
        }
    }

    private static PageReceipt Receipt(Scope page)
    {
        if (page.Element.ValueKind != JsonValueKind.Object)
        {
            throw NotAManifest("a page of 'pages' is not an object");
        }

        var sha256 = page.Text("sha256", "the SHA-256 of the page's body");

        if (!IsFingerprint(sha256))
        {
            throw NotAManifest("a page's 'sha256' is not the SHA-256 of its body as sixty-four lower-case hexadecimal characters");
        }

        return new PageReceipt(
            page.Moment("from", "the start moment the page was asked for"),
            page.Count("rows", "how many candles the page held"),
            sha256,
            page.Optional("date"),
            page.Optional("uuid"));
    }

    private static string Spelled(DateTime moment, string format) => moment.ToString(format, CultureInfo.InvariantCulture);

    private static bool IsFingerprint(string text) => text.Length == 64 && text.All(letter => letter is (>= '0' and <= '9') or (>= 'a' and <= 'f'));

    private static FormatException NotAManifest(string why) => new($"This is not a landing's manifest: {why}.");

    // One object of the document, and where it stands in it, so a refusal names the key as the document spells it.
    private readonly record struct Scope(JsonElement Element, string Where)
    {
        public Scope Part(string key, JsonValueKind kind, string meaning) => new(Property(key, kind, meaning), $"{Where}{key}.");

        public string Text(string key, string meaning)
        {
            var found = Property(key, JsonValueKind.String, meaning).GetString();

            return string.IsNullOrEmpty(found) ? throw NotAManifest($"'{Where}{key}' is empty — {meaning}") : found;
        }

        public string? Optional(string key) => Element.TryGetProperty(key, out var found) && found.ValueKind == JsonValueKind.String ? found.GetString() : null;

        public int Count(string key, string meaning)
        {
            if (!Element.TryGetProperty(key, out var found) || found.ValueKind != JsonValueKind.Number || !found.TryGetInt32(out var count) || count < 0)
            {
                throw NotAManifest($"'{Where}{key}' is missing, or is not a whole number of nought or more — {meaning}");
            }

            return count;
        }

        public DateTime Moment(string key, string meaning)
        {
            var text = Text(key, meaning);

            return DateTime.TryParse(text, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var moment)
                ? DateTime.SpecifyKind(moment, DateTimeKind.Utc)
                : throw NotAManifest($"'{Where}{key}' is '{text}', which is not a moment — {meaning}");
        }

        private JsonElement Property(string key, JsonValueKind kind, string meaning)
        {
            if (!Element.TryGetProperty(key, out var found) || found.ValueKind != kind)
            {
                throw NotAManifest($"'{Where}{key}' is missing, or is not {(kind == JsonValueKind.Array ? "a list" : kind == JsonValueKind.Object ? "an object" : "text")} — {meaning}");
            }

            return found;
        }
    }
}
