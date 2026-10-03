// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The text of a file being read, surveyed once: where each of its parts stands, every key written twice, and every fault
/// at its line and column.
/// </summary>
/// <remarks>
/// <para>
/// The text is walked token by token before anything is read from it: that is where the places come from, and where a key
/// written twice is caught, because the document the values are then read from keeps neither the places nor the second of
/// two keys. Reading collects every fault before it refuses, so a person is told everything that is wrong at once.
/// </para>
/// <para>
/// A file can also hold what is being read inside it, as the value of a key at its top — a trained network beside the
/// pipeline it was trained behind. The whole file is read first, so text that stops being JSON anywhere in it is said where
/// it stops; the value is then surveyed as a file of its own, its places kept as places in the whole text. Of all of the
/// larger file only that value's own bytes are kept: the file around it may be a network of millions of numbers, and a copy
/// of every byte of it would cost more than the value to read past them.
/// </para>
/// <para>
/// The same duty as <c>DeepSharp.Networks.NetworkText</c> — kept here rather than referenced there, since a pipeline never
/// carries the deep-learning core as a dependency.
/// </para>
/// </remarks>
internal sealed class SurveyedText
{
    private static readonly long[] Nowhere = [];

    private readonly string _json;
    private readonly List<Noted> _faults = [];

    // The UTF-8 of the part of the text this is, and where that part starts in the whole text: all of the text, unless what
    // is being read stands inside a larger file as the value of one of its keys, when it is that value alone. Every place is
    // a place in the whole text, so a fault is at the line and column of the file a person opens.
    private readonly byte[] _text;
    private readonly long _start;

    // Where the parts of the text stand, as the survey that opens every read found them.
    private Places _places = new();

    private SurveyedText(string json, byte[] text, long start)
    {
        _json = json;
        _text = text;
        _start = start;
    }

    /// <summary>The UTF-8 of the part of the text this is: what a document is parsed from.</summary>
    public byte[] Bytes => _text;

    /// <summary>Where the value the text is stands.</summary>
    public long Root => _places.Root;

    /// <summary>A file's own text, all of it surveyed.</summary>
    /// <param name="json">The text.</param>
    /// <returns>The text, surveyed.</returns>
    /// <exception cref="PipelineFileException">The text is not JSON, or a key is written twice.</exception>
    public static SurveyedText Whole(string json) => new SurveyedText(json, Encoding.UTF8.GetBytes(json), 0).Surveyed();

    /// <summary>What a larger file holds under a key at its top, surveyed as a file of its own.</summary>
    /// <param name="json">The larger file.</param>
    /// <param name="property">The key the value stands under.</param>
    /// <returns>That value's text, surveyed, its places kept as places in the larger file.</returns>
    /// <exception cref="PipelineFileException">
    /// The larger file is not JSON, not one object, or holds the key not once; or the value itself is not JSON or holds a
    /// key twice. Every fault is at its line and column in the larger file.
    /// </exception>
    public static SurveyedText Within(string json, string property)
    {
        var (part, start) = new SurveyedText(json, [], 0).PartUnder(property);

        return new SurveyedText(json, part, start).Surveyed();
    }

    /// <summary>Where a key at the top of the text stands.</summary>
    /// <param name="key">The key.</param>
    public long Of(string key) => _places.Keys[key];

    /// <summary>Where each element of the list under a key at the top of the text stands.</summary>
    /// <param name="key">The key.</param>
    /// <returns>A place for each element, in the order they stand; none when the text holds no such list.</returns>
    public IReadOnlyList<long> ElementsUnder(string key) =>
        _places.Elements.TryGetValue(key, out var elements) ? elements : Nowhere;

    /// <summary>Notes a fault about the thing at a place.</summary>
    /// <param name="offset">The place in the whole text of the thing the fault is about.</param>
    /// <param name="message">What is wrong.</param>
    public void Fault(long offset, string message) => _faults.Add(new Noted(offset, message));

    /// <summary>Refuses the text when anything has been found wrong with it.</summary>
    /// <exception cref="PipelineFileException">Anything was found wrong; every fault is named.</exception>
    public void ThrowIfFaulty()
    {
        if (_faults.Count > 0)
        {
            throw Refused();
        }
    }

    /// <summary>The refusal the faults found so far are, each at its line and column.</summary>
    public PipelineFileException Refused() => new([.. Placed([.. _faults.OrderBy(fault => fault.Offset)])]);

    // The reader ends its message with where it stopped, counted from nought; the fault carries the place
    // counted from one, as an editor shows it, and saying it twice in two ways would only confuse.
    private static string WithoutItsPlace(string message) => message.Split(" LineNumber:")[0];

    private SurveyedText Surveyed()
    {
        Survey();
        ThrowIfFaulty();

        return this;
    }

    /// <summary>Walks the text token by token: where everything stands, kept as this text's places, and every key written twice.</summary>
    /// <remarks>
    /// Of the part of the text this is, each place counted from the start of the whole text. A part inside a larger file is
    /// only ever taken once the whole file has been read as JSON, so text that stops being JSON is met here in a text that
    /// is the whole of itself.
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
                if (reader.CurrentDepth == 2 && under is not null
                    && reader.TokenType is not (JsonTokenType.PropertyName or JsonTokenType.EndObject or JsonTokenType.EndArray))
                {
                    places.Under(under).Add(at);
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
                            Fault(at, WrittenTwice(name));
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
            Fault(OffsetOf(fault), StopsBeingJson(fault));

            throw Refused();
        }

        _places = places;
    }

    // A key written twice, and the one sentence that says so wherever it is found.
    private static string WrittenTwice(string name) => $"'{name.Quoted()}' is written twice here, and only one of the two would be read.";

    private static string StopsBeingJson(JsonException fault) =>
        $"The text stops being JSON here: {WithoutItsPlace(fault.Message).Quoted()}";

    /// <summary>The value a larger file holds under a key at its top, taken as it passes, and where it starts.</summary>
    private (byte[] Part, long Start) PartUnder(string property)
    {
        var pieces = new TextPieces(_json);
        var taken = new ArrayBufferWriter<byte>();
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
                            Fault(root, $"A file that holds a pipeline in place is one JSON object, with the pipeline under '{property.Quoted()}'.");

                            throw Refused();
                        }
                    }
                    else if (key >= 0)
                    {
                        // The value of the key the pipeline stands under: the first is the pipeline, taken as it passes.
                        if (start >= 0)
                        {
                            Fault(key, WrittenTwice(property));
                        }
                        else
                        {
                            start = at;
                            from = (int)reader.TokenStartIndex;
                            taking = reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray;

                            if (!taking)
                            {
                                taken.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                            }
                        }

                        key = -1;
                    }
                    else if (taking && reader.CurrentDepth == 1 && reader.TokenType is JsonTokenType.EndObject or JsonTokenType.EndArray)
                    {
                        taken.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                        taking = false;
                    }
                    else if (reader.TokenType == JsonTokenType.PropertyName && reader.CurrentDepth == 1 && reader.ValueTextEquals(property))
                    {
                        key = at;
                    }
                }

                // A value that goes on into the next piece: what of it this piece held.
                if (taking)
                {
                    taken.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                }

                pieces.Taken(ref reader);
            }
            while (!pieces.Last);
        }
        catch (JsonException fault)
        {
            // Nothing after this point can be read, so it is the last fault there is.
            Fault(OffsetOf(fault), StopsBeingJson(fault));

            throw Refused();
        }

        if (start < 0)
        {
            Fault(root, $"This file holds no '{property.Quoted()}', which is where the pipeline in it stands.");
        }

        ThrowIfFaulty();

        return (taken.WrittenSpan.ToArray(), start);
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

    /// <summary>
    /// Each fault at its line and column, both from one, the column in characters rather than bytes: the text walked once, a
    /// character at a time, each counted as the bytes of UTF-8 it is written in, up to each fault's offset in turn.
    /// </summary>
    /// <param name="faults">The faults, in the order they stand; each offset one within the text or at its end.</param>
    private IEnumerable<PipelineFileFault> Placed(Noted[] faults)
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

    /// <summary>A fault noted while reading, and the offset in the text of the thing it is about.</summary>
    private readonly record struct Noted(long Offset, string Message);

    /// <summary>Where the parts of a file stand in its text, as offsets of their first byte.</summary>
    private sealed class Places
    {
        /// <summary>The value the file is.</summary>
        public long Root { get; set; }

        /// <summary>Each key at the top of the file.</summary>
        public Dictionary<string, long> Keys { get; } = new(StringComparer.Ordinal);

        /// <summary>Each element of the list under a key at the top of the file, by that key.</summary>
        public Dictionary<string, List<long>> Elements { get; } = new(StringComparer.Ordinal);

        /// <summary>Where the elements under a key are kept.</summary>
        /// <param name="key">The key.</param>
        public List<long> Under(string key) => Elements.TryGetValue(key, out var elements) ? elements : Elements[key] = [];
    }
}
