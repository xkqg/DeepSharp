// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The top of the one file, surveyed before any part of it is read: that it is JSON, one object, of a version this library
/// reads, holding its parts and nothing else — where each part stands, so a fault about a part is placed at it, and the text
/// of the pipeline it carries, as the file carries it.
/// </summary>
/// <remarks>
/// Each part is read by what reads it — the network by the network document, the pipeline as its own file is — and each
/// places its own faults; this places the file's. The file is walked a piece of its UTF-8 at a time and only the pipeline's
/// own bytes are kept: the network's part beside it holds millions of numbers, and a copy of every byte of the file, with
/// where each of its lines starts, cost more than half as much again as reading the network itself.
/// </remarks>
internal sealed class FileText
{
    // How many bytes of the file are made at a time while its top is walked: the walk holds no more of it than this and the
    // token it has reached.
    private const int Piece = 1 << 16;

    private static readonly string[] Parts = [TrainedNetwork.VersionKey, TrainedNetwork.NetworkKey, TrainedNetwork.PipelineKey, TrainedNetwork.TrainingKey];

    private readonly string _json;
    private readonly Dictionary<string, long> _places = new(StringComparer.Ordinal);
    private byte[]? _pipeline;

    /// <summary>Surveys a file's top.</summary>
    /// <exception cref="NetworkFileException">The text is not JSON, not one object, of no version this library reads, or holds what is no part of it.</exception>
    public FileText(string json)
    {
        _json = json;

        var faults = Survey();

        if (faults.Count > 0)
        {
            throw new NetworkFileException(Placed(faults));
        }
    }

    /// <summary>A refusal of the file, placed at the part a key stands for.</summary>
    /// <remarks>Only ever asked of a part its reader has read, so the key stands in the file.</remarks>
    public NetworkFileException Refused(string key, string message) => new(Placed([new Noted(_places[key], message)]));

    /// <summary>The pipeline the file carries, its text taken from the file as it stands there.</summary>
    /// <exception cref="NetworkFileException">The file carries no pipeline: nothing under its key, or no object.</exception>
    public PipelineText Pipeline() =>
        _pipeline is { } pipeline
            ? PipelineText.Of(pipeline)
            : throw new NetworkFileException(Placed([new Noted(
                _places.GetValueOrDefault(TrainedNetwork.PipelineKey),
                $"This file carries no pipeline: a trained network's file carries the pipeline it was trained behind, as an object under '{TrainedNetwork.PipelineKey}'.")]));

    private List<Noted> Survey()
    {
        var faults = new List<Noted>();
        var pieces = new Pieces(_json);
        var taken = new ArrayBufferWriter<byte>();
        var opened = false;
        var ended = false;
        var within = false;
        var taking = false;
        string? key = null;
        var at = 0L;
        int? version = null;

        try
        {
            // Handed the last piece, a reader reads to the end of the file's object, or throws where the text stops being JSON.
            do
            {
                var reader = pieces.Next();
                var from = 0;

                while (!ended && reader.Read())
                {
                    if (!opened)
                    {
                        if (reader.TokenType != JsonTokenType.StartObject)
                        {
                            faults.Add(new Noted(0, "A trained network's file is one JSON object: its version, its network and the pipeline it was trained behind."));

                            return faults;
                        }

                        opened = true;
                    }
                    else if (within)
                    {
                        // Inside a part that is an object or a list, until it ends.
                        if (reader.CurrentDepth == 1 && reader.TokenType is JsonTokenType.EndObject or JsonTokenType.EndArray)
                        {
                            within = false;
                            Passed(ref reader, from);
                        }
                    }
                    else if (reader.TokenType == JsonTokenType.EndObject)
                    {
                        // Outside every part, only the file's own object ends: nothing after it is read.
                        ended = true;
                    }
                    else if (key is null)
                    {
                        key = reader.GetString()!;
                        at = pieces.Passed + reader.TokenStartIndex;
                        _places.TryAdd(key, at);
                    }
                    else
                    {
                        // The value of the key just read.
                        if (key == TrainedNetwork.VersionKey)
                        {
                            version = reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out var number) ? number : 0;
                        }

                        within = reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray;
                        taking = key == TrainedNetwork.PipelineKey && reader.TokenType == JsonTokenType.StartObject && _pipeline is null;
                        from = (int)reader.TokenStartIndex;

                        if (!within)
                        {
                            Passed(ref reader, from);
                        }
                    }
                }

                // A pipeline that goes on into the next piece: what of it this piece held.
                if (taking)
                {
                    taken.Write(pieces.Bytes(from, (int)reader.BytesConsumed));
                }

                pieces.Taken(ref reader);
            }
            while (!ended);
        }
        catch (JsonException fault)
        {
            faults.Add(new Noted(OffsetOf(fault), $"The text stops being JSON here: {fault.Message.Split(" LineNumber:")[0]}"));

            return faults;
        }

        if (version is not >= 1)
        {
            faults.Insert(0, new Noted(
                _places.GetValueOrDefault(TrainedNetwork.VersionKey),
                $"A trained network's file names the whole number of the version it was written against, from 1, under '{TrainedNetwork.VersionKey}'."));
        }
        else if (version > TrainedNetwork.Version)
        {
            faults.Insert(0, new Noted(
                _places[TrainedNetwork.VersionKey],
                string.Create(CultureInfo.InvariantCulture, $"This file was written against version {version}, by a newer DeepSharp than this one, which reads up to version {TrainedNetwork.Version}: read it with that DeepSharp.")));
        }

        return faults;

        // A part has been read past: the pipeline's bytes kept when they are the pipeline's, and a key that is no part named.
        void Passed(ref Utf8JsonReader reader, int start)
        {
            if (taking)
            {
                taken.Write(pieces.Bytes(start, (int)reader.BytesConsumed));
                _pipeline = taken.WrittenSpan.ToArray();
                taking = false;
            }

            if (!Parts.Contains(key))
            {
                faults.Add(new Noted(at, $"'{key}' is no part of a trained network's file: it holds its version, its network, the pipeline it was trained behind and, for a checkpoint, what the run needs to go on."));
            }

            key = null;
        }
    }

    /// <summary>Where a reader's fault is, from the line and the byte in that line it names, both from nought.</summary>
    /// <remarks>
    /// Counted in the file's UTF-8 by walking the text itself, up to its end: the reader was handed that UTF-8, so the line it
    /// names is one the text has.
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
    /// character at a time, each counted as the bytes of UTF-8 it is written in, up to each fault's offset in the order they
    /// stand — the faults handed back in the order they were noted.
    /// </summary>
    private NetworkFileFault[] Placed(IReadOnlyList<Noted> faults)
    {
        var placed = new NetworkFileFault[faults.Count];
        var bytes = 0L;
        var line = 1;
        var lineStart = 0;
        var at = 0;

        foreach (var each in Enumerable.Range(0, faults.Count).OrderBy(index => faults[index].Offset))
        {
            while (bytes < faults[each].Offset)
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

            placed[each] = new NetworkFileFault(line, at - lineStart + 1, faults[each].Message);
        }

        return placed;
    }

    private readonly record struct Noted(long Offset, string Message);

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

        // The next piece, behind what the reader left unread of the one before, and told whether it ends the text: the buffer
        // grows for a token longer than it.
        public Utf8JsonReader Next()
        {
            if (_buffer.Length - _held < WidestLetter)
            {
                Array.Resize(ref _buffer, _buffer.Length * 2);
            }

            _encoder.Convert(text.AsSpan(_made), _buffer.AsSpan(_held), flush: true, out var letters, out var bytes, out _);
            _made += letters;
            _held += bytes;

            return new Utf8JsonReader(_buffer.AsSpan(0, _held), isFinalBlock: _made == text.Length, _state);
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
}
