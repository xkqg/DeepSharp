// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// The text of a file a network is read from, surveyed once: where every key and every element of a list stands, so a
/// fault found while reading is placed at its line and column, and every key written twice.
/// </summary>
/// <remarks>
/// A place is known by its path from the top of the file — the keys and the places in lists on the way down — so the part
/// of a larger file a network stands in is placed in that file's own lines.
/// </remarks>
internal sealed class NetworkText
{
    private const char Separator = '\u001f';

    private readonly byte[] _bytes;
    private readonly int[] _lineStarts;
    private readonly Dictionary<string, int> _places = new(StringComparer.Ordinal);
    private readonly List<Noted> _faults = [];

    /// <summary>Surveys a file's text.</summary>
    /// <exception cref="NetworkFileException">The text is not JSON, or a key is written twice.</exception>
    public NetworkText(string json)
    {
        ArgumentNullException.ThrowIfNull(json);

        _bytes = Encoding.UTF8.GetBytes(json);
        _lineStarts = LineStarts(_bytes);
        Survey();
        ThrowIfFaulty();
    }

    /// <summary>The text, parsed.</summary>
    public JsonDocument Parsed() => JsonDocument.Parse(_bytes);

    /// <summary>Notes a fault at the place a path leads to, or the nearest place above it the survey found.</summary>
    public void Fault(IReadOnlyList<string> path, string message) => _faults.Add(new Noted(PlaceOf(path), message));

    /// <summary>
    /// Notes a refusal as a fault at a place, in the file's own words: its first line, without the name of a C# parameter
    /// the file never had.
    /// </summary>
    public void Fault(IReadOnlyList<string> path, Exception refusal)
    {
        var message = refusal.Message.Split(Environment.NewLine)[0];
        var parameter = message.IndexOf(" (Parameter '", StringComparison.Ordinal);

        Fault(path, parameter < 0 ? message : message[..parameter]);
    }

    /// <summary>Whether a fault has been noted.</summary>
    public bool Faulty => _faults.Count > 0;

    /// <summary>Refuses the file when a fault has been noted.</summary>
    /// <exception cref="NetworkFileException">A fault was noted: every one is named.</exception>
    public void ThrowIfFaulty()
    {
        if (Faulty)
        {
            throw Refused();
        }
    }

    /// <summary>The refusal of the file: every fault noted, in the order they stand, each at its line and column.</summary>
    public NetworkFileException Refused() => new([.. _faults.OrderBy(fault => fault.Offset).Select(Placed)]);

    private int PlaceOf(IReadOnlyList<string> path)
    {
        for (var length = path.Count; length > 0; length--)
        {
            if (_places.TryGetValue(string.Join(Separator, path.Take(length)), out var offset))
            {
                return offset;
            }
        }

        return 0;
    }

    private void Survey()
    {
        var reader = new Utf8JsonReader(_bytes);
        var frames = new Stack<Frame>();
        var property = string.Empty;

        try
        {
            while (reader.Read())
            {
                var at = (int)reader.TokenStartIndex;

                switch (reader.TokenType)
                {
                    case JsonTokenType.PropertyName:
                        var frame = frames.Peek();
                        property = reader.GetString()!;
                        string[] named = [.. frame.Path, property];

                        if (!frame.Names.Add(property))
                        {
                            _faults.Add(new Noted(at, $"'{property}' is written twice here, and only one of the two would be read."));
                        }

                        _places.TryAdd(string.Join(Separator, named), at);
                        break;

                    case JsonTokenType.EndObject or JsonTokenType.EndArray:
                        frames.Pop();
                        break;

                    default:
                        var path = PathOfValue(frames, property, at);

                        if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                        {
                            frames.Push(new Frame(path, reader.TokenType == JsonTokenType.StartArray));
                        }

                        break;
                }
            }
        }
        catch (JsonException fault)
        {
            var line = (int)Math.Min(fault.LineNumber.GetValueOrDefault(), _lineStarts.Length - 1);
            var offset = (int)Math.Min(_lineStarts[line] + fault.BytePositionInLine.GetValueOrDefault(), _bytes.Length);

            _faults.Add(new Noted(offset, $"The text stops being JSON here: {fault.Message.Split(" LineNumber:")[0]}"));
        }
    }

    // The path of a value starting here: an element of the list it stands in, the value of the key just read, or the top.
    private string[] PathOfValue(Stack<Frame> frames, string property, int at)
    {
        if (frames.Count == 0)
        {
            return [];
        }

        var frame = frames.Peek();

        if (!frame.IsList)
        {
            return [.. frame.Path, property];
        }

        string[] element = [.. frame.Path, frame.Next().ToString(System.Globalization.CultureInfo.InvariantCulture)];
        _places.TryAdd(string.Join(Separator, element), at);

        return element;
    }

    /// <summary>A fault at its line and column, both from one; the column in characters rather than bytes.</summary>
    private NetworkFileFault Placed(Noted fault)
    {
        var line = Array.BinarySearch(_lineStarts, fault.Offset);

        if (line < 0)
        {
            line = ~line - 1;
        }

        return new NetworkFileFault(line + 1, Encoding.UTF8.GetCharCount(_bytes, _lineStarts[line], fault.Offset - _lineStarts[line]) + 1, fault.Message);
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

    private readonly record struct Noted(int Offset, string Message);

    // An object or a list the survey is inside: its path, and the keys read in it or the place of its next element.
    private sealed class Frame(string[] path, bool isList)
    {
        private int _next;

        public string[] Path { get; } = path;

        public bool IsList { get; } = isList;

        public HashSet<string> Names { get; } = new(StringComparer.Ordinal);

        public int Next() => _next++;
    }
}
