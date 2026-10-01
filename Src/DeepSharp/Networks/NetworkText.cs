// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// The text of a file a network is read from, surveyed once: where every key, every object and every list stands, so a
/// fault found while reading is placed at its line and column, and every key written twice.
/// </summary>
/// <remarks>
/// <para>
/// A place is known by its path from the top of the file — the keys and the places in lists on the way down — so the part
/// of a larger file a network stands in is placed in that file's own lines. A number or a word standing in a list is not
/// kept: a network's file holds millions of them, one for every value of every slot, and keeping where each stood cost
/// several times what the file itself does. Where one stands is found by walking its list when a fault is placed there,
/// as are the lines, which are counted only when a fault is placed at all.
/// </para>
/// <para>
/// The lists its reader says it reads apart — a tensor's values — are left out of the text that is parsed, as empty lists,
/// and their numbers are read from the text where they stand, when the reader asks for them. A parsed document keeps a row
/// for every value, and takes its rows from .NET's pool of arrays at the size of the whole text, rounded up: parsing a
/// checkpoint's numbers cost more than twice the text itself, where reading them where they stand costs the numbers alone.
/// </para>
/// </remarks>
internal sealed class NetworkText
{
    private const char Separator = '\u001f';

    private readonly byte[] _bytes;
    private readonly Dictionary<string, int> _places = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _lists = new(StringComparer.Ordinal);
    private readonly List<Range> _apart = [];
    private readonly List<Noted> _faults = [];
    private int[]? _lineStarts;

    /// <summary>Surveys a file's text.</summary>
    /// <param name="json">The text.</param>
    /// <param name="readsApart">Whether its reader reads the numbers of the list at a path apart from the rest of the text.</param>
    /// <exception cref="NetworkFileException">The text is not JSON, or a key is written twice.</exception>
    public NetworkText(string json, Func<string[], bool> readsApart)
    {
        ArgumentNullException.ThrowIfNull(json);

        _bytes = Encoding.UTF8.GetBytes(json);
        Survey(readsApart);
        ThrowIfFaulty();
    }

    /// <summary>The text, parsed: every list read apart standing in it as an empty list.</summary>
    public JsonDocument Parsed() => JsonDocument.Parse(_apart.Count == 0 ? _bytes : Kept());

    /// <summary>How many elements the list at a path holds, counted where they stand.</summary>
    /// <param name="list">The path of a list the text holds.</param>
    public int CountOf(IReadOnlyList<string> list)
    {
        var reader = ListAt(list);
        var count = 0;

        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
        {
            reader.Skip();
            count++;
        }

        return count;
    }

    /// <summary>
    /// The elements of the list at a path, each read where it stands as the finite number a float holds; nothing when any is
    /// not one, with a fault noted at each that is not.
    /// </summary>
    /// <param name="list">The path of a list the text holds.</param>
    /// <param name="count">How many elements it holds.</param>
    /// <param name="message">What is said of an element that is not such a number.</param>
    public float[]? FloatsOf(IReadOnlyList<string> list, int count, string message)
    {
        var start = _lists[Key(list, list.Count)];
        var reader = ListAt(list);
        var numbers = new float[count];
        var readable = true;

        for (var place = 0; place < count; place++)
        {
            reader.Read();

            if (reader.TokenType == JsonTokenType.Number && reader.TryGetSingle(out var number) && float.IsFinite(number))
            {
                numbers[place] = number;

                continue;
            }

            _faults.Add(new Noted(start + (int)reader.TokenStartIndex, message));
            reader.Skip();
            readable = false;
        }

        return readable ? numbers : null;
    }

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

        Fault(path, (parameter < 0 ? message : message[..parameter]).Quoted());
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

    // Where each line of the text starts, counted the first time a fault is placed.
    private int[] LineStarts => _lineStarts ??= LinesOf(_bytes);

    private static string Key(IReadOnlyList<string> path, int length) => string.Join(Separator, path.Take(length));

    private int PlaceOf(IReadOnlyList<string> path)
    {
        for (var length = path.Count; length > 0; length--)
        {
            if (_places.TryGetValue(Key(path, length), out var offset))
            {
                return offset;
            }

            // A number or a word in a list: a place under a list is always the index of an element its reader found there.
            if (length > 1 && _lists.TryGetValue(Key(path, length - 1), out var list))
            {
                return ElementAt(list, int.Parse(path[length - 1], CultureInfo.InvariantCulture));
            }
        }

        return 0;
    }

    // A reader of the text standing at the start of the list at a path, its opening read.
    private Utf8JsonReader ListAt(IReadOnlyList<string> list)
    {
        var reader = new Utf8JsonReader(_bytes.AsSpan(_lists[Key(list, list.Count)]));

        reader.Read();

        return reader;
    }

    // The text a document is parsed from: every list read apart standing in it as an empty list.
    private byte[] Kept()
    {
        var kept = new byte[_bytes.Length - _apart.Sum(range => range.End.Value - range.Start.Value)];
        var from = 0;
        var to = 0;

        foreach (var range in _apart)
        {
            _bytes.AsSpan(from, range.Start.Value - from).CopyTo(kept.AsSpan(to));
            to += range.Start.Value - from;
            from = range.End.Value;
        }

        _bytes.AsSpan(from).CopyTo(kept.AsSpan(to));

        return kept;
    }

    // Where the element at an index of the list starting at a place stands, the list walked over every element before it.
    private int ElementAt(int list, int index)
    {
        var reader = new Utf8JsonReader(_bytes.AsSpan(list));

        reader.Read();
        reader.Read();

        for (var place = 0; place < index; place++)
        {
            reader.Skip();
            reader.Read();
        }

        return list + (int)reader.TokenStartIndex;
    }

    private void Survey(Func<string[], bool> readsApart)
    {
        var reader = new Utf8JsonReader(_bytes);
        var frames = new Stack<Frame>();
        var property = string.Empty;

        // The list being read apart, while the survey is inside it: the depth it stands at and the byte after its opening.
        var apartDepth = -1;
        var apartFrom = 0;

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
                            _faults.Add(new Noted(at, $"'{property.Quoted()}' is written twice here, and only one of the two would be read."));
                        }

                        _places.TryAdd(string.Join(Separator, named), at);
                        break;

                    case JsonTokenType.EndObject or JsonTokenType.EndArray:
                        frames.Pop();

                        if (reader.CurrentDepth == apartDepth)
                        {
                            _apart.Add(new Range(apartFrom, at));
                            apartDepth = -1;
                        }

                        break;

                    case JsonTokenType.StartObject or JsonTokenType.StartArray:
                        var path = PathOfHolder(frames, property, at);
                        var isList = reader.TokenType == JsonTokenType.StartArray;

                        if (isList)
                        {
                            _lists.TryAdd(string.Join(Separator, path), at);

                            if (apartDepth < 0 && readsApart(path))
                            {
                                apartDepth = reader.CurrentDepth;
                                apartFrom = at + 1;
                            }
                        }

                        frames.Push(new Frame(path, isList));
                        break;

                    default:
                        // A number or a word: in a list it is counted, so the elements after it keep their places, and
                        // never kept.
                        if (frames.Count > 0 && frames.Peek().IsList)
                        {
                            frames.Peek().Next();
                        }

                        break;
                }
            }
        }
        catch (JsonException fault)
        {
            var line = (int)Math.Min(fault.LineNumber.GetValueOrDefault(), LineStarts.Length - 1);
            var offset = (int)Math.Min(LineStarts[line] + fault.BytePositionInLine.GetValueOrDefault(), _bytes.Length);

            _faults.Add(new Noted(offset, $"The text stops being JSON here: {fault.Message.Split(" LineNumber:")[0].Quoted()}"));
        }
    }

    // The path of an object or a list starting here: an element of the list it stands in, whose place is kept; the value
    // of the key just read, placed at that key; or the top.
    private string[] PathOfHolder(Stack<Frame> frames, string property, int at)
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

        string[] element = [.. frame.Path, frame.Next().ToString(CultureInfo.InvariantCulture)];
        _places.TryAdd(string.Join(Separator, element), at);

        return element;
    }

    /// <summary>A fault at its line and column, both from one; the column in characters rather than bytes.</summary>
    private NetworkFileFault Placed(Noted fault)
    {
        var starts = LineStarts;
        var line = Array.BinarySearch(starts, fault.Offset);

        if (line < 0)
        {
            line = ~line - 1;
        }

        return new NetworkFileFault(line + 1, Encoding.UTF8.GetCharCount(_bytes, starts[line], fault.Offset - starts[line]) + 1, fault.Message);
    }

    private static int[] LinesOf(byte[] text)
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
