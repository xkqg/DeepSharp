// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// A text read as JSON a piece of its UTF-8 at a time: each piece handed to a reader behind what the reader left unread of
/// the one before, so no more of the text is ever held as bytes than a piece and the token it ends in.
/// </summary>
/// <remarks>
/// The file around a pipeline may be a network of millions of numbers, and holding all of its UTF-8 at once to read past it
/// would cost more than the pipeline does. A token longer than a piece is the one thing a piece cannot hold, so room is
/// made for it rather than the walk stopping short of it.
/// </remarks>
/// <param name="text">The text.</param>
internal sealed class TextPieces(string text)
{
    // How many bytes of the text are made at a time: the walk holds no more of it than this and the token it has reached.
    private const int Piece = 1 << 16;

    // The most bytes of UTF-8 one character of the text is written in.
    private const int WidestLetter = 4;

    private readonly Encoder _encoder = Encoding.UTF8.GetEncoder();
    private byte[] _buffer = new byte[Piece];
    private JsonReaderState _state;
    private int _made;
    private int _held;

    /// <summary>How many bytes of the text come before the piece a reader was last handed.</summary>
    public long Passed { get; private set; }

    /// <summary>Whether the piece a reader was last handed ends the text.</summary>
    public bool Last { get; private set; }

    /// <summary>The next piece, behind what the reader left unread of the one before: the buffer grows for a token longer than it.</summary>
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

    /// <summary>The bytes of the piece a reader was last handed, from one place in it to another.</summary>
    /// <param name="from">Where in the piece to take from.</param>
    /// <param name="to">Where in the piece to take to.</param>
    public ReadOnlySpan<byte> Bytes(int from, int to) => _buffer.AsSpan(from, to - from);

    /// <summary>What a reader took of its piece: what it left is kept, in front of the next.</summary>
    /// <param name="reader">The reader the piece was handed to.</param>
    public void Taken(ref Utf8JsonReader reader)
    {
        var taken = (int)reader.BytesConsumed;

        _state = reader.CurrentState;
        _buffer.AsSpan(taken, _held - taken).CopyTo(_buffer);
        _held -= taken;
        Passed += taken;
    }
}
