// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A text as one small vector it learns: each id of a row looked up in a table of vectors, and the vectors of the row's words averaged.
/// </summary>
/// <remarks>
/// <para>
/// A row is the ids a tokenizer wrote — whole numbers from nought to one below the vocabulary, as many as the length. Id nought is
/// padding: it is no word, it is left out of the average and learns nothing, so a short text is not pulled towards a vector that
/// stands for nothing, and a row of nothing but padding comes out as the vector of nothing, noughts. Every other id, an unfamiliar word
/// among them, has a vector of its own. The average is taken over the words the row holds, so a long text and a short one are on the same
/// scale; the order of the words is not kept, which is a bag of words with the learned vectors in place of the counts.
/// </para>
/// <para>
/// The lookup is a product the seam already has: a matrix with a row for each text and a column for each id, holding the share of the
/// text each id is, times the weight. The weight is PyTorch's <c>weight</c> of an embedding bag in its mean mode — a row for each id, by
/// the width of a vector, starting from a normal curve of spread one as PyTorch starts it — so what a gradient sends back reaches
/// only the vectors of the ids the rows hold, the same numbers a lookup would send.
/// </para>
/// </remarks>
public sealed class TokenEmbedding : Layer, ISaved<TokenEmbedding>
{
    private const string LengthKey = "length";
    private const string VocabularyKey = "vocabulary";
    private const string WidthKey = "width";

    /// <summary>An embedding of the ids of a text, its vectors starting from the given draws.</summary>
    /// <param name="length">How many ids a row holds: one at least.</param>
    /// <param name="vocabulary">How many ids there are, padding and the id of a word never seen among them: three at least.</param>
    /// <param name="width">How many numbers an id's vector holds: one at least.</param>
    /// <param name="draws">The draws the vectors start from.</param>
    /// <exception cref="ArgumentOutOfRangeException">A row holds no id, the vocabulary is below three, or a vector holds no number.</exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public TokenEmbedding(int length, int vocabulary, int width, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(length, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabulary, 3);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);

        Length = length;
        Weight = AddParameter("weight", Tensor.From(new Shape(vocabulary, width), [.. Enumerable.Range(0, vocabulary * width).Select(_ => (float)draws.NextNormal())]));
    }

    /// <summary>How many ids a row holds.</summary>
    public int Length { get; }

    /// <summary>How many ids there are: padding, the id of a word never seen, and the words.</summary>
    public int Vocabulary => Weight.Value.Shape[0];

    /// <summary>How many numbers an id's vector holds.</summary>
    public int Width => Weight.Value.Shape[1];

    /// <summary>The vectors: a row for each id, by the width of a vector.</summary>
    public Parameter Weight { get; }

    /// <inheritdoc />
    public static string Name => "tokenEmbedding";

    /// <inheritdoc />
    public static TokenEmbedding Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, LengthKey), rebuilding.Whole(settings, VocabularyKey), rebuilding.Whole(settings, WidthKey), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(LengthKey, Length);
        writer.WriteNumber(VocabularyKey, Vocabulary);
        writer.WriteNumber(WidthKey, Width);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// The input is not a batch of rows of as many ids as the layer was built for, or a value of it is no id of the vocabulary: not a
    /// whole number, below nought, or not below the vocabulary's size.
    /// </exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2 || input.Shape[1] != Length)
        {
            throw new ArgumentException($"A token embedding of rows of {Length} ids takes a batch of such rows, and was handed a {input.Shape} one.", nameof(input));
        }

        var rows = input.Shape[0];
        var vocabulary = Vocabulary;
        var ids = input.Values;
        var shares = new float[rows * vocabulary];

        for (var row = 0; row < rows; row++)
        {
            var words = 0;

            for (var at = 0; at < Length; at++)
            {
                var id = IdAt(ids[(row * Length) + at], row, at);

                if (id != 0)
                {
                    shares[(row * vocabulary) + id]++;
                    words++;
                }
            }

            for (var id = 1; id < vocabulary && words > 0; id++)
            {
                shares[(row * vocabulary) + id] /= words;
            }
        }

        return pass.Backend.MatMul(Tensor.From(new Shape(rows, vocabulary), shares), Weight.Value);
    }

    private int IdAt(float value, int row, int at)
    {
        var id = (int)value;

        return id == value && id >= 0 && id < Vocabulary
            ? id
            : throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"Row {row + 1}, place {at + 1} holds {value}, which is no id of a vocabulary of {Vocabulary}: an id is a whole number from nought to {Vocabulary - 1}."),
                nameof(value));
    }
}
