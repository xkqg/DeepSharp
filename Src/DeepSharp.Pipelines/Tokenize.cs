// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Cuts a column of text into the ids an embedding takes: the words the training rows held, most often first, each a whole number, and
/// every row written as the same number of them.
/// </summary>
/// <remarks>
/// <para>
/// A word is a run of letters and digits, kept in lower case. The vocabulary is learned from the training rows alone, which is what
/// makes this a step that belongs after the split: the words with the most occurrences there are kept, the most frequent first and
/// those equally frequent in the order of their letters, up to the size said. Place nought is padding, and a row shorter than the
/// length is filled up with it; place one is any word the training rows never held, so an unfamiliar word has somewhere to go;
/// the words themselves stand from place two. A row longer than the length is cut at it, and an empty cell is nothing but padding.
/// </para>
/// <para>
/// The column keeps its name and holds an array of that many ids, one a row, as a column of pictures holds its pixels. The ids are
/// whole numbers handed over as they are, never scaled into a range: a network reads them with a <c>tokenEmbedding</c>, which turns
/// each id into the vector it learned.
/// </para>
/// </remarks>
public sealed record TokenizeStep : IFittedStep, IPipelineStep<TokenizeStep>, IDescribesColumns
{
    // The first place a word has: nought pads a row out, one stands for a word the training rows never held.
    private const int FirstWord = 2;

    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column of text to cut into ids.", "review", [ColumnKind.Text]);

    private static readonly WholeNumberParameter LengthKey = new(
        "length", "How many ids every row is written as: a longer text is cut at it, a shorter one filled up with the padding id, nought.", 32) { AtLeast = 1 };

    private static readonly WholeNumberParameter VocabularyKey = new(
        "vocabulary", "How many ids there are, the padding id (nought) and the id of a word never seen (one) among them: the words the training rows held most often take the rest.", 5000) { AtLeast = FirstWord + 1 };

    /// <summary>Declares that a column of text is cut into ids.</summary>
    /// <param name="column">The column of text.</param>
    /// <param name="length">How many ids every row is written as.</param>
    /// <param name="vocabulary">How many ids there are, the two kept for padding and for a word never seen among them: three at the least.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The length is below one, or the vocabulary below three.</exception>
    public TokenizeStep(string column, int length, int vocabulary)
    {
        Column = ColumnKey.Require(column);
        Length = LengthKey.Require(length);
        Vocabulary = VocabularyKey.Require(vocabulary);
    }

    /// <summary>The column of text.</summary>
    public string Column { get; }

    /// <summary>How many ids every row is written as.</summary>
    public int Length { get; }

    /// <summary>How many ids there are, padding and the unseen place among them.</summary>
    public int Vocabulary { get; }

    /// <inheritdoc />
    public static string Name => "tokenize";

    /// <inheritdoc />
    public static string Purpose => "Cuts a column of text into ids of the words the training rows held most often, every row the same number of them, for an embedding to take.";

    /// <inheritdoc />
    public static int Since => 12;

    /// <inheritdoc />
    public static StepParameters<TokenizeStep> Parameters { get; } = new StepParameters<TokenizeStep>()
        .With(ColumnKey, step => step.Column)
        .With(LengthKey, step => step.Length)
        .With(VocabularyKey, step => step.Vocabulary);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>The column stays where it stands, an array of ids a row; the ids are whole numbers and land in no range.</remarks>
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Column, ColumnKind.Media, Form.Signed);
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">No training row of the column holds a word.</exception>
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var column = table[Column];
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        for (var row = 0; row < table.RowCount; row++)
        {
            if (parts[row] != Part.Train || column.TextAt(row) is not { } text)
            {
                continue;
            }

            foreach (var word in WordsOf(text))
            {
                counts[word] = counts.GetValueOrDefault(word) + 1;
            }
        }

        if (counts.Count == 0)
        {
            throw new InvalidOperationException($"No training row of '{Column}' holds a word, so there is no vocabulary to learn.");
        }

        var learned = new FittedStepValues();

        learned.Learned(
            "tokens",
            [.. counts.OrderByDescending(each => each.Value).ThenBy(each => each.Key, StringComparer.Ordinal).Take(Vocabulary - FirstWord).Select(each => each.Key)]);

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        var places = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var word in fitted.List("tokens"))
        {
            places[word] = places.Count + FirstWord;
        }

        var column = table[Column];
        var cells = new MediaCell?[table.RowCount];

        for (var row = 0; row < cells.Length; row++)
        {
            var ids = new int[Length];
            var at = 0;

            foreach (var word in column.TextAt(row) is { } text ? WordsOf(text) : [])
            {
                if (at == Length)
                {
                    break;
                }

                ids[at++] = places.GetValueOrDefault(word, 1);
            }

            cells[row] = MediaCell.Parse(MediaCell.Ids([Length], ids));
        }

        table.Remove(Column);
        table.Put(new MediaColumn(Column, cells));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A key is missing or is of the wrong kind.</exception>
    public static TokenizeStep ReadFrom(JsonElement element) => new(ColumnKey.Read(element), LengthKey.Read(element), VocabularyKey.Read(element));

    // The words of a text, in lower case, in the order they are written: a run of letters and digits is one.
    private static IEnumerable<string> WordsOf(string text)
    {
        var word = new StringBuilder();

        foreach (var character in text)
        {
            if (char.IsLetterOrDigit(character))
            {
                word.Append(char.ToLowerInvariant(character));
            }
            else if (word.Length > 0)
            {
                yield return word.ToString();

                word.Clear();
            }
        }

        if (word.Length > 0)
        {
            yield return word.ToString();
        }
    }
}
