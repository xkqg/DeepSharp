// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import.Parts;

/// <summary>A line of a test file, split into its words, numbered from one as an editor counts lines.</summary>
/// <param name="Number">Its number.</param>
/// <param name="Words">Its words.</param>
internal readonly record struct Line(int Number, string[] Words)
{
    /// <summary>The number a line holds for one slot: its path, its shape as lengths joined by <c>x</c>, and its values.</summary>
    /// <returns>The entry, placed at this line.</returns>
    public SlotEntry Entry() => new(
        Words[0],
        Tensor.From(new Shape([.. Words[1].Split('x').Select(length => int.Parse(length, CultureInfo.InvariantCulture))]), [.. Words[2..].Select(Value)]),
        string.Create(CultureInfo.InvariantCulture, $"line {Number}"));

    /// <summary>A whole number the line holds.</summary>
    /// <param name="at">Which word.</param>
    /// <returns>The number.</returns>
    public int Whole(int at) => int.Parse(Words[at], CultureInfo.InvariantCulture);

    private static float Value(string word) => float.Parse(word, CultureInfo.InvariantCulture);
}

/// <summary>The lines of a file, as a test file writes them.</summary>
internal static class LinesExtensions
{
    extension(Stream file)
    {
        /// <summary>Every line that holds a word, split into its words.</summary>
        public IEnumerable<Line> Lines()
        {
            using var reader = new StreamReader(file, Encoding.UTF8, leaveOpen: true);
            var number = 0;

            for (var text = reader.ReadLine(); text is not null; text = reader.ReadLine())
            {
                number++;

                if (text.Split(' ', StringSplitOptions.RemoveEmptyEntries) is [_, ..] words)
                {
                    yield return new Line(number, words);
                }
            }
        }
    }
}
