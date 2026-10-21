// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>The words that read a series a step at a time: an LSTM, a GRU, and the mask a row of ids makes.</summary>
public sealed partial class Sequential
{
    /// <summary>A long short-term memory layer over each series, as PyTorch's <c>nn.LSTM</c>.</summary>
    /// <param name="hidden">How many numbers the state it carries along a series holds.</param>
    /// <param name="bothWays">Whether a second state reads each series back from its own end, the two laid side by side; one direction, unless said.</param>
    /// <param name="layers">How many layers are stacked, each reading the steps of the one below; one, unless said.</param>
    /// <param name="sequences">Whether it makes the state at every step instead of the last; the last, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The state or the layers hold fewer than one number.</exception>
    /// <remarks>
    /// It takes each example as a series, steps by channels. Given a mask as its second take — <c>.Takes("steps", "mask")</c> — a batch of
    /// series of unequal lengths is read each as it would be read alone.
    /// </remarks>
    public Sequential Lstm(int hidden, bool bothWays = false, int layers = 1, bool sequences = false) =>
        Append(Recurrence(new RecurrentWord("lstm", hidden, layers, bothWays, sequences)));

    /// <summary>A gated recurrent unit over each series, as PyTorch's <c>nn.GRU</c>.</summary>
    /// <param name="hidden">How many numbers the state it carries along a series holds.</param>
    /// <param name="bothWays">Whether a second state reads each series back from its own end, the two laid side by side; one direction, unless said.</param>
    /// <param name="layers">How many layers are stacked, each reading the steps of the one below; one, unless said.</param>
    /// <param name="sequences">Whether it makes the state at every step instead of the last; the last, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The state or the layers hold fewer than one number.</exception>
    public Sequential Gru(int hidden, bool bothWays = false, int layers = 1, bool sequences = false) =>
        Append(Recurrence(new RecurrentWord("gru", hidden, layers, bothWays, sequences)));

    /// <summary>The mask of a row of ids: a one where an id is a word, a nought where it is padding.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>A recurrent layer takes it as its second step, from the same ids the embedding before it reads.</remarks>
    public Sequential Unpadded() => Append(new MaskWord());

    private static RecurrentWord Recurrence(RecurrentWord word)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(word.Hidden, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(word.Layers, 1);

        return word;
    }

    /// <summary>A recurrent layer over each series.</summary>
    private sealed record RecurrentWord(string Kind, int Hidden, int Layers, bool BothWays, bool Sequences) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == 2 ? null : $"{Kind}, takes each example as a series of steps and channels, and each reaching it is {each}.";

        public string? Refusal(Shape[] inputs)
        {
            if (inputs.Length > 2)
            {
                return $"{Kind}, takes a series and, besides, the mask of its steps, and was given the work of {inputs.Length} steps.";
            }

            return Refusal(inputs[0]) ?? (inputs.Length == 2 && inputs[1] != new Shape(inputs[0][0])
                ? $"{Kind}, takes a mask of a one or a nought for each of the {inputs[0][0]} steps of a series, and the mask handed to it is {inputs[1]}."
                : null);
        }

        public Shape After(Shape each)
        {
            var wide = Hidden * (BothWays ? 2 : 1);

            return Sequences ? new Shape(each[0], wide) : new Shape(wide);
        }

        public Layer Make(Shape each, Draws draws)
        {
            var build = new Recurrence(each[1], Hidden) { Layers = Layers, BothWays = BothWays, Sequences = Sequences };

            return Kind == "lstm" ? new Lstm(build, draws) : new Gru(build, draws);
        }
    }

    /// <summary>The mask of a row of ids.</summary>
    private sealed record MaskWord : IWord
    {
        public string? Refusal(Shape each) => each.Rank == 1 ? null : $"unpadded, takes each example as a row of ids, and each reaching it is {each}.";

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new Unpadded();
    }
}
