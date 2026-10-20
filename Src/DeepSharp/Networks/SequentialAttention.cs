// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What an encoder block is made of: the heads of its attention, the width of its feed-forward layer, and how it is put together.</summary>
/// <param name="Heads">How many heads its attention has; they divide the width of a step.</param>
/// <param name="FeedForward">How wide the hidden layer of its feed-forward part is.</param>
public readonly record struct EncoderShape(int Heads, int FeedForward)
{
    /// <summary>The share of values left out while the block trains, after the attention, in the feed-forward part and after it, and of the attention's weights; none, unless said.</summary>
    public double Dropout { get; init; }

    /// <summary>Whether each norm stands before the part it guards, as PyTorch's <c>norm_first</c> has it; it stands after the sum, unless said.</summary>
    public bool NormFirst { get; init; }

    /// <summary>Whether the feed-forward part bends with a GELU; it bends with a ReLU, unless said.</summary>
    public bool Gelu { get; init; }

    /// <summary>Whether the block is handed the mask of the padding as its second take, which its attention looks past; it is handed none, unless said.</summary>
    public bool Masked { get; init; }
}

/// <summary>The words that look across the steps of a series: attention, the positions of the steps, and the encoder block made of them.</summary>
public sealed partial class Sequential
{
    /// <summary>Multi-head attention over each series, as PyTorch's <c>nn.MultiheadAttention</c>.</summary>
    /// <param name="how">The heads, whether a step may look only at the steps up to its own, and the share of the weights left out as it trains.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head, or the dropout is not a share below one.</exception>
    /// <remarks>
    /// It takes each example as a series, steps by channels, and makes a series of the same shape. Given a mask as its second take —
    /// <c>.Takes("steps", "mask")</c> — a batch of series of unequal lengths is read each as it would be read alone.
    /// </remarks>
    public Sequential MultiHeadAttention(Attending how) => Append(new AttentionWord(Attended(how), Projects: true));

    /// <summary>Multi-head attention over each series, every step looking at every step.</summary>
    /// <param name="heads">How many heads the width of a step is split into; they divide it.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head.</exception>
    public Sequential MultiHeadAttention(int heads) => MultiHeadAttention(new Attending(heads));

    /// <summary>Scaled dot-product attention, with no number to learn.</summary>
    /// <param name="how">The heads, whether a step may look only at the steps up to its own, and the share of the weights left out as it trains.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head, or the dropout is not a share below one.</exception>
    /// <remarks>
    /// Given one step it takes the series as the queries, the keys and the values alike; given three — <c>.Takes("queries", "keys", "values")</c>
    /// — as PyTorch's <c>scaled_dot_product_attention</c> takes them; a mask of the steps that take part is the last take.
    /// </remarks>
    public Sequential Attention(Attending how) => Append(new AttentionWord(Attended(how), Projects: false));

    /// <summary>Scaled dot-product attention, every step looking at every step.</summary>
    /// <param name="heads">How many heads the width of a step is split into; they divide it.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head.</exception>
    public Sequential Attention(int heads) => Attention(new Attending(heads));

    /// <summary>Adds to each step of a series the place it stands at, as the sine and cosine waves of 'Attention Is All You Need' write it.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential SinePositions() => Append(new PositionsWord());

    /// <summary>
    /// An encoder block, as PyTorch's <c>nn.TransformerEncoderLayer</c> is made: attention over the steps, a feed-forward part over each step, the
    /// input added back after each and a layer normalisation guarding each.
    /// </summary>
    /// <param name="shape">The heads of its attention, the width of its feed-forward layer, and how it is put together.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head or one feed-forward unit, or the dropout is not a share below one.</exception>
    /// <remarks>
    /// The block is one step. Its numbers are found under the names PyTorch gives a layer's state: <c>self_attn.in_proj_weight</c>,
    /// <c>self_attn.out_proj.weight</c>, <c>linear1.weight</c>, <c>linear2.weight</c>, <c>norm1.weight</c>, <c>norm2.weight</c>, each with its bias, under the
    /// name the step is given. A block begun with <see cref="EncoderShape.Masked"/> is given the mask of the padding as its second take.
    /// </remarks>
    public Sequential EncoderBlock(EncoderShape shape)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(shape.FeedForward, 1);
        _ = Attended(new Attending(shape.Heads) { Dropout = shape.Dropout });

        return Append(new BlockWord(each => EncoderWords(shape, each[each.Rank - 1])));
    }

    /// <summary>The words of an encoder block, to be lowered on their own, as a block is, or put in a description by <see cref="EncoderBlock"/>.</summary>
    /// <param name="shape">The heads of its attention, the width of its feed-forward layer, and how it is put together.</param>
    /// <param name="embed">How many channels a step holds, which the heads divide.</param>
    /// <returns>The description: each step called as the module of PyTorch's layer is.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one head, one channel or one feed-forward unit, or the dropout is not a share below one.</exception>
    public static Sequential EncoderWords(EncoderShape shape, int embed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(shape.FeedForward, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(embed, 1);

        return Encoder(shape, Attended(new Attending(shape.Heads) { Dropout = shape.Dropout }), embed);
    }

    private static Attending Attended(Attending how)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(how.Heads, 1);
        _ = global::DeepSharp.Networks.Dropout.RequireRate(how.Dropout);

        return how;
    }

    // The words of an encoder block, each step called as the module of PyTorch's layer is: the sums are added back to the block's input.
    private static Sequential Encoder(EncoderShape shape, Attending how, int embed)
    {
        string[] mask = shape.Masked ? [Step.Mask] : [];
        var inner = new Sequential();
        var rate = shape.Dropout;

        if (shape.NormFirst)
        {
            inner.LayerNorm().Named("norm1");
            inner.MultiHeadAttention(how).Named("self_attn").Takes(["norm1", .. mask]);
            inner.Add(Step.Input, Dropped(inner, rate, "dropout1", "self_attn")).Named("sum1");
            inner.LayerNorm().Named("norm2");
            inner.Add("sum1", FeedForward(inner, shape, embed)).Named("sum2");

            return inner;
        }

        inner.MultiHeadAttention(how).Named("self_attn").Takes([Step.Input, .. mask]);
        inner.Add(Step.Input, Dropped(inner, rate, "dropout1", "self_attn")).Named("sum1");
        inner.LayerNorm().Named("norm1");
        inner.Add("norm1", FeedForward(inner, shape, embed)).Named("sum2");
        inner.LayerNorm().Named("norm2");

        return inner;
    }

    // The feed-forward part over each step: a layer wide, the bend, a layer back to the width of the step, and the dropouts; the name of its last step.
    private static string FeedForward(Sequential inner, EncoderShape shape, int embed)
    {
        inner.Dense(shape.FeedForward).Named("linear1");
        _ = shape.Gelu ? inner.Gelu() : inner.Relu();

        _ = Dropped(inner, shape.Dropout, "dropout", string.Empty);
        inner.Dense(embed).Named("linear2");

        return Dropped(inner, shape.Dropout, "dropout2", "linear2");
    }

    // A dropout after the step named, when the block leaves anything out; the name of the step whose work goes on.
    private static string Dropped(Sequential inner, double rate, string name, string from)
    {
        if (rate <= 0)
        {
            return from;
        }

        inner.Dropout(rate).Named(name);

        return name;
    }

    /// <summary>Attention over a series, with the projections of the heads or without.</summary>
    private sealed record AttentionWord(Attending How, bool Projects) : IMaskedWord
    {
        private string Kind => Projects ? "multiHeadAttention" : "attention";

        public string? Refusal(Shape each) =>
            each.Rank != 2
                ? $"{Kind}, takes each example as a series of steps and channels, and each reaching it is {each}."
                : each[1] % How.Heads != 0
                    ? string.Create(CultureInfo.InvariantCulture, $"{Kind}, splits a step of {each[1]} channels between {How.Heads} heads, and that is not a whole number of channels for each head.")
                    : null;

        public string? Refusal(Shape[] inputs)
        {
            var seen = inputs.Length;

            if (seen is 0 or > 4 || (Projects && seen > 2))
            {
                return $"{Kind}, takes a series and, besides, the mask of its steps{(Projects ? string.Empty : ", or queries, keys and values, and then that mask")}, and was given the work of {seen} steps.";
            }

            var keys = seen >= 3 ? inputs[1] : inputs[0];
            var mask = seen is 2 or 4 ? inputs[seen - 1] : (Shape?)null;

            var refusal = Refusal(inputs[0]) ?? (seen >= 3 ? Refusal(keys) : null);

            if (refusal is not null)
            {
                return refusal;
            }

            if (seen >= 3 && (inputs[2] != keys || keys[1] != inputs[0][1]))
            {
                return $"{Kind}, takes keys of as wide a step as its queries and values of the very shape of its keys: these are {inputs[0]}, {keys} and {inputs[2]}.";
            }

            return mask is { } masked && masked != new Shape(keys[0])
                ? $"{Kind}, takes a mask of a one or a nought for each of the {keys[0]} steps of a series, and the mask handed to it is {masked}."
                : null;
        }

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => Projects ? new MultiHeadAttention(each[1], How, draws) : new Attention(How);
    }

    private sealed record PositionsWord : IWord
    {
        public string? Refusal(Shape each) => each.Rank == 2 ? null : $"sinePositions, takes each example as a series of steps and channels, and each reaching it is {each}.";

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new SinePositions();
    }
}
