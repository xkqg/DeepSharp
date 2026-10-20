// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// The words 0.11.0 adds to a description: the smooth activations a transformer is made of, the leaky, parametric and
/// self-normalising ones, softmax as a layer, the normalisations over groups of channels, over one example's channels and by the
/// root mean square, and the periodic embedding of a row of numbers.
/// </summary>
public sealed partial class Sequential
{
    /// <summary>Weighs each value by the share of a normal curve below it: the GELU, exactly as PyTorch and Keras leave it.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Gelu() => Gelu(approximate: false);

    /// <summary>The GELU, exactly or by the hyperbolic tangent's approximation: Keras's <c>approximate</c>, PyTorch's <c>approximate='tanh'</c>.</summary>
    /// <param name="approximate">Whether it is the hyperbolic tangent's approximation.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Gelu(bool approximate) => Append(new PlainWord(() => new Gelu(approximate)));

    /// <summary>Weighs each value by its own logistic curve: the SiLU, also called swish.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Silu() => Append(new PlainWord(() => new Silu()));

    /// <summary>Weighs each value by the hyperbolic tangent of its softplus: mish.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Mish() => Append(new PlainWord(() => new Mish()));

    /// <summary>Keeps each value above nothing and lets a hundredth of every other through, as PyTorch leaves the leaky rectifier.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>Keras's layer lets three tenths through unless told, and its activation word two: a description of either says so.</remarks>
    public Sequential LeakyRelu() => LeakyRelu(0.01);

    /// <summary>Keeps each value above nothing and lets the given share of every other through.</summary>
    /// <param name="negativeSlope">The share let through, nothing or more.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The share is not a finite number of nothing or more.</exception>
    public Sequential LeakyRelu(double negativeSlope)
    {
        var layer = new LeakyRelu(negativeSlope);

        return Append(new PlainWord(() => new LeakyRelu(layer.NegativeSlope)));
    }

    /// <summary>Keeps each value above nothing and lets a share of every other through that it learns, one share for every value, as PyTorch leaves it.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential PRelu() => PRelu(eachChannel: false);

    /// <summary>The parametric rectifier, learning one share for every value, or one for each channel of the last axis.</summary>
    /// <param name="eachChannel">
    /// Whether each channel has a share of its own — PyTorch's <c>num_parameters</c> set to the channels, and Keras's
    /// <c>PReLU</c> shared over every axis but the channels'.
    /// </param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>Each share starts at a quarter, as PyTorch starts it; Keras starts it at nothing.</remarks>
    public Sequential PRelu(bool eachChannel) => Append(new ParametricWord(eachChannel));

    /// <summary>Keeps each value above nothing and bends every other towards minus one along an exponential: the ELU.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Elu() => Elu(1);

    /// <summary>The ELU, bending towards minus the given alpha.</summary>
    /// <param name="alpha">What the values below nothing bend towards, below nothing: nothing or more.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Alpha is not a finite number of nothing or more.</exception>
    public Sequential Elu(double alpha)
    {
        var layer = new Elu(alpha);

        return Append(new PlainWord(() => new Elu(layer.Alpha)));
    }

    /// <summary>The self-normalising ELU, at Klambauer et al.'s alpha and scale.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Selu() => Append(new PlainWord(() => new Selu()));

    /// <summary>Turns the last axis of each example into shares of one whole.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>
    /// Last, before a loss that applies the softmax itself — <see cref="CrossEntropy"/> — compiling the description lifts it into the
    /// loss, as it lifts a sigmoid before a binary cross-entropy: the network ends before it, and every prediction goes through one
    /// softmax, the loss's.
    /// </remarks>
    public Sequential Softmax() => Append(new PlainWord(() => new Softmax()));

    /// <summary>Normalises each example over every place it holds, its channels — the last axis — in so many groups.</summary>
    /// <param name="groups">How many groups the channels stand in; a number that divides them.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one group.</exception>
    /// <remarks>As PyTorch leaves it: an epsilon of a hundred-thousandth; Keras adds a thousandth, and a description of it says so.</remarks>
    public Sequential GroupNorm(int groups) => GroupNorm(groups, 1e-5, default);

    /// <summary>Normalises each example over every place it holds, its channels in so many groups, with the epsilon and the scale and the shift it is told to keep.</summary>
    /// <param name="groups">How many groups the channels stand in; a number that divides them.</param>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing.</param>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one group, or the epsilon is not a finite number above nothing.</exception>
    public Sequential GroupNorm(int groups, double epsilon, Affine affine)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        Normalisation.RequireEpsilon(epsilon, nameof(epsilon));

        return Append(new GroupWord(groups, channels => new GroupNorm(channels, groups, affine) { Epsilon = epsilon }));
    }

    /// <summary>Normalises each channel of each example — the last axis — over every place the example holds.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>With a scale and a shift, as Keras's group normalisation of a group for every channel keeps them; PyTorch's keeps neither unless told.</remarks>
    public Sequential InstanceNorm() => InstanceNorm(1e-5, default);

    /// <summary>Normalises each channel of each example over every place it holds, with the epsilon and the scale and the shift it is told to keep.</summary>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing.</param>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epsilon is not a finite number above nothing.</exception>
    public Sequential InstanceNorm(double epsilon, Affine affine)
    {
        Normalisation.RequireEpsilon(epsilon, nameof(epsilon));

        return Append(new NormalisationWord(channels => new InstanceNorm(channels, affine) { Epsilon = epsilon }));
    }

    /// <summary>Divides each row of the last axis by the root of the mean of its squares, and scales it, as PyTorch leaves it.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential RmsNorm() => RmsNorm(global::DeepSharp.Networks.RmsNorm.SingleStep);

    /// <summary>Divides each row of the last axis by the root of the mean of its squares and the given epsilon, and scales it: Keras's adds a millionth.</summary>
    /// <param name="epsilon">What is added to the mean of the squares before its root is taken, above nothing.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epsilon is not a finite number above nothing.</exception>
    public Sequential RmsNorm(double epsilon) => RmsNorm(epsilon, scale: true);

    /// <summary>Divides each row of the last axis by the root of the mean of its squares and the given epsilon, and scales it or not.</summary>
    /// <param name="epsilon">What is added to the mean of the squares before its root is taken, above nothing.</param>
    /// <param name="scale">Whether it scales each feature by a number it learns — PyTorch's <c>elementwise_affine</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epsilon is not a finite number above nothing.</exception>
    public Sequential RmsNorm(double epsilon, bool scale)
    {
        Normalisation.RequireEpsilon(epsilon, nameof(epsilon));

        return Append(new NormalisationWord(features => new RmsNorm(features, scale) { Epsilon = epsilon }));
    }

    /// <summary>
    /// Turns each number of a row into its cosines and sines at frequencies it learns, through a linear layer: the periodic
    /// embedding of Gorishniy et al., their PLR once a <see cref="Relu()"/> follows it.
    /// </summary>
    /// <param name="periodicity">How many frequencies each number has, the spread they start from, how wide each comes out, and whether the linear layer is shared.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one frequency or number out, or the spread is not a finite number above nothing.</exception>
    public Sequential Periodic(Periodicity periodicity)
    {
        // Built once, so a periodicity it cannot hold is refused where it is written rather than where it is lowered.
        _ = new Periodic(1, periodicity, new RandomStream(0).Draw("periodicity", 0, 0));

        return Append(new PeriodicWord(periodicity));
    }

    /// <summary>
    /// Replaces the columns of a row that hold one category's ones and noughts by a vector of so many numbers it learns for each
    /// category: an embedding, over the block an encoder writes a column of words down as.
    /// </summary>
    /// <param name="block">The places of the block's columns in the row as it reaches the word, counting from nought, in the order of the categories.</param>
    /// <param name="width">How many numbers a category's vector holds.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A vector holds fewer than one number.</exception>
    /// <exception cref="ArgumentException">The block holds fewer than two columns, or one twice.</exception>
    /// <remarks>
    /// A pipeline names the block of a column it encoded — <c>prepared.CategoryBlock("sex")</c> — and a network declared in it says
    /// the column, which is turned into the places where the rows are handed over.
    /// </remarks>
    public Sequential Embedding(IReadOnlyList<int> block, int width)
    {
        ArgumentNullException.ThrowIfNull(block);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);

        if (block.Count < 2 || block.Distinct().Count() != block.Count)
        {
            throw new ArgumentException($"An embedding reads a category from two columns of the row at least, each once, and was told [{string.Join(", ", block)}].", nameof(block));
        }

        return Append(new EmbeddingWord([.. block], width));
    }

    /// <summary>A text as one small vector: the ids of a row looked up in vectors it learns, and the vectors of the words averaged.</summary>
    /// <param name="vocabulary">How many ids there are, padding and the id of a word never seen among them: three at least.</param>
    /// <param name="width">How many numbers each id's vector holds.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The vocabulary is below three, or a vector holds fewer than one number.</exception>
    /// <remarks>
    /// A row is the ids a tokenizer wrote (<c>Tokenize</c> in a pipeline), id nought being padding that is left out of the average. The
    /// layer takes each example as a row of ids, and so stands first.
    /// </remarks>
    public Sequential TokenEmbedding(int vocabulary, int width) => TokenEmbedding(vocabulary, width, keepSteps: false);

    /// <summary>The ids of a text as small vectors it learns, either averaged into one or kept a step at a time.</summary>
    /// <param name="vocabulary">How many ids there are, padding and the id of a word never seen among them: three at least.</param>
    /// <param name="width">How many numbers each id's vector holds.</param>
    /// <param name="keepSteps">Whether each id keeps its own vector at its step, the padding nothing, for a recurrent layer to read; the average of the words' otherwise.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The vocabulary is below three, or a vector holds fewer than one number.</exception>
    public Sequential TokenEmbedding(int vocabulary, int width, bool keepSteps)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabulary, 3);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);

        return Append(new TokenEmbeddingWord(vocabulary, width) { KeepSteps = keepSteps });
    }

    /// <summary>A parametric rectifier, of one share or of one for each channel of the last axis.</summary>
    private sealed record ParametricWord(bool EachChannel) : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new PRelu(EachChannel ? each[each.Rank - 1] : 1);
    }

    /// <summary>A group normalisation, of as many channels as the last axis of an example holds, in groups that must divide them.</summary>
    private sealed record GroupWord(int Groups, Func<int, Layer> Made) : IWord
    {
        public string? Refusal(Shape each) =>
            each[each.Rank - 1] % Groups == 0
                ? null
                : string.Create(CultureInfo.InvariantCulture, $"groupNorm, puts the channels in {Groups} groups, and the {each[each.Rank - 1]} channels of each example reaching it do not go into them.");

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => Made(each[each.Rank - 1]);
    }

    /// <summary>The ids of a text averaged as the vectors they stand for: a row of ids comes out as one vector.</summary>
    private sealed record TokenEmbeddingWord(int Vocabulary, int Width) : IWord
    {
        public bool KeepSteps { get; init; }

        public string? Refusal(Shape each) => each.Rank == 1 ? null : $"tokenEmbedding, takes each example as a row of ids, and each reaching it is {each}.";

        public Shape After(Shape each) => KeepSteps ? new Shape(each[0], Width) : new Shape(Width);

        public Layer Make(Shape each, Draws draws) => new TokenEmbedding(each[0], Vocabulary, Width, draws) { KeepsSteps = KeepSteps };
    }

    /// <summary>An embedding of a category held by a block of a row's columns: the block gone, and a vector in its place.</summary>
    private sealed record EmbeddingWord(int[] Block, int Width) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != 1
                ? $"embedding, takes each example as a row of numbers, and each reaching it is {each}."
                : Block.Max() >= each[0]
                    ? string.Create(CultureInfo.InvariantCulture, $"embedding, reads a category from column {Block.Max() + 1} of the row, and each row reaching it holds {each[0]}.")
                    : null;

        public Shape After(Shape each) => new(each[0] - Block.Length + Width);

        public Layer Make(Shape each, Draws draws) => new Embedding(each[0], Block, Width, draws);
    }

    /// <summary>A periodic embedding of a row of numbers: each number comes out as so many.</summary>
    private sealed record PeriodicWord(Periodicity Periodicity) : IWord
    {
        public string? Refusal(Shape each) => each.Rank == 1 ? null : $"periodic, takes each example as a row of numbers, and each reaching it is {each}: flatten it first.";

        public Shape After(Shape each) => new(each[0] * Periodicity.Width);

        public Layer Make(Shape each, Draws draws) => new Periodic(each[0], Periodicity, draws);
    }
}
