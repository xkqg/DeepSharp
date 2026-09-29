// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A network described in Keras's words — a stack of layers, each named by what it does and how many values it makes —
/// read once and lowered onto the <see cref="LayerStack"/> a network written as code is.
/// </summary>
/// <remarks>
/// Only the words are written here. Every width a layer takes is worked out from the shape of an example as it reaches the
/// layer, as Keras works it out, and every layer that learns starts from draws out of a stream by the place it stands at —
/// the draws a network written as code would take for the same place. Once lowered, nothing downstream knows which door
/// the network came through: the loop, the file and the charts see one kind of network.
/// </remarks>
public sealed class Sequential
{
    private const string NoLayers = "This network has no layers yet: write them in before it is lowered.";

    private readonly List<IWord> _words = [];

    /// <summary>A description that holds no word yet.</summary>
    public Sequential()
    {
    }

    // The description as it stands, kept apart from the one that goes on being written.
    private Sequential(Sequential description)
    {
        _words = [.. description._words];
        InputShape = description.InputShape;
    }

    /// <summary>The shape of one example, when stated; nothing when it is taken from the rows the network first meets.</summary>
    public Shape? InputShape { get; private set; }

    /// <summary>
    /// States the shape of one example: every word is checked against it where the description is compiled, and the rows it
    /// is fitted on are held to it.
    /// </summary>
    /// <param name="shape">The shape of one example, without the batch's axis: 14 numbers, or an image of 28x28x1.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The shape holds no value.</exception>
    /// <exception cref="InvalidOperationException">A layer is written already, or the input was stated already: it stands first, once.</exception>
    public Sequential Input(Shape shape)
    {
        global::DeepSharp.Networks.Reshape.RequireEach(shape);

        if (InputShape is not null || _words.Count > 0)
        {
            throw new InvalidOperationException("The input stands first, once: it is stated before any layer, and only once.");
        }

        InputShape = shape;

        return this;
    }

    /// <summary>A fully connected layer: every value an example holds, weighed into each of so many.</summary>
    /// <param name="units">How many values it makes of each example.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one value.</exception>
    public Sequential Dense(int units) => Add(new DenseWord(units));

    /// <summary>Keeps each value above nothing, and nothing otherwise.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Relu() => Add(new PlainWord(() => new Relu()));

    /// <summary>Bends each value into the span between minus one and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Tanh() => Add(new PlainWord(() => new Tanh()));

    /// <summary>Bends each value into the span between nothing and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Sigmoid() => Add(new PlainWord(() => new Sigmoid()));

    /// <summary>Leaves out a share of the values while training, each afresh on every pass.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential Dropout(double rate) => Add(new DropoutWord(global::DeepSharp.Networks.Dropout.RequireRate(rate)));

    /// <summary>Normalises each feature — the last axis of an example — over the batch.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential BatchNorm() => Add(new NormalisationWord(features => new BatchNorm(features)));

    /// <summary>Normalises each example over its features — the last axis.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential LayerNorm() => Add(new NormalisationWord(features => new LayerNorm(features)));

    /// <summary>Lays each example out in another shape holding as many values: a row of pixels as an image, say.</summary>
    /// <param name="shape">The shape of one example, without the batch's axis.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The shape holds no value.</exception>
    public Sequential Reshape(Shape shape) => Add(new ReshapeWord(global::DeepSharp.Networks.Reshape.RequireEach(shape)));

    /// <summary>A convolution: a window slid over each image, making so many channels at every place it stands.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The window cannot stand anywhere.</exception>
    public Sequential Conv2D(int filters, Window window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        global::DeepSharp.Networks.Conv2D.RequireWindow(window);

        return Add(new ConvolutionWord(filters, window));
    }

    /// <summary>Lays each example out as one row of its values.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Flatten() => Add(new FlattenWord());

    /// <summary>
    /// Lowers the description onto a stack of layers for examples of the given shape, each layer that learns starting from
    /// draws out of the stream by the place it stands at.
    /// </summary>
    /// <param name="input">The shape of one example, without the batch's axis.</param>
    /// <param name="stream">The stream the layers' starting values are drawn from: <c>initialise:</c> and the layer's place.</param>
    /// <returns>The layers, as a network written as code would stack them.</returns>
    /// <exception cref="ArgumentException">
    /// The shape holds no value, is not the input the description states, or an example of it cannot reach a layer — a row
    /// handed to a convolution, an image to a dense layer, a window larger than the image.
    /// </exception>
    /// <exception cref="InvalidOperationException">The description holds no layer.</exception>
    public LayerStack Lower(Shape input, RandomStream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        Check(input);

        var layers = new List<Layer>(_words.Count);
        var each = input;

        for (var place = 0; place < _words.Count; place++)
        {
            layers.Add(_words[place].Make(each, stream.Draw(string.Create(CultureInfo.InvariantCulture, $"initialise:{place}"), 0, 0)));
            each = _words[place].After(each);
        }

        return new LayerStack(layers);
    }

    /// <summary>
    /// Makes the description ready to be fitted, as Keras's <c>compile</c> does: what moves its parameters, what it is trained to
    /// bring down, and how the rate changes. The description is taken as it stands; a word written after this changes nothing.
    /// </summary>
    /// <param name="optimizer">What moves its parameters.</param>
    /// <param name="loss">What it is trained to bring down: named, never assumed.</param>
    /// <param name="schedule">How the optimizer's rate changes from epoch to epoch; it stays as it is, unless said.</param>
    /// <returns>
    /// The compiled network. Its layers are built at its first fit, from the shape of the rows it learns from — or the input
    /// stated — and the run's seed, exactly as <see cref="Lower"/> builds them from a stream of that seed.
    /// </returns>
    /// <exception cref="InvalidOperationException">The description holds no layer.</exception>
    /// <exception cref="ArgumentException">Its input is stated, and an example of it cannot reach a layer.</exception>
    /// <remarks>
    /// Nothing is drawn here: the seed that decides what the layers start at is the run's, given to the fit, so one recorded
    /// number reproduces the whole run — the start, the shuffles and the dropouts.
    /// </remarks>
    public CompiledNetwork Compile(Optimizer optimizer, Loss loss, LearningRateSchedule? schedule = null)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(loss);

        if (_words.Count == 0)
        {
            throw new InvalidOperationException(NoLayers);
        }

        if (InputShape is { } stated)
        {
            Check(stated);
        }

        return new CompiledNetwork(new Sequential(this), optimizer, loss, schedule);
    }

    // Refuses a shape the description cannot be lowered at: one holding no value, another than its input says, or one an
    // example of which cannot reach one of its words.
    private void Check(Shape input)
    {
        global::DeepSharp.Networks.Reshape.RequireEach(input);

        if (InputShape is { } stated && stated != input)
        {
            throw new ArgumentException($"This network takes each example as {stated}, as its input says, and cannot be lowered at {input}.", nameof(input));
        }

        if (_words.Count == 0)
        {
            throw new InvalidOperationException(NoLayers);
        }

        var each = input;

        for (var place = 0; place < _words.Count; place++)
        {
            var word = _words[place];

            if (word.Refusal(each) is { } refusal)
            {
                throw new ArgumentException(
                    string.Create(CultureInfo.InvariantCulture, $"This network cannot be lowered at {input}: word {place + 1}, {refusal}"), nameof(input));
            }

            each = word.After(each);
        }
    }

    private Sequential Add(IWord word)
    {
        _words.Add(word);

        return this;
    }

    /// <summary>One word of a description: what it cannot take, what it hands on, and the layer it becomes for what it can take.</summary>
    private interface IWord
    {
        /// <summary>The word, and what is wrong with an example of this shape reaching it; nothing when it can take it.</summary>
        string? Refusal(Shape each);

        /// <summary>The shape of one example the word hands on, for an example of this shape.</summary>
        Shape After(Shape each);

        /// <summary>The layer the word becomes for examples of this shape, starting from the given draws when it learns.</summary>
        Layer Make(Shape each, Draws draws);
    }

    /// <summary>A word whose layer takes any example and hands it on in the same shape, learning nothing.</summary>
    private sealed record PlainWord(Func<Layer> Make) : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        Layer IWord.Make(Shape each, Draws draws) => Make();
    }

    private sealed record DenseWord : IWord
    {
        public DenseWord(int units)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(units, 1);
            Units = units;
        }

        public int Units { get; }

        public string? Refusal(Shape each) => each.Rank == 1 ? null : $"dense, takes each example as a row of numbers, and each reaching it is {each}: flatten it first.";

        public Shape After(Shape each) => new(Units);

        public Layer Make(Shape each, Draws draws) => new Dense(each[0], Units, draws);
    }

    private sealed record DropoutWord(double Rate) : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new Dropout(Rate);
    }

    /// <summary>A normalisation, of as many features as the last axis of an example holds.</summary>
    private sealed record NormalisationWord(Func<int, Layer> Make) : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        Layer IWord.Make(Shape each, Draws draws) => Make(each[each.Rank - 1]);
    }

    private sealed record ReshapeWord(Shape Each) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Count == Each.Count
                ? null
                : string.Create(CultureInfo.InvariantCulture, $"reshape, makes each example {Each}, which holds {Each.Count} values, and each reaching it holds {each.Count}.");

        public Shape After(Shape each) => Each;

        public Layer Make(Shape each, Draws draws) => new Reshape(Each);
    }

    private sealed record ConvolutionWord(int Filters, Window Window) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank != 3 ? $"conv2d, takes each example as an image of rows, columns and channels, and each reaching it is {each}."
            : Window.RowsOver(each[0]) < 1 || Window.ColumnsOver(each[1]) < 1 ? $"conv2d, slides a {Window} over each example, and an example {each} is smaller than it."
            : null;

        public Shape After(Shape each) => new(Window.RowsOver(each[0]), Window.ColumnsOver(each[1]), Filters);

        public Layer Make(Shape each, Draws draws) => new Conv2D(each[2], Filters, Window, draws);
    }

    private sealed record FlattenWord : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => new(each.Count);

        public Layer Make(Shape each, Draws draws) => new Flatten();
    }
}
