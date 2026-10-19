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
public sealed partial class Sequential
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
    public Sequential Dense(int units) => Dense(units, new Beginning());

    /// <summary>A fully connected layer whose weights start as the given initialiser draws them.</summary>
    /// <param name="units">How many values it makes of each example.</param>
    /// <param name="weights">How the weights start; the bias starts as PyTorch starts it, as it does without this.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one value.</exception>
    /// <exception cref="ArgumentNullException">There is no initialiser.</exception>
    /// <remarks>
    /// How a layer starts is not written into a network's file, which holds the numbers it learned: the initialiser is a
    /// matter of the code that describes the network, under the run's seed, and a network read back starts nowhere.
    /// </remarks>
    public Sequential Dense(int units, Initialiser weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        return Dense(units, new Beginning { Weights = weights });
    }

    /// <summary>A fully connected layer that starts as it is told to, or has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="units">How many values it makes of each example.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one value.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    /// <remarks>
    /// A beginning here is built without draws, as <c>new Beginning { WithoutBias = true }</c>: the draws a described network's
    /// layers start from are the run's, counted from its seed at the place of each layer, so the description says only how.
    /// A layer written as code is handed a beginning that names draws and no more.
    /// </remarks>
    public Sequential Dense(int units, Beginning beginning)
    {
        RequireUnseeded(beginning);

        return Add(new DenseWord(units, beginning));
    }

    /// <summary>Keeps each value above nothing, and nothing otherwise.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Relu() => Add(new PlainWord(() => new Relu()));

    /// <summary>Bends each value into the span between minus one and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Tanh() => Add(new PlainWord(() => new Tanh()));

    /// <summary>Bends each value into the span between nothing and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>
    /// Last, before a loss that applies the sigmoid itself — <see cref="BinaryCrossEntropy"/> — it is Keras's habit, the
    /// activation on the network and a loss that takes what it gives, and compiling the description lifts it into the loss:
    /// the network ends before it, and every prediction goes through one sigmoid, the loss's.
    /// </remarks>
    public Sequential Sigmoid() => Add(new PlainWord(() => new Sigmoid()));

    /// <summary>Leaves out a share of the values while training, each afresh on every pass.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential Dropout(double rate) => Add(new DropoutWord(global::DeepSharp.Networks.Dropout.RequireRate(rate)));

    /// <summary>Mirrors images at random while training, each by a coin of its own across the axes it is told to.</summary>
    /// <param name="horizontal">Whether an image is mirrored across its columns.</param>
    /// <param name="vertical">Whether an image is mirrored across its rows.</param>
    /// <param name="seed">The number the flips are drawn from: the layer's own, which the run's seed moves nothing of.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential RandomFlip(bool horizontal, bool vertical, long seed) => Add(new FlipWord(horizontal, vertical, seed));

    /// <summary>Shifts images at random while training, by padding each and cutting a window of its own size out at a place drawn for it.</summary>
    /// <param name="pad">How many places an image is padded by on every side, and so how far it can move.</param>
    /// <param name="seed">The number the shifts are drawn from: the layer's own.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The border is below nought.</exception>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential RandomCrop(int pad, long seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(pad);

        return Add(new CropWord(pad, seed));
    }

    /// <summary>Adds noise to every value while training: normal, of the spread it is told.</summary>
    /// <param name="spread">The standard deviation of the noise; nought for none.</param>
    /// <param name="seed">The number the noise is drawn from: the layer's own.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The spread is not a number of nought or more.</exception>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential GaussianNoise(double spread, long seed) => Add(new NoiseWord(new GaussianNoise(spread, seed).Spread, seed));

    /// <summary>Normalises each feature — the last axis of an example — over the batch.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>As PyTorch leaves it: a momentum of a tenth, in PyTorch's meaning, and an epsilon of a hundred-thousandth.</remarks>
    public Sequential BatchNorm() => BatchNorm(default(Affine));

    /// <summary>Normalises each feature — the last axis of an example — over the batch, keeping the scale and the shift it is told to.</summary>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one — Keras's <c>scale</c> and <c>center</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential BatchNorm(Affine affine) => Add(new NormalisationWord(features => new BatchNorm(features, affine)));

    /// <summary>Normalises each feature — the last axis of an example — over the batch, with Keras's momentum and epsilon.</summary>
    /// <param name="momentum">
    /// Keras's momentum: the share of the running statistics each training batch leaves as they were, between nothing and
    /// one — 0.99, unless Keras is told otherwise. The layer keeps its complement, PyTorch's meaning of the word.
    /// </param>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing — 0.001 in Keras.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The momentum is not a share, or the epsilon not a finite number above nothing.</exception>
    /// <remarks>
    /// A batch normalisation Keras trained answers here as it answered there, from its scale, its shift and its running
    /// statistics. Trained here, its running variance takes each batch's variance as PyTorch counts it, over one row fewer.
    /// </remarks>
    public Sequential BatchNorm(double momentum, double epsilon) => BatchNorm(momentum, epsilon, default);

    /// <summary>Normalises each feature over the batch, with Keras's momentum and epsilon, keeping the scale and the shift it is told to.</summary>
    /// <param name="momentum">Keras's momentum: the share of the running statistics each training batch leaves as they were, between nothing and one.</param>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing — 0.001 in Keras.</param>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The momentum is not a share, or the epsilon not a finite number above nothing.</exception>
    public Sequential BatchNorm(double momentum, double epsilon, Affine affine)
    {
        if (!global::DeepSharp.Networks.BatchNorm.IsShare(momentum))
        {
            throw new ArgumentOutOfRangeException(
                nameof(momentum), momentum, "Keras's momentum is the share of the running statistics a training batch leaves as they were, between nothing and one.");
        }

        Normalisation.RequireEpsilon(epsilon, nameof(epsilon));

        return Add(new NormalisationWord(features => new BatchNorm(features, affine) { Momentum = 1 - momentum, Epsilon = epsilon }));
    }

    /// <summary>Normalises each example over its features — the last axis.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>As PyTorch leaves it: an epsilon of a hundred-thousandth.</remarks>
    public Sequential LayerNorm() => LayerNorm(default(Affine));

    /// <summary>Normalises each example over its features — the last axis — keeping the scale and the shift it is told to.</summary>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one — Keras's <c>scale</c> and <c>center</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential LayerNorm(Affine affine) => Add(new NormalisationWord(features => new LayerNorm(features, affine)));

    /// <summary>Normalises each example over its features — the last axis — with Keras's epsilon.</summary>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing — 0.001 in Keras.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epsilon is not a finite number above nothing.</exception>
    public Sequential LayerNorm(double epsilon) => LayerNorm(epsilon, default);

    /// <summary>Normalises each example over its features with Keras's epsilon, keeping the scale and the shift it is told to.</summary>
    /// <param name="epsilon">What is added to each variance before its root is taken, above nothing — 0.001 in Keras.</param>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epsilon is not a finite number above nothing.</exception>
    public Sequential LayerNorm(double epsilon, Affine affine)
    {
        Normalisation.RequireEpsilon(epsilon, nameof(epsilon));

        return Add(new NormalisationWord(features => new LayerNorm(features, affine) { Epsilon = epsilon }));
    }

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
    public Sequential Conv2D(int filters, Window window) => Conv2D(filters, window, 1, new Beginning());

    /// <summary>The same with the channels in groups: each group of channels in reaching its own group of channels out.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The channels do not go into the groups, or the window cannot stand anywhere.</exception>
    public Sequential Conv2D(int filters, Window window, int groups) => Conv2D(filters, window, groups, new Beginning());

    /// <summary>The same with a say in how the layer starts, or that it has no bias: Keras's <c>use_bias=False</c>.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv2D(int filters, Window window, Beginning beginning) => Conv2D(filters, window, 1, beginning);

    /// <summary>The same with the channels in groups and a say in how the layer starts, or that it has no bias.</summary>
    /// <param name="filters">How many channels it makes.</param>
    /// <param name="window">The window, and how it walks.</param>
    /// <param name="groups">How many groups the channels are split into; as many as channels in is a depthwise convolution.</param>
    /// <param name="beginning">How its kernel and its bias start — or that it has no bias; the draws are the run's, at the layer's place, and are not said here.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It makes fewer than one channel, or splits them into fewer than one group.</exception>
    /// <exception cref="ArgumentException">The beginning names draws, which a described network takes from its run, or wants no bias and says how it starts.</exception>
    public Sequential Conv2D(int filters, Window window, int groups, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(filters, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        RequireUnseeded(beginning);

        return Add(new ConvolutionWord(
            "conv2d",
            "an image of rows, columns and channels",
            filters,
            new PlaneWalk(window),
            (channels, draws) => new Conv2D(new Channels { In = channels, Out = filters, Groups = groups }, window, beginning with { Draws = draws })));
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
    /// <remarks>
    /// Every word is lowered, a last activation as well: it is <see cref="Compile"/> that leaves out one the loss applies
    /// itself, since only the loss says which that is.
    /// </remarks>
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
    /// stated — and the run's seed, exactly as <see cref="Lower"/> builds them from a stream of that seed; a last word whose
    /// activation the loss applies itself is left out of them.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// The description holds no layer, or none but the activation its loss applies itself.
    /// </exception>
    /// <exception cref="ArgumentException">Its input is stated, and an example of it cannot reach a layer.</exception>
    /// <remarks>
    /// Nothing is drawn here: the seed that decides what the layers start at is the run's, given to the fit, so one recorded
    /// number reproduces the whole run — the start, the shuffles and the dropouts.
    /// <para>
    /// Keras's habit is to end a network in the activation its loss is trained with — <c>Dense(1, activation='sigmoid')</c>
    /// before a binary cross-entropy, which there takes what the network gives. A loss here takes the logits and applies that
    /// activation itself, so a last <see cref="Sigmoid()"/> before <see cref="BinaryCrossEntropy"/>, or a last <see cref="Softmax()"/>
    /// before <see cref="CrossEntropy"/>, is lifted into the loss: the network ends before it, and every prediction goes through
    /// the loss's activation once. Before any other loss it stays.
    /// </para>
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

        return new CompiledNetwork(Lifted(loss), optimizer, loss, schedule);
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

    // How a described network's layers start is told without the draws: they are the run's, at the place of each layer. A
    // beginning that names draws, or that wants no bias and says how to start one, is refused where it is handed in.
    private static void RequireUnseeded(Beginning beginning)
    {
        if (beginning.Draws is not null)
        {
            throw new ArgumentException(
                "A described network starts every layer from the draws of its run, at the place the layer stands: say how its layers start, and build the beginning without draws.",
                nameof(beginning));
        }

        if (beginning.WithoutBias && beginning.Bias is not null)
        {
            throw new ArgumentException("A layer without a bias has none to start: either it has no bias or it says how the bias starts.", nameof(beginning));
        }
    }

    private Sequential Add(IWord word)
    {
        if (word is IVariationWord && _words.Exists(each => each is not IVariationWord))
        {
            throw new InvalidOperationException(
                "A variation is of the rows themselves, made on the values as the batch carries them: it stands first, before any layer that learns.");
        }

        _words.Add(word);

        return this;
    }

    // The description as it stands, for a network compiled with the loss: a last word whose activation the loss applies
    // itself left out of it, so the network ends where the loss takes over.
    private Sequential Lifted(Loss loss)
    {
        var lifted = new Sequential(this);
        var applied = loss.Activation.ToString().ToLowerInvariant();

        if (lifted._words[^1].IsAppliedBy(loss))
        {
            lifted._words.RemoveAt(lifted._words.Count - 1);
        }

        return lifted._words.Count > 0
            ? lifted
            : throw new InvalidOperationException(
                $"This network holds nothing but the {applied} its loss applies itself: write in the layers before it, and the loss takes its outputs as the logits it applies the {applied} to.");
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

        /// <summary>Whether the word is the activation the loss applies to a network's outputs itself; a word is not, unless it says so.</summary>
        bool IsAppliedBy(Loss loss) => false;
    }

    /// <summary>A word whose layer takes any example and hands it on in the same shape, learning nothing.</summary>
    private sealed record PlainWord(Func<Layer> Make) : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        Layer IWord.Make(Shape each, Draws draws) => Make();

        // The loss says which layer it applies itself, by the one rule it is compiled with a stack by.
        public bool IsAppliedBy(Loss loss) => loss.Applies(Make());
    }

    private sealed record DenseWord : IWord
    {
        public DenseWord(int units, Beginning beginning)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(units, 1);
            Units = units;
            Begins = beginning;
        }

        public int Units { get; }

        // How the layer starts, without the draws: PyTorch's start when nothing is said.
        public Beginning Begins { get; }

        public string? Refusal(Shape each) => each.Rank == 1 ? null : $"dense, takes each example as a row of numbers, and each reaching it is {each}: flatten it first.";

        public Shape After(Shape each) => new(Units);

        public Layer Make(Shape each, Draws draws) => new Dense(each[0], Units, Begins with { Draws = draws });
    }

    // A word that varies the rows while a network trains, and so stands before every other.
    private interface IVariationWord : IWord
    {
    }

    private sealed record FlipWord(bool Horizontal, bool Vertical, long Seed) : IVariationWord
    {
        public string? Refusal(Shape each) => each.Rank == 3 ? null : $"randomflip, takes each example as an image of rows, columns and channels, and each reaching it is {each}.";

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new RandomFlip(Horizontal, Vertical, Seed);
    }

    private sealed record CropWord(int Pad, long Seed) : IVariationWord
    {
        public string? Refusal(Shape each) => each.Rank == 3 ? null : $"randomcrop, takes each example as an image of rows, columns and channels, and each reaching it is {each}.";

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new RandomCrop(Pad, Seed);
    }

    private sealed record NoiseWord(double Spread, long Seed) : IVariationWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => each;

        public Layer Make(Shape each, Draws draws) => new GaussianNoise(Spread, Seed);
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

    private sealed record FlattenWord : IWord
    {
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => new(each.Count);

        public Layer Make(Shape each, Draws draws) => new Flatten();
    }
}
