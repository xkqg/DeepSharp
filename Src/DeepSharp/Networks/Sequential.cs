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
    private readonly List<Said> _said = [];

    /// <summary>A description that holds no word yet.</summary>
    public Sequential()
    {
    }

    // The description as it stands, kept apart from the one that goes on being written.
    private Sequential(Sequential description)
    {
        _words = [.. description._words];
        _said = [.. description._said];
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

        return Append(new DenseWord(units, beginning));
    }

    /// <summary>Keeps each value above nothing, and nothing otherwise.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Relu() => Append(new PlainWord(() => new Relu()));

    /// <summary>Bends each value into the span between minus one and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Tanh() => Append(new PlainWord(() => new Tanh()));

    /// <summary>Bends each value into the span between nothing and one.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>
    /// Last, before a loss that applies the sigmoid itself — <see cref="BinaryCrossEntropy"/> — it is Keras's habit, the
    /// activation on the network and a loss that takes what it gives, and compiling the description lifts it into the loss:
    /// the network ends before it, and every prediction goes through one sigmoid, the loss's.
    /// </remarks>
    public Sequential Sigmoid() => Append(new PlainWord(() => new Sigmoid()));

    /// <summary>Leaves out a share of the values while training, each afresh on every pass.</summary>
    /// <param name="rate">The share left out, from nothing to below one.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not a share below one.</exception>
    public Sequential Dropout(double rate) => Append(new DropoutWord(global::DeepSharp.Networks.Dropout.RequireRate(rate)));

    /// <summary>Mirrors images at random while training, each by a coin of its own across the axes it is told to.</summary>
    /// <param name="horizontal">Whether an image is mirrored across its columns.</param>
    /// <param name="vertical">Whether an image is mirrored across its rows.</param>
    /// <param name="seed">The number the flips are drawn from: the layer's own, which the run's seed moves nothing of.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential RandomFlip(bool horizontal, bool vertical, long seed) => Append(new FlipWord(horizontal, vertical, seed));

    /// <summary>Shifts images at random while training, by padding each and cutting a window of its own size out at a place drawn for it.</summary>
    /// <param name="pad">How many places an image is padded by on every side, and so how far it can move.</param>
    /// <param name="seed">The number the shifts are drawn from: the layer's own.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The border is below nought.</exception>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential RandomCrop(int pad, long seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(pad);

        return Append(new CropWord(pad, seed));
    }

    /// <summary>Adds noise to every value while training: normal, of the spread it is told.</summary>
    /// <param name="spread">The standard deviation of the noise; nought for none.</param>
    /// <param name="seed">The number the noise is drawn from: the layer's own.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The spread is not a number of nought or more.</exception>
    /// <exception cref="InvalidOperationException">A layer that is no variation is written already: a variation stands first.</exception>
    public Sequential GaussianNoise(double spread, long seed) => Append(new NoiseWord(new GaussianNoise(spread, seed).Spread, seed));

    /// <summary>Normalises each feature — the last axis of an example — over the batch.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>As PyTorch leaves it: a momentum of a tenth, in PyTorch's meaning, and an epsilon of a hundred-thousandth.</remarks>
    public Sequential BatchNorm() => BatchNorm(default(Affine));

    /// <summary>Normalises each feature — the last axis of an example — over the batch, keeping the scale and the shift it is told to.</summary>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one — Keras's <c>scale</c> and <c>center</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential BatchNorm(Affine affine) => Append(new NormalisationWord(features => new BatchNorm(features, affine)));

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

        return Append(new NormalisationWord(features => new BatchNorm(features, affine) { Momentum = 1 - momentum, Epsilon = epsilon }));
    }

    /// <summary>Normalises each example over its features — the last axis.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <remarks>As PyTorch leaves it: an epsilon of a hundred-thousandth.</remarks>
    public Sequential LayerNorm() => LayerNorm(default(Affine));

    /// <summary>Normalises each example over its features — the last axis — keeping the scale and the shift it is told to.</summary>
    /// <param name="affine">Whether it scales and shifts what it normalised: both, unless it says it keeps neither or one — Keras's <c>scale</c> and <c>center</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential LayerNorm(Affine affine) => Append(new NormalisationWord(features => new LayerNorm(features, affine)));

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

        return Append(new NormalisationWord(features => new LayerNorm(features, affine) { Epsilon = epsilon }));
    }

    /// <summary>Lays each example out in another shape holding as many values: a row of pixels as an image, say.</summary>
    /// <param name="shape">The shape of one example, without the batch's axis.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">The shape holds no value.</exception>
    public Sequential Reshape(Shape shape) => Append(new ReshapeWord(global::DeepSharp.Networks.Reshape.RequireEach(shape)));

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

        return Append(new ConvolutionWord(
            "conv2d",
            "an image of rows, columns and channels",
            filters,
            new PlaneWalk(window),
            (channels, draws) => new Conv2D(new Channels { In = channels, Out = filters, Groups = groups }, window, beginning with { Draws = draws })));
    }

    /// <summary>Lays each example out as one row of its values.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    public Sequential Flatten() => Append(new FlattenWord());

    /// <summary>Gives the last word's step a name, so a later word can take what it made, and the numbers it learns are found under it.</summary>
    /// <param name="name">The name: some text with no dot or comma in it, which is no whole number and is not <c>input</c>.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">No word is written yet.</exception>
    /// <exception cref="ArgumentException">The name cannot stand in a path, or is a place, or is the input's.</exception>
    /// <remarks>A step called second holds second.weight, as the module a network written as code calls second does.</remarks>
    public Sequential Named(string name)
    {
        RequireStep();
        LayerStack.RequireName(name);

        _said[^1] = _said[^1] with { Name = name };

        return this;
    }

    /// <summary>Has the last word's step take the work of the named steps instead of the one before it.</summary>
    /// <param name="steps">The names of the steps — <c>input</c> for the rows the network is handed; the place of a step nobody named.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">No word is written yet.</exception>
    /// <exception cref="ArgumentException">No step is named, or one has no name.</exception>
    public Sequential Takes(params string[] steps)
    {
        RequireStep();
        RequireNames(steps);

        _said[^1] = _said[^1] with { Takes = [.. steps] };

        return this;
    }

    /// <summary>Adds up what the named steps made, value by value: a residual block's way of handing on what it was given beside what it made.</summary>
    /// <param name="steps">The names of the steps added up: two or more, <c>input</c> for the rows the network is handed.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">No step is named.</exception>
    public Sequential Add(params string[] steps)
    {
        RequireNames(steps);

        return Append(new JoinWord("add", shapes => shapes.Distinct().Count() == 1 ? null : $"takes steps of one shape, and these are {string.Join(" and ", shapes)}.", shapes => shapes[0], () => new Sum())).Takes(steps);
    }

    /// <summary>Lays what the named steps made side by side, along the last axis, in the order they are named.</summary>
    /// <param name="steps">The names of the steps laid side by side: two or more, <c>input</c> for the rows the network is handed.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentException">No step is named.</exception>
    public Sequential Concatenate(params string[] steps)
    {
        RequireNames(steps);

        return Append(new JoinWord("concatenate", AlikeButTheLast, Wider, () => new Concatenation())).Takes(steps);
    }

    /// <summary>Takes some of the columns of the last axis, as a step of its own: how a network of two inputs reads each from the row it is handed.</summary>
    /// <param name="first">The first column taken, counted from nought.</param>
    /// <param name="count">How many columns: one at least.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The first column is below nought, or no column is taken.</exception>
    public Sequential Columns(int first, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);

        return Append(new ColumnsWord(first, count));
    }

    /// <summary>A description run over every step of a series: the steps laid out as rows, the layers called once, the answers put back as the series.</summary>
    /// <param name="inner">The words of the layers every step is run through; held by this description as they stand.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentNullException">There is no description.</exception>
    /// <exception cref="InvalidOperationException">The description holds no word.</exception>
    /// <remarks>Keras's <c>TimeDistributed</c>. A dense layer or a normalisation works on the last axis and needs no wrapper; this is for layers that take a row. The numbers are found under <c>layer</c>.</remarks>
    public Sequential EveryStep(Sequential inner)
    {
        ArgumentNullException.ThrowIfNull(inner);

        return inner._words.Count == 0 ? throw new InvalidOperationException(NoLayers) : Append(new EveryStepWord(new Sequential(inner)));
    }

    /// <summary>A description inside this one, as one step: its words lowered at the shape the step is handed, under the step's name.</summary>
    /// <param name="inner">The words of the block; held by this description as they stand, a word written after this changes nothing.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="ArgumentNullException">There is no block.</exception>
    /// <exception cref="InvalidOperationException">The block holds no word.</exception>
    public Sequential Block(Sequential inner)
    {
        ArgumentNullException.ThrowIfNull(inner);

        if (inner._words.Count == 0)
        {
            throw new InvalidOperationException(NoLayers);
        }

        var held = new Sequential(inner);

        return Append(new BlockWord(_ => held));
    }

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
    public LayerStack Lower(Shape input, RandomStream stream) => Lowered(input, stream, string.Empty, null);

    /// <summary>Lowers a description that is a block onto a stack of layers for examples of the given shape and the mask of their padding.</summary>
    /// <param name="input">The shape of one example, without the batch's axis.</param>
    /// <param name="mask">The shape of the mask of one example: a one for each step that takes part, a nought for each that is padding.</param>
    /// <param name="stream">The stream the layers' starting values are drawn from: <c>initialise:</c> and the layer's place.</param>
    /// <returns>The layers, a stack that takes the mask as its second take and whose steps name it <see cref="Step.Mask"/>.</returns>
    /// <exception cref="ArgumentException">The shape holds no value, or an example of it, or the mask, cannot reach a layer.</exception>
    /// <exception cref="InvalidOperationException">The description holds no layer.</exception>
    public LayerStack Lower(Shape input, Shape mask, RandomStream stream) => Lowered(input, stream, string.Empty, mask);

    // The description lowered, each layer that learns starting from the draws of its place: the place alone, or the place of the block
    // it stands in and then its own, so a layer's start moves with nothing a name or another layer says.
    private LayerStack Lowered(Shape input, RandomStream stream, string within, Shape? mask)
    {
        ArgumentNullException.ThrowIfNull(stream);

        var steps = new List<Step>(_words.Count);

        foreach (var (place, visit) in Resolve(input, mask).Select((visit, place) => (place, visit)))
        {
            var purpose = string.Create(CultureInfo.InvariantCulture, $"initialise:{within}{place}");
            var layer = visit.Word switch
            {
                BlockWord block => block.Builds(visit.Inputs[0]).Lowered(visit.Inputs[0], stream, string.Create(CultureInfo.InvariantCulture, $"{within}{place}."), visit.Inputs.Length > 1 ? visit.Inputs[1] : null),
                EveryStepWord every => new EveryStep(every.Inner.Lowered(every.EachStep(visit.Inputs[0]), stream, string.Create(CultureInfo.InvariantCulture, $"{within}{place}."), null)),
                IJoinWord join => join.Make(visit.Inputs),
                var word => word.Make(visit.Inputs[0], stream.Draw(purpose, 0, 0)),
            };

            steps.Add(new Step(layer, visit.Said.Name, visit.Said.Takes));
        }

        return new LayerStack(steps);
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
    /// stated — and the run's seed, exactly as <see cref="Lower(Shape, RandomStream)"/> builds them from a stream of that seed; a last word whose
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

            // A description that names its steps or joins them is held to the rules of steps where it is compiled, as a stack is where it is
            // built: lowered once under a stream of its own, which nothing keeps, so a mistake is found before a fit is asked for.
            if (Branches)
            {
                _ = Lowered(stated, new RandomStream(0), string.Empty, null);
            }
        }

        return new CompiledNetwork(Lifted(loss), optimizer, loss, schedule);
    }

    // Refuses a shape the description cannot be lowered at: one holding no value, another than its input says, or one an
    // example of which cannot reach one of its words.
    private void Check(Shape input, Shape? mask = null) => Resolve(input, mask);

    // What each word takes and hands on for an example of the given shape: the shape of what each step it takes made, and of what it
    // makes, worked out down the description and refused where an example cannot reach a word.
    private List<Visited> Resolve(Shape input, Shape? mask = null)
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

        var shapes = new Dictionary<string, Shape>(StringComparer.Ordinal) { [Step.Input] = input };
        var visits = new List<Visited>(_words.Count);

        if (mask is { } maskShape)
        {
            shapes[Step.Mask] = maskShape;
        }

        for (var place = 0; place < _words.Count; place++)
        {
            var word = _words[place];
            var said = _said[place];
            var name = said.Name ?? place.ToString(CultureInfo.InvariantCulture);
            IReadOnlyList<string> takes = said.Takes ?? [place == 0 ? Step.Input : visits[place - 1].Name];

            if (takes.FirstOrDefault(taken => !shapes.ContainsKey(taken)) is { } unknown)
            {
                throw new ArgumentException(
                    string.Create(CultureInfo.InvariantCulture, $"This network cannot be lowered at {input}: word {place + 1} takes '{unknown}', which is not the input or a word before it."), nameof(input));
            }

            Shape[] inputs = [.. takes.Select(taken => shapes[taken])];
            var refusal = word switch
            {
                IJoinWord join => join.Refusal(inputs),
                IMaskedWord masked => masked.Refusal(inputs),
                _ when inputs.Length != 1 => string.Create(CultureInfo.InvariantCulture, $"takes the work of one step, and was given {inputs.Length}."),
                _ => word.Refusal(inputs[0]),
            };

            if (refusal is not null)
            {
                throw new ArgumentException(
                    string.Create(CultureInfo.InvariantCulture, $"This network cannot be lowered at {input}: word {place + 1}, {refusal}"), nameof(input));
            }

            var after = word switch
            {
                IJoinWord joined => joined.After(inputs),
                IMaskedWord masks => masks.After(inputs),
                _ => word.After(inputs[0]),
            };

            shapes[name] = after;
            visits.Add(new Visited(name, said, word, inputs, after));
        }

        return visits;
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

    // Whether any word names its step, takes other than the step before it, or is a join or a block: a description that is not a straight line.
    private bool Branches => _said.Any(said => said != default) || _words.Any(word => word is IJoinWord or BlockWord or EveryStepWord);

    private void RequireStep()
    {
        if (_words.Count == 0)
        {
            throw new InvalidOperationException("Write the word first: a name, or the steps taken, go to the step the last word makes.");
        }
    }

    private static void RequireNames(string[] steps)
    {
        ArgumentNullException.ThrowIfNull(steps);

        if (steps.Length == 0 || steps.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("Name the steps whose work is taken: one or more, each with a name.", nameof(steps));
        }
    }

    // How steps laid side by side must be alike, and how wide they are together.
    private static string? AlikeButTheLast(IReadOnlyList<Shape> shapes) =>
        shapes.Select(shape => shape.Rank).Distinct().Count() == 1
        && shapes[0].Rank >= 1
        && shapes.All(shape => shape.Axes[..^1].SequenceEqual(shapes[0].Axes[..^1]))
            ? null
            : $"lays steps side by side along the last axis, and they must be alike in every other: these are {string.Join(" and ", shapes)}.";

    private static Shape Wider(IReadOnlyList<Shape> shapes) =>
        new([.. shapes[0].Axes[..^1], shapes.Sum(shape => shape.Axes[^1])]);

    private Sequential Append(IWord word)
    {
        if (word is IVariationWord && _words.Exists(each => each is not IVariationWord))
        {
            throw new InvalidOperationException(
                "A variation is of the rows themselves, made on the values as the batch carries them: it stands first, before any layer that learns.");
        }

        _words.Add(word);
        _said.Add(default);

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
            lifted._said.RemoveAt(lifted._said.Count - 1);
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

        // It works on the last axis, so a series or an image keeps its places and makes Units of its channels at each of them.
        public string? Refusal(Shape each) => null;

        public Shape After(Shape each) => new([.. each.Axes[..^1].ToArray(), Units]);

        public Layer Make(Shape each, Draws draws) => new Dense(each[each.Rank - 1], Units, Begins with { Draws = draws });
    }

    // A word that reads a series and may be given, as a second take, the mask of which steps are steps of a series.
    private interface IMaskedWord : IWord
    {
        /// <summary>What is wrong with the work of the steps it takes — the series, and the mask when there is one; nothing when it can take them.</summary>
        string? Refusal(Shape[] inputs);

        /// <summary>The shape of one example the word hands on, for the work of the steps it takes: that of the first, unless the word says otherwise.</summary>
        Shape After(Shape[] inputs) => After(inputs[0]);
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

    /// <summary>What the description says of a word's step: the name it goes by and the steps it takes, when it says either.</summary>
    private readonly record struct Said(string? Name, IReadOnlyList<string>? Takes);

    /// <summary>A word with what it takes and hands on, worked out for one shape of example.</summary>
    private readonly record struct Visited(string Name, Said Said, IWord Word, Shape[] Inputs, Shape After);

    /// <summary>A word that takes the work of several steps: what it cannot take, what it hands on, and the layer it becomes.</summary>
    private interface IJoinWord : IWord
    {
        string? Refusal(IReadOnlyList<Shape> inputs);

        Shape After(IReadOnlyList<Shape> inputs);

        Layer Make(IReadOnlyList<Shape> inputs);
    }

    private sealed record JoinWord(string Word, Func<IReadOnlyList<Shape>, string?> Refuses, Func<IReadOnlyList<Shape>, Shape> Hands, Func<Layer> Build) : IJoinWord
    {
        public string? Refusal(IReadOnlyList<Shape> inputs) => inputs.Count >= 2 ? Refuses(inputs) is { } refusal ? $"{Word}, {refusal}" : null : $"{Word}, takes the work of two steps or more, and was given {inputs.Count}.";

        public Shape After(IReadOnlyList<Shape> inputs) => Hands(inputs);

        public Layer Make(IReadOnlyList<Shape> inputs) => Build();

        string? IWord.Refusal(Shape each) => throw new InvalidOperationException("A join takes the work of several steps.");

        Shape IWord.After(Shape each) => throw new InvalidOperationException("A join takes the work of several steps.");

        Layer IWord.Make(Shape each, Draws draws) => throw new InvalidOperationException("A join takes the work of several steps.");
    }

    private sealed record ColumnsWord(int First, int Count) : IWord
    {
        public string? Refusal(Shape each) =>
            each.Rank >= 1 && each[each.Rank - 1] >= First + Count
                ? null
                : string.Create(CultureInfo.InvariantCulture, $"columns, takes {Count} columns from column {First}, and each example reaching it is {each}, whose last axis is not as long as {First + Count}.");

        public Shape After(Shape each) => new([.. each.Axes[..^1], Count]);

        public Layer Make(Shape each, Draws draws) => new Columns(First, Count);
    }

    /// <summary>A description run over every step of a series: whole examples laid out as rows, the description lowered at the shape of one step.</summary>
    private sealed record EveryStepWord(Sequential Inner) : IWord
    {
        public Shape EachStep(Shape each) => new(each.Axes[1..].ToArray());

        public string? Refusal(Shape each)
        {
            if (each.Rank < 2)
            {
                return $"everyStep, takes each example as a series of steps, and each reaching it is {each}.";
            }

            try
            {
                Inner.Check(EachStep(each));

                return null;
            }
            catch (ArgumentException refused)
            {
                return $"everyStep, {refused.Message}";
            }
        }

        public Shape After(Shape each) => new([each[0], .. Inner.Resolve(EachStep(each))[^1].After.Axes.ToArray()]);

        public Layer Make(Shape each, Draws draws) => throw new InvalidOperationException("A layer run over every step is lowered with the stream of its description, at its own place.");
    }

    private sealed record BlockWord(Func<Shape, Sequential> Builds) : IMaskedWord
    {
        public string? Refusal(Shape each) => Refusal([each]);

        public string? Refusal(Shape[] inputs)
        {
            if (inputs.Length > 2)
            {
                return $"block, takes a series and, besides, the mask of its steps, and was given the work of {inputs.Length} steps.";
            }

            try
            {
                Builds(inputs[0]).Check(inputs[0], inputs.Length == 2 ? inputs[1] : null);

                return null;
            }
            catch (ArgumentException refused)
            {
                return $"block, {refused.Message}";
            }
        }

        public Shape After(Shape each) => Builds(each).Resolve(each)[^1].After;

        public Shape After(Shape[] inputs) => Builds(inputs[0]).Resolve(inputs[0], inputs.Length == 2 ? inputs[1] : null)[^1].After;

        public Layer Make(Shape each, Draws draws) => throw new InvalidOperationException("A block is lowered with the stream of its description, at its own place.");
    }
}
