// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Declaring a network in the chain that prepares its rows, in either of the two vocabularies.
/// </summary>
/// <remarks>
/// The course from raw data to a validated model is one artefact: the steps that prepare the rows, and then the network
/// those rows train. Both doors write the same declaration — the words TensorFlow and Keras use and the words PyTorch
/// uses lower onto one model, and both run on whichever engine the declaration names. Where a caller leaves a number
/// unsaid, each door fills in the number its own library leaves there, and the declaration writes it down, so the file
/// says which run was asked for rather than which library asked for it.
/// </remarks>
public static class NetworkChainExtensions
{
    extension(FittingBuilder fitting)
    {
        /// <summary>Declares the network this pipeline trains, in the words PyTorch uses.</summary>
        /// <param name="network">The network: its layers, what moves them, what judges them, and how the run goes.</param>
        /// <returns>The chain, with the network declared.</returns>
        /// <exception cref="ArgumentNullException">There is no chain, or nothing describes the network.</exception>
        /// <exception cref="DeclarationException">The declaration names no answer yet, or already names a learner.</exception>
        public FittingBuilder WithTorch(Func<NetworkDeclaration, NetworkDeclaration> network)
        {
            ArgumentNullException.ThrowIfNull(fitting);
            ArgumentNullException.ThrowIfNull(network);

            return fitting.Add(network(new NetworkDeclaration(Vocabularies.Torch)).Step());
        }

        /// <summary>Declares the network this pipeline trains, in the words TensorFlow and Keras use.</summary>
        /// <param name="network">The network: its layers, what moves them, what judges them, and how the run goes.</param>
        /// <returns>The chain, with the network declared.</returns>
        /// <exception cref="ArgumentNullException">There is no chain, or nothing describes the network.</exception>
        /// <exception cref="DeclarationException">The declaration names no answer yet, or already names a learner.</exception>
        public FittingBuilder WithTensorflow(Func<NetworkDeclaration, NetworkDeclaration> network)
        {
            ArgumentNullException.ThrowIfNull(fitting);
            ArgumentNullException.ThrowIfNull(network);

            return fitting.Add(network(new NetworkDeclaration(Vocabularies.Keras)).Step());
        }
    }
}

/// <summary>Which library's words a network is declared in, and whose numbers fill in what a caller leaves unsaid.</summary>
public enum Vocabularies
{
    /// <summary>PyTorch's: Adam's epsilon a hundred-millionth, as <c>torch.optim.Adam</c> leaves it.</summary>
    Torch,

    /// <summary>TensorFlow's and Keras's: Adam's epsilon a ten-millionth, as <c>keras.optimizers.Adam</c> leaves it.</summary>
    Keras,
}

/// <summary>
/// A network written out word by word: add the layers, say what moves them and what judges them, say how the run goes.
/// </summary>
/// <remarks>
/// The shape both libraries already use — stack the layers, compile the model, fit it to the data — written as one
/// chain, and it ends as a step in the pipeline's own declaration. Every number is written down, the ones a caller
/// names and the ones their library leaves unsaid alike.
/// </remarks>
public sealed class NetworkDeclaration
{
    // What PyTorch adds to the mean of the squares of a root-mean-square normalisation unless told otherwise.
    private const double RmsNormSingleStep = 1.1920928955078125e-07;

    private readonly List<PartDeclaration> _layers = [];
    private readonly Vocabularies _words;

    private PartDeclaration _optimizer;
    private PartDeclaration _loss;
    private PartDeclaration _stopping;
    private PartDeclaration _schedule = NetworkWords.Schedules[0].Declared();
    private double _clip;
    private double _clipValue;
    private int _stepEvery = 1;
    private double _average;
    private bool _judgedByAverage;
    private double? _decay;
    private double? _dampening;
    private bool _nesterov;
    private bool _amsgrad;
    private bool _centered;
    private string _engine = LearnNetworkStep.LightEngine;
    private int _seed = 20260929;
    private int _epochs = 100;
    private int _batch = 32;

    internal NetworkDeclaration(Vocabularies words)
    {
        _words = words;
        _optimizer = default;
        _loss = default;
        _stopping = NetworkWords.Stopping[0].Declared();
    }

    /// <summary>A layer that answers with this many numbers, each a weighted sum of the ones below it, plus a bias of its own.</summary>
    /// <param name="units">How many numbers it answers with.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>The form 0.9.0 had, kept so that a program built against it still finds it; the one below says the rest.</remarks>
    public NetworkDeclaration Dense(int units) => Dense(units, bias: true);

    /// <summary>A layer that answers with this many numbers, each a weighted sum of the ones below it.</summary>
    /// <param name="units">How many numbers it answers with.</param>
    /// <param name="bias">Whether the layer adds a bias of its own to every sum; it does, unless said.</param>
    /// <param name="weights">How the weights start; as PyTorch starts a linear layer's, unless said.</param>
    /// <param name="biasStart">How the bias starts, when there is one; as PyTorch starts it, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentException">The layer has no bias and says how its bias starts.</exception>
    /// <remarks>
    /// Saying none of the three is the layer this method always declared, written as it always was. Keras starts its dense
    /// layers at Glorot's weights and a bias of nought; neither vocabulary changes what is left unsaid, so a declaration that
    /// wants Keras's start says so: <c>weights: InitialiserWord.GlorotUniform, biasStart: InitialiserWord.Zeros</c>.
    /// </remarks>
    public NetworkDeclaration Dense(int units, bool bias = true, InitialiserWord? weights = null, InitialiserWord? biasStart = null)
    {
        var layer = Kind("dense").Declared().With("units", PartValue.Of(units)).With("bias", PartValue.Of(bias));

        if (weights is { } startWeights)
        {
            layer = layer.With("weights", PartValue.Of(startWeights.Word()));
        }

        if (biasStart is { } startBias)
        {
            layer = layer.With("biasStart", PartValue.Of(startBias.Word()));
        }

        return Layer(layer);
    }

    /// <summary>Adds a share of the sum of the absolute values of the weights of the layer written last to the loss, as Keras's L1 regularizer does.</summary>
    /// <param name="penalty">The share: nought or more.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last has no weights to penalise.</exception>
    /// <remarks>The reported loss includes it, in training and in validation, and only for numbers that learn.</remarks>
    public NetworkDeclaration L1(double penalty) => OnTheLastLayer("L1", "l1", new SlotRules { L1 = penalty }.L1);

    /// <summary>Adds a share of the sum of the squares of the weights of the layer written last to the loss, as Keras's L2 regularizer does.</summary>
    /// <param name="penalty">The share: nought or more.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a finite number.</exception>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last has no weights to penalise.</exception>
    /// <remarks>
    /// Keras's L2 has no half in it, and neither has this: a share of the squares themselves, which is not the weight decay an
    /// optimizer adds to the gradient — that one takes a half of it, so the two are the same number twice over.
    /// </remarks>
    public NetworkDeclaration L2(double penalty) => OnTheLastLayer("L2", "l2", new SlotRules { L2 = penalty }.L2);

    /// <summary>Holds the numbers of each unit of the layer written last under a norm after every step, as Keras's MaxNorm does.</summary>
    /// <param name="bound">The most norm a unit's numbers may have: above nothing.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last has no weights to bound.</exception>
    public NetworkDeclaration MaxNorm(double bound) => OnTheLastLayer("MaxNorm", "maxNorm", new SlotRules { MaxNorm = bound }.MaxNorm!.Value);

    /// <summary>Holds the numbers of the layer written last still, as Keras's trainable = False does: no step moves them.</summary>
    /// <returns>The declaration.</returns>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last has no numbers to hold.</exception>
    public NetworkDeclaration Held() => OnTheLastLayer("Held", "learns", false);

    /// <summary>Gives the numbers of the layer written last a share of the run's rate to move at: a learning rate of their own for a part of the network.</summary>
    /// <param name="share">The share of the rate: above nothing; one is the whole rate, as a layer that says none has.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last has no numbers to give a share.</exception>
    /// <remarks>
    /// PyTorch's learning rate of a parameter group, said of a layer, so a network read from elsewhere can be trained further with
    /// its first layers moving slower than its last. A step moves the layer's numbers by the rate the schedule gives the epoch
    /// times the share. To hold numbers still, say <see cref="Held"/>: a share of nothing is refused.
    /// </remarks>
    public NetworkDeclaration RateShare(double share) => OnTheLastLayer("RateShare", "rateShare", new SlotRules { RateShare = share }.RateShare!.Value);

    /// <summary>Keeps a low-rank pair beside the held kernel of the dense layer written last, as Keras's <c>enable_lora</c> does.</summary>
    /// <param name="rank">The rank of the pair: at least one.</param>
    /// <param name="alpha">What the pair's product is scaled by over the rank; the rank itself, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The rank is below one, or the alpha is not a finite number above nothing.</exception>
    /// <exception cref="InvalidOperationException">No layer is written yet.</exception>
    /// <exception cref="NotSupportedException">The layer written last is no dense layer.</exception>
    /// <remarks>
    /// The kernel is held still and the pair and the bias learn; the layer answers with the kernel's product plus the pair's,
    /// scaled by alpha over the rank. The network's file then names the seventh version of the network's part.
    /// </remarks>
    public NetworkDeclaration LowRank(int rank, double? alpha = null)
    {
        _ = new LowRank(rank, alpha);

        var ranked = OnTheLastLayer("LowRank", "rank", (double)rank);

        return alpha is { } scaled ? ranked.OnTheLastLayer("LowRank", "alpha", scaled) : ranked;
    }

    /// <summary>A layer that leaves what is below nought at nought.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Relu() => Layer("relu");

    /// <summary>A layer that squashes every number between minus one and one.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Tanh() => Layer("tanh");

    /// <summary>A layer that squashes every number between nought and one.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Sigmoid() => Layer("sigmoid");

    /// <summary>A layer that leaves a share of the numbers out at every pass while training.</summary>
    /// <param name="rate">The share.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Dropout(double rate) => Layer("dropout", new PartSetting("rate", PartValue.Of(rate)));

    /// <summary>A variation that mirrors images at random while the network trains, and not while it answers.</summary>
    /// <param name="seed">The number the flips are drawn from: the layer's own, which the run's seed moves nothing of.</param>
    /// <param name="horizontal">Whether an image is mirrored across its columns.</param>
    /// <param name="vertical">Whether an image is mirrored across its rows.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>It varies the rows themselves, so it stands first, before any layer that learns.</remarks>
    public NetworkDeclaration RandomFlip(int seed, bool horizontal = true, bool vertical = false) => Layer(
        "randomFlip",
        new PartSetting("horizontal", PartValue.Of(horizontal)),
        new PartSetting("vertical", PartValue.Of(vertical)),
        new PartSetting("seed", PartValue.Of(seed)));

    /// <summary>A variation that shifts images at random while the network trains: each padded and a window of its own size cut out.</summary>
    /// <param name="pad">How many places an image is padded by on every side, and so how far it can move.</param>
    /// <param name="seed">The number the shifts are drawn from: the layer's own.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration RandomCrop(int pad, int seed) => Layer(
        "randomCrop", new PartSetting("pad", PartValue.Of(pad)), new PartSetting("seed", PartValue.Of(seed)));

    /// <summary>A variation that adds normal noise to every number while the network trains.</summary>
    /// <param name="spread">The standard deviation of the noise.</param>
    /// <param name="seed">The number the noise is drawn from: the layer's own.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration GaussianNoise(double spread, int seed) => Layer(
        "gaussianNoise", new PartSetting("spread", PartValue.Of(spread)), new PartSetting("seed", PartValue.Of(seed)));

    /// <summary>A layer that lays an example out as one row of numbers: an image as the row of its pixels.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Flatten() => Layer("flatten");

    /// <summary>A layer that brings every number to the mean and spread of the rows it was trained on.</summary>
    /// <param name="momentum">How much of what was measured before carries on; each library's own, unless said.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>The form 0.9.0 had, kept so that a program built against it still finds it; it keeps a scale and a shift.</remarks>
    public NetworkDeclaration BatchNorm(double? momentum = null, double? epsilon = null) => BatchNorm(momentum, epsilon, scale: true);

    /// <summary>A layer that brings every number to the mean and spread of the rows it was trained on, with the libraries' own momentum and epsilon.</summary>
    /// <param name="scale">Whether it multiplies what it normalised by a scale of its own.</param>
    /// <param name="shift">Whether it adds a shift of its own; it does, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BatchNorm(bool scale, bool shift = true) => BatchNorm(null, null, scale, shift);

    /// <summary>A layer that brings every number to the mean and spread of the rows it was trained on.</summary>
    /// <param name="momentum">How much of what was measured before carries on; each library's own, unless said.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <param name="scale">Whether it multiplies what it normalised by a scale of its own; it does, unless said.</param>
    /// <param name="shift">Whether it adds a shift of its own; it does, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BatchNorm(double? momentum, double? epsilon, bool scale, bool shift = true) => Layer(
        Kind("batchNorm").Declared()
            .With("momentum", PartValue.Of(momentum ?? (_words == Vocabularies.Keras ? 0.99 : 0.9)))
            .With("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-3 : 1e-5)))
            .With("scale", PartValue.Of(scale))
            .With("shift", PartValue.Of(shift)));

    /// <summary>A layer that brings every number of one row to that row's own mean and spread.</summary>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>The form 0.9.0 had, kept so that a program built against it still finds it; it keeps a scale and a shift.</remarks>
    public NetworkDeclaration LayerNorm(double? epsilon = null) => LayerNorm(epsilon, scale: true);

    /// <summary>A layer that brings every number of one row to that row's own mean and spread, with the libraries' own epsilon.</summary>
    /// <param name="scale">Whether it multiplies what it normalised by a scale of its own.</param>
    /// <param name="shift">Whether it adds a shift of its own; it does, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration LayerNorm(bool scale, bool shift = true) => LayerNorm(null, scale, shift);

    /// <summary>A layer that brings every number of one row to that row's own mean and spread.</summary>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <param name="scale">Whether it multiplies what it normalised by a scale of its own; it does, unless said.</param>
    /// <param name="shift">Whether it adds a shift of its own; it does, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration LayerNorm(double? epsilon, bool scale, bool shift = true) => Layer(
        Kind("layerNorm").Declared()
            .With("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-3 : 1e-5)))
            .With("scale", PartValue.Of(scale))
            .With("shift", PartValue.Of(shift)));

    /// <summary>A layer that weighs every number by the share of a normal curve below it: the GELU.</summary>
    /// <param name="approximate">Whether it is the hyperbolic tangent's approximation; exact, as both libraries leave it, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Gelu(bool approximate = false) => Layer("gelu", new PartSetting("approximate", PartValue.Of(approximate)));

    /// <summary>A layer that weighs every number by its own sigmoid: the SiLU, also called swish.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Silu() => Layer("silu");

    /// <summary>A layer that weighs every number by the hyperbolic tangent of its softplus: mish.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Mish() => Layer("mish");

    /// <summary>A layer that leaves what is above nought and lets a share of what is below it through.</summary>
    /// <param name="negativeSlope">The share; each library's own, unless said: a hundredth in PyTorch, three tenths in Keras's layer.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration LeakyRelu(double? negativeSlope = null) =>
        Layer("leakyRelu", new PartSetting("negativeSlope", PartValue.Of(negativeSlope ?? (_words == Vocabularies.Keras ? 0.3 : 0.01))));

    /// <summary>A layer that leaves what is above nought and lets a share of what is below it through that it learns.</summary>
    /// <param name="eachFeature">
    /// Whether every number of the row learns a share of its own; each library's own, unless said: one share for them all in
    /// PyTorch, one for each in Keras.
    /// </param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration PRelu(bool? eachFeature = null) =>
        Layer("prelu", new PartSetting("eachFeature", PartValue.Of(eachFeature ?? _words == Vocabularies.Keras)));

    /// <summary>A layer that leaves what is above nought and bends what is below it towards minus alpha: the ELU.</summary>
    /// <param name="alpha">What it bends towards, below nought; one, as both libraries leave it, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Elu(double alpha = 1) => Layer("elu", new PartSetting("alpha", PartValue.Of(alpha)));

    /// <summary>A layer that bends as the ELU does at the alpha and the scale that keep a network's numbers normalised: the SELU.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Selu() => Layer("selu");

    /// <summary>A layer that turns the numbers of a row into shares of one whole.</summary>
    /// <returns>The declaration.</returns>
    /// <remarks>Last, before a cross-entropy, it is lifted into the loss, which applies the softmax itself.</remarks>
    public NetworkDeclaration Softmax() => Layer("softmax");

    /// <summary>A layer that divides every number of one row by the root of the mean of the row's squares, and scales it.</summary>
    /// <param name="epsilon">The small number added to the mean of the squares; each library's own, unless said.</param>
    /// <param name="scale">Whether it scales what it normalised by a scale of its own; it does, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration RmsNorm(double? epsilon = null, bool scale = true) => Layer(
        Kind("rmsNorm").Declared()
            .With("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-6 : RmsNormSingleStep)))
            .With("scale", PartValue.Of(scale)));

    /// <summary>
    /// A layer that turns every number of the row into its cosines and sines at frequencies it learns, through a linear layer:
    /// Gorishniy et al.'s periodic embedding, their PLR once a <see cref="Relu"/> follows it.
    /// </summary>
    /// <param name="sigma">The spread of the normal curve the frequencies start from: said every time, since the right one depends on the data.</param>
    /// <param name="frequencies">How many frequencies each number has; forty-eight, as their package leaves it, unless said.</param>
    /// <param name="width">How many numbers each number comes out as; twenty-four, as their package leaves it, unless said.</param>
    /// <param name="lite">Whether one linear layer is shared by every number of the row; it is not, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Periodic(double sigma, int frequencies = 48, int width = 24, bool lite = false) => Layer(
        "periodic",
        new PartSetting("frequencies", PartValue.Of(frequencies)),
        new PartSetting("sigma", PartValue.Of(sigma)),
        new PartSetting("width", PartValue.Of(width)),
        new PartSetting("lite", PartValue.Of(lite)));

    /// <summary>
    /// A layer that turns the columns a category was written down as into a small vector it learns for each category: an
    /// embedding, over the block the pipeline's encoder wrote, its own vector for a category the training rows never held and for
    /// a gap.
    /// </summary>
    /// <param name="column">The column of words, as it was named before the pipeline wrote it down one column per category.</param>
    /// <param name="width">How many numbers each category's vector holds.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>It reads the rows as the pipeline hands them over, so every embedding stands before the first layer that is not one.</remarks>
    public NetworkDeclaration Embedding(string column, int width) => Layer(
        "embedding",
        new PartSetting("column", PartValue.Of(column)),
        new PartSetting("width", PartValue.Of(width)));

    /// <summary>Moves the network's numbers against their gradients.</summary>
    /// <param name="rate">How far one step moves them; each library's own, unless said.</param>
    /// <param name="momentum">How much of the last step's velocity carries on; none, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Sgd(double? rate = null, double momentum = 0)
    {
        _optimizer = new PartDeclaration("sgd", [
            new PartSetting("rate", PartValue.Of(rate ?? (_words == Vocabularies.Keras ? 0.01 : 0.001))),
            new PartSetting("momentum", PartValue.Of(momentum)),
        ]);

        return this;
    }

    /// <summary>Moves the network's numbers by a running mean of their gradients over the root of a running mean of their squares.</summary>
    /// <param name="rate">How far one step moves them; a thousandth, unless said, as both libraries leave it.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Adam(double rate = 0.001, double? epsilon = null)
    {
        _optimizer = new PartDeclaration("adam", [
            new PartSetting("rate", PartValue.Of(rate)),
            new PartSetting("firstMoment", PartValue.Of(0.9)),
            new PartSetting("secondMoment", PartValue.Of(0.999)),
            new PartSetting("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-7 : 1e-8))),
        ]);

        return this;
    }

    /// <summary>
    /// Moves the network's numbers as Adam does, every number shrunk towards nought before its step, apart from its gradient:
    /// PyTorch's AdamW.
    /// </summary>
    /// <param name="rate">How far one step moves them; a thousandth, unless said.</param>
    /// <param name="weightDecay">How far every number is shrunk at each step, as a share of the step's rate; a hundredth, unless said.</param>
    /// <param name="betas">How much of each running mean carries on; nine tenths and 0.999, unless said.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; a hundred-millionth, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>
    /// The numbers left unsaid are PyTorch's in both vocabularies. Keras's AdamW decays the same way and leaves a decay of
    /// 0.004 and an epsilon of a ten-millionth unsaid; a declaration in Keras's words that wants those names them.
    /// </remarks>
    public NetworkDeclaration AdamW(double rate = 0.001, double weightDecay = 0.01, Betas? betas = null, double epsilon = 1e-8)
    {
        var said = betas ?? new Betas(0.9, 0.999);

        _optimizer = new PartDeclaration("adamw", [
            new PartSetting("rate", PartValue.Of(rate)),
            new PartSetting("firstMoment", PartValue.Of(said.First)),
            new PartSetting("secondMoment", PartValue.Of(said.Second)),
            new PartSetting("epsilon", PartValue.Of(epsilon)),
            new PartSetting("weightDecay", PartValue.Of(weightDecay)),
        ]);

        return this;
    }

    /// <summary>
    /// Moves the network's numbers by their gradients over the root of a running mean of the squares of their gradients:
    /// PyTorch's RMSprop. Declared in PyTorch's words only.
    /// </summary>
    /// <param name="rate">How far one step moves them; a hundredth, unless said.</param>
    /// <param name="alpha">How much of the running mean of the squares carries on; 0.99, unless said.</param>
    /// <param name="momentum">How much of the last step carries into the next; none, unless said.</param>
    /// <param name="epsilon">The small number added to the root so nothing is divided by nought; a hundred-millionth, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">
    /// The network is declared in Keras's words: Keras's RMSprop adds its epsilon under the root and carries the rate in its
    /// momentum, so its words would name an optimizer this library does not run.
    /// </exception>
    public NetworkDeclaration RmsProp(double rate = 0.01, double alpha = 0.99, double momentum = 0, double epsilon = 1e-8)
    {
        RequirePyTorchsWords("RMSprop", "RmsProp");

        _optimizer = new PartDeclaration("rmsprop", [
            new PartSetting("rate", PartValue.Of(rate)),
            new PartSetting("alpha", PartValue.Of(alpha)),
            new PartSetting("epsilon", PartValue.Of(epsilon)),
            new PartSetting("momentum", PartValue.Of(momentum)),
        ]);

        return this;
    }

    /// <summary>
    /// Moves the network's numbers as Adam does, with Nesterov's momentum warmed up over the steps: PyTorch's NAdam. Declared
    /// in PyTorch's words only.
    /// </summary>
    /// <param name="rate">How far one step moves them; two thousandths, unless said.</param>
    /// <param name="betas">How much of each running mean carries on; nine tenths and 0.999, unless said.</param>
    /// <param name="momentumDecay">How fast the momentum warms up over the steps; four thousandths, unless said.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; a hundred-millionth, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">
    /// The network is declared in Keras's words: Keras's Nadam holds its momentum's decay fixed and leaves other numbers
    /// unsaid, so its words would name an optimizer this library does not run.
    /// </exception>
    public NetworkDeclaration Nadam(double rate = 0.002, Betas? betas = null, double momentumDecay = 0.004, double epsilon = 1e-8)
    {
        RequirePyTorchsWords("Nadam", "Nadam");

        var said = betas ?? new Betas(0.9, 0.999);

        _optimizer = new PartDeclaration("nadam", [
            new PartSetting("rate", PartValue.Of(rate)),
            new PartSetting("firstMoment", PartValue.Of(said.First)),
            new PartSetting("secondMoment", PartValue.Of(said.Second)),
            new PartSetting("epsilon", PartValue.Of(epsilon)),
            new PartSetting("momentumDecay", PartValue.Of(momentumDecay)),
        ]);

        return this;
    }

    /// <summary>
    /// Adds a share of every number to its gradient before the step moves it, as PyTorch's <c>weight_decay</c> does for SGD, Adam,
    /// RMSprop and NAdam. Declared in PyTorch's words only.
    /// </summary>
    /// <param name="decay">The share: nought or more.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">The network is declared in Keras's words, where the weight decay is decoupled from the gradient.</exception>
    /// <remarks>Said before or after the optimizer; it is held until the step is made, and refused then by an optimizer that has none (AdamW has its own).</remarks>
    public NetworkDeclaration WeightDecay(double decay)
    {
        RequirePyTorchsOption("WeightDecay", "adds a share of every number to its gradient, which Keras's weight_decay does not: it decays the number itself");
        _decay = decay;

        return this;
    }

    /// <summary>Lets the velocity take in less of each new gradient, but not the first: PyTorch's SGD <c>dampening</c>. Declared in PyTorch's words only.</summary>
    /// <param name="dampening">The share left out: from nothing to one.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">The network is declared in Keras's words, where SGD has no dampening.</exception>
    public NetworkDeclaration Dampening(double dampening)
    {
        RequirePyTorchsOption("Dampening", "lets the velocity take in less of each new gradient, which Keras's SGD has no word for");
        _dampening = dampening;

        return this;
    }

    /// <summary>Takes the step from the gradient plus the momentum times the velocity, as Nesterov has it: PyTorch's SGD <c>nesterov</c>. Declared in PyTorch's words only.</summary>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">The network is declared in Keras's words.</exception>
    /// <remarks>It needs a momentum above nothing and no dampening, and the step that is made refuses it without.</remarks>
    public NetworkDeclaration Nesterov()
    {
        RequirePyTorchsOption("Nesterov", "takes the step from the gradient plus the momentum times the velocity");
        _nesterov = true;

        return this;
    }

    /// <summary>Takes the step over the largest mean of the squares there has been, as AMSGrad has it: PyTorch's <c>amsgrad</c> for Adam and AdamW. Declared in PyTorch's words only.</summary>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">The network is declared in Keras's words, whose AMSGrad places its epsilon otherwise.</exception>
    public NetworkDeclaration AmsGrad()
    {
        RequirePyTorchsOption("AmsGrad", "takes the step over the largest mean of the squares there has been, which Keras's Adam works out with its epsilon placed otherwise");
        _amsgrad = true;

        return this;
    }

    /// <summary>Takes the root of the mean of the squares less the square of the mean of the gradients: PyTorch's RMSprop <c>centered</c>. Declared in PyTorch's words only.</summary>
    /// <returns>The declaration.</returns>
    /// <exception cref="NotSupportedException">The network is declared in Keras's words.</exception>
    public NetworkDeclaration Centered()
    {
        RequirePyTorchsOption("Centered", "takes the root of the mean of the squares less the square of the mean of the gradients");
        _centered = true;

        return this;
    }

    /// <summary>Judges the answers by the mean of their squared differences: an answer that is an amount.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration MeanSquaredError() => Judged("meanSquaredError");

    /// <summary>Judges one answer among several classes, read through a softmax.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration CrossEntropy() => Judged("crossEntropy");

    /// <summary>Judges an answer that is a chance, read through a sigmoid.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BinaryCrossEntropy() => Judged("binaryCrossEntropy");

    /// <summary>Judges an answer of shares in an order by how far its shares have to move along the order, read through a softmax.</summary>
    /// <returns>The declaration.</returns>
    /// <remarks>Only an output whose answers are in an order can mean it: a distribution said to be ordered.</remarks>
    public NetworkDeclaration EarthMoversDistance() => Judged("earthMoversDistance");

    /// <summary>Judges one answer among several classes, read through a softmax, against answers softened by a share spread evenly over the classes.</summary>
    /// <param name="smoothing">The share: nought or more, below one.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration CrossEntropy(double smoothing) => Judged("crossEntropy", Smoothed(smoothing));

    /// <summary>Judges an answer that is a chance, read through a sigmoid, against answers moved a share of the way towards a half.</summary>
    /// <param name="smoothing">The share: nought or more, below one.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BinaryCrossEntropy(double smoothing) => Judged("binaryCrossEntropy", Smoothed(smoothing));

    /// <summary>Judges the answers by the mean of the size of their differences: an amount, some far off.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration MeanAbsoluteError() => Judged("meanAbsoluteError");

    /// <summary>Judges the answers by half the square of the difference near them and its size further off: an amount with outliers.</summary>
    /// <param name="delta">How far from the answer the loss stops being squared; one, as both libraries leave it, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Huber(double delta = 1) => Judged("huber", new PartSetting("delta", PartValue.Of(delta)));

    /// <summary>Judges the answers by the pinball loss of a quantile: the value the answer stays below that share of the time.</summary>
    /// <param name="quantile">The share: above nought and below one.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Quantile(double quantile) => Judged("quantile", new PartSetting("quantile", PartValue.Of(quantile)));

    /// <summary>Judges labels by a binary cross-entropy that weighs down the ones the network is already sure of, read through a sigmoid: a rare class.</summary>
    /// <param name="gamma">How hard a sure label is weighed down; two, as both libraries leave it, unless said.</param>
    /// <param name="alpha">
    /// What a one is weighed by, and a nought by one less it, or minus one for no weight, as torchvision writes it; each library's own,
    /// unless said: a quarter in torchvision, none in Keras.
    /// </param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Focal(double gamma = 2, double? alpha = null) => Judged(
        "focal",
        new PartSetting("gamma", PartValue.Of(gamma)),
        new PartSetting("alpha", PartValue.Of(alpha ?? (_words == Vocabularies.Keras ? global::DeepSharp.Networks.Focal.NoWeight : 0.25))));

    /// <summary>Judges labels by how far each answer falls short of a margin of one on its side of nought, as Keras's hinge does; read by the side it falls.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Hinge() => Judged("hinge");

    /// <summary>Judges counts by the negative log-likelihood of a Poisson count.</summary>
    /// <param name="logInput">
    /// Whether the network answers with the logarithm of each count's rate; each library's own, unless said: the logarithm in
    /// PyTorch, the rate itself in Keras.
    /// </param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Poisson(bool? logInput = null) => Judged(
        "poisson",
        new PartSetting("logInput", PartValue.Of(logInput ?? _words != Vocabularies.Keras)),
        new PartSetting("epsilon", PartValue.Of(_words == Vocabularies.Keras ? 1e-7 : 1e-8)));

    /// <summary>How the run goes: the number every draw is worked out from, how many passes, and how many rows a step.</summary>
    /// <param name="seed">The number every random draw of the run is worked out from.</param>
    /// <param name="epochs">How many times the run goes over the training rows, at most; a hundred, unless said.</param>
    /// <param name="batch">How many rows one step is worked out from; thirty-two, unless said, as both libraries leave it.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Run(int seed, int epochs = 100, int batch = 32)
    {
        _seed = seed;
        _epochs = epochs;
        _batch = batch;

        return this;
    }

    /// <summary>Stops the run once the validation loss has not fallen for as many passes as it waits.</summary>
    /// <param name="patience">How many passes without a better one it waits.</param>
    /// <param name="least">How far the loss has to fall below the best for a pass to count as better; nothing, unless said.</param>
    /// <param name="restoreBest">Whether the run ends holding the numbers of its best pass; it does, unless said otherwise.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration StoppingAfter(int patience, double least = 0, bool restoreBest = true)
    {
        _stopping = new PartDeclaration("patience", [
            new PartSetting("patience", PartValue.Of(patience)),
            new PartSetting("least", PartValue.Of(least)),
            new PartSetting("best", PartValue.Of(restoreBest)),
        ]);

        return this;
    }

    /// <summary>Multiplies the rate by a factor every so many epochs: PyTorch's StepLR.</summary>
    /// <param name="every">How many epochs pass between two falls.</param>
    /// <param name="factor">What the rate is multiplied by at each fall; a tenth, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>One schedule a declaration: the last that is said is the one written.</remarks>
    public NetworkDeclaration StepDecay(int every, double factor = 0.1) =>
        Scheduled("stepDecay", new PartSetting("every", PartValue.Of(every)), new PartSetting("factor", PartValue.Of(factor)));

    /// <summary>Multiplies the rate by a factor every epoch: PyTorch's ExponentialLR.</summary>
    /// <param name="factor">What the rate is multiplied by from one epoch to the next.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>One schedule a declaration: the last that is said is the one written.</remarks>
    public NetworkDeclaration ExponentialDecay(double factor) => Scheduled("exponentialDecay", new PartSetting("factor", PartValue.Of(factor)));

    /// <summary>Lowers the rate along half a cosine to its least over so many epochs: PyTorch's CosineAnnealingLR.</summary>
    /// <param name="epochs">How many epochs the fall takes.</param>
    /// <param name="minimum">The least rate, reached at the last of them; nought, unless said.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>One schedule a declaration: the last that is said is the one written.</remarks>
    public NetworkDeclaration CosineDecay(int epochs, double minimum = 0) =>
        Scheduled("cosineDecay", new PartSetting("epochs", PartValue.Of(epochs)), new PartSetting("minimum", PartValue.Of(minimum)));

    /// <summary>
    /// Warms the rate up in a straight line from a share of itself to the whole of it over so many epochs, and holds it after:
    /// PyTorch's LinearLR.
    /// </summary>
    /// <param name="epochs">How many epochs the warm-up takes.</param>
    /// <param name="start">The share of the rate the first epoch takes, above nought and at most one; a third, unless said, as PyTorch leaves it.</param>
    /// <returns>The declaration.</returns>
    /// <remarks>
    /// One schedule a declaration: the last that is said is the one written, so a warm-up declared here hands over to the
    /// optimizer's own rate. A warm-up that hands over to a decay is written as code, as a <c>LinearWarmup</c> holding the
    /// schedule that follows it.
    /// </remarks>
    public NetworkDeclaration WarmUp(int epochs, double start = 1.0 / 3) =>
        Scheduled("linearWarmup", new PartSetting("epochs", PartValue.Of(epochs)), new PartSetting("start", PartValue.Of(start)));

    /// <summary>
    /// Clips every step's gradients to the given norm together before they move the network, as PyTorch's
    /// <c>clip_grad_norm_</c> clips them.
    /// </summary>
    /// <param name="norm">The most norm the gradients may have together.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The norm is not a number above nought.</exception>
    public NetworkDeclaration ClipGradients(double norm)
    {
        _clip = new GradientClip(norm).MaxNorm;

        return this;
    }

    /// <summary>
    /// Clips every value of a step's gradients to the given bound either side of nothing before they move the network, as
    /// PyTorch's <c>clip_grad_value_</c> does; before the whole norm is clipped, when it is.
    /// </summary>
    /// <param name="bound">The most any value may reach: a number above nothing.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The bound is not a finite number above nothing.</exception>
    public NetworkDeclaration ClipValue(double bound)
    {
        _clipValue = new FitOptions(0) { ClipValue = bound }.ClipValue!.Value;

        return this;
    }

    /// <summary>Takes a step from the mean of the gradients of so many batches, as accumulating gradients does.</summary>
    /// <param name="batches">How many batches go into a step: one or more.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is below one.</exception>
    /// <remarks>A window never goes across the end of an epoch: the last batches of one are stepped as a window of however many are left.</remarks>
    public NetworkDeclaration StepEvery(int batches)
    {
        _stepEvery = new FitOptions(0) { StepEvery = batches }.StepEvery;

        return this;
    }

    /// <summary>Keeps an exponential moving average of the weights, written over them when the run ends.</summary>
    /// <param name="share">The share of itself the average keeps at every step: above nothing and below one.</param>
    /// <returns>The declaration.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share above nothing and below one.</exception>
    public NetworkDeclaration Average(double share)
    {
        _average = new FitOptions(0) { Average = share }.Average!.Value;

        return this;
    }

    /// <summary>
    /// Judges every epoch by the average of the weights rather than by the weights as they are, so early stopping watches — and,
    /// where it restores the best, brings back — the weights the run ends holding.
    /// </summary>
    /// <returns>The declaration.</returns>
    /// <remarks>Said of a run that keeps no <see cref="Average"/>, the step is refused where it is made.</remarks>
    public NetworkDeclaration JudgedByAverage()
    {
        _judgedByAverage = true;

        return this;
    }

    /// <summary>Runs on the engine that goes by this name.</summary>
    /// <param name="engine">The engine's name, as the application gives it — and with it, the device it works on.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration On(string engine)
    {
        _engine = engine;

        return this;
    }

    // The step this declaration stands as, held to every rule a step written by hand is held to.
    internal LearnNetworkStep Step() => new(_layers, WithOptions(_optimizer), _loss)
    {
        Schedule = _schedule,
        Clip = _clip,
        ClipValue = _clipValue,
        StepEvery = _stepEvery,
        Average = _average,
        JudgedByAverage = _judgedByAverage,
        Stopping = _stopping,
        Engine = _engine,
        Seed = _seed,
        Epochs = _epochs,
        Batch = _batch,
    };

    private NetworkDeclaration Layer(string kind, params PartSetting[] settings) => Layer(new PartDeclaration(kind, settings));

    // A rule for the layer written last, said as a setting its word takes: refused by name by a word that takes none.
    private NetworkDeclaration OnTheLastLayer(string door, string key, double value) => OnTheLastLayer(door, key, PartValue.Of(value));

    private NetworkDeclaration OnTheLastLayer(string door, string key, bool value) => OnTheLastLayer(door, key, PartValue.Of(value));

    private NetworkDeclaration OnTheLastLayer(string door, string key, PartValue value)
    {
        if (_layers.Count == 0)
        {
            throw new InvalidOperationException($"{door} is said of the layer written last, and none is written yet.");
        }

        var last = _layers[^1];

        if (Kind(last.Kind).Setting(key) is null)
        {
            throw new NotSupportedException($"A '{last.Kind}' takes no '{key}', so {door} cannot be said of it: say it of a layer that learns numbers.");
        }

        _layers[^1] = last.With(key, value);

        return this;
    }

    private NetworkDeclaration Layer(PartDeclaration layer)
    {
        _layers.Add(layer);

        return this;
    }

    // The kind of layer a name stands for, so that a layer is declared from what its kind starts with rather than from keys
    // written out again here.
    private static PartKind Kind(string name) => NetworkWords.Layers.Single(kind => kind.Name == name);

    private NetworkDeclaration Judged(string kind, params PartSetting[] settings)
    {
        _loss = new PartDeclaration(kind, settings);

        return this;
    }

    // The smoothing of a loss of classes, written only where there is some, so an unsoftened loss is declared as it always was.
    private static PartSetting[] Smoothed(double smoothing) => smoothing is 0d ? [] : [new PartSetting("smoothing", PartValue.Of(smoothing))];


    private NetworkDeclaration Scheduled(string kind, params PartSetting[] settings)
    {
        _schedule = new PartDeclaration(kind, settings);

        return this;
    }

    // The optimizer with the options said beside it, each held until now: an option the optimizer's word does not take is refused
    // by name, here, where the optimizer is known.
    private PartDeclaration WithOptions(PartDeclaration optimizer)
    {
        optimizer = Optioned(optimizer, "WeightDecay", "weightDecay", _decay is { } decay ? PartValue.Of(decay) : null);
        optimizer = Optioned(optimizer, "Dampening", "dampening", _dampening is { } dampening ? PartValue.Of(dampening) : null);
        optimizer = Optioned(optimizer, "Nesterov", "nesterov", _nesterov ? PartValue.Of(true) : null);
        optimizer = Optioned(optimizer, "AmsGrad", "amsgrad", _amsgrad ? PartValue.Of(true) : null);

        return Optioned(optimizer, "Centered", "centered", _centered ? PartValue.Of(true) : null);
    }

    private static PartDeclaration Optioned(PartDeclaration optimizer, string door, string key, PartValue? said)
    {
        if (said is not { } value)
        {
            return optimizer;
        }

        if (optimizer.Kind is null || NetworkWords.Optimizers.Single(kind => kind.Name == optimizer.Kind).Setting(key) is null)
        {
            throw new NotSupportedException($"{door} is not something {(optimizer.Kind is null ? "no optimizer" : $"the '{optimizer.Kind}' optimizer")} takes: name an optimizer that does.");
        }

        // AdamW's weight decay is its own, decoupled from the gradient, and said as AdamW's own number.
        if (optimizer.Kind == "adamw" && key == "weightDecay")
        {
            throw new NotSupportedException("WeightDecay adds a share of every number to its gradient, which AdamW does not: its weight decay is its own, said as AdamW(weightDecay: …).");
        }

        return optimizer.With(key, value);
    }

    // An option of PyTorch's optimizers that Keras's words cannot say, refused in them rather than written as something else.
    private void RequirePyTorchsOption(string door, string meaning)
    {
        if (_words == Vocabularies.Keras)
        {
            throw new NotSupportedException(
                $"{door} {meaning}, and this library runs it as PyTorch's optimizers do, so a network declared in Keras's words cannot say it: declare it in PyTorch's, .WithTorch(network => network.{door}(…)).");
        }
    }

    // An optimizer whose Keras namesake runs other arithmetic is declared in PyTorch's words only, and refused in Keras's
    // rather than written down as an optimizer they would name otherwise.
    private void RequirePyTorchsWords(string keras, string door)
    {
        if (_words == Vocabularies.Keras)
        {
            throw new NotSupportedException(
                $"Keras's {keras} works its steps out otherwise than PyTorch's, which this library runs, so a network declared in Keras's words cannot say it: declare it in PyTorch's, .WithTorch(network => network.{door}(…)).");
        }
    }
}
