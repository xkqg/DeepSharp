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
    private readonly List<PartDeclaration> _layers = [];
    private readonly Vocabularies _words;

    private PartDeclaration _optimizer;
    private PartDeclaration _loss;
    private PartDeclaration _stopping;
    private PartDeclaration _schedule = NetworkWords.Schedules[0].Declared();
    private double _clip;
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

    /// <summary>A layer that answers with this many numbers, each a weighted sum of the ones below it.</summary>
    /// <param name="units">How many numbers it answers with.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration Dense(int units) => Layer("dense", new PartSetting("units", PartValue.Of(units)));

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

    /// <summary>A layer that brings every number to the mean and spread of the rows it was trained on.</summary>
    /// <param name="momentum">How much of what was measured before carries on; each library's own, unless said.</param>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BatchNorm(double? momentum = null, double? epsilon = null) => Layer(
        "batchNorm",
        new PartSetting("momentum", PartValue.Of(momentum ?? (_words == Vocabularies.Keras ? 0.99 : 0.9))),
        new PartSetting("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-3 : 1e-5))));

    /// <summary>A layer that brings every number of one row to that row's own mean and spread.</summary>
    /// <param name="epsilon">The small number added so nothing is divided by nought; each library's own, unless said.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration LayerNorm(double? epsilon = null) =>
        Layer("layerNorm", new PartSetting("epsilon", PartValue.Of(epsilon ?? (_words == Vocabularies.Keras ? 1e-3 : 1e-5))));

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

    /// <summary>Runs on the engine that goes by this name.</summary>
    /// <param name="engine">The engine's name, as the application gives it — and with it, the device it works on.</param>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration On(string engine)
    {
        _engine = engine;

        return this;
    }

    // The step this declaration stands as, held to every rule a step written by hand is held to.
    internal LearnNetworkStep Step() => new(_layers, _optimizer, _loss)
    {
        Schedule = _schedule,
        Clip = _clip,
        Stopping = _stopping,
        Engine = _engine,
        Seed = _seed,
        Epochs = _epochs,
        Batch = _batch,
    };

    private NetworkDeclaration Layer(string kind, params PartSetting[] settings)
    {
        _layers.Add(new PartDeclaration(kind, settings));

        return this;
    }

    private NetworkDeclaration Judged(string kind)
    {
        _loss = new PartDeclaration(kind, []);

        return this;
    }

    private NetworkDeclaration Scheduled(string kind, params PartSetting[] settings)
    {
        _schedule = new PartDeclaration(kind, settings);

        return this;
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
