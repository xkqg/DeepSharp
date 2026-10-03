// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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

    /// <summary>Judges the answers by the mean of their squared differences: an answer that is an amount.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration MeanSquaredError() => Judged("meanSquaredError");

    /// <summary>Judges one answer among several classes, read through a softmax.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration CrossEntropy() => Judged("crossEntropy");

    /// <summary>Judges an answer that is a chance, read through a sigmoid.</summary>
    /// <returns>The declaration.</returns>
    public NetworkDeclaration BinaryCrossEntropy() => Judged("binaryCrossEntropy");

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
}
