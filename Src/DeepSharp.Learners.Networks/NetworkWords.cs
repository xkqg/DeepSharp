// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The words a declared network is written in: which layers there are, which optimizer, which loss and when a run
/// stops, each with the settings it takes.
/// </summary>
/// <remarks>
/// One vocabulary, and it is the one a network's own file already speaks — a layer is written under the name the saved
/// network writes it under, so a declaration and the network it trains say 'dense' and 'adam' alike and nothing has to
/// be translated between them. Every setting a word takes is written down: a declaration is replayed from what it says,
/// and a number nobody wrote would make two runs of one declaration differ.
/// <para>
/// The layers are the ones a pipeline's rows can reach. A pipeline hands over a flat row of numbers, so the words that
/// work on a row are here and the ones that work on an image — a window over it, a reshaping of it — are not: a network
/// that takes images is written as code and trained on data a pipeline did not prepare.
/// </para>
/// </remarks>
public static class NetworkWords
{
    private static readonly WholeNumberParameter UnitsKey = new("units", "How many numbers the layer answers with.", 16) { AtLeast = 1 };

    private static readonly NumberParameter RateKey = new("rate", "The share of numbers left out at every pass while training, between nothing and one.", 0.2) { Above = 0 };

    private static readonly NumberParameter MomentumKey = new("momentum", "How much of what was measured before carries on.", 0.9) { AtLeast = 0 };

    private static readonly NumberParameter EpsilonKey = new("epsilon", "The small number added so nothing is divided by nought.", 1e-5) { Above = 0 };

    private static readonly NumberParameter StepKey = new("rate", "How far one step moves the numbers.", 0.001) { Above = 0 };

    private static readonly NumberParameter CarriedKey = new("momentum", "How much of the last step's velocity carries into the next.", 0) { AtLeast = 0 };

    private static readonly NumberParameter FirstKey = new("firstMoment", "How much of the running mean of the gradients carries on.", 0.9) { Above = 0 };

    private static readonly NumberParameter SecondKey = new("secondMoment", "How much of the running mean of their squares carries on.", 0.999) { Above = 0 };

    private static readonly NumberParameter TinyKey = new("epsilon", "The small number added so nothing is divided by nought.", 1e-8) { Above = 0 };

    private static readonly WholeNumberParameter PatienceKey = new("patience", "How many passes without a better one the run waits before it stops.", 10) { AtLeast = 0 };

    private static readonly NumberParameter LeastKey = new("least", "How far the validation loss has to fall below the best for a pass to count as better.", 0) { AtLeast = 0 };

    private static readonly TrueOrFalseParameter BestKey = new("best", "Whether the run ends holding the numbers of its best pass rather than its last.", true);

    /// <summary>The layers a declared network is built from, in the words a saved network writes them under.</summary>
    public static IReadOnlyList<PartKind> Layers { get; } =
    [
        new("dense", "A layer that answers with as many numbers as it is given, each a weighted sum of the ones below it.", [UnitsKey]),
        new("relu", "Leaves what is below nought at nought.", []),
        new("tanh", "Squashes every number between minus one and one.", []),
        new("sigmoid", "Squashes every number between nought and one.", []),
        new("dropout", "Leaves a share of the numbers out at every pass while training, and none while serving.", [RateKey]),
        new("batchNorm", "Brings every number to the mean and spread of the rows it was trained on.", [MomentumKey, EpsilonKey]),
        new("layerNorm", "Brings every number of one row to that row's own mean and spread.", [EpsilonKey]),
    ];

    /// <summary>The optimizers a declared network is moved by.</summary>
    public static IReadOnlyList<PartKind> Optimizers { get; } =
    [
        new("sgd", "Moves every number against its gradient, carrying some of the last step's velocity.", [StepKey, CarriedKey]),
        new("adam", "Moves every number by a running mean of its gradients over the root of a running mean of their squares.", [StepKey, FirstKey, SecondKey, TinyKey]),
    ];

    /// <summary>The losses a declared network is judged by.</summary>
    public static IReadOnlyList<PartKind> Losses { get; } =
    [
        new("meanSquaredError", "The mean of the squared differences: an answer that is an amount.", []),
        new("crossEntropy", "The loss of one answer among several classes, read through a softmax.", []),
        new("binaryCrossEntropy", "The loss of an answer that is a chance, read through a sigmoid.", []),
    ];

    /// <summary>When a run stops: at its last pass, or once the validation loss stops falling.</summary>
    public static IReadOnlyList<PartKind> Stopping { get; } =
    [
        new("never", "The run goes on to its last pass.", []),
        new("patience", "The run stops once the validation loss has not fallen for as many passes as it waits.", [PatienceKey, LeastKey, BestKey]),
    ];

    /// <summary>The network a declaration's layers describe, in the words TensorFlow and Keras use.</summary>
    /// <param name="layers">The layers, from the one the rows reach first.</param>
    /// <returns>The description, ready to be compiled.</returns>
    /// <exception cref="ArgumentNullException">There are no layers.</exception>
    /// <exception cref="NotSupportedException">A layer gives a name these words do not know.</exception>
    public static Sequential Described(this IReadOnlyList<PartDeclaration> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);

        var described = new Sequential();

        foreach (var layer in layers)
        {
            described = Word(LayerWords, layer, Layers)(described, layer);
        }

        return described;
    }

    /// <summary>The optimizer a declared part names.</summary>
    /// <param name="optimizer">The part.</param>
    /// <returns>The optimizer.</returns>
    /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
    public static Optimizer Moved(this PartDeclaration optimizer) => Word(OptimizerWords, optimizer, Optimizers)(optimizer);

    /// <summary>The loss a declared part names.</summary>
    /// <param name="loss">The part.</param>
    /// <returns>The loss.</returns>
    /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
    public static Loss Judged(this PartDeclaration loss) => Word(LossWords, loss, Losses)(loss);

    /// <summary>When the run a declared part names stops, or nothing when it goes on to its last pass.</summary>
    /// <param name="stopping">The part.</param>
    /// <returns>The stopping, or nothing.</returns>
    /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
    public static EarlyStopping? Stops(this PartDeclaration stopping) => Word(StoppingWords, stopping, Stopping)(stopping);

    // What one word makes, looked up by the name the part gives: one table per family, so a word is added by writing it
    // beside the name it is written under rather than by widening a decision somewhere else.
    private static TMade Word<TMade>(IReadOnlyDictionary<string, TMade> words, PartDeclaration part, IReadOnlyList<PartKind> kinds) =>
        words.TryGetValue(part.Kind, out var made)
            ? made
            : throw new NotSupportedException(
                $"'{part.Kind}' is not a word a declared network is written in: {string.Join(", ", kinds.Select(kind => $"'{kind.Name}'"))}.");

    private static IReadOnlyDictionary<string, Func<Sequential, PartDeclaration, Sequential>> LayerWords { get; } =
        new Dictionary<string, Func<Sequential, PartDeclaration, Sequential>>(StringComparer.Ordinal)
        {
            ["dense"] = (described, layer) => described.Dense(layer.Whole(UnitsKey.Key)),
            ["relu"] = (described, _) => described.Relu(),
            ["tanh"] = (described, _) => described.Tanh(),
            ["sigmoid"] = (described, _) => described.Sigmoid(),
            ["dropout"] = (described, layer) => described.Dropout(layer.Number(RateKey.Key)),
            ["batchNorm"] = (described, layer) => described.BatchNorm(layer.Number(MomentumKey.Key), layer.Number(EpsilonKey.Key)),
            ["layerNorm"] = (described, layer) => described.LayerNorm(layer.Number(EpsilonKey.Key)),
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, Optimizer>> OptimizerWords { get; } =
        new Dictionary<string, Func<PartDeclaration, Optimizer>>(StringComparer.Ordinal)
        {
            ["sgd"] = optimizer => new Sgd(optimizer.Number(StepKey.Key)) { Momentum = optimizer.Number(CarriedKey.Key) },
            ["adam"] = optimizer => new Adam(optimizer.Number(StepKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
            },
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, Loss>> LossWords { get; } =
        new Dictionary<string, Func<PartDeclaration, Loss>>(StringComparer.Ordinal)
        {
            ["meanSquaredError"] = _ => new MeanSquaredError(),
            ["crossEntropy"] = _ => new CrossEntropy(),
            ["binaryCrossEntropy"] = _ => new BinaryCrossEntropy(),
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, EarlyStopping?>> StoppingWords { get; } =
        new Dictionary<string, Func<PartDeclaration, EarlyStopping?>>(StringComparer.Ordinal)
        {
            ["never"] = _ => null,
            ["patience"] = stopping => new EarlyStopping
            {
                Patience = stopping.Whole(PatienceKey.Key),
                MinDelta = stopping.Number(LeastKey.Key),
                RestoreBest = stopping.YesOrNo(BestKey.Key),
            },
        };
}
