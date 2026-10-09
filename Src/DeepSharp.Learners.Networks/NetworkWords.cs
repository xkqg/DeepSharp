// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The words a declared network is written in: which layers there are, which optimizer, which loss, how the rate changes
/// and when a run stops, each with the settings it takes.
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

    private static readonly NumberParameter DecayKey = new("weightDecay", "How far every number is shrunk towards nought at each step, as a share of the step's rate.", 0.01) { AtLeast = 0 };

    private static readonly NumberParameter SquaresRateKey = new("rate", "How far one step moves the numbers.", 0.01) { Above = 0 };

    private static readonly NumberParameter AlphaKey = new("alpha", "How much of the running mean of the squares of the gradients carries on.", 0.99) { AtLeast = 0 };

    private static readonly NumberParameter NesterovRateKey = new("rate", "How far one step moves the numbers.", 0.002) { Above = 0 };

    private static readonly NumberParameter WarmingKey = new("momentumDecay", "How fast the momentum warms up over the steps.", 0.004) { AtLeast = 0 };

    private static readonly WholeNumberParameter EveryKey = new("every", "How many epochs pass between two falls of the rate.", 10) { AtLeast = 1 };

    private static readonly NumberParameter FallKey = new("factor", "What the rate is multiplied by at each fall.", 0.1) { Above = 0 };

    private static readonly NumberParameter EpochFactorKey = new("factor", "What the rate is multiplied by from one epoch to the next.", 0.9) { Above = 0 };

    private static readonly WholeNumberParameter CosineKey = new("epochs", "How many epochs the rate falls over, along half a cosine.", 100) { AtLeast = 1 };

    private static readonly NumberParameter LeastRateKey = new("minimum", "The least rate, reached at the end of the fall.", 0) { AtLeast = 0 };

    private static readonly WholeNumberParameter WarmupKey = new("epochs", "How many epochs the rate warms up over.", 5) { AtLeast = 1 };

    private static readonly NumberParameter StartKey = new("start", "The share of the rate the first epoch takes: above nought, and at most one.", 1.0 / 3) { Above = 0 };

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
        new("adamw", "Adam, with every number shrunk towards nought before its step, apart from its gradient.", [StepKey, FirstKey, SecondKey, TinyKey, DecayKey]),
        new("rmsprop", "Moves every number by its gradient over the root of a running mean of the squares of its gradients.", [SquaresRateKey, AlphaKey, TinyKey, CarriedKey]),
        new("nadam", "Adam with Nesterov's momentum, warmed up over the steps.", [NesterovRateKey, FirstKey, SecondKey, TinyKey, WarmingKey]),
    ];

    /// <summary>The losses a declared network is judged by.</summary>
    public static IReadOnlyList<PartKind> Losses { get; } =
    [
        new("meanSquaredError", "The mean of the squared differences: an answer that is an amount.", []),
        new("crossEntropy", "The loss of one answer among several classes, read through a softmax.", []),
        new("binaryCrossEntropy", "The loss of an answer that is a chance, read through a sigmoid.", []),
        new("earthMoversDistance", "How far the shares have to move along the order of the bands to be the answer's: an answer of shares in an order, read through a softmax.", []),
    ];

    /// <summary>How the rate a declared network is moved at changes from epoch to epoch, in the words a network's file writes it in.</summary>
    /// <remarks>
    /// Each is worked out from the epoch alone, so none listens to the validation loss. A warm-up declared here hands over to
    /// the optimizer's own rate; one that hands over to a decay is written as code, where a <see cref="LinearWarmup"/> holds
    /// the schedule that follows it.
    /// </remarks>
    public static IReadOnlyList<PartKind> Schedules { get; } =
    [
        new("constant", "The rate the optimizer starts at, every epoch.", []),
        new("stepDecay", "The rate multiplied by a factor every so many epochs.", [EveryKey, FallKey]),
        new("exponentialDecay", "The rate multiplied by a factor every epoch.", [EpochFactorKey]),
        new("cosineDecay", "The rate falling along half a cosine to its least over so many epochs.", [CosineKey, LeastRateKey]),
        new("linearWarmup", "The rate rising in a straight line from a share of itself to the whole of it over so many epochs, then held.", [WarmupKey, StartKey]),
    ];

    /// <summary>When a run stops: at its last pass, or once the validation loss stops falling.</summary>
    public static IReadOnlyList<PartKind> Stopping { get; } =
    [
        new("never", "The run goes on to its last pass.", []),
        new("patience", "The run stops once the validation loss has not fallen for as many passes as it waits.", [PatienceKey, LeastKey, BestKey]),
    ];

    extension(IReadOnlyList<PartDeclaration> layers)
    {
        /// <summary>The network a declaration's layers describe, in the words TensorFlow and Keras use.</summary>
        /// <returns>The description, ready to be compiled.</returns>
        /// <exception cref="ArgumentNullException">There are no layers.</exception>
        /// <exception cref="NotSupportedException">A layer gives a name these words do not know.</exception>
        public Sequential Described()
        {
            ArgumentNullException.ThrowIfNull(layers);

            var described = new Sequential();

            foreach (var layer in layers)
            {
                described = Word(LayerWords, layer, Layers)(described, layer);
            }

            return described;
        }
    }

    extension(PartDeclaration optimizer)
    {
        /// <summary>The optimizer a declared part names.</summary>
        /// <returns>The optimizer.</returns>
        /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
        public Optimizer Moved() => Word(OptimizerWords, optimizer, Optimizers)(optimizer);
    }

    extension(PartDeclaration loss)
    {
        /// <summary>The loss a declared part names.</summary>
        /// <returns>The loss.</returns>
        /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
        public Loss Judged() => Word(LossWords, loss, Losses)(loss);
    }

    extension(PartDeclaration schedule)
    {
        /// <summary>How the rate changes from epoch to epoch, as a declared part names it.</summary>
        /// <returns>The schedule.</returns>
        /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
        public LearningRateSchedule Scheduled() => Word(ScheduleWords, schedule, Schedules)(schedule);
    }

    extension(PartDeclaration stopping)
    {
        /// <summary>When the run a declared part names stops, or nothing when it goes on to its last pass.</summary>
        /// <returns>The stopping, or nothing.</returns>
        /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
        public EarlyStopping? Stops() => Word(StoppingWords, stopping, Stopping)(stopping);
    }

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
            ["adamw"] = optimizer => new AdamW(optimizer.Number(StepKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
                WeightDecay = optimizer.Number(DecayKey.Key),
            },
            ["rmsprop"] = optimizer => new RmsProp(optimizer.Number(SquaresRateKey.Key))
            {
                Alpha = optimizer.Number(AlphaKey.Key),
                Epsilon = optimizer.Number(TinyKey.Key),
                Momentum = optimizer.Number(CarriedKey.Key),
            },
            ["nadam"] = optimizer => new Nadam(optimizer.Number(NesterovRateKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
                MomentumDecay = optimizer.Number(WarmingKey.Key),
            },
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, Loss>> LossWords { get; } =
        new Dictionary<string, Func<PartDeclaration, Loss>>(StringComparer.Ordinal)
        {
            ["meanSquaredError"] = _ => new MeanSquaredError(),
            ["crossEntropy"] = _ => new CrossEntropy(),
            ["binaryCrossEntropy"] = _ => new BinaryCrossEntropy(),
            ["earthMoversDistance"] = _ => new EarthMoversDistance(),
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, LearningRateSchedule>> ScheduleWords { get; } =
        new Dictionary<string, Func<PartDeclaration, LearningRateSchedule>>(StringComparer.Ordinal)
        {
            ["constant"] = _ => new ConstantRate(),
            ["stepDecay"] = schedule => new StepDecay(schedule.Whole(EveryKey.Key), schedule.Number(FallKey.Key)),
            ["exponentialDecay"] = schedule => new ExponentialDecay(schedule.Number(EpochFactorKey.Key)),
            ["cosineDecay"] = schedule => new CosineDecay(schedule.Whole(CosineKey.Key), schedule.Number(LeastRateKey.Key)),
            ["linearWarmup"] = schedule => new LinearWarmup(schedule.Whole(WarmupKey.Key), schedule.Number(StartKey.Key)),
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
