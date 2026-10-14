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

    private static readonly TrueOrFalseParameter BiasKey = new("bias", "Whether the layer adds a bias of its own to every sum; it does, unless it says it does not.", true) { LeftOut = true };

    private static readonly OneOfParameter<InitialiserWord> WeightsKey =
        new("weights", "How the weights of the layer start; as PyTorch starts a linear layer's, unless said.", InitialiserWord.KaimingUniform) { LeftOut = InitialiserWord.KaimingUniform };

    private static readonly OneOfParameter<InitialiserWord> BiasStartKey =
        new("biasStart", "How the bias of the layer starts, when it has one; as PyTorch starts it, unless said.", InitialiserWord.FanInUniform) { LeftOut = InitialiserWord.FanInUniform };

    private static readonly TrueOrFalseParameter ScaleKey = new("scale", "Whether it multiplies what it normalised by a scale of its own; it does, unless it says it does not.", true) { LeftOut = true };

    private static readonly TrueOrFalseParameter ShiftKey = new("shift", "Whether it adds a shift of its own to what it normalised; it does, unless it says it does not.", true) { LeftOut = true };

    private static readonly NumberParameter L1Key =
        new("l1", "How much of the sum of the absolute values of the layer's weights the loss adds, as Keras's L1 regularizer does; none, unless said.", 0) { AtLeast = 0, LeftOut = 0 };

    private static readonly NumberParameter L2Key =
        new("l2", "How much of the sum of the squares of the layer's weights the loss adds, as Keras's L2 regularizer does; none, unless said.", 0) { AtLeast = 0, LeftOut = 0 };

    private static readonly NumberParameter MaxNormKey =
        new("maxNorm", "The most norm each unit's weights may have after a step, as Keras's MaxNorm holds them to; nought, or left out, for no bound.", 0) { AtLeast = 0, LeftOut = 0 };

    private static readonly TrueOrFalseParameter LearnsKey =
        new("learns", "Whether the layer's numbers learn, or are held still as Keras's trainable = False holds them; they learn, unless said.", true) { LeftOut = true };

    private static readonly NumberParameter RateKey = new("rate", "The share of numbers left out at every pass while training, between nothing and one.", 0.2) { Above = 0 };

    private static readonly NumberParameter MomentumKey = new("momentum", "How much of what was measured before carries on.", 0.9) { AtLeast = 0 };

    private static readonly NumberParameter EpsilonKey = new("epsilon", "The small number added so nothing is divided by nought.", 1e-5) { Above = 0 };

    private static readonly NumberParameter StepKey = new("rate", "How far one step moves the numbers.", 0.001) { Above = 0 };

    private static readonly NumberParameter CarriedKey = new("momentum", "How much of the last step's velocity carries into the next.", 0) { AtLeast = 0 };

    private static readonly NumberParameter FirstKey = new("firstMoment", "How much of the running mean of the gradients carries on.", 0.9) { Above = 0 };

    private static readonly NumberParameter SecondKey = new("secondMoment", "How much of the running mean of their squares carries on.", 0.999) { Above = 0 };

    private static readonly NumberParameter TinyKey = new("epsilon", "The small number added so nothing is divided by nought.", 1e-8) { Above = 0 };

    private static readonly NumberParameter DecayKey = new("weightDecay", "How far every number is shrunk towards nought at each step, as a share of the step's rate.", 0.01) { AtLeast = 0 };

    private static readonly NumberParameter CoupledDecayKey =
        new("weightDecay", "How much of every number is added to its gradient before the step, as a share of the number itself; none, unless said.", 0) { AtLeast = 0, LeftOut = 0 };

    private static readonly NumberParameter DampeningKey =
        new("dampening", "How little of each new gradient the velocity takes in, from nothing to one; none, unless said.", 0) { AtLeast = 0, LeftOut = 0 };

    private static readonly TrueOrFalseParameter NesterovKey =
        new("nesterov", "Whether the step is taken from the gradient plus the momentum times the velocity, as Nesterov has it; it is not, unless said.", false) { LeftOut = false };

    private static readonly TrueOrFalseParameter AmsGradKey =
        new("amsgrad", "Whether the step is taken over the largest mean of the squares there has been, as AMSGrad has it; it is not, unless said.", false) { LeftOut = false };

    private static readonly TrueOrFalseParameter CenteredKey =
        new("centered", "Whether the root is taken of the mean of the squares less the square of the mean of the gradients; it is not, unless said.", false) { LeftOut = false };

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
        new("dense", "A layer that answers with as many numbers as it is given, each a weighted sum of the ones below it.", [UnitsKey, BiasKey, WeightsKey, BiasStartKey, L1Key, L2Key, MaxNormKey, LearnsKey]),
        new("relu", "Leaves what is below nought at nought.", []),
        new("tanh", "Squashes every number between minus one and one.", []),
        new("sigmoid", "Squashes every number between nought and one.", []),
        new("dropout", "Leaves a share of the numbers out at every pass while training, and none while serving.", [RateKey]),
        new("batchNorm", "Brings every number to the mean and spread of the rows it was trained on.", [MomentumKey, EpsilonKey, ScaleKey, ShiftKey, LearnsKey]),
        new("layerNorm", "Brings every number of one row to that row's own mean and spread.", [EpsilonKey, ScaleKey, ShiftKey, LearnsKey]),
    ];

    /// <summary>The optimizers a declared network is moved by.</summary>
    public static IReadOnlyList<PartKind> Optimizers { get; } =
    [
        new("sgd", "Moves every number against its gradient, carrying some of the last step's velocity.", [StepKey, CarriedKey, CoupledDecayKey, DampeningKey, NesterovKey]),
        new("adam", "Moves every number by a running mean of its gradients over the root of a running mean of their squares.", [StepKey, FirstKey, SecondKey, TinyKey, CoupledDecayKey, AmsGradKey]),
        new("adamw", "Adam, with every number shrunk towards nought before its step, apart from its gradient.", [StepKey, FirstKey, SecondKey, TinyKey, DecayKey, AmsGradKey]),
        new("rmsprop", "Moves every number by its gradient over the root of a running mean of the squares of its gradients.", [SquaresRateKey, AlphaKey, TinyKey, CarriedKey, CoupledDecayKey, CenteredKey]),
        new("nadam", "Adam with Nesterov's momentum, warmed up over the steps.", [NesterovRateKey, FirstKey, SecondKey, TinyKey, WarmingKey, CoupledDecayKey]),
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
                // A layer written by hand may leave out what a file may leave out: it is lowered as the layer it means.
                var made = Word(LayerWords, layer, Layers);

                described = made(described, Layers.Single(kind => kind.Name == layer.Kind).Completed(layer));
            }

            return described;
        }

        /// <summary>Refuses layers that contradict themselves: a bias that is not there, and told how to start.</summary>
        /// <exception cref="ArgumentException">A dense layer has no bias and says how its bias starts.</exception>
        internal void RequireCoherent()
        {
            foreach (var layer in layers.Where(each => each.Kind == "dense" && !each.YesOrNo(BiasKey.Key) && each.Text(BiasStartKey.Key) != BiasStartKey.LeftOut?.Word()))
            {
                throw new ArgumentException(
                    $"A dense layer with 'bias' false has no bias to start, and this one gives a '{BiasStartKey.Key}' of '{layer.Text(BiasStartKey.Key)}': either it has no bias or it says how the bias starts.",
                    nameof(layers));
            }
        }
    }

    extension(InitialiserWord word)
    {
        /// <summary>The initialiser a word names, with the numbers it leaves unsaid.</summary>
        /// <returns>The initialiser.</returns>
        /// <exception cref="ArgumentOutOfRangeException">The value is none of the words: a number cast to the set.</exception>
        public Initialiser Made() => word switch
        {
            InitialiserWord.KaimingUniform => new KaimingUniform(),
            InitialiserWord.GlorotUniform => new GlorotUniform(),
            InitialiserWord.FanInUniform => new FanInUniform(),
            InitialiserWord.Zeros => new Zeros(),
            InitialiserWord.Ones => new Ones(),
            InitialiserWord.Normal => new Normal(),
            InitialiserWord.TruncatedNormal => new TruncatedNormal(),
            InitialiserWord.Orthogonal => new Orthogonal(),
            _ => throw new ArgumentOutOfRangeException(nameof(word), word, "No such way of starting numbers is a word of a declared network."),
        };
    }

    extension(PartDeclaration optimizer)
    {
        /// <summary>The optimizer a declared part names.</summary>
        /// <returns>The optimizer.</returns>
        /// <exception cref="NotSupportedException">It gives a name these words do not know.</exception>
        /// <exception cref="ArgumentException">A setting is outside what the optimizer takes, or its options do not go together.</exception>
        public Optimizer Moved()
        {
            var made = Word(OptimizerWords, optimizer, Optimizers);
            var moved = made(Optimizers.Single(kind => kind.Name == optimizer.Kind).Completed(optimizer));

            moved.Validate();

            return moved;
        }
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

    // How a declared dense layer starts: the weights and the bias by the word each is told, or no bias at all.
    private static Beginning StartOf(PartDeclaration layer) => new()
    {
        Weights = Enum.Parse<InitialiserWord>(layer.Text(WeightsKey.Key), ignoreCase: true).Made(),
        Bias = layer.YesOrNo(BiasKey.Key) ? Enum.Parse<InitialiserWord>(layer.Text(BiasStartKey.Key), ignoreCase: true).Made() : null,
        WithoutBias = !layer.YesOrNo(BiasKey.Key),
    };

    // The rules a declared layer says of its numbers, put on the layer the description just made: a penalty and a bound on its
    // weights, and a hold on all of them. A layer whose word takes none of them is left as it was made.
    private static Sequential GovernedBy(PartDeclaration layer, Sequential described)
    {
        var rules = layer.Holds(L1Key.Key)
            ? new SlotRules { L1 = layer.Number(L1Key.Key), L2 = layer.Number(L2Key.Key), MaxNorm = layer.Number(MaxNormKey.Key) is > 0 and var bound ? bound : null }
            : default;

        if (rules != default)
        {
            described = described.Ruled(rules);
        }

        return layer.YesOrNo(LearnsKey.Key) ? described : described.Held();
    }

    // What a declared normalisation keeps: both a scale and a shift, unless it says it keeps neither or one.
    private static Affine KeptBy(PartDeclaration layer) => new()
    {
        WithoutScale = !layer.YesOrNo(ScaleKey.Key),
        WithoutShift = !layer.YesOrNo(ShiftKey.Key),
    };

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
            ["dense"] = (described, layer) => GovernedBy(layer, described.Dense(layer.Whole(UnitsKey.Key), StartOf(layer))),
            ["relu"] = (described, _) => described.Relu(),
            ["tanh"] = (described, _) => described.Tanh(),
            ["sigmoid"] = (described, _) => described.Sigmoid(),
            ["dropout"] = (described, layer) => described.Dropout(layer.Number(RateKey.Key)),
            ["batchNorm"] = (described, layer) => GovernedBy(layer, described.BatchNorm(layer.Number(MomentumKey.Key), layer.Number(EpsilonKey.Key), KeptBy(layer))),
            ["layerNorm"] = (described, layer) => GovernedBy(layer, described.LayerNorm(layer.Number(EpsilonKey.Key), KeptBy(layer))),
        };

    private static IReadOnlyDictionary<string, Func<PartDeclaration, Optimizer>> OptimizerWords { get; } =
        new Dictionary<string, Func<PartDeclaration, Optimizer>>(StringComparer.Ordinal)
        {
            ["sgd"] = optimizer => new Sgd(optimizer.Number(StepKey.Key))
            {
                Momentum = optimizer.Number(CarriedKey.Key),
                WeightDecay = optimizer.Number(CoupledDecayKey.Key),
                Dampening = optimizer.Number(DampeningKey.Key),
                Nesterov = optimizer.YesOrNo(NesterovKey.Key),
            },
            ["adam"] = optimizer => new Adam(optimizer.Number(StepKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
                WeightDecay = optimizer.Number(CoupledDecayKey.Key),
                AmsGrad = optimizer.YesOrNo(AmsGradKey.Key),
            },
            ["adamw"] = optimizer => new AdamW(optimizer.Number(StepKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
                WeightDecay = optimizer.Number(DecayKey.Key),
                AmsGrad = optimizer.YesOrNo(AmsGradKey.Key),
            },
            ["rmsprop"] = optimizer => new RmsProp(optimizer.Number(SquaresRateKey.Key))
            {
                Alpha = optimizer.Number(AlphaKey.Key),
                Epsilon = optimizer.Number(TinyKey.Key),
                Momentum = optimizer.Number(CarriedKey.Key),
                WeightDecay = optimizer.Number(CoupledDecayKey.Key),
                Centered = optimizer.YesOrNo(CenteredKey.Key),
            },
            ["nadam"] = optimizer => new Nadam(optimizer.Number(NesterovRateKey.Key))
            {
                Betas = new Betas(optimizer.Number(FirstKey.Key), optimizer.Number(SecondKey.Key)),
                Epsilon = optimizer.Number(TinyKey.Key),
                MomentumDecay = optimizer.Number(WarmingKey.Key),
                WeightDecay = optimizer.Number(CoupledDecayKey.Key),
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
