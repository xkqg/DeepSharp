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
/// that takes images is written as code and trained on data a pipeline did not prepare. A normalisation over groups of
/// channels, or over one example's channels, measures each over the places an image holds, and a row has one place, so
/// neither is a word here either.
/// </para>
/// </remarks>
public static class NetworkWords
{
    // The word of a layer that reads a category from its block of columns: lowered only where a pipeline hands its rows over.
    private const string EmbeddingWord = "embedding";

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

    private static readonly NumberParameter RateShareKey =
        new("rateShare", "The share of the run's rate the layer's numbers move at, as a learning rate of their own for a part of the network; the whole rate, unless said.", 1) { Above = 0, LeftOut = 1 };

    private static readonly WholeNumberParameter RankKey =
        new("rank", "The rank of a low-rank pair kept beside the layer's held kernel, as Keras's enable_lora keeps one; nought, or left out, for no pair.", 4) { AtLeast = 1, LeftOut = 0 };

    private static readonly NumberParameter PairAlphaKey =
        new("alpha", "What the pair's product is scaled by over the rank; nought, or left out, for the rank itself, which scales by one.", 8) { AtLeast = 0, LeftOut = 0 };

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

    private static readonly TrueOrFalseParameter ApproximateKey =
        new("approximate", "Whether it is the hyperbolic tangent's approximation of the curve rather than the exact one, which PyTorch and Keras use unless told.", false);

    private static readonly NumberParameter SlopeKey =
        new("negativeSlope", "The share of a number below nought let through: a hundredth in PyTorch, three tenths in Keras's layer.", 0.01) { AtLeast = 0 };

    private static readonly TrueOrFalseParameter EachFeatureKey =
        new("eachFeature", "Whether every number of a row learns a share of its own, rather than one share for them all, as PyTorch starts it.", false);

    private static readonly NumberParameter BendKey = new("alpha", "What the numbers below nought bend towards, below nought.", 1) { AtLeast = 0 };

    private static readonly NumberParameter RootEpsilonKey =
        new("epsilon", "The small number added to the mean of the squares so nothing is divided by nought; PyTorch's is the step a single-precision number takes above one.", 1.1920928955078125e-07) { Above = 0 };

    private static readonly WholeNumberParameter FrequenciesKey = new("frequencies", "How many frequencies each number is turned round at.", 48) { AtLeast = 1 };

    private static readonly NumberParameter SigmaKey =
        new("sigma", "The spread of the normal curve the frequencies start from: the setting the embedding is most sensitive to, searched for every dataset by its authors, and with no default.", 0.01) { Above = 0 };

    private static readonly WholeNumberParameter WidthKey = new("width", "How many numbers each number of the row comes out as.", 24) { AtLeast = 1 };

    private static readonly TrueOrFalseParameter LiteKey =
        new("lite", "Whether one linear layer is shared by every number of the row, rather than one of each number's own.", false);

    private static readonly TextParameter EmbeddingColumnKey =
        new("column", "The column of words whose categories the layer turns into vectors, as it was named before the pipeline wrote it down one column per category.", "category");

    private static readonly WholeNumberParameter EmbeddingWidthKey = new("width", "How many numbers each category's vector holds.", 4) { AtLeast = 1 };

    private static readonly TrueOrFalseParameter FlipAcrossKey =
        new("horizontal", "Whether an image is mirrored across its columns, left for right, with a chance of a half.", true);

    private static readonly TrueOrFalseParameter FlipDownKey =
        new("vertical", "Whether an image is mirrored across its rows, top for bottom, with a chance of a half.", false);

    private static readonly WholeNumberParameter VariationSeedKey =
        new("seed", "The number the variation is drawn from: its own, which the run's seed moves nothing of, and with no default.", 1);

    private static readonly WholeNumberParameter PadKey =
        new("pad", "How many places an image is padded by on every side, and so how far it can move; what moves in from outside is nought.", 2) { AtLeast = 0 };

    private static readonly NumberParameter SpreadKey =
        new("spread", "The standard deviation of the normal noise added to every number.", 0.1) { AtLeast = 0 };

    private static readonly NumberParameter DeltaKey = new("delta", "How far from the answer the loss stops being squared and grows in a straight line.", 1) { Above = 0 };

    private static readonly NumberParameter QuantileKey = new("quantile", "The share of the time the answer is to stay below the value the model gives: above nought and below one.", 0.5) { Above = 0 };

    private static readonly NumberParameter GammaKey = new("gamma", "How hard a label the network is already sure of is weighed down.", 2) { AtLeast = 0 };

    private static readonly NumberParameter BalanceKey =
        new("alpha", "What a one is weighed by, and a nought by one less it — or minus one for no weight, as torchvision writes it.", 0.25) { AtLeast = -1 };

    private static readonly TrueOrFalseParameter LogInputKey =
        new("logInput", "Whether the network answers with the logarithm of each count's rate, as PyTorch takes it, rather than the rate itself, as Keras does.", true);

    private static readonly NumberParameter RateEpsilonKey =
        new("epsilon", "The small number added to a rate before its logarithm is taken: a hundred-millionth in PyTorch, a ten-millionth in Keras.", 1e-8) { Above = 0 };

    private static readonly NumberParameter SmoothingKey =
        new("smoothing", "How much of the answers is spread evenly before they are compared, so no label is taken as quite certain; none, unless said.", 0) { AtLeast = 0, LeftOut = 0 };

    /// <summary>The layers a declared network is built from, in the words a saved network writes them under.</summary>
    public static IReadOnlyList<PartKind> Layers { get; } =
    [
        new("dense", "A layer that answers with as many numbers as it is given, each a weighted sum of the ones below it.", [UnitsKey, BiasKey, WeightsKey, BiasStartKey, L1Key, L2Key, MaxNormKey, LearnsKey, RateShareKey, RankKey, PairAlphaKey]),
        new("relu", "Leaves what is below nought at nought.", []),
        new("tanh", "Squashes every number between minus one and one.", []),
        new("sigmoid", "Squashes every number between nought and one.", []),
        new("dropout", "Leaves a share of the numbers out at every pass while training, and none while serving.", [RateKey]),
        new("batchNorm", "Brings every number to the mean and spread of the rows it was trained on.", [MomentumKey, EpsilonKey, ScaleKey, ShiftKey, LearnsKey, RateShareKey]),
        new("layerNorm", "Brings every number of one row to that row's own mean and spread.", [EpsilonKey, ScaleKey, ShiftKey, LearnsKey, RateShareKey]),
        new("gelu", "Weighs every number by the share of a normal curve below it: smooth where relu bends.", [ApproximateKey]),
        new("silu", "Weighs every number by its own sigmoid; also called swish.", []),
        new("mish", "Weighs every number by the hyperbolic tangent of its softplus.", []),
        new("leakyRelu", "Leaves what is above nought, and lets a share of what is below it through.", [SlopeKey]),
        new("prelu", "Leaves what is above nought, and lets a share of what is below it through that it learns.", [EachFeatureKey]),
        new("elu", "Leaves what is above nought, and bends what is below it towards minus alpha.", [BendKey]),
        new("selu", "Bends as elu does, at the alpha and the scale that keep a network's numbers normalised.", []),
        new("softmax", "Turns the numbers of a row into shares of one whole.", []),
        new("rmsNorm", "Divides every number of one row by the root of the mean of the row's squares, and scales it.", [RootEpsilonKey, ScaleKey, LearnsKey, RateShareKey]),
        new("periodic", "Turns every number of the row into its cosines and sines at frequencies it learns, through a linear layer: a relu after it makes it PLR.", [FrequenciesKey, SigmaKey, WidthKey, LiteKey]),
        new("randomFlip", "Mirrors images at random while training and not while serving; it varies the rows themselves, so it stands first.", [FlipAcrossKey, FlipDownKey, VariationSeedKey]),
        new("randomCrop", "Shifts images at random while training and not while serving, padding each and cutting a window of its own size out; it stands first.", [PadKey, VariationSeedKey]),
        new("gaussianNoise", "Adds normal noise to every number while training and not while serving; it stands first.", [SpreadKey, VariationSeedKey]),
        new("flatten", "Lays an example out as one row of numbers: an image as the row of its pixels.", []),
        new(EmbeddingWord, "Turns the columns a category was written down as into a small vector it learns for each category; it reads the rows as the pipeline hands them over, so it stands first.", [EmbeddingColumnKey, EmbeddingWidthKey]),
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
        new("crossEntropy", "The loss of one answer among several classes, read through a softmax.", [SmoothingKey]),
        new("binaryCrossEntropy", "The loss of an answer that is a chance, read through a sigmoid.", [SmoothingKey]),
        new("earthMoversDistance", "How far the shares have to move along the order of the bands to be the answer's: an answer of shares in an order, read through a softmax.", []),
        new("meanAbsoluteError", "The mean of the size of the differences: an answer that is an amount, some far off.", []),
        new("huber", "Half the square of the difference near the answer, its size further off: an amount with outliers.", [DeltaKey]),
        new("quantile", "The pinball loss: the value the answer stays below a share of the time, rather than its mean.", [QuantileKey]),
        new("focal", "A binary cross-entropy that weighs down the labels the network is already sure of: a rare class, read through a sigmoid.", [GammaKey, BalanceKey]),
        new("hinge", "How far each answer falls short of a margin of one on its side of nought: a label, read by the side it falls.", []),
        new("poisson", "The loss of a count: an answer that is how many times something happened, read through an exponential when it is a logarithm.", [LogInputKey, RateEpsilonKey]),
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

        /// <summary>
        /// The network a declaration's layers describe, for the rows a run of its pipeline hands over: an embedding's column turned
        /// into the places of its block among the columns those rows hold.
        /// </summary>
        /// <param name="prepared">The run whose rows the network is trained on.</param>
        /// <returns>The description, ready to be compiled.</returns>
        /// <exception cref="InvalidOperationException">
        /// An embedding stands after a layer that is not one, or names a column the pipeline did not write down one column per
        /// category, or one whose block the rows do not hold whole.
        /// </exception>
        /// <remarks>
        /// An embedding reads a row as the pipeline hands it over, so the embeddings stand first; each replaces its block by its
        /// vector, and the next one finds its block among the columns that leaves.
        /// </remarks>
        internal Sequential Described(PreparedData prepared)
        {
            var embeddings = layers.TakeWhile(layer => layer.Kind == EmbeddingWord).Count();

            if (layers.Skip(embeddings).Any(layer => layer.Kind == EmbeddingWord))
            {
                throw new InvalidOperationException(
                    "An embedding reads a category from the row as the pipeline hands it over, so every embedding stands before the first layer that is not one.");
            }

            if (embeddings == 0)
            {
                return layers.Described();
            }

            List<string> columns = [.. prepared.Batch(Part.Train, TrainedNetwork.FeatureNeeds).FeatureNames];
            var described = new Sequential();

            foreach (var layer in layers.Take(embeddings))
            {
                var column = layer.Text(EmbeddingColumnKey.Key);
                var width = layer.Whole(EmbeddingWidthKey.Key);
                var names = prepared.CategoryBlock(column);
                var block = names.Select(name => columns.IndexOf(name)).ToArray();

                if (block.Any(place => place < 0))
                {
                    throw new InvalidOperationException(
                        $"An embedding of '{column}' reads its block of categories, and the rows handed over do not hold '{names[Array.IndexOf(block, -1)]}' of it: leave the block whole.");
                }

                described = described.Embedding(block, width);

                var first = block.Min();
                columns = [
                    .. columns.Take(first).Where(name => !names.Contains(name)),
                    .. Enumerable.Range(0, width).Select(number => $"{column}_vector{number}"),
                    .. columns.Skip(first).Where(name => !names.Contains(name)),
                ];
            }

            foreach (var layer in layers.Skip(embeddings))
            {
                described = Word(LayerWords, layer, Layers)(described, Layers.Single(kind => kind.Name == layer.Kind).Completed(layer));
            }

            return described;
        }

        /// <summary>Refuses layers that contradict themselves: a bias that is not there, and told how to start.</summary>
        /// <exception cref="ArgumentException">A dense layer has no bias and says how its bias starts.</exception>
        internal void RequireCoherent()
        {
            var learned = -1;

            for (var at = 0; at < layers.Count; at++)
            {
                var varies = layers[at].Kind is "randomFlip" or "randomCrop" or "gaussianNoise";

                if (!varies && learned < 0)
                {
                    learned = at;
                }
                else if (varies && learned >= 0)
                {
                    throw new ArgumentException(
                        $"A '{layers[at].Kind}' varies the rows themselves, made on the values as the batch carries them: it stands first, before the '{layers[learned].Kind}' written ahead of it.",
                        nameof(layers));
                }
            }

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
        public Loss Judged() => Word(LossWords, loss, Losses)(Losses.Single(kind => kind.Name == loss.Kind).Completed(loss));
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
        LowRank = layer.Whole(RankKey.Key) is > 0 and var rank ? new LowRank(rank, layer.Number(PairAlphaKey.Key) is > 0 and var alpha ? alpha : null) : null,
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

        if (layer.Holds(RateShareKey.Key) && layer.Number(RateShareKey.Key) is var share && share != RateShareKey.LeftOut)
        {
            described = described.RateShare(share);
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
            : throw new NotSupportedException(Refused.TryGetValue(part.Kind, out var why)
                ? $"'{part.Kind}' is not a word a declared network is written in: {why}"
                : $"'{part.Kind}' is not a word a declared network is written in: {string.Join(", ", kinds.Select(kind => $"'{kind.Name}'"))}.");

    // The words left out on purpose, each with why and with the word that says it.
    private static IReadOnlyDictionary<string, string> Refused { get; } = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["smoothL1"] =
            "PyTorch's SmoothL1Loss is the Huber loss over its beta, so it is one word, 'huber', with its delta at that beta — the same network learned, its loss beta times as large.",
    };

    private static IReadOnlyDictionary<string, Func<Sequential, PartDeclaration, Sequential>> LayerWords { get; } =
        new Dictionary<string, Func<Sequential, PartDeclaration, Sequential>>(StringComparer.Ordinal)
        {
            ["dense"] = (described, layer) => GovernedBy(layer, described.Dense(layer.Whole(UnitsKey.Key), StartOf(layer))),
            ["relu"] = (described, _) => described.Relu(),
            ["tanh"] = (described, _) => described.Tanh(),
            ["sigmoid"] = (described, _) => described.Sigmoid(),
            ["dropout"] = (described, layer) => described.Dropout(layer.Number(RateKey.Key)),
            ["randomFlip"] = (described, layer) => described.RandomFlip(layer.YesOrNo(FlipAcrossKey.Key), layer.YesOrNo(FlipDownKey.Key), layer.Whole(VariationSeedKey.Key)),
            ["randomCrop"] = (described, layer) => described.RandomCrop(layer.Whole(PadKey.Key), layer.Whole(VariationSeedKey.Key)),
            ["gaussianNoise"] = (described, layer) => described.GaussianNoise(layer.Number(SpreadKey.Key), layer.Whole(VariationSeedKey.Key)),
            ["flatten"] = (described, _) => described.Flatten(),
            ["batchNorm"] = (described, layer) => GovernedBy(layer, described.BatchNorm(layer.Number(MomentumKey.Key), layer.Number(EpsilonKey.Key), KeptBy(layer))),
            ["layerNorm"] = (described, layer) => GovernedBy(layer, described.LayerNorm(layer.Number(EpsilonKey.Key), KeptBy(layer))),
            ["gelu"] = (described, layer) => described.Gelu(layer.YesOrNo(ApproximateKey.Key)),
            ["silu"] = (described, _) => described.Silu(),
            ["mish"] = (described, _) => described.Mish(),
            ["leakyRelu"] = (described, layer) => described.LeakyRelu(layer.Number(SlopeKey.Key)),
            ["prelu"] = (described, layer) => described.PRelu(layer.YesOrNo(EachFeatureKey.Key)),
            ["elu"] = (described, layer) => described.Elu(layer.Number(BendKey.Key)),
            ["selu"] = (described, _) => described.Selu(),
            ["softmax"] = (described, _) => described.Softmax(),
            ["rmsNorm"] = (described, layer) => GovernedBy(layer, described.RmsNorm(layer.Number(RootEpsilonKey.Key), layer.YesOrNo(ScaleKey.Key))),
            ["periodic"] = (described, layer) => described.Periodic(
                new Periodicity(layer.Whole(FrequenciesKey.Key), layer.Number(SigmaKey.Key), layer.Whole(WidthKey.Key)) { Lite = layer.YesOrNo(LiteKey.Key) }),
            [EmbeddingWord] = (_, layer) => throw new InvalidOperationException(
                $"An embedding of '{layer.Text(EmbeddingColumnKey.Key)}' reads its block from the rows a pipeline hands over, so it is lowered where they are: train the network behind its pipeline."),
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
            ["crossEntropy"] = loss => new CrossEntropy { Smoothing = loss.Number(SmoothingKey.Key) },
            ["binaryCrossEntropy"] = loss => new BinaryCrossEntropy { Smoothing = loss.Number(SmoothingKey.Key) },
            ["earthMoversDistance"] = _ => new EarthMoversDistance(),
            ["meanAbsoluteError"] = _ => new MeanAbsoluteError(),
            ["huber"] = loss => new Huber(loss.Number(DeltaKey.Key)),
            ["quantile"] = loss => new Quantile(loss.Number(QuantileKey.Key)),
            ["focal"] = loss => new Focal(loss.Number(GammaKey.Key)) { Alpha = Focal.FromWritten(loss.Number(BalanceKey.Key)) },
            ["hinge"] = _ => new Hinge(),
            ["poisson"] = loss => new Poisson(loss.YesOrNo(LogInputKey.Key)) { Epsilon = loss.Number(RateEpsilonKey.Key) },
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
