// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>What a network's outputs go through on their way out as predictions.</summary>
public enum OutputActivation
{
    /// <summary>Nothing: an output is the prediction, as it is for a number.</summary>
    Identity,

    /// <summary>Each row's shares: an output is one class's logit among its row's.</summary>
    Softmax,

    /// <summary>The logistic curve: an output is the logit of a one.</summary>
    Sigmoid,

    /// <summary>The exponential: an output is the logarithm of a count's rate, as a Poisson loss on logarithms takes it.</summary>
    Exp,

    /// <summary>
    /// Which side of nought it falls: an output above nought is a one and any other a nought, as a hinge loss reads its margin —
    /// a class, never a chance.
    /// </summary>
    Margin,
}

/// <summary>
/// What a network is trained to bring down: one value for a batch, from its outputs and the answers it should have given.
/// </summary>
/// <remarks>
/// A loss is named by the person who trains, as Keras's <c>compile</c> has it; nothing picks one for them. It takes the
/// network's raw outputs — the logits, for a loss of classes — and applies the activation it is written with itself,
/// which is steadier than a network ending in one and is what PyTorch does; <see cref="Predictions"/> applies the same
/// activation to what is served. So a network compiled with a loss does not end in the activation the loss applies: a stack
/// whose last layer is that activation is refused where it is compiled, and a description in Keras's words that ends in it,
/// as Keras's habit is, leaves it out. Each loss also says which answers it could have meant, and a row handed over with
/// any other is refused before anything is trained on it.
/// </remarks>
public abstract class Loss
{
    /// <summary>What the outputs go through on their way out as predictions.</summary>
    public abstract OutputActivation Activation { get; }

    /// <summary>
    /// Whether a layer is the output activation this loss applies to a network's outputs itself — a sigmoid, for a loss whose
    /// outputs are the logits of a one — so that a network compiled with it ends before one.
    /// </summary>
    /// <param name="layer">The layer.</param>
    /// <returns>
    /// Whether it is a <see cref="Sigmoid"/> before a loss whose outputs are logits of a one, or a <see cref="Softmax"/> before one
    /// whose outputs are a row's logits.
    /// </returns>
    public bool Applies(Layer layer) =>
        (layer is Sigmoid && Activation == OutputActivation.Sigmoid) || (layer is Softmax && Activation == OutputActivation.Softmax);

    /// <summary>The loss of a batch.</summary>
    /// <param name="outputs">The network's raw outputs: a row for each example.</param>
    /// <param name="answers">The answers it should have given, of the same shape.</param>
    /// <param name="backend">The backend the arithmetic runs on: a recording one, for a loss whose gradients are wanted.</param>
    /// <returns>One value with no axes: the mean over the batch.</returns>
    /// <exception cref="ArgumentException">The outputs and the answers are of different shapes.</exception>
    public Tensor Of(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(outputs);
        ArgumentNullException.ThrowIfNull(answers);
        ArgumentNullException.ThrowIfNull(backend);

        if (outputs.Shape != answers.Shape)
        {
            throw new ArgumentException(
                $"A loss compares each output with its answer, and was handed {outputs.Shape} outputs and {answers.Shape} answers.", nameof(answers));
        }

        return Computed(outputs, answers, backend);
    }

    /// <summary>The predictions the outputs stand for: the outputs through <see cref="Activation"/>.</summary>
    /// <param name="outputs">The network's raw outputs.</param>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <returns>The predictions, of the outputs' shape.</returns>
    public Tensor Predictions(Tensor outputs, ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(outputs);
        ArgumentNullException.ThrowIfNull(backend);

        return Activation switch
        {
            OutputActivation.Softmax => backend.Exp(backend.LogSoftmax(outputs)),
            OutputActivation.Sigmoid => backend.Sigmoid(outputs),
            OutputActivation.Exp => backend.Exp(outputs),
            OutputActivation.Margin => backend.Positive(outputs),
            _ => outputs,
        };
    }

    /// <summary>Why a row's answers cannot be what this loss is trained against, when they cannot.</summary>
    /// <param name="answers">One row's answers, each a finite number.</param>
    /// <returns>The reason, or nothing when the row can be trained on.</returns>
    public string? Refusal(IReadOnlyList<double> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);

        return Refused(answers);
    }

    /// <summary>The loss of a batch whose outputs and answers are of one shape.</summary>
    /// <param name="outputs">The network's raw outputs.</param>
    /// <param name="answers">The answers, of the same shape.</param>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <returns>One value with no axes.</returns>
    protected abstract Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend);

    /// <summary>Why a row's answers cannot be trained against; nothing, unless the loss says otherwise.</summary>
    /// <param name="answers">One row's answers.</param>
    /// <returns>The reason, or nothing.</returns>
    protected virtual string? Refused(IReadOnlyList<double> answers) => null;
}

/// <summary>The mean of the squared difference between every output and its answer: for answers that are numbers.</summary>
public sealed class MeanSquaredError : Loss, ISaved<MeanSquaredError>
{
    /// <inheritdoc />
    public static string Name => "meanSquaredError";

    /// <inheritdoc />
    public static MeanSquaredError Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Identity;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var misses = backend.Subtract(outputs, answers);

        return backend.Mean(backend.Multiply(misses, misses));
    }
}

/// <summary>
/// The cross-entropy of each row's logits against its answers, taken as how the row's whole is divided among its classes:
/// one class, or shares of several.
/// </summary>
/// <remarks>
/// Against shares it is the Kullback–Leibler divergence plus the entropy of the answers, which no network moves, so one
/// loss serves a row that is one class and a row that is a distribution. The logits go through a log-softmax that shifts
/// each row by its largest value first, and a prediction is each row's shares.
/// </remarks>
public sealed class CrossEntropy : Loss, ISaved<CrossEntropy>
{
    private readonly double _smoothing;

    /// <summary>
    /// How much of each row's answers is spread evenly over its classes before it is compared: PyTorch's <c>label_smoothing</c> —
    /// a row of one class keeps one less this share on that class, and every class gets this share over how many there are.
    /// Nothing, unless said.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share of nothing or more and below one.</exception>
    /// <remarks>Written into the network's file only when there is some, which raises the part to the sixth version.</remarks>
    public double Smoothing
    {
        get => _smoothing;
        init => _smoothing = value.RequireSmoothing(nameof(value));
    }

    /// <inheritdoc />
    public static string Name => "crossEntropy";

    /// <inheritdoc />
    public static CrossEntropy Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { Smoothing = rebuilding.Number(settings, SmoothingExtensions.Key, 0) };
    }

    /// <inheritdoc />
    /// <remarks>How much each row's answers are spread, only where they are.</remarks>
    public void WriteSettings(Utf8JsonWriter writer) => writer.WriteSmoothing(Smoothing);

    /// <inheritdoc />
    public int FileVersion => Smoothing.VersionOfSmoothing();

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Softmax;

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The outputs are not a matrix of rows and classes.</exception>
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var classes = outputs.Shape.Rank == 2 ? outputs.Shape[1] : 1;
        var compared = Smoothing == 0
            ? answers
            : backend.Add(backend.Scale(answers, backend.Scalar(1 - Smoothing)), backend.Fill(answers.Shape, (float)(Smoothing / classes)));

        return backend.Scale(backend.Mean(backend.Multiply(compared, backend.LogSoftmax(outputs))), backend.Fill(new Shape(), -outputs.Shape[1]));
    }

    /// <inheritdoc />
    /// <remarks>Every share at least nothing, and a row's shares one, within the rounding of their sum.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) => answers.NotShares();
}

/// <summary>
/// The earth mover's distance between each row's shares and its answer, along the order of its columns: how far, column by
/// column, the predicted shares have to move to be the answer's — for answers that are shares of a whole in an order, as
/// bands of weight are.
/// </summary>
/// <remarks>
/// The size of the difference between the shares added up from the first column and the answers added up the same way,
/// summed over every column but the last — where both have added up to one — and averaged over the batch: the Wasserstein
/// distance scipy measures between the two over the columns' places, a distance of one being one column. A prediction is
/// each row's shares, as <see cref="CrossEntropy"/> gives them, and a row is refused as it refuses one. Only what the backend
/// already does is used: the shares are added up by multiplying with a triangle of ones, and the size of a number is the
/// rectified number and the rectified opposite, whose slope at nought is nought, as the slope of PyTorch's size is there.
/// <para>
/// With a remainder, the last answer is what is left of a whole once the columns before it are counted, which stands outside
/// their order: the columns before it are compared as shares of what they hold together, and what is left by the size of how
/// far its share is off, so a prediction is trained on the shape of the columns and on how much of the whole they hold, and
/// moving a share into what is left is never taken for moving it along the order.
/// </para>
/// </remarks>
public sealed class EarthMoversDistance : Loss, ISaved<EarthMoversDistance>
{
    private const string RemainderKey = "remainder";

    /// <summary>The distance along the order of every column.</summary>
    public EarthMoversDistance()
        : this(remainder: false)
    {
    }

    /// <summary>The distance along the order of the columns, the last of them what is left of the whole when said.</summary>
    /// <param name="remainder">Whether the last answer is what is left of the whole, outside the order of the others.</param>
    public EarthMoversDistance(bool remainder) => Remainder = remainder;

    /// <summary>Whether the last answer is what is left of the whole, outside the order of the others.</summary>
    public bool Remainder { get; }

    /// <inheritdoc />
    public static string Name => "earthMoversDistance";

    /// <inheritdoc />
    public static EarthMoversDistance Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Holds(settings, RemainderKey));
    }

    /// <inheritdoc />
    /// <remarks>Whether the last answer is what is left of the whole, only when it is.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        if (Remainder)
        {
            writer.WriteBoolean(RemainderKey, true);
        }
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Softmax;

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The outputs are not a matrix of rows and columns, or hold fewer than two columns in the order.</exception>
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var ordered = outputs.Shape.Rank == 2 ? outputs.Shape[1] - (Remainder ? 1 : 0) : 0;

        if (ordered < 2)
        {
            throw new ArgumentException(
                $"A distance along an order runs over a matrix of rows with two columns in the order at least{(Remainder ? ", and what is left after them" : string.Empty)}; these outputs are {outputs.Shape}.",
                nameof(outputs));
        }

        var shares = backend.Exp(backend.LogSoftmax(outputs));

        if (!Remainder)
        {
            return Summed(backend, backend.MatMul(backend.Subtract(shares, answers), Below(ordered)), ordered - 1);
        }

        var columns = Taking(outputs.Shape[1], ordered);
        var shape = backend.Subtract(AsShares(backend, backend.MatMul(shares, columns), ordered), AsShares(backend, backend.MatMul(answers, columns), ordered));
        var left = backend.MatMul(backend.Subtract(shares, answers), Last(outputs.Shape[1]));

        return backend.Add(Summed(backend, backend.MatMul(shape, Below(ordered)), ordered - 1), backend.Mean(backend.SizeOf(left)));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Every share at least nothing, and a row's shares one, as <see cref="CrossEntropy"/> refuses them; with a remainder, the
    /// columns before it hold something, or they have no shape to compare.
    /// </remarks>
    protected override string? Refused(IReadOnlyList<double> answers) =>
        answers.NotShares() ?? (Remainder && answers.Take(answers.Count - 1).Sum() <= 0
            ? "the columns before what is left hold nothing, so they have no order of shares to be measured along."
            : null);

    // Each row's columns as shares of what they hold together: each over the row's sum, which a square of ones spreads to
    // every column.
    private static Tensor AsShares(ITensorBackend backend, Tensor columns, int count) =>
        backend.Divide(columns, backend.MatMul(columns, Filled(count, count, (_, _) => true)));

    // The batch's mean of each row's sizes summed: the mean of every size, times how many a row holds.
    private static Tensor Summed(ITensorBackend backend, Tensor differences, int perRow) =>
        backend.Scale(backend.Mean(backend.SizeOf(differences)), backend.Fill(new Shape(), perRow));

    // A column added up to each threshold but the last: column i counts towards threshold j when it is not after it.
    private static Tensor Below(int count) => Filled(count, count - 1, (row, column) => row <= column);

    // The first columns of a row, as they are, and none after them.
    private static Tensor Taking(int count, int taken) => Filled(count, taken, (row, column) => row == column);

    // The last column of a row alone.
    private static Tensor Last(int count) => Filled(count, 1, (row, _) => row == count - 1);

    private static Tensor Filled(int rows, int columns, Func<int, int, bool> one) =>
        Tensor.From(new Shape(rows, columns), [.. Enumerable.Range(0, rows * columns).Select(at => one(at / columns, at % columns) ? 1f : 0f)]);
}

/// <summary>What a row's answers must be to be shares of one whole.</summary>
internal static class ShareExtensions
{
    extension(IReadOnlyList<double> answers)
    {
        /// <summary>Why a row's answers are no shares of one whole, when they are not.</summary>
        /// <returns>
        /// The reason — a share below nothing, or shares that do not sum to one within the rounding of their sum — or nothing.
        /// </returns>
        public string? NotShares()
        {
            if (answers.FirstOrDefault(share => share < 0) is var below and < 0)
            {
                return string.Create(CultureInfo.InvariantCulture, $"a share of {below} is below nothing, and a class's share of a row never is.");
            }

            var sum = answers.Sum();

            return Math.Abs(sum - 1) <= 1e-6 * Math.Max(1, answers.Count)
                ? null
                : string.Create(CultureInfo.InvariantCulture, $"its shares sum to {sum}, and a row's classes share one whole.");
        }
    }
}

/// <summary>
/// The cross-entropy of each output's logit against an answer between nothing and one: for answers that are one thing or
/// the other, one column for each.
/// </summary>
/// <remarks>
/// Worked out as the softplus of the logit less the logit where the answer is one, as PyTorch's loss on logits is: at a
/// logit of nothing it sends back a half less the answer, and no logit overflows it. A prediction is the probability of a
/// one.
/// </remarks>
public sealed class BinaryCrossEntropy : Loss, ISaved<BinaryCrossEntropy>
{
    private readonly double _smoothing;

    /// <summary>
    /// How far every answer is moved towards a half before it is compared: Keras's <c>label_smoothing</c> — an answer keeps one
    /// less this share of itself and gains half of it. Nothing, unless said.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share of nothing or more and below one.</exception>
    /// <remarks>
    /// PyTorch's loss on logits has no such setting, so this is Keras's. Written into the network's file only where there is some,
    /// which raises the part to the sixth version.
    /// </remarks>
    public double Smoothing
    {
        get => _smoothing;
        init => _smoothing = value.RequireSmoothing(nameof(value));
    }

    /// <inheritdoc />
    public static string Name => "binaryCrossEntropy";

    /// <inheritdoc />
    public static BinaryCrossEntropy Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new() { Smoothing = rebuilding.Number(settings, SmoothingExtensions.Key, 0) };
    }

    /// <inheritdoc />
    /// <remarks>How far each answer is moved towards a half, only where it is.</remarks>
    public void WriteSettings(Utf8JsonWriter writer) => writer.WriteSmoothing(Smoothing);

    /// <inheritdoc />
    public int FileVersion => Smoothing.VersionOfSmoothing();

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Sigmoid;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var compared = Smoothing == 0
            ? answers
            : backend.Add(backend.Scale(answers, backend.Scalar(1 - Smoothing)), backend.Fill(answers.Shape, (float)(Smoothing / 2)));

        return backend.Mean(backend.Subtract(backend.Softplus(outputs), backend.Multiply(outputs, compared)));
    }

    /// <inheritdoc />
    /// <remarks>Every answer between nothing and one: a label, or the probability of one.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) => answers.NotChances();
}

/// <summary>How a loss of classes says how much its answers are softened, in its file and in its settings.</summary>
internal static class SmoothingExtensions
{
    /// <summary>The setting a loss of classes writes its smoothing under.</summary>
    internal const string Key = "smoothing";

    extension(double smoothing)
    {
        /// <summary>The smoothing, when it is a share of nothing or more and below one.</summary>
        /// <exception cref="ArgumentOutOfRangeException">It is not.</exception>
        internal double RequireSmoothing(string parameter) =>
            smoothing is >= 0 and < 1
                ? smoothing
                : throw new ArgumentOutOfRangeException(parameter, smoothing, "How much the answers are softened is a share of nothing or more, below one.");

        /// <summary>The version of a network's part that can say this smoothing: the sixth where there is some, the second where there is none.</summary>
        internal int VersionOfSmoothing() => smoothing == 0 ? NetworkDocument.SecondVersion : NetworkDocument.SixthVersion;
    }

    extension(Utf8JsonWriter writer)
    {
        /// <summary>Writes the smoothing where there is some, and nothing where there is none, so an unsoftened loss is written as it always was.</summary>
        internal void WriteSmoothing(double smoothing)
        {
            ArgumentNullException.ThrowIfNull(writer);

            if (smoothing != 0)
            {
                writer.WriteNumber(Key, smoothing);
            }
        }
    }

    extension(IReadOnlyList<double> answers)
    {
        /// <summary>Why a row's answers are not each between nothing and one, when they are not.</summary>
        internal string? NotChances() =>
            answers.FirstOrDefault(answer => answer is < 0 or > 1, 0) is var outside and (< 0 or > 1)
                ? string.Create(CultureInfo.InvariantCulture, $"an answer of {outside} is not between nothing and one.")
                : null;

        /// <summary>Why a row's answers are not each a label of nought or one, when they are not.</summary>
        internal string? NotLabels() =>
            answers.FirstOrDefault(answer => answer is not (0 or 1), 0) is var other and not (0 or 1)
                ? string.Create(CultureInfo.InvariantCulture, $"an answer of {other} is no label: it is nought or one.")
                : null;
    }
}

/// <summary>The mean of the size of the difference between every output and its answer: for answers that are numbers, PyTorch's <c>L1Loss</c>.</summary>
/// <remarks>The size is the rectified difference and the rectified opposite, whose slope at nought is nought, as PyTorch's is.</remarks>
public sealed class MeanAbsoluteError : Loss, ISaved<MeanAbsoluteError>
{
    /// <inheritdoc />
    public static string Name => "meanAbsoluteError";

    /// <inheritdoc />
    public static MeanAbsoluteError Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Identity;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend) =>
        backend.Mean(backend.SizeOf(backend.Subtract(outputs, answers)));
}

/// <summary>
/// Half the square of the difference near nought and the difference's size away from it: PyTorch's <c>HuberLoss</c> and Keras's
/// <c>Huber</c>, for answers that are numbers some of which are far off.
/// </summary>
/// <remarks>
/// Within delta of the answer it is half the square of the miss, beyond it delta times the size of the miss less half of delta,
/// so the two meet with the same slope. PyTorch's <c>SmoothL1Loss</c> is this over delta, its <c>beta</c>, and is not a second kind.
/// </remarks>
public sealed class Huber : Loss, ISaved<Huber>
{
    private const string DeltaKey = "delta";

    /// <summary>The Huber loss bending at the given distance.</summary>
    /// <param name="delta">How far from the answer it stops being squared: a number above nothing, one in both PyTorch and Keras.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    public Huber(double delta = 1) =>
        Delta = double.IsFinite(delta) && delta > 0
            ? delta
            : throw new ArgumentOutOfRangeException(nameof(delta), delta, "A Huber loss bends at a distance from the answer above nothing.");

    /// <summary>How far from the answer it stops being squared.</summary>
    public double Delta { get; }

    /// <inheritdoc />
    public static string Name => "huber";

    /// <inheritdoc />
    public static Huber Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, DeltaKey));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(DeltaKey, Delta);
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Identity;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        // The size of the miss, the part of it within delta, and what is left beyond: half the square of the first, and delta
        // times the rest.
        var size = backend.SizeOf(backend.Subtract(outputs, answers));
        var within = backend.Subtract(size, backend.Relu(backend.Subtract(size, backend.Fill(size.Shape, (float)Delta))));
        var squared = backend.Scale(backend.Multiply(within, within), backend.Scalar(0.5));

        return backend.Mean(backend.Add(squared, backend.Scale(backend.Subtract(size, within), backend.Scalar(Delta))));
    }
}

/// <summary>
/// The pinball loss of a quantile: a miss above the answer weighed by one less the quantile, one below it by the quantile — for a
/// model that answers with the value the answer stays below that share of the time, rather than with its mean.
/// </summary>
/// <remarks>
/// Scikit-learn's <c>mean_pinball_loss</c> at that alpha; neither PyTorch nor Keras has one, so a network read from either is told
/// it by its caller. Each miss is a rectified difference, whose slope at nought is nought.
/// </remarks>
public sealed class Quantile : Loss, ISaved<Quantile>
{
    private const string QuantileKey = "quantile";

    /// <summary>The pinball loss of the given quantile.</summary>
    /// <param name="quantile">The share of the time the answer stays below the value asked for: above nothing and below one.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share above nothing and below one.</exception>
    public Quantile(double quantile) =>
        Share = quantile is > 0 and < 1
            ? quantile
            : throw new ArgumentOutOfRangeException(nameof(quantile), quantile, "A quantile is a share above nothing and below one.");

    /// <summary>The share of the time the answer stays below the value asked for.</summary>
    public double Share { get; }

    /// <inheritdoc />
    public static string Name => "quantile";

    /// <inheritdoc />
    public static Quantile Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, QuantileKey));
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(QuantileKey, Share);
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Identity;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var below = backend.Scale(backend.Relu(backend.Subtract(answers, outputs)), backend.Scalar(Share));
        var above = backend.Scale(backend.Relu(backend.Subtract(outputs, answers)), backend.Scalar(1 - Share));

        return backend.Mean(backend.Add(below, above));
    }
}

/// <summary>
/// The focal loss: a binary cross-entropy on logits, each label's weighed down by how sure of it the network already is — for
/// answers where one class is rare. The torchvision <c>sigmoid_focal_loss</c> of Lin et al.
/// </summary>
/// <remarks>
/// Each label's cross-entropy times one less the chance given to it, raised to gamma, and, where it is told one, times alpha for a
/// one and one less alpha for a nought. The answers are labels, nought or one, as torchvision's are: the power is the exponential of
/// gamma times the logarithm of a chance, and the logarithm of the chance a label is given is minus a softplus of the logit, which
/// is never nothing, so a network certain of a label sends back no number that is not one. A prediction is the chance of a one.
/// </remarks>
public sealed class Focal : Loss, ISaved<Focal>
{
    private const string GammaKey = "gamma";
    private const string AlphaKey = "alpha";

    private readonly double? _alpha;

    /// <summary>The focal loss of the given gamma, without a weight for the rare class.</summary>
    /// <param name="gamma">How hard a label the network is sure of is weighed down: nothing or more, two in torchvision and Keras.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number of nothing or more.</exception>
    public Focal(double gamma = 2) =>
        Gamma = double.IsFinite(gamma) && gamma >= 0
            ? gamma
            : throw new ArgumentOutOfRangeException(nameof(gamma), gamma, "A focal loss weighs a sure label down by a power of nothing or more.");

    /// <summary>How hard a label the network is sure of is weighed down.</summary>
    public double Gamma { get; }

    /// <summary>
    /// What a one is weighed by, and a nought by one less it: a share between nothing and one, a quarter in torchvision; nothing,
    /// unless said, for no weight at all, as Keras leaves it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share between nothing and one.</exception>
    public double? Alpha
    {
        get => _alpha;
        init => _alpha = value is null or (>= 0 and <= 1)
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "A focal loss weighs a one by a share between nothing and one.");
    }

    /// <inheritdoc />
    public static string Name => "focal";

    /// <inheritdoc />
    public static Focal Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Number(settings, GammaKey)) { Alpha = FromWritten(rebuilding.Number(settings, AlphaKey)) };
    }

    /// <inheritdoc />
    /// <remarks>Gamma and alpha, alpha written as minus one where there is no weight, as torchvision writes it.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(GammaKey, Gamma);
        writer.WriteNumber(AlphaKey, Alpha ?? NoWeight);
    }

    /// <summary>How torchvision, a network's file and a declaration write a focal loss without a weight for the rare class: an alpha of minus one.</summary>
    public const double NoWeight = -1;

    /// <summary>The alpha a written number stands for: nothing for minus one, the number itself otherwise.</summary>
    /// <param name="written">The number written.</param>
    /// <returns>The alpha.</returns>
    public static double? FromWritten(double written) => written == NoWeight ? null : written;

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Sigmoid;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var ones = backend.Fill(answers.Shape, 1f);
        var noughts = backend.Subtract(ones, answers);
        var crossEntropy = backend.Subtract(backend.Softplus(outputs), backend.Multiply(outputs, answers));

        // The logarithm of one less the chance given to the label: of the chance of a nought where it is a one, and of a one
        // where it is a nought — each minus a softplus, which no logit makes infinite.
        var missed = backend.Add(
            backend.Multiply(answers, backend.Scale(backend.Softplus(outputs), backend.Scalar(-1))),
            backend.Multiply(noughts, backend.Scale(backend.Softplus(backend.Scale(outputs, backend.Scalar(-1))), backend.Scalar(-1))));
        var focused = backend.Multiply(crossEntropy, backend.Exp(backend.Scale(missed, backend.Scalar(Gamma))));

        return Alpha is not { } alpha
            ? backend.Mean(focused)
            : backend.Mean(backend.Multiply(focused, backend.Add(backend.Scale(answers, backend.Scalar(alpha)), backend.Scale(noughts, backend.Scalar(1 - alpha)))));
    }

    /// <inheritdoc />
    /// <remarks>Every answer a label, nought or one, as torchvision's focal loss takes them.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) => answers.NotLabels();
}

/// <summary>
/// The hinge: how far each output falls short of a margin of one on its label's side of nought — Keras's <c>hinge</c>, for answers
/// that are one thing or the other.
/// </summary>
/// <remarks>
/// A label of nought is taken as minus one and a label of one as one, as Keras takes them, and each output loses the rectified
/// difference between one and itself times that sign. What a network trained with it gives is a margin, not a chance, so a
/// prediction is the side of nought it falls: a one above nought, a nought at or below it.
/// </remarks>
public sealed class Hinge : Loss, ISaved<Hinge>
{
    /// <inheritdoc />
    public static string Name => "hinge";

    /// <inheritdoc />
    public static Hinge Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Margin;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend)
    {
        var ones = backend.Fill(answers.Shape, 1f);
        var signs = backend.Subtract(backend.Scale(answers, backend.Scalar(2)), ones);

        return backend.Mean(backend.Relu(backend.Subtract(ones, backend.Multiply(signs, outputs))));
    }

    /// <inheritdoc />
    /// <remarks>Every answer a label, nought or one.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) => answers.NotLabels();
}

/// <summary>
/// The negative log-likelihood of a Poisson count: for answers that are counts, each output the logarithm of its rate or the rate
/// itself — PyTorch's <c>PoissonNLLLoss</c>, and Keras's <c>poisson</c> on rates.
/// </summary>
/// <remarks>
/// On logarithms it is the exponential of the output less the count times the output, and a prediction is the exponential of the
/// output: the rate. On rates it is the output less the count times the logarithm of the output and a small number added, so a rate
/// of nought has a logarithm; a network that answers with a rate keeps it above nought itself. The constant terms of the
/// likelihood are left out, as both leave them out unless told otherwise.
/// </remarks>
public sealed class Poisson : Loss, ISaved<Poisson>
{
    private const string LogInputKey = "logInput";
    private const string EpsilonKey = "epsilon";

    private readonly double _epsilon = 1e-8;

    /// <summary>The Poisson loss on the logarithms of the rates, or on the rates.</summary>
    /// <param name="logInput">Whether every output is the logarithm of a rate, as PyTorch takes it unless told otherwise; or the rate itself, as Keras takes it.</param>
    public Poisson(bool logInput) => LogInput = logInput;

    /// <summary>Whether every output is the logarithm of a rate rather than the rate itself.</summary>
    public bool LogInput { get; }

    /// <summary>What is added to a rate before its logarithm is taken: a hundred-millionth, unless said, as PyTorch leaves it; Keras adds a ten-millionth.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a finite number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = double.IsFinite(value) && value > 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "What is added to a rate before its logarithm is taken is a number above nothing.");
    }

    /// <inheritdoc />
    public static string Name => "poisson";

    /// <inheritdoc />
    public static Poisson Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Holds(settings, LogInputKey)) { Epsilon = rebuilding.Number(settings, EpsilonKey) };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBoolean(LogInputKey, LogInput);
        writer.WriteNumber(EpsilonKey, Epsilon);
    }

    /// <inheritdoc />
    public override OutputActivation Activation => LogInput ? OutputActivation.Exp : OutputActivation.Identity;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend) =>
        LogInput
            ? backend.Mean(backend.Subtract(backend.Exp(outputs), backend.Multiply(answers, outputs)))
            : backend.Mean(backend.Subtract(outputs, backend.Multiply(answers, backend.Log(backend.Add(outputs, backend.Fill(outputs.Shape, (float)Epsilon))))));

    /// <inheritdoc />
    /// <remarks>Every answer a count: nothing or more.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) =>
        answers.FirstOrDefault(answer => answer < 0, 0) is var below and < 0
            ? string.Create(CultureInfo.InvariantCulture, $"an answer of {below} is below nothing, and a count never is.")
            : null;
}
