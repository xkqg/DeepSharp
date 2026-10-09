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
    internal bool Applies(Layer layer) => layer is Sigmoid && Activation == OutputActivation.Sigmoid;

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
    /// <inheritdoc />
    public static string Name => "crossEntropy";

    /// <inheritdoc />
    public static CrossEntropy Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Softmax;

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The outputs are not a matrix of rows and classes.</exception>
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend) =>
        backend.Scale(backend.Mean(backend.Multiply(answers, backend.LogSoftmax(outputs))), backend.Fill(new Shape(), -outputs.Shape[1]));

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

        return backend.Add(Summed(backend, backend.MatMul(shape, Below(ordered)), ordered - 1), backend.Mean(Size(backend, left)));
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
        backend.Scale(backend.Mean(Size(backend, differences)), backend.Fill(new Shape(), perRow));

    // The size of every number: the rectified number and the rectified opposite, whose slope at nought is nought.
    private static Tensor Size(ITensorBackend backend, Tensor values) =>
        backend.Add(backend.Relu(values), backend.Relu(backend.Scale(values, backend.Fill(new Shape(), -1))));

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
    /// <inheritdoc />
    public static string Name => "binaryCrossEntropy";

    /// <inheritdoc />
    public static BinaryCrossEntropy Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    public override OutputActivation Activation => OutputActivation.Sigmoid;

    /// <inheritdoc />
    protected override Tensor Computed(Tensor outputs, Tensor answers, ITensorBackend backend) =>
        backend.Mean(backend.Subtract(backend.Softplus(outputs), backend.Multiply(outputs, answers)));

    /// <inheritdoc />
    /// <remarks>Every answer between nothing and one: a label, or the probability of one.</remarks>
    protected override string? Refused(IReadOnlyList<double> answers) =>
        answers.FirstOrDefault(answer => answer is < 0 or > 1, 0) is var outside and (< 0 or > 1)
            ? string.Create(CultureInfo.InvariantCulture, $"an answer of {outside} is not between nothing and one.")
            : null;
}
