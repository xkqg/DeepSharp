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
/// activation to what is served. Each loss also says which answers it could have meant, and a row handed over with any
/// other is refused before anything is trained on it.
/// </remarks>
public abstract class Loss
{
    /// <summary>What the outputs go through on their way out as predictions.</summary>
    public abstract OutputActivation Activation { get; }

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
    protected override string? Refused(IReadOnlyList<double> answers)
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
