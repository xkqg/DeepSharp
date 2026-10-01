// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Brings the values of the last axis to a mean of nothing and a spread of one, then scales and shifts each feature by
/// numbers it learns.
/// </summary>
/// <remarks>
/// The last axis is a row's features, or an image's channels, which come last here. What the two normalisations share is
/// written once, here — taking the batch as rows of that axis, the scale and the shift, and what is added to a variance
/// before its root is taken — and each says only what it measures the mean and the spread over.
/// </remarks>
public abstract class Normalisation : Layer
{
    private readonly double _epsilon = 1e-5;

    private protected Normalisation(int features)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(features, 1);

        Weight = AddParameter("weight", Filled(features, 1f));
        Bias = AddParameter("bias", Filled(features, 0f));
    }

    /// <summary>What each normalised feature is scaled by; one, to start with.</summary>
    public Parameter Weight { get; }

    /// <summary>What each scaled feature is shifted by; nothing, to start with.</summary>
    public Parameter Bias { get; }

    /// <summary>How many features, or channels, the last axis holds.</summary>
    public int Features => Weight.Value.Shape[0];

    /// <summary>What is added to each variance before its root is taken, so nothing is divided by nothing; a hundred-thousandth, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public double Epsilon
    {
        get => _epsilon;
        init => _epsilon = RequireEpsilon(value, nameof(value));
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The last axis holds another number of features.</exception>
    protected sealed override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank < 2 || input.Shape[input.Shape.Rank - 1] != Features)
        {
            throw new ArgumentException(
                $"A normalisation of {Features} features takes a batch whose last axis holds them, and was handed a {input.Shape} one.",
                nameof(input));
        }

        var backend = pass.Backend;
        var rows = backend.Reshape(input, new Shape(input.Shape.Count / Features, Features));
        var normalised = Normalise(rows, pass);

        return backend.Reshape(
            backend.AddRow(backend.Multiply(normalised, backend.RowsOf(Weight.Value, rows.Shape[0])), Bias.Value), input.Shape);
    }

    /// <summary>The rows brought to a mean of nothing and a spread of one, over whatever this normalisation measures them.</summary>
    /// <param name="rows">A row for every place, a column for every feature.</param>
    /// <param name="pass">The pass: whether it trains, and the backend.</param>
    /// <returns>The normalised rows, before the scale and the shift.</returns>
    private protected abstract Tensor Normalise(Tensor rows, Pass pass);

    /// <summary>The spread a variance stands for: its root, once <see cref="Epsilon"/> is added.</summary>
    private protected Tensor Spread(ITensorBackend backend, Tensor variance) =>
        backend.Sqrt(backend.Add(variance, backend.Fill(variance.Shape, (float)Epsilon)));

    private protected static Tensor Filled(int count, float value) => Tensor.From(new Shape(count), [.. Enumerable.Repeat(value, count)]);

    /// <summary>What may be added to a variance before its root is taken: a finite number above nothing, refused otherwise.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not one.</exception>
    internal static double RequireEpsilon(double value, string parameter) =>
        double.IsFinite(value) && value > 0
            ? value
            : throw new ArgumentOutOfRangeException(parameter, value, "What is added to a variance is a number above nothing.");
}

/// <summary>
/// A batch normalisation: each feature measured over the batch while the network trains, and over the running mean and
/// variance those measurements left once it has trained.
/// </summary>
/// <remarks>
/// The running statistics are kept as PyTorch keeps them: moved a tenth of the way to each training batch's mean, and to
/// its variance counted over one row fewer. They are measured on the training rows alone, so the rows a network is
/// measured on shape nothing it learned — the same rule the pipeline keeps for everything it learns — and an evaluation
/// pass uses them without moving them. Keras moves its running variance towards the batch's variance counted over every
/// row, so one described in Keras's words keeps the running mean Keras keeps, and a running variance that takes each
/// batch's in a share larger by the batch's rows over one fewer; what it answers from them is Keras's.
/// </remarks>
public sealed class BatchNorm : Normalisation, ISaved<BatchNorm>
{
    private readonly double _momentum = 0.1;

    /// <summary>A batch normalisation of so many features, starting at a scale of one and a shift of nothing.</summary>
    /// <param name="features">How many features, or channels, the last axis holds.</param>
    /// <exception cref="ArgumentOutOfRangeException">It holds fewer than one.</exception>
    public BatchNorm(int features)
        : base(features)
    {
        RunningMean = AddRunningStatistic("running_mean", Filled(features, 0f));
        RunningVariance = AddRunningStatistic("running_var", Filled(features, 1f));
    }

    /// <inheritdoc />
    public static string Name => "batchNorm";

    /// <summary>Each feature's mean, as the training batches measured it.</summary>
    public RunningStatistic RunningMean { get; }

    /// <summary>Each feature's variance, as the training batches measured it.</summary>
    public RunningStatistic RunningVariance { get; }

    /// <summary>How far the running statistics move towards each training batch's: a tenth, unless said, as PyTorch leaves it.</summary>
    /// <remarks>
    /// PyTorch's meaning of the word, the share of the new measurement; Keras's momentum is the share of the old one, and a
    /// description in Keras's words, <see cref="Sequential.BatchNorm(double, double)"/>, takes Keras's and keeps its
    /// complement here.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">It is not a share between nothing and one.</exception>
    public double Momentum
    {
        get => _momentum;
        init => _momentum = IsShare(value)
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "Momentum is a share between nothing and one.");
    }

    /// <inheritdoc />
    public static BatchNorm Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "features"))
        {
            Momentum = rebuilding.Number(settings, "momentum"),
            Epsilon = rebuilding.Number(settings, "epsilon"),
        };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("features", Features);
        writer.WriteNumber("momentum", Momentum);
        writer.WriteNumber("epsilon", Epsilon);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">A training pass hands it a single row, which has no spread to measure.</exception>
    private protected override Tensor Normalise(Tensor rows, Pass pass)
    {
        var backend = pass.Backend;
        var count = rows.Shape[0];
        Tensor mean;
        Tensor variance;

        if (pass.Mode == PassMode.Training)
        {
            if (count < 2)
            {
                throw new ArgumentException(
                    "A batch normalisation measures each feature over the batch, and a batch of one row has no spread to measure.",
                    nameof(rows));
            }

            mean = backend.Scale(backend.SumRows(rows), backend.Scalar(1.0 / count));
            var centred = backend.Subtract(rows, backend.RowsOf(mean, count));
            variance = backend.Scale(backend.SumRows(backend.Multiply(centred, centred)), backend.Scalar(1.0 / count));

            RunningMean.Update(Moved(backend, RunningMean.Value, mean, 1), pass);
            RunningVariance.Update(Moved(backend, RunningVariance.Value, variance, count / (count - 1.0)), pass);
        }
        else
        {
            mean = RunningMean.Value;
            variance = RunningVariance.Value;
        }

        return backend.Divide(backend.Subtract(rows, backend.RowsOf(mean, count)), backend.RowsOf(Spread(backend, variance), count));
    }

    /// <summary>Whether a momentum is a share, between nothing and one: PyTorch's, or Keras's, its complement, alike.</summary>
    internal static bool IsShare(double momentum) => momentum is >= 0 and <= 1;

    // The running statistic moved towards what this batch measured, the measurement first corrected by the given factor.
    private Tensor Moved(ITensorBackend backend, Tensor running, Tensor measured, double correction) =>
        backend.Add(backend.Scale(running, backend.Scalar(1 - Momentum)), backend.Scale(measured, backend.Scalar(Momentum * correction)));
}

/// <summary>
/// A layer normalisation: each row measured over its own features, the same in training and afterwards.
/// </summary>
/// <remarks>It keeps no running statistic, and does not care how many rows a batch holds.</remarks>
public sealed class LayerNorm : Normalisation, ISaved<LayerNorm>
{
    /// <summary>A layer normalisation of so many features, starting at a scale of one and a shift of nothing.</summary>
    /// <param name="features">How many features the last axis holds.</param>
    /// <exception cref="ArgumentOutOfRangeException">It holds fewer than one.</exception>
    public LayerNorm(int features)
        : base(features)
    {
    }

    /// <inheritdoc />
    public static string Name => "layerNorm";

    /// <inheritdoc />
    public static LayerNorm Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, "features")) { Epsilon = rebuilding.Number(settings, "epsilon") };
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("features", Features);
        writer.WriteNumber("epsilon", Epsilon);
    }

    /// <inheritdoc />
    private protected override Tensor Normalise(Tensor rows, Pass pass)
    {
        var backend = pass.Backend;
        var share = backend.Scalar(1.0 / Features);

        var mean = backend.Scale(backend.SumRows(backend.Transpose(rows)), share);
        var centred = backend.Subtract(rows, backend.ColumnsOf(mean, Features));
        var variance = backend.Scale(backend.SumRows(backend.Transpose(backend.Multiply(centred, centred))), share);

        return backend.Divide(centred, backend.ColumnsOf(Spread(backend, variance), Features));
    }
}
