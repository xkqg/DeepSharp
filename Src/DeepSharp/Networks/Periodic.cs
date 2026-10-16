// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How a periodic embedding is shaped: how many frequencies each feature has, how they start, and how wide each feature comes out.</summary>
/// <param name="Frequencies">How many frequencies each feature is turned round at: one at least.</param>
/// <param name="Sigma">
/// The spread of the normal curve the frequencies start from: a number above nothing, and one that has no default, since the right
/// one depends on the data — its authors searched it for every dataset, and it is the setting the embedding is most sensitive to.
/// </param>
/// <param name="Width">How many numbers each feature comes out as: one at least.</param>
public readonly record struct Periodicity(int Frequencies, double Sigma, int Width)
{
    /// <summary>Whether one linear layer is shared by every feature, rather than one of each feature's own: the authors' 'lite' form.</summary>
    public bool Lite { get; init; }
}

/// <summary>
/// The periodic embedding of a number: each feature times frequencies it learns, a whole turn round, as its cosines and its sines,
/// and those through a linear layer — of each feature's own, or one shared by every feature.
/// </summary>
/// <remarks>
/// Gorishniy, Rubachev and Babenko's embedding for numbers in a network over a table ("On Embeddings for Numerical Features in
/// Tabular Deep Learning", 2022), as their package <c>rtdl_num_embeddings</c> writes it: for a feature v and its frequencies c,
/// the cosines of 2πcv followed by the sines, then a linear layer. Followed by a rectifier, it is their PLR, and that is a
/// <see cref="Relu"/> after it here. A row of features comes out as a row of each feature's numbers, one feature after another.
/// <para>
/// Its slots are <c>frequencies</c> — the features by the frequencies, starting from a normal curve of the spread it is told, cut
/// three spreads out as the authors cut it — and <c>weight</c> and <c>bias</c>. The weight is the cosines and the sines of one
/// feature by every feature's numbers, so a feature's own block of it is that feature's linear layer laid on its side; shared, it
/// is the cosines and the sines by the numbers, as a <see cref="Dense"/> layer lays its weights. Both start as a linear layer of
/// that many inputs does in PyTorch. Every feature reaches only its own numbers, by products of values rather than by a matrix
/// of mostly noughts, so it costs what the arithmetic costs and nothing is added to the seam.
/// </para>
/// </remarks>
public sealed class Periodic : Layer, ISaved<Periodic>
{
    private const string FeaturesKey = "features";
    private const string FrequenciesKey = "frequencies";
    private const string SigmaKey = "sigma";
    private const string WidthKey = "width";
    private const string LiteKey = "lite";

    /// <summary>A periodic embedding of so many features, starting from the given draws: the frequencies first, then the weight and the bias.</summary>
    /// <param name="features">How many features a row holds.</param>
    /// <param name="periodicity">How many frequencies each feature has, the spread they start from, and how wide each feature comes out.</param>
    /// <param name="draws">The draws its start is taken from.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// There are fewer than one feature, frequency or number out, or the spread is not a finite number above nothing.
    /// </exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public Periodic(int features, Periodicity periodicity, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(features, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(periodicity.Frequencies, 1, nameof(periodicity));
        ArgumentOutOfRangeException.ThrowIfLessThan(periodicity.Width, 1, nameof(periodicity));

        if (!double.IsFinite(periodicity.Sigma) || periodicity.Sigma <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(periodicity), periodicity.Sigma, "The frequencies start from a normal curve whose spread is a number above nothing.");
        }

        Features = features;
        Periodicity = periodicity;

        var waves = 2 * periodicity.Frequencies;
        var outputs = periodicity.Lite ? periodicity.Width : features * periodicity.Width;
        var bound = 1 / Math.Sqrt(waves);

        Frequencies = AddParameter(FrequenciesKey, Drawn(new Shape(features, periodicity.Frequencies), () => Cut(draws, periodicity.Sigma)));
        Weight = AddParameter("weight", Drawn(new Shape(waves, outputs), () => bound * ((2 * draws.NextDouble()) - 1)));
        Bias = AddParameter("bias", Drawn(new Shape(outputs), () => bound * ((2 * draws.NextDouble()) - 1)));
    }

    /// <summary>How many features a row holds.</summary>
    public int Features { get; }

    /// <summary>How many frequencies each feature has, the spread they started from, and how wide each feature comes out.</summary>
    public Periodicity Periodicity { get; }

    /// <summary>The frequencies: the features by the frequencies of each.</summary>
    public Parameter Frequencies { get; }

    /// <summary>
    /// The linear layer's weight: the cosines and the sines of a feature by every feature's numbers, each feature's own block its own;
    /// or, shared, by the numbers of one.
    /// </summary>
    public Parameter Weight { get; }

    /// <summary>The linear layer's bias: every feature's numbers, or, shared, one feature's.</summary>
    public Parameter Bias { get; }

    /// <inheritdoc />
    public static string Name => "periodic";

    /// <inheritdoc />
    public static Periodic Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var periodicity = new Periodicity(rebuilding.Whole(settings, FrequenciesKey), rebuilding.Number(settings, SigmaKey), rebuilding.Whole(settings, WidthKey))
        {
            Lite = rebuilding.Holds(settings, LiteKey),
        };

        return new(rebuilding.Whole(settings, FeaturesKey), periodicity, rebuilding.Draws);
    }

    /// <inheritdoc />
    /// <remarks>Every setting, whether shared or not included: a new kind writes all it was built with.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(FeaturesKey, Features);
        writer.WriteNumber(FrequenciesKey, Periodicity.Frequencies);
        writer.WriteNumber(SigmaKey, Periodicity.Sigma);
        writer.WriteNumber(WidthKey, Periodicity.Width);
        writer.WriteBoolean(LiteKey, Periodicity.Lite);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of rows of as many features as the embedding has.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2 || input.Shape[1] != Features)
        {
            throw new ArgumentException($"A periodic embedding of {Features} features takes a batch of rows of them, and was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var rows = input.Shape[0];
        var count = Periodicity.Frequencies;

        // Every feature beside each of its frequencies, times it, a whole turn round: a row of the features' angles, feature by feature.
        var spread = backend.Reshape(backend.MatMul(backend.Reshape(input, new Shape(rows * Features, 1)), backend.Fill(new Shape(1, count), 1f)), new Shape(rows, Features * count));
        var angles = backend.Scale(backend.Multiply(spread, backend.RowsOf(backend.Reshape(Frequencies.Value, new Shape(Features * count)), rows)), backend.Scalar(2 * Math.PI));
        var cosines = backend.Sin(backend.Add(angles, backend.Fill(angles.Shape, (float)(Math.PI / 2))));
        var sines = backend.Sin(angles);

        var made = backend.Add(Through(backend, cosines, Half(backend, first: true), rows), Through(backend, sines, Half(backend, first: false), rows));

        return Periodicity.Lite
            ? backend.Reshape(backend.AddRow(made, Bias.Value), new Shape(rows, Features * Periodicity.Width))
            : backend.AddRow(made, Bias.Value);
    }

    // The cosines' or the sines' rows of the weight, taken by a product with a matrix that picks them, so the gradient reaches the
    // rows the weight holds.
    private Tensor Half(ITensorBackend backend, bool first)
    {
        var count = Periodicity.Frequencies;
        var picks = new float[count * 2 * count];

        for (var at = 0; at < count; at++)
        {
            picks[(at * 2 * count) + at + (first ? 0 : count)] = 1f;
        }

        return backend.MatMul(Tensor.From(new Shape(count, 2 * count), picks), Weight.Value);
    }

    // The waves of every feature through the half of the weight that reads them. Shared, every feature's waves are a row of their
    // own times the weight. Each feature's own, a wave meets the weight only in its own feature's block: the waves and the weight
    // are each laid out along every row, feature and number out, a wave a row, multiplied, and the waves added up.
    private Tensor Through(ITensorBackend backend, Tensor waves, Tensor weight, int rows)
    {
        var count = Periodicity.Frequencies;
        var width = Periodicity.Width;
        var each = backend.Reshape(waves, new Shape(rows * Features, count));

        if (Periodicity.Lite)
        {
            return backend.MatMul(each, weight);
        }

        var laid = rows * Features * width;
        var wavesLaid = backend.Reshape(
            backend.MatMul(backend.Reshape(backend.Transpose(each), new Shape(count * rows * Features, 1)), backend.Fill(new Shape(1, width), 1f)),
            new Shape(count, laid));
        var weightLaid = backend.Transpose(backend.Reshape(
            backend.MatMul(backend.Fill(new Shape(rows, 1), 1f), backend.Reshape(backend.Transpose(weight), new Shape(1, Features * width * count))),
            new Shape(laid, count)));

        return backend.Reshape(backend.SumRows(backend.Multiply(wavesLaid, weightLaid)), new Shape(rows, Features * width));
    }

    // A number from a normal curve of the given spread, drawn again until it lies within three spreads of nought, as PyTorch's
    // truncated normal cuts it.
    private static double Cut(Draws draws, double sigma)
    {
        double value;

        do
        {
            value = draws.NextNormal();
        }
        while (Math.Abs(value) > 3);

        return value * sigma;
    }

    private static Tensor Drawn(Shape shape, Func<double> next) => Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(_ => (float)next())]);
}
