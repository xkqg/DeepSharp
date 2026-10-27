// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// One set of weights answering as several: every row of a batch multiplied, before the layers that follow, by a
/// different one of several learned adapters, so a shared stack of layers after it runs once and answers as that many
/// virtual members. Wen, Tran and Ba's BatchEnsemble; the one adapter TabM-mini keeps of it (Gorishniy, Kotelnikov and
/// Babenko's TabM, which reports TabM-mini performing as well as the fuller form that also shrinks and shifts).
/// </summary>
/// <remarks>
/// Its own adapter is the only part not shared: a matrix of as many rows as there are variants, each row a learned scale
/// for every input. A batch of rows in is <see cref="Variants"/> times as many rows out, laid out variant by variant
/// within the place of each row of the batch that made them — row <c>b * Variants + v</c> is the batch's row <c>b</c>
/// times the adapter's row <c>v</c> — so the shared layers after it (a <see cref="Dense"/>, an activation, a
/// <see cref="Dropout"/>) read it as an ordinary, larger batch and need no change at all to answer every variant at
/// once: each already treats any row the same as any other. <see cref="VariantHeads"/> is what closes the variants
/// again, at the end.
/// </remarks>
public sealed class BatchEnsemble : Layer, ISaved<BatchEnsemble>
{
    private const string VariantsKey = "variants";
    private const string InputsKey = "inputs";

    /// <summary>A BatchEnsemble adapter for the given width, starting at a random sign for every variant and every input.</summary>
    /// <param name="variants">How many virtual members the adapter gives the shared layers after it; at least one.</param>
    /// <param name="inputs">How many values each row holds.</param>
    /// <param name="draws">The draws its start is taken from.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one variant or one input.</exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public BatchEnsemble(int variants, int inputs, Draws draws)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(variants, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(inputs, 1);
        ArgumentNullException.ThrowIfNull(draws);

        Variants = variants;
        Inputs = inputs;
        Adapter = AddParameter("adapter", Signs(variants, inputs, draws));
    }

    /// <summary>How many virtual members the adapter gives the shared layers after it.</summary>
    public int Variants { get; }

    /// <summary>How many values each row holds.</summary>
    public int Inputs { get; }

    /// <summary>The adapter: as many rows as there are variants, each a learned scale for every input.</summary>
    public Parameter Adapter { get; }

    /// <inheritdoc />
    public static string Name => "batchEnsemble";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static BatchEnsemble Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, VariantsKey), rebuilding.Whole(settings, InputsKey), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(VariantsKey, Variants);
        writer.WriteNumber(InputsKey, Inputs);
    }

    /// <inheritdoc />
    /// <remarks>Takes a batch of rows of <see cref="Inputs"/> values, and hands the layers after it a batch of rows of <see cref="Variants"/>
    /// of them — the variants axis prepended to the one row every example was, as a series' steps are.</remarks>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2 || input.Shape[1] != Inputs)
        {
            throw new ArgumentException(
                $"A BatchEnsemble adapter of {Inputs} inputs takes a batch of rows of them, and was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var batch = input.Shape[0];
        var tiledInput = backend.MatMul(backend.Transpose(RepeatEachRow(batch, Variants)), input);
        var tiledAdapter = backend.MatMul(backend.Transpose(TileWhole(batch, Variants)), Adapter.Value);
        var laid = backend.Multiply(tiledInput, tiledAdapter);

        return backend.Reshape(laid, new Shape(batch, Variants, Inputs));
    }

    /// <summary>
    /// A matrix of noughts and ones that, by a product with a batch's rows, repeats each one <paramref name="times"/> times
    /// in a row: row <c>b * times + t</c> of the product is row <c>b</c> of the batch, for every <c>t</c>.
    /// </summary>
    internal static Tensor RepeatEachRow(int rows, int times) =>
        Selection.Of(rows, rows * times, (row, column) => (column / times) == row);

    /// <summary>
    /// A matrix of noughts and ones that, by a product with a matrix of <paramref name="variants"/> rows, lays the whole of
    /// it out <paramref name="copies"/> times: row <c>c * variants + v</c> of the product is row <c>v</c> of the matrix, for
    /// every <c>c</c>.
    /// </summary>
    internal static Tensor TileWhole(int copies, int variants) =>
        Selection.Of(variants, copies * variants, (row, column) => (column % variants) == row);

    // Every variant's share of every input starts at a random sign, as Wen, Tran and Ba's own BatchEnsemble draws it.
    private static Tensor Signs(int variants, int inputs, Draws draws)
    {
        var values = new float[variants * inputs];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = draws.NextDouble() < 0.5 ? -1f : 1f;
        }

        return Tensor.From(new Shape(variants, inputs), values);
    }
}

/// <summary>
/// Closes the variants a <see cref="BatchEnsemble"/> opened: as many separate, un-shared projections as there are
/// variants, each answering only its own share of the rows, its outputs then the mean of them all — TabM's own non-shared
/// prediction heads (Gorishniy, Kotelnikov and Babenko: "the k prediction branches should not interact with each other").
/// </summary>
/// <remarks>
/// Every variant's projection is its own weights and its own bias; none are shared, unlike a <see cref="Dense"/> after a
/// <see cref="BatchEnsemble"/>, which is the same for every variant. While a pass trains, <see cref="LaidOverVariants.OfVariants"/>
/// takes a loss across every variant's own answer, the mean of <see cref="Variants"/> losses rather than the loss of their
/// mean, as TabM-mini's own training does; while it only answers, <see cref="LaidOverVariants.VariantPredictions"/> brings each
/// variant's raw output through the loss's own activation first and takes the mean of those, in the answer's own units, as a
/// deep ensemble's members are averaged.
/// </remarks>
public sealed class VariantHeads : Layer, ISaved<VariantHeads>
{
    private const string VariantsKey = "variants";
    private const string InputsKey = "inputs";
    private const string OutputsKey = "outputs";

    /// <summary>As many separate projections as there are variants, starting as PyTorch's linear layers do.</summary>
    /// <param name="variants">How many separate projections it holds; at least one.</param>
    /// <param name="inputs">How many values each variant's own row holds.</param>
    /// <param name="outputs">How many values each variant's projection makes of its row.</param>
    /// <param name="draws">The draws its start is taken from.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than one variant, one input or one output.</exception>
    /// <exception cref="ArgumentNullException">There are no draws.</exception>
    public VariantHeads(int variants, int inputs, int outputs, Draws draws)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(variants, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(inputs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outputs, 1);
        ArgumentNullException.ThrowIfNull(draws);

        Variants = variants;
        Inputs = inputs;
        Outputs = outputs;

        var fans = new Fans(inputs, outputs);
        var weights = new float[variants * inputs * outputs];
        var biases = new float[variants * outputs];

        for (var variant = 0; variant < variants; variant++)
        {
            var one = new KaimingUniform().Draw(new Shape(inputs, outputs), fans, draws);
            var bias = new FanInUniform().Draw(new Shape(outputs), fans, draws);

            Array.Copy(one.Values.ToArray(), 0, weights, variant * inputs * outputs, inputs * outputs);
            Array.Copy(bias.Values.ToArray(), 0, biases, variant * outputs, outputs);
        }

        Weight = AddParameter("weight", Tensor.From(new Shape(variants * inputs, outputs), weights));
        Bias = AddParameter("bias", Tensor.From(new Shape(variants, outputs), biases));
    }

    /// <summary>How many separate projections it holds.</summary>
    public int Variants { get; }

    /// <summary>How many values each variant's own row holds.</summary>
    public int Inputs { get; }

    /// <summary>How many values each variant's projection makes of its row.</summary>
    public int Outputs { get; }

    /// <summary>Every variant's weights, laid end to end: <see cref="Variants"/> times <see cref="Inputs"/> rows by <see cref="Outputs"/> columns.</summary>
    public Parameter Weight { get; }

    /// <summary>Every variant's bias, a row each.</summary>
    public Parameter Bias { get; }

    /// <inheritdoc />
    public static string Name => "variantHeads";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static VariantHeads Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            rebuilding.Whole(settings, VariantsKey),
            rebuilding.Whole(settings, InputsKey),
            rebuilding.Whole(settings, OutputsKey),
            rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(VariantsKey, Variants);
        writer.WriteNumber(InputsKey, Inputs);
        writer.WriteNumber(OutputsKey, Outputs);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of rows laid out by as many variants of as many inputs as it holds.</exception>
    /// <remarks>Takes the <see cref="Variants"/>-laid-out batch a <see cref="BatchEnsemble"/> opened — or the layers between it and this one
    /// left as they laid it out — and hands on a batch laid out the same way, its last axis <see cref="Outputs"/> wide instead.</remarks>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 3 || input.Shape[1] != Variants || input.Shape[2] != Inputs)
        {
            throw new ArgumentException(
                $"Variant heads for {Variants} variants of {Inputs} inputs take a batch laid out by them, and was handed a {input.Shape} one.",
                nameof(input));
        }

        var backend = pass.Backend;
        var batch = input.Shape[0];
        var laid = backend.Reshape(input, new Shape(batch * Variants, Inputs));
        var joined = default(Tensor);

        for (var variant = 0; variant < Variants; variant++)
        {
            var extract = ExtractVariant(batch, Variants, variant);
            var extracted = backend.MatMul(extract, laid);
            var weight = backend.MatMul(SelectRows(Inputs, Variants, variant), Weight.Value);
            var bias = backend.MatMul(SelectRows(1, Variants, variant), Bias.Value);
            var answered = backend.AddRow(backend.MatMul(extracted, weight), backend.Reshape(bias, new Shape(Outputs)));
            var scattered = backend.MatMul(backend.Transpose(extract), answered);

            joined = joined is null ? scattered : backend.Add(joined, scattered);
        }

        return backend.Reshape(joined!, new Shape(batch, Variants, Outputs));
    }

    // A matrix of noughts and ones that, by a product with the batch laid out variant by variant, picks out the one
    // variant's rows alone, in the order the batch they came from had.
    private static Tensor ExtractVariant(int batch, int variants, int variant) =>
        Selection.Of(batch, batch * variants, (row, column) => column == (row * variants) + variant);

    // A matrix of noughts and ones that, by a product with a matrix of as many rows as there are variants (or groups of
    // them), picks out the rows belonging to the one variant alone.
    private static Tensor SelectRows(int width, int variants, int variant) =>
        Selection.Of(width, width * variants, (row, column) => column == (variant * width) + row);
}

/// <summary>
/// How a loss is taken, and a prediction made, of the raw output of a network that ends in <see cref="VariantHeads"/>: laid
/// out variant by variant within the place of each row of the batch that made them, as <see cref="BatchEnsemble"/> laid them
/// out to begin with.
/// </summary>
internal static class LaidOverVariants
{
    extension(Loss loss)
    {
        /// <summary>
        /// The loss of a batch whose raw outputs are several answers for each row: the mean of those losses for each row,
        /// not the loss of their mean, as TabM-mini's own training takes it.
        /// </summary>
        /// <param name="outputs">The network's raw outputs, a batch laid out by its variants, rank for rank.</param>
        /// <param name="answers">The batch's answers, a row each.</param>
        /// <param name="backend">The backend the arithmetic runs on.</param>
        /// <returns>One value with no axes: the mean over the batch of the mean, over its variants, of the loss.</returns>
        internal Tensor OfVariants(Tensor outputs, Tensor answers, ITensorBackend backend)
        {
            var variants = outputs.Shape[1];
            var flattened = Flattened(outputs, backend);
            var repeated = backend.MatMul(backend.Transpose(BatchEnsemble.RepeatEachRow(answers.Shape[0], variants)), answers);

            return loss.Of(flattened, repeated, backend);
        }

        /// <summary>
        /// The predictions of a batch whose raw outputs are several answers for each row: each variant's own raw output
        /// through this loss's <see cref="Loss.Activation"/> first, then the mean of them, in the answer's own units — as a
        /// deep ensemble's members are averaged.
        /// </summary>
        /// <param name="outputs">The network's raw outputs, a batch laid out by its variants, rank for rank.</param>
        /// <param name="backend">The backend the arithmetic runs on.</param>
        /// <returns>A row of predictions for each row of the batch, of the outputs' own width.</returns>
        internal Tensor VariantPredictions(Tensor outputs, ITensorBackend backend)
        {
            var variants = outputs.Shape[1];
            var flattened = Flattened(outputs, backend);
            var predicted = loss.Predictions(flattened, backend);
            var batch = flattened.Shape[0] / variants;
            var summed = backend.MatMul(BatchEnsemble.RepeatEachRow(batch, variants), predicted);

            return backend.Scale(summed, backend.Scalar(1.0 / variants));
        }
    }

    // The raw output a network ending in VariantHeads gives is laid out by its variants already, rank for rank; flattened
    // to a batch of rows is what a loss, written for an ordinary batch, takes.
    private static Tensor Flattened(Tensor outputs, ITensorBackend backend) =>
        outputs.Shape.Rank == 3
            ? backend.Reshape(outputs, new Shape(outputs.Shape[0] * outputs.Shape[1], outputs.Shape[2]))
            : outputs;

    extension(Network network)
    {
        /// <summary>How many variants a network's raw output is laid out by, where it ends in <see cref="VariantHeads"/>; nothing otherwise.</summary>
        internal int? Variants => network is LayerStack { Layers: [.., VariantHeads heads] } ? heads.Variants : null;
    }
}
