// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A linear layer: every output is a weighted sum of every input, plus a bias of its own — or without one, where it is said to
/// have none.
/// </summary>
/// <remarks>
/// The weights are laid out as many rows as there are inputs by as many columns as there are outputs, so a batch of rows
/// times the weights is the batch's outputs; PyTorch keeps the same numbers turned round. The slots are named as PyTorch
/// names them, <c>weight</c> and <c>bias</c>. A layer without a bias keeps the <c>weight</c> alone, Keras's
/// <c>use_bias=False</c> and PyTorch's <c>bias=False</c>, and writes into its file that it has none.
/// <para>
/// A layer begun with a <see cref="LowRank"/> keeps a pair of matrices beside its kernel, <c>lora_a</c> and <c>lora_b</c>, as
/// Keras's <c>enable_lora</c> does: the kernel is held still and the pair and the bias learn, and the layer answers with the
/// kernel's product plus the pair's, scaled by alpha over the rank. It writes its rank and its alpha into its file, which makes
/// the network's part the seventh version.
/// </para>
/// </remarks>
public sealed class Dense : Layer, ISaved<Dense>
{
    private const string RankKey = "rank";
    private const string AlphaKey = "alpha";

    /// <summary>A linear layer that starts as PyTorch's does: weights and bias evenly within one over the root of the inputs.</summary>
    /// <param name="inputs">How many values each row it reads holds.</param>
    /// <param name="outputs">How many values it makes of each row.</param>
    /// <param name="draws">The draws its start is taken from: the weights first, then the bias.</param>
    /// <param name="weights">How the weights start, when not as PyTorch starts them.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one value.</exception>
    public Dense(int inputs, int outputs, Draws draws, Initialiser? weights = null)
        : this(inputs, outputs, new Beginning(draws ?? throw new ArgumentNullException(nameof(draws))) { Weights = weights })
    {
    }

    /// <summary>A linear layer that starts as it is told to: as PyTorch's does, unless the beginning says otherwise.</summary>
    /// <param name="inputs">How many values each row it reads holds.</param>
    /// <param name="outputs">How many values it makes of each row.</param>
    /// <param name="beginning">The draws its start is taken from, and how its weights and its bias start — or that it has no bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one value.</exception>
    /// <exception cref="ArgumentNullException">The beginning holds no draws.</exception>
    /// <exception cref="ArgumentException">The beginning wants no bias and says how its bias starts.</exception>
    public Dense(int inputs, int outputs, Beginning beginning)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inputs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outputs, 1);

        var begun = beginning.Required();
        var fans = new Fans(inputs, outputs);

        Weight = AddParameter("weight", begun.WeightsOf(new Shape(inputs, outputs), fans));
        Bias = begun.BiasOf(new Shape(outputs), fans) is { } bias ? AddParameter("bias", bias) : null;

        if (begun.LowRank is { } pair)
        {
            // Keras's enable_lora: the kernel is held, A starts evenly as He's uniform initialisation draws it and B at nothing, so
            // the answer at the first step is the kernel's own.
            LowRank = pair;
            Weight.Rules = Weight.Rules with { Held = true };
            A = AddParameter("lora_a", new KaimingUniform(0).Draw(new Shape(inputs, pair.Rank), fans, begun.Draws));
            B = AddParameter("lora_b", Tensor.Zeros(new Shape(pair.Rank, outputs)));
        }
    }

    /// <summary>A linear layer that starts at the given numbers.</summary>
    /// <param name="weights">As many rows as there are inputs, by as many columns as there are outputs.</param>
    /// <param name="bias">One value for each output; nothing for a layer without a bias.</param>
    /// <exception cref="ArgumentException">The weights are not a matrix, or the bias is not a row as long as they are wide.</exception>
    public Dense(Tensor weights, Tensor? bias)
    {
        ArgumentNullException.ThrowIfNull(weights);

        if (weights.Shape.Rank != 2 || (bias is not null && bias.Shape != new Shape(weights.Shape[1])))
        {
            throw new ArgumentException(
                $"A linear layer's weights are a matrix and its bias a row as long as the matrix is wide, and these are {weights.Shape} and {bias?.Shape.ToString() ?? "no bias"}.",
                nameof(bias));
        }

        Weight = AddParameter("weight", weights);
        Bias = bias is null ? null : AddParameter("bias", bias);
    }

    /// <summary>The weights: inputs by outputs.</summary>
    public Parameter Weight { get; }

    /// <summary>The bias: one value for each output; nothing for a layer built without one.</summary>
    public Parameter? Bias { get; }

    /// <summary>The low-rank pair beside the held kernel: its rank and its alpha; nothing for a layer built without one.</summary>
    public LowRank? LowRank { get; }

    /// <summary>The pair's first matrix: as many rows as the layer has inputs, the rank columns; nothing for a layer without a pair.</summary>
    public Parameter? A { get; }

    /// <summary>The pair's second matrix: the rank rows, as many columns as the layer has outputs; nothing for a layer without a pair.</summary>
    public Parameter? B { get; }

    /// <summary>How many values each row it reads holds.</summary>
    public int Inputs => Weight.Value.Shape[0];

    /// <summary>How many values it makes of each row.</summary>
    public int Outputs => Weight.Value.Shape[1];

    /// <inheritdoc />
    public static string Name => "dense";

    /// <inheritdoc />
    public static Dense Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(
            rebuilding.Whole(settings, "inputs"),
            rebuilding.Whole(settings, "outputs"),
            new Beginning(rebuilding.Draws)
            {
                WithoutBias = !rebuilding.Yes(settings, "bias"),
                LowRank = PairIn(settings, rebuilding),
            });
    }

    // The pair a layer was written with: its rank, and its alpha where it is not the rank; none for a layer written without a rank.
    private static LowRank? PairIn(JsonElement settings, Rebuilding rebuilding)
    {
        if (settings.Member(RankKey) is null)
        {
            return settings.Member(AlphaKey) is null ? null : throw rebuilding.Refuse(AlphaKey, "written only beside a rank, the rank of the pair it scales");
        }

        var rank = rebuilding.Whole(settings, RankKey);

        if (rank < 1)
        {
            throw rebuilding.Refuse(RankKey, "a whole number from one");
        }

        var alpha = rebuilding.Number(settings, AlphaKey, rank);

        return double.IsFinite(alpha) && alpha > 0 ? new LowRank(rank, alpha) : throw rebuilding.Refuse(AlphaKey, "a number above nothing");
    }

    /// <inheritdoc />
    public int FileVersion => LowRank is not null ? NetworkDocument.SeventhVersion : Bias is null ? NetworkDocument.FifthVersion : NetworkDocument.SecondVersion;

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inputs", Inputs);
        writer.WriteNumber("outputs", Outputs);

        if (Bias is null)
        {
            writer.WriteBoolean("bias", false);
        }

        if (LowRank is { } pair)
        {
            writer.WriteNumber(RankKey, pair.Rank);
            writer.WriteNumber(AlphaKey, pair.Alpha);
        }
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        var product = pass.Backend.MatMul(input, Weight.Value);

        if (LowRank is { } pair)
        {
            // x W + (x A B) · alpha / rank: what Keras's Dense answers once enable_lora has been called.
            var beside = pass.Backend.MatMul(pass.Backend.MatMul(input, A!.Value), B!.Value);

            product = pass.Backend.Add(product, pair.Scale == 1 ? beside : pass.Backend.Scale(beside, pass.Backend.Fill(new Shape(), (float)pair.Scale)));
        }

        return Bias is null ? product : pass.Backend.AddRow(product, Bias.Value);
    }
}
