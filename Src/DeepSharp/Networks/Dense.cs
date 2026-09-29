// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A linear layer: every output is a weighted sum of every input, plus a bias of its own.
/// </summary>
/// <remarks>
/// The weights are laid out as many rows as there are inputs by as many columns as there are outputs, so a batch of rows
/// times the weights is the batch's outputs; PyTorch keeps the same numbers turned round. The slots are named as PyTorch
/// names them, <c>weight</c> and <c>bias</c>.
/// </remarks>
public sealed class Dense : Layer, ISaved<Dense>
{
    /// <summary>A linear layer that starts as PyTorch's does: weights and bias evenly within one over the root of the inputs.</summary>
    /// <param name="inputs">How many values each row it reads holds.</param>
    /// <param name="outputs">How many values it makes of each row.</param>
    /// <param name="draws">The draws its start is taken from: the weights first, then the bias.</param>
    /// <param name="weights">How the weights start, when not as PyTorch starts them.</param>
    /// <exception cref="ArgumentOutOfRangeException">It reads or makes fewer than one value.</exception>
    public Dense(int inputs, int outputs, Draws draws, Initialiser? weights = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(inputs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(outputs, 1);
        ArgumentNullException.ThrowIfNull(draws);

        var fans = new Fans(inputs, outputs);

        Weight = AddParameter("weight", (weights ?? new KaimingUniform()).Draw(new Shape(inputs, outputs), fans, draws));
        Bias = AddParameter("bias", new FanInUniform().Draw(new Shape(outputs), fans, draws));
    }

    /// <summary>A linear layer that starts at the given numbers.</summary>
    /// <param name="weights">As many rows as there are inputs, by as many columns as there are outputs.</param>
    /// <param name="bias">One value for each output.</param>
    /// <exception cref="ArgumentException">The weights are not a matrix, or the bias is not a row as long as they are wide.</exception>
    public Dense(Tensor weights, Tensor bias)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(bias);

        if (weights.Shape.Rank != 2 || bias.Shape != new Shape(weights.Shape[1]))
        {
            throw new ArgumentException(
                $"A linear layer's weights are a matrix and its bias a row as long as the matrix is wide, and these are {weights.Shape} and {bias.Shape}.",
                nameof(bias));
        }

        Weight = AddParameter("weight", weights);
        Bias = AddParameter("bias", bias);
    }

    /// <summary>The weights: inputs by outputs.</summary>
    public Parameter Weight { get; }

    /// <summary>The bias: one value for each output.</summary>
    public Parameter Bias { get; }

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

        return new(rebuilding.Whole(settings, "inputs"), rebuilding.Whole(settings, "outputs"), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("inputs", Inputs);
        writer.WriteNumber("outputs", Outputs);
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.AddRow(pass.Backend.MatMul(input, Weight.Value), Bias.Value);
}
