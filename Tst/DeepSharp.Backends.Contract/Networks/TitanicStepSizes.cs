// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// How large the terms are that each total of one Titanic training step adds up: for every output, the loss and every
/// gradient, the same expression worked out in double precision on the sizes of what went into it, each value's own
/// rounding carried on by the size of its effect. A float total lies within a few units of rounding of that size from any
/// other float total of the same terms, whatever order they were added in; that is the scale two engines are held to.
/// </summary>
/// <param name="Outputs">The size behind each row's output.</param>
/// <param name="Loss">The size behind the loss.</param>
/// <param name="Gradients">The size behind each value of each gradient, in the network's order of slots.</param>
internal sealed record TitanicStepSizes(double[] Outputs, double Loss, double[][] Gradients)
{
    /// <summary>
    /// The sizes of the step a network of fourteen inputs, a dense layer of sixteen, a rectifier and one output takes over a
    /// batch with a binary cross-entropy on its logits, from the parameters it started at.
    /// </summary>
    /// <param name="batch">The rows and their answers.</param>
    /// <param name="start">The four parameters it started at: the hidden weights, their bias, the output weights, theirs.</param>
    /// <returns>The sizes.</returns>
    public static TitanicStepSizes Of(TitanicBatch batch, float[][] start)
    {
        var x = batch.Features;
        var y = batch.Answers;
        var n = batch.Rows;
        var w = batch.Width;
        var hiddenWeights = start[0];
        var hiddenBias = start[1];
        var outputWeights = start[2];
        var outputBias = start[3][0];
        var hidden = hiddenBias.Length;

        // The hidden layer: each value, the size behind it, and whether the rectifier passes it.
        var z = new double[n * hidden];
        var zSize = new double[n * hidden];

        for (var row = 0; row < n; row++)
        {
            for (var unit = 0; unit < hidden; unit++)
            {
                double value = hiddenBias[unit], size = Math.Abs(hiddenBias[unit]);

                for (var input = 0; input < w; input++)
                {
                    var term = (double)x[(row * w) + input] * hiddenWeights[(input * hidden) + unit];
                    value += term;
                    size += Math.Abs(term);
                }

                z[(row * hidden) + unit] = value;
                zSize[(row * hidden) + unit] = size;
            }
        }

        // The outputs, the loss, and the loss's gradient at each output: a sigmoid's slope is at most a quarter.
        var outputSize = new double[n];
        var toOutput = new double[n];
        var toOutputSize = new double[n];
        var loss = 0d;

        for (var row = 0; row < n; row++)
        {
            double value = outputBias, size = Math.Abs(outputBias);

            for (var unit = 0; unit < hidden; unit++)
            {
                if (z[(row * hidden) + unit] > 0)
                {
                    value += z[(row * hidden) + unit] * outputWeights[unit];
                    size += zSize[(row * hidden) + unit] * Math.Abs(outputWeights[unit]);
                }
            }

            var probability = 1 / (1 + Math.Exp(-value));
            outputSize[row] = size;
            loss += Math.Abs(Softplus(value)) + Math.Abs(value * y[row]) + (Math.Abs(probability - y[row]) * size);
            toOutput[row] = (probability - y[row]) / n;
            toOutputSize[row] = (probability + Math.Abs(y[row]) + (0.25 * size)) / n;
        }

        // Back through the output layer and the rectifier to the hidden layer.
        var outputWeightsSize = new double[hidden];
        var toHiddenSize = new double[n * hidden];

        for (var row = 0; row < n; row++)
        {
            for (var unit = 0; unit < hidden; unit++)
            {
                var at = (row * hidden) + unit;

                if (z[at] > 0)
                {
                    outputWeightsSize[unit] += (zSize[at] * Math.Abs(toOutput[row])) + (Math.Abs(z[at]) * toOutputSize[row]);
                    toHiddenSize[at] = toOutputSize[row] * Math.Abs(outputWeights[unit]);
                }
            }
        }

        var hiddenWeightsSize = new double[w * hidden];
        var hiddenBiasSize = new double[hidden];

        for (var row = 0; row < n; row++)
        {
            for (var unit = 0; unit < hidden; unit++)
            {
                hiddenBiasSize[unit] += toHiddenSize[(row * hidden) + unit];

                for (var input = 0; input < w; input++)
                {
                    hiddenWeightsSize[(input * hidden) + unit] += Math.Abs((double)x[(row * w) + input]) * toHiddenSize[(row * hidden) + unit];
                }
            }
        }

        return new TitanicStepSizes(outputSize, loss / n, [hiddenWeightsSize, hiddenBiasSize, outputWeightsSize, [toOutputSize.Sum()]]);
    }

    // The softplus of a value, worked out so that neither end loses it.
    private static double Softplus(double value) => Math.Max(value, 0) + Math.Log(1 + Math.Exp(-Math.Abs(value)));
}
