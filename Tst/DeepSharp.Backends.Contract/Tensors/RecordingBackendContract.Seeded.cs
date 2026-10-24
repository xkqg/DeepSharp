// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

public abstract partial class RecordingBackendContract
{
    [Fact]
    public void ASeedOfTwoToTheSixteenth_ScalesEveryGradientExactly_ToTheBit()
    {
        // A power of two moves nothing but the exponent: what is sent back scaled by it and scaled down by it again is what was sent back.
        var weights = Weights();
        var bias = Bias();
        var plain = WorkedOut(weights, bias, seed: null);
        var seeded = WorkedOut(weights, bias, seed: _backend.Fill(new Shape(), 65536f));

        foreach (var tensor in new[] { weights, bias })
        {
            var one = plain[tensor].Values.ToArray();
            var scaled = seeded[tensor].Values.ToArray();

            Assert.Equal(one.Length, scaled.Length);

            for (var at = 0; at < one.Length; at++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(one[at] * 65536f), BitConverter.SingleToInt32Bits(scaled[at]));
            }
        }
    }

    [Fact]
    public void ASeedAsShapedAsALossWithAxes_IsTheProductOfTheSeedWithTheWayBack()
    {
        // The loss is two answers; the seed says how much each of them is worth, and the weights' gradient is the days turned round
        // times the seed: 0.7993437 + 2 * 0.7866853 and -0.5730045 + 2 * -0.6739606.
        var weights = Weights();
        var pass = new RecordingBackend(_backend);
        var answers = pass.MatMul(Days(), weights);

        var gradients = pass.GradientsOf(answers, [weights], Tensor.From(new Shape(2, 1), [1f, 2f]));

        Assert.Equal(new Shape(2, 1), gradients[weights].Shape);
        Assert.Equal(2.3727143, gradients[weights].Values[0], 5);
        Assert.Equal(-1.9209257, gradients[weights].Values[1], 5);
    }

    [Fact]
    public void ASeedNotShapedAsTheLoss_IsRefused_NamingBothShapes()
    {
        var weights = Weights();
        var pass = new RecordingBackend(_backend);
        var loss = pass.Mean(pass.MatMul(Days(), weights));

        var wrong = Assert.Throws<ArgumentException>(() => pass.GradientsOf(loss, [weights], Tensor.From(new Shape(2), [1f, 1f])));

        Assert.Contains("2", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("seed", wrong.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ALossWithAxes_StillNeedsASeed_ToBeDifferentiated()
    {
        var weights = Weights();
        var pass = new RecordingBackend(_backend);
        var answers = pass.MatMul(Days(), weights);

        Assert.Throws<ArgumentException>(() => pass.GradientsOf(answers, [weights]));
    }

    private Gradients WorkedOut(Tensor weights, Tensor bias, Tensor? seed)
    {
        var pass = new RecordingBackend(_backend);
        var answers = pass.AddRow(pass.MatMul(Days(), weights), bias);
        var misses = pass.Subtract(answers, Returns());
        var loss = pass.Mean(pass.Multiply(misses, misses));

        return seed is null ? pass.GradientsOf(loss, [weights, bias]) : pass.GradientsOf(loss, [weights, bias], seed);
    }
}
