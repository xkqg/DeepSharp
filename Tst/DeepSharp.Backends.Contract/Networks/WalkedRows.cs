// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The rows the published samples hand over through the pipeline, and the fixed weights they were walked through: every
/// number a network test here expects was worked out by PyTorch from exactly these.
/// </summary>
public static class WalkedRows
{
    /// <summary>Titanic's first four training rows, as <c>Batch(Part.Train, Needs.OneScale)</c> hands them over.</summary>
    public static Tensor Passengers() => Tensor.From(new Shape(4, 14),
    [
        -0.75f, -1.0f, -0.4576526880264282f, -0.9716978669166565f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        -0.75f, -1.0f, -0.055541593581438065f, -0.721728503704071f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        -1.0f, -1.0f, -0.3571248948574066f, -0.969062864780426f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        -0.75f, -1.0f, -0.13093742728233337f, -0.7927113771438599f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    ]);

    /// <summary>Whether each of the four survived.</summary>
    public static Tensor Survived() => Tensor.From(new Shape(4, 1), [0f, 1f, 1f, 1f]);

    /// <summary>The logits PyTorch's walked network reaches for the four, through the hidden and output weights below.</summary>
    public static float[] PassengersLogits() => [0.2f, 0.2f, 0.22550064f, 0.2f];

    /// <summary>
    /// How large the terms are that the four's binary cross-entropy averages: each one's softplus of its logit and, where it
    /// survived, its logit — what a total of them is held to three roundings of, on any engine.
    /// </summary>
    public static double SurvivedLossTermsSize()
    {
        var logits = PassengersLogits();
        var survived = Survived().Values;
        var total = 0d;

        for (var passenger = 0; passenger < logits.Length; passenger++)
        {
            double logit = logits[passenger];
            total += Math.Log(1 + Math.Exp(logit)) + Math.Abs(logit * survived[passenger]);
        }

        return total / logits.Length;
    }

    /// <summary>The hidden layer's weights in the Titanic walk: ((7 × row + 3 × column) mod 11 − 5) / 20.</summary>
    public static Tensor HiddenWeights() =>
        Tensor.From(new Shape(14, 4), [.. Enumerable.Range(0, 14 * 4).Select(at => (((7 * (at / 4)) + (3 * (at % 4))) % 11 - 5) / 20f)]);

    public static Tensor HiddenBias() => Tensor.From(new Shape(4), [0.1f, -0.1f, 0.05f, 0f]);

    public static Tensor OutputWeights() => Tensor.From(new Shape(4, 1), [-0.3f, 0.2f, 0f, -0.2f]);

    public static Tensor OutputBias() => Tensor.From(new Shape(1), [0.2f]);

    /// <summary>The first four training days of the price series, as the pipeline hands them over.</summary>
    public static Tensor Days() => Tensor.From(new Shape(4, 4),
    [
        0.7576184868812561f, -0.3281572461128235f, 0.9749279022216797f, -0.22252093255519867f,
        0.7993437051773071f, -0.5730045437812805f, 0.4338837265968323f, -0.9009688496589661f,
        0.7866852879524231f, -0.6739606261253357f, -0.4338837265968323f, -0.9009688496589661f,
        0.8359118700027466f, -0.5186105370521545f, -0.9749279022216797f, -0.22252093255519867f,
    ]);

    /// <summary>Each day's return five days on.</summary>
    public static Tensor Returns() =>
        Tensor.From(new Shape(4, 1), [0.03395130857825279f, 0.0005437539075501263f, 0.015336714684963226f, -0.008030833676457405f]);

    /// <summary>The tanh layer's weights in the price walk: ((3 × row + 2 × column) mod 7 − 3) / 10.</summary>
    public static Tensor TanhWeights() =>
        Tensor.From(new Shape(4, 3), [.. Enumerable.Range(0, 4 * 3).Select(at => (((3 * (at / 3)) + (2 * (at % 3))) % 7 - 3) / 10f)]);

    public static Tensor TanhBias() => Tensor.From(new Shape(3), [0f, 0.1f, -0.1f]);

    public static Tensor ReturnWeights() => Tensor.From(new Shape(3, 1), [-0.2f, 0f, 0.2f]);

    public static Tensor ReturnBias() => Tensor.From(new Shape(1), [0f]);

    /// <summary>One batch of four rows of three, to normalise.</summary>
    public static Tensor Batch() => Tensor.From(new Shape(4, 3), [1f, 2f, -1f, 0.5f, -2f, 3f, 2f, 0f, 1f, -1.5f, 4f, 0f]);
}
