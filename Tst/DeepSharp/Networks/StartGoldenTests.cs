// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The numbers a layer starts at when nothing is said of how it starts, pinned as 0.9.0 drew them: PyTorch's start, weights
/// evenly within one over the root of the inputs and the bias within the same, out of the draws the layer is handed. A release
/// that lets a layer say how it starts, or that it has no bias, must leave a layer that says none of it starting at exactly
/// these numbers, to the bit — otherwise every network anyone trained from a seed is another network the day they upgrade.
/// </summary>
public class StartGoldenTests
{
    [Fact]
    public void ADenseLayerThatSaysNothingOfHowItStarts_StartsAsItDid()
    {
        var dense = new Dense(14, 16, new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(Golden.DenseWeights, Held(dense.Weight.Value, 6));
        Assert.Equal(Golden.DenseBias, Held(dense.Bias!.Value, 6));
        Assert.Equal(Golden.DenseWeightsTotal, Total(dense.Weight.Value));
        Assert.Equal(Golden.DenseBiasTotal, Total(dense.Bias.Value));
    }

    [Fact]
    public void AConvolutionThatSaysNothingOfHowItStarts_StartsAsItDid()
    {
        var conv = new Conv2D(1, 2, new Window(3, 3), new RandomStream(42).Draw("initialise:0", 0, 0));

        Assert.Equal(Golden.ConvKernel, Held(conv.Weight.Value, 6));
        Assert.Equal(Golden.ConvBias, Held(conv.Bias!.Value, 2));
        Assert.Equal(Golden.ConvKernelTotal, Total(conv.Weight.Value));
        Assert.Equal(Golden.ConvBiasTotal, Total(conv.Bias.Value));
    }

    [Fact]
    public void ANetworkDescribedInKerasWordsAndLowered_StartsEveryLayerAsItDid()
    {
        var lowered = new Sequential().Dense(8).Relu().Dense(1).Lower(new Shape(4), new RandomStream(7));
        var slots = lowered.Slots().ToDictionary(named => named.Path, named => named.Slot.Value);

        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], slots.Keys.ToArray());
        Assert.Equal(Golden.LoweredFirstWeights, Held(slots["0.weight"], 6));
        Assert.Equal(Golden.LoweredFirstBias, Held(slots["0.bias"], 6));
        Assert.Equal(Golden.LoweredSecondWeights, Held(slots["2.weight"], 6));
        Assert.Equal(Golden.LoweredSecondBias, Held(slots["2.bias"], 1));
    }

    private static string Held(Tensor tensor, int count) =>
        string.Join(' ', tensor.Values[..count].ToArray().Select(value => value.ToString("R", CultureInfo.InvariantCulture)));

    private static string Total(Tensor tensor) =>
        tensor.Values.ToArray().Sum(value => (double)value).ToString("R", CultureInfo.InvariantCulture);

    // Captured at 56dfb72, before any edit of this release: the first numbers of each slot, and the total of all of them.
    private static class Golden
    {
        public const string DenseWeights = "-0.12765878 -0.10788446 -0.09855747 -0.17743157 0.117237195 -0.21938328";
        public const string DenseBias = "0.014421146 0.025959555 -0.068329 -0.22409895 -0.2010724 0.20639539";
        public const string DenseWeightsTotal = "0.8924913227092475";
        public const string DenseBiasTotal = "-0.8798309527337551";
        public const string ConvKernel = "-0.15921848 -0.13455556 -0.12292276 -0.22129604 0.14622048 -0.27361903";
        public const string ConvBias = "-0.072245486 0.12440931";
        public const string ConvKernelTotal = "-0.9445622079074383";
        public const string ConvBiasTotal = "0.05216382443904877";
        public const string LoweredFirstWeights = "-0.041240573 -0.4635439 -0.0065649203 0.10280301 -0.45245698 0.38425565";
        public const string LoweredFirstBias = "0.22774267 -0.47882658 -0.17252837 -0.16657864 0.3401946 0.13576752";
        public const string LoweredSecondWeights = "-0.24339956 0.2938011 0.3516626 0.034315825 -0.2470853 -0.24831533";
        public const string LoweredSecondBias = "0.14474821";
    }
}
