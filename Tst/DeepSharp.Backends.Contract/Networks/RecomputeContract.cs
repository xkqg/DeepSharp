// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A layer whose workings are thrown away after its forward pass and worked out again when the way back reaches it trains a network
/// to what the same layer kept would, to the bit, on every engine: the draws are counted rather than kept, so a dropout drops the same
/// values, a normalisation moves its running statistics once and not twice, and a layer that reads several steps or a series — attention,
/// an LSTM — is run again as it was run.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class RecomputeContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void ARecomputedSegment_TrainsToTheBit_WhatTheSameSegmentNotRecomputedTrains()
    {
        var plain = Trained(Rows(), new Sequential().Dense(6).Relu().Dense(6).Relu().Dense(1), recompute: null);
        var again = Trained(Rows(), new Sequential().Dense(6).Relu().Dense(6).Relu().Dense(1), recompute: Recompute.Layers("0", "1", "2", "3"));

        Assert.Equal(Bits(plain), Bits(again));
    }

    [Fact]
    public void ARecomputedNormalisation_MovesItsRunningStatisticsOnce_NotTwice()
    {
        var plain = Trained(Rows(), new Sequential().Dense(6).BatchNorm().Relu().Dense(1), recompute: null);
        var again = Trained(Rows(), new Sequential().Dense(6).BatchNorm().Relu().Dense(1), recompute: Recompute.Layers("1"));

        Assert.Equal(Bits(plain), Bits(again));
        Assert.Contains(again.Network.Slots(), named => named.Path.EndsWith("running_mean", StringComparison.Ordinal));
    }

    [Fact]
    public void ARecomputedDropout_DropsTheValuesItDroppedTheFirstTime_BecauseItsDrawsAreCountedNotKept()
    {
        var plain = Trained(Rows(), new Sequential().Dense(8).Dropout(0.5).Relu().Dense(1), recompute: null);
        var again = Trained(Rows(), new Sequential().Dense(8).Dropout(0.5).Relu().Dense(1), recompute: Recompute.Layers("1"));

        Assert.Equal(Bits(plain), Bits(again));
    }

    [Fact]
    public void ARecomputedLayerThatReadsASeries_AnLstm_TrainsToTheBit()
    {
        var series = Series();
        var plain = Trained(series, new Sequential().Lstm(4).Dense(1), recompute: null);
        var again = Trained(series, new Sequential().Lstm(4).Dense(1), recompute: Recompute.Layers("0"));

        Assert.Equal(Bits(plain), Bits(again));
    }

    [Fact]
    public void ARecomputedAttention_TrainsToTheBit()
    {
        var series = Series();
        var plain = Trained(series, new Sequential().MultiHeadAttention(2).Flatten().Dense(1), recompute: null);
        var again = Trained(series, new Sequential().MultiHeadAttention(2).Flatten().Dense(1), recompute: Recompute.Layers("0"));

        Assert.Equal(Bits(plain), Bits(again));
    }

    [Fact]
    public void EveryLayer_AskedToBeRecomputedAtOnce_TrainsToTheBit_WhatTheSameNetworkNotRecomputedTrains()
    {
        var plain = Trained(Rows(), new Sequential().Dense(6).BatchNorm().Relu().Dense(6).Dropout(0.5).Relu().Dense(1), recompute: null);
        var again = Trained(Rows(), new Sequential().Dense(6).BatchNorm().Relu().Dense(6).Dropout(0.5).Relu().Dense(1), recompute: Recompute.Everything);

        Assert.Equal(Bits(plain), Bits(again));
    }

    [Fact]
    public void ALayerAskedToBeRecomputed_ThatTheNetworkDoesNotHave_IsRefused_NamingItAndTheLayersItDoesHave()
    {
        var compiled = new Sequential().Dense(3).Relu().Dense(1).Compile(new Sgd(0.01), new MeanSquaredError());

        var refused = Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(), validation: null, new FitOptions(seed: 3) { Backend = _backend, Recompute = Recompute.Layers("nowhere") }));

        Assert.Contains("nowhere", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'0'", refused.Message, StringComparison.Ordinal);
    }

    private CompiledNetwork Trained(TrainingData rows, Sequential description, Recompute? recompute)
    {
        var compiled = description.Compile(new Adam(0.05), new MeanSquaredError());

        _ = compiled.Fit(rows, validation: null, new FitOptions(seed: 7) { Epochs = 3, BatchSize = 4, Backend = _backend, Recompute = recompute });

        return compiled;
    }

    private static IEnumerable<int[]> Bits(CompiledNetwork compiled) =>
        compiled.Network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());

    private static TrainingData Rows() => new(
        Tensor.From(new Shape(8, 5), [.. Enumerable.Range(0, 40).Select(at => (float)Math.Sin((at * 0.37) + 0.1))]),
        Tensor.From(new Shape(8, 1), [.. Enumerable.Range(0, 8).Select(at => (float)Math.Cos(at * 0.51))]));

    private static TrainingData Series() => new(
        Tensor.From(new Shape(8, 5, 4), [.. Enumerable.Range(0, 160).Select(at => (float)Math.Sin((at * 0.23) + 0.3))]),
        Tensor.From(new Shape(8, 1), [.. Enumerable.Range(0, 8).Select(at => (float)Math.Cos(at * 0.41))]));
}
