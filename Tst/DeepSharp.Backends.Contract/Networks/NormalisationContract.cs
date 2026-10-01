// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The two normalisations, over the last axis — a row's features, or an image's channels when they come last: how each
/// starts, what it keeps, and what it refuses, on any engine.
/// </summary>
/// <param name="engine">The backend the passes run on.</param>
public abstract class NormalisationContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void ABatchNormalisation_StartsAtAScaleOfOneAndAShiftOfNothing_AndSaysWhatItKeeps()
    {
        var norm = new BatchNorm(3);

        Assert.Equal([1f, 1f, 1f], norm.Weight.Value.Values.ToArray());
        Assert.Equal([0f, 0f, 0f], norm.Bias.Value.Values.ToArray());
        Assert.Equal([0f, 0f, 0f], norm.RunningMean.Value.Values.ToArray());
        Assert.Equal([1f, 1f, 1f], norm.RunningVariance.Value.Values.ToArray());
        Assert.Equal(["weight", "bias", "running_mean", "running_var"], norm.Slots().Select(slot => slot.Path));
        Assert.Equal(0.1, norm.Momentum);
        Assert.Equal(1e-5, norm.Epsilon);
    }

    [Fact]
    public void ABatchOfOneRow_CannotBeMeasured_InATrainingPass()
    {
        var norm = new BatchNorm(3);

        Assert.Throws<ArgumentException>(() => norm.Forward(Tensor.Zeros(new Shape(1, 3)), Training()));
    }

    [Fact]
    public void ANormalisationOfFeaturesItDoesNotHave_OrOfNone_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new BatchNorm(3).Forward(Tensor.Zeros(new Shape(4, 2)), Training()));
        Assert.Throws<ArgumentException>(() => new LayerNorm(3).Forward(Tensor.Zeros(new Shape(4, 2)), Training()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerNorm(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(3) { Momentum = 1.5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchNorm(3) { Epsilon = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new LayerNorm(3) { Epsilon = -1 });
    }

    [Fact]
    public void ALayerNormalisation_KeepsAScaleAndAShift_AndNoRunningStatistic()
    {
        var norm = new LayerNorm(3);

        Assert.Equal(["weight", "bias"], norm.Slots().Select(slot => slot.Path));
        Assert.Equal(1e-5, norm.Epsilon);
    }

    private Pass Training() => Pass.Training(_backend, new RandomStream(1), 0, 0);
}
