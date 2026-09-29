// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What a layer's numbers start at before it has learned anything: drawn from the run's stream, so the same seed starts
/// the same network, and within the bounds PyTorch draws them from — a linear layer's weights and bias both within one
/// over the root of how many inputs it reads.
/// </summary>
public class InitialiserTests
{
    private static Draws DrawsFor(string purpose) => new RandomStream(42).Draw(purpose, 0, 0);

    [Fact]
    public void PyTorchsLinearWeights_LieWithinOneOverTheRootOfTheInputs_AndReachNearIt()
    {
        var weights = new KaimingUniform().Draw(new Shape(14, 16), new Fans(14, 16), DrawsFor("0.weight"));

        Assert.Equal(new Shape(14, 16), weights.Shape);
        Assert.All(weights.Values.ToArray(), value => Assert.InRange(value, -0.267261f, 0.267261f));
        Assert.True(weights.Values.ToArray().Max(MathF.Abs) > 0.25f);
    }

    [Fact]
    public void KaimingUniform_WithTheSlopeOfARectifier_IsHesBound()
    {
        // The root of six over the inputs, for a slope of nothing: He's bound for layers a rectifier follows.
        var weights = new KaimingUniform(negativeSlope: 0).Draw(new Shape(14, 16), new Fans(14, 16), DrawsFor("0.weight"));

        Assert.All(weights.Values.ToArray(), value => Assert.InRange(value, -0.654654f, 0.654654f));
        Assert.True(weights.Values.ToArray().Max(MathF.Abs) > 0.6f);
    }

    [Fact]
    public void PyTorchsLinearBias_LiesWithinOneOverTheRootOfTheInputs()
    {
        var bias = new FanInUniform().Draw(new Shape(16), new Fans(14, 16), DrawsFor("0.bias"));

        Assert.All(bias.Values.ToArray(), value => Assert.InRange(value, -0.267261f, 0.267261f));
    }

    [Fact]
    public void GlorotUniform_LiesWithinTheRootOfSixOverTheInputsAndOutputs()
    {
        var weights = new GlorotUniform().Draw(new Shape(14, 16), new Fans(14, 16), DrawsFor("0.weight"));

        Assert.All(weights.Values.ToArray(), value => Assert.InRange(value, -0.447214f, 0.447214f));
        Assert.True(weights.Values.ToArray().Max(MathF.Abs) > 0.4f);
    }

    [Fact]
    public void Zeros_AreNothing()
    {
        var weights = new Zeros().Draw(new Shape(3, 2), new Fans(3, 2), DrawsFor("0.weight"));

        Assert.All(weights.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void TheSameSeedAndPurpose_DrawTheSameNumbers_AndAnotherPurposeOthers()
    {
        var first = new KaimingUniform().Draw(new Shape(4, 3), new Fans(4, 3), DrawsFor("0.weight"));
        var again = new KaimingUniform().Draw(new Shape(4, 3), new Fans(4, 3), DrawsFor("0.weight"));
        var other = new KaimingUniform().Draw(new Shape(4, 3), new Fans(4, 3), DrawsFor("2.weight"));

        Assert.Equal(first.Values.ToArray(), again.Values.ToArray());
        Assert.NotEqual(first.Values.ToArray(), other.Values.ToArray());
    }

    [Fact]
    public void FansOfNothing_OrASlopeThatIsNotANumber_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KaimingUniform().Draw(new Shape(1), new Fans(0, 1), DrawsFor("x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GlorotUniform().Draw(new Shape(1), new Fans(1, 0), DrawsFor("x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new KaimingUniform(double.NaN));
    }

    [Fact]
    public void AnInitialiser_NeedsItsDraws()
    {
        Assert.Throws<ArgumentNullException>(() => new Zeros().Draw(new Shape(1), new Fans(1, 1), null!));
    }
}
