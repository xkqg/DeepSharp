// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A spatial dropout leaves whole channels out while a network trains, rather than single values: the feature maps of a
/// series, an image or a volume, where neighbouring values are so alike that leaving one out changes nothing the next does
/// not give back. A channel is left out for an example, in every place at once, and what is kept is scaled by one over the
/// share kept, as dropout's is. Which channels is drawn from the run's stream for the layer, the epoch and the step.
/// </summary>
public abstract class SpatialDropoutContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    private static Tensor Ones(params int[] axes) => Tensor.From(new Shape(axes), [.. Enumerable.Repeat(1f, new Shape(axes).Count)]);

    [Fact]
    public void InATrainingPass_EveryChannelOfAnExampleIsLeftOutInEveryPlaceOrKeptInEveryPlace_AndScaledByOneOverTheShareKept()
    {
        var series = new SpatialDropout1D(0.25).Forward(Ones(4, 6, 8), Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0));
        var image = new SpatialDropout2D(0.25).Forward(Ones(4, 3, 3, 8), Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0));
        var volume = new SpatialDropout3D(0.25).Forward(Ones(4, 2, 3, 2, 8), Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0));

        Tensor[] outputs = [series, image, volume];
        int[] placesOf = [6, 9, 12];

        for (var which = 0; which < outputs.Length; which++)
        {
            var values = outputs[which].Values.ToArray();
            var places = placesOf[which];

            Assert.Contains(0f, values);
            Assert.Contains(1.3333334f, values);

            for (var example = 0; example < 4; example++)
            {
                for (var channel = 0; channel < 8; channel++)
                {
                    var across = Enumerable.Range(0, places).Select(place => values[(((example * places) + place) * 8) + channel]).Distinct().ToArray();

                    Assert.Single(across);
                    Assert.True(across[0] == 0f || across[0] == 1.3333334f, $"{across[0]}");
                }
            }
        }
    }

    [Fact]
    public void TheSameSeedEpochAndStep_LeaveOutTheSameChannels_AndTheNextStepOthers()
    {
        var dropout = new SpatialDropout2D(0.5);
        var images = Ones(4, 2, 2, 8);

        var first = dropout.Forward(images, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 5)).Values.ToArray();
        var again = dropout.Forward(images, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 5)).Values.ToArray();
        var next = dropout.Forward(images, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 6)).Values.ToArray();

        Assert.Equal(first, again);
        Assert.NotEqual(first, next);
    }

    [Fact]
    public void TwoSpatialDropoutsInOneNetwork_LeaveOutDifferentChannels()
    {
        var stack = new LayerStack(new SpatialDropout1D(0.5), new SpatialDropout1D(0.5));
        var series = Ones(4, 3, 16);
        var pass = Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0);

        Assert.NotEqual(stack.Layers[0].Forward(series, pass).Values.ToArray(), stack.Layers[1].Forward(series, pass).Values.ToArray());
    }

    [Fact]
    public void TheGradient_PassesWhereAChannelWasKept_ScaledAsItWas()
    {
        var images = Ones(2, 2, 2, 4);
        var recording = new RecordingBackend(_backend);

        var dropped = new SpatialDropout2D(0.5).Forward(images, Pass.Training(recording, new RandomStream(42), epoch: 0, step: 0));
        var gradient = recording.GradientsOf(recording.Scale(recording.Mean(dropped), recording.Fill(new Shape(), dropped.Shape.Count)), [images])[images];

        Assert.Equal(dropped.Values.ToArray(), gradient.Values.ToArray());
    }

    [Fact]
    public void InAnEvaluationPass_OrAtARateOfNothing_NothingIsLeftOut()
    {
        var series = Ones(2, 5, 3);

        Assert.Same(series, new SpatialDropout1D(0.25).Forward(series, Pass.Evaluation(_backend)));
        Assert.Same(series, new SpatialDropout1D(0).Forward(series, Pass.Training(_backend, new RandomStream(42), 0, 0)));
    }

    [Fact]
    public void Input_OfAnotherRankThanTheSpatialDropoutWalks_IsRefused()
    {
        var pass = Pass.Training(_backend, new RandomStream(42), 0, 0);

        Assert.Throws<ArgumentException>(() => new SpatialDropout1D(0.5).Forward(Ones(2, 3, 3, 3), pass));
        Assert.Throws<ArgumentException>(() => new SpatialDropout2D(0.5).Forward(Ones(2, 3, 3), pass));
        Assert.Throws<ArgumentException>(() => new SpatialDropout3D(0.5).Forward(Ones(2, 3, 3, 3), pass));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1)]
    [InlineData(double.NaN)]
    public void ARateThatIsNotAShareBelowOne_IsRefused(double rate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpatialDropout1D(rate));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpatialDropout2D(rate));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpatialDropout3D(rate));
    }

    [Fact]
    public void ASpatialDropout_SaysItsRate_AndLearnsNothing()
    {
        Assert.Equal(0.25, new SpatialDropout2D(0.25).Rate);
        Assert.Empty(new SpatialDropout3D(0.25).Parameters());
    }
}
