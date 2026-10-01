// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// Dropout leaves a value out at random while a network trains, and scales what it keeps so the sum a layer sees stays
/// what it would have been — PyTorch's inverted dropout. Which values it leaves out is drawn from the run's stream for
/// the layer, the epoch and the step, so the same run drops the same values; once the network has trained, it drops
/// nothing.
/// </summary>
public abstract class DropoutContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    private static Tensor Ones() => Tensor.From(new Shape(2, 4), [.. Enumerable.Repeat(1f, 8)]);

    [Fact]
    public void InATrainingPass_EveryValueIsLeftOutOrKeptAndScaledByOneOverTheShareKept()
    {
        var dropped = new Dropout(0.25).Forward(Ones(), Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0));

        Assert.All(dropped.Values.ToArray(), value => Assert.True(value == 0f || value == 1.3333334f, $"{value}"));
        Assert.Contains(0f, dropped.Values.ToArray());
        Assert.Contains(1.3333334f, dropped.Values.ToArray());
    }

    [Fact]
    public void TheSameSeedEpochAndStep_LeaveOutTheSameValues_AndTheNextStepOthers()
    {
        var dropout = new Dropout(0.5);
        var rows = Tensor.From(new Shape(4, 8), [.. Enumerable.Repeat(1f, 32)]);

        var first = dropout.Forward(rows, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 5)).Values.ToArray();
        var again = dropout.Forward(rows, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 5)).Values.ToArray();
        var next = dropout.Forward(rows, Pass.Training(_backend, new RandomStream(42), epoch: 2, step: 6)).Values.ToArray();

        Assert.Equal(first, again);
        Assert.NotEqual(first, next);
    }

    [Fact]
    public void TwoDropoutsInOneNetwork_LeaveOutDifferentValues()
    {
        var stack = new LayerStack(new Dropout(0.5), new Dropout(0.5));
        var rows = Tensor.From(new Shape(4, 8), [.. Enumerable.Repeat(1f, 32)]);
        var pass = Pass.Training(_backend, new RandomStream(42), epoch: 0, step: 0);

        var first = stack.Layers[0].Forward(rows, pass).Values.ToArray();
        var second = stack.Layers[1].Forward(rows, pass).Values.ToArray();

        Assert.NotEqual(first, second);
    }

    [Fact]
    public void TheGradient_PassesWhereAValueWasKept_ScaledAsItWas()
    {
        var rows = Ones();
        var recording = new RecordingBackend(_backend);

        var dropped = new Dropout(0.25).Forward(rows, Pass.Training(recording, new RandomStream(42), epoch: 0, step: 0));
        var gradient = recording.GradientsOf(recording.Scale(recording.Mean(dropped), recording.Fill(new Shape(), 8f)), [rows])[rows];

        Assert.Equal(dropped.Values.ToArray(), gradient.Values.ToArray());
    }

    [Fact]
    public void InAnEvaluationPass_NothingIsLeftOut()
    {
        var rows = Ones();

        Assert.Same(rows, new Dropout(0.25).Forward(rows, Pass.Evaluation(_backend)));
    }

    [Fact]
    public void ARateOfNothing_LeavesNothingOut_EvenInTraining()
    {
        var rows = Ones();

        Assert.Same(rows, new Dropout(0).Forward(rows, Pass.Training(_backend, new RandomStream(42), 0, 0)));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1)]
    [InlineData(double.NaN)]
    public void ARateThatIsNotAShareBelowOne_IsRefused(double rate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dropout(rate));
    }

    [Fact]
    public void ADropout_SaysItsRate()
    {
        Assert.Equal(0.25, new Dropout(0.25).Rate);
    }
}
