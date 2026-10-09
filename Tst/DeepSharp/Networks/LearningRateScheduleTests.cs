// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// How far each epoch's steps move the weights: the rate an optimizer starts at, and how it falls from one epoch to the
/// next. Every rate is PyTorch's own scheduler's for the same settings, epoch by epoch, out to the fortieth.
/// </summary>
public class LearningRateScheduleTests
{
    [Fact]
    public void AConstantRate_IsTheRateItStartedAt_EveryEpoch()
    {
        var schedule = new ConstantRate();

        Assert.All(Enumerable.Range(0, 41), epoch => Assert.Equal(0.1, schedule.RateAt(epoch, 0.1)));
    }

    [Fact]
    public void AStepDecay_MultipliesTheRateByItsFactor_EverySoManyEpochs()
    {
        var schedule = new StepDecay(every: 3, factor: 0.5);

        AssertRates([0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0125], schedule);
        Assert.Equal(0.0015625, schedule.RateAt(20, 0.1), 1e-12);
        Assert.Equal(1.220703125e-05, schedule.RateAt(40, 0.1), 1e-15);
    }

    [Fact]
    public void AStepDecay_FallsToATenth_UnlessItsFactorIsSaid()
    {
        var schedule = new StepDecay(every: 10);

        Assert.Equal(0.1, schedule.RateAt(9, 0.1), 1e-12);
        Assert.Equal(0.0010000000000000002, schedule.RateAt(20, 0.1), 1e-12);
        Assert.Equal(1.0000000000000004e-05, schedule.RateAt(40, 0.1), 1e-15);
    }

    [Fact]
    public void AnExponentialDecay_MultipliesTheRateByItsFactor_EveryEpoch()
    {
        var schedule = new ExponentialDecay(0.9);

        AssertRates(
            [0.1, 0.09000000000000001, 0.08100000000000002, 0.07290000000000002, 0.06561000000000002, 0.05904900000000002,
             0.05314410000000002, 0.04782969000000002, 0.043046721000000024, 0.03874204890000002],
            schedule);
        Assert.Equal(0.01215766545905694, schedule.RateAt(20, 0.1), 1e-12);
        Assert.Equal(0.0014780882941434613, schedule.RateAt(40, 0.1), 1e-12);
    }

    [Fact]
    public void ACosineDecay_FallsAlongHalfACosineToItsLeastRate_AndRisesAgainAfterIt()
    {
        // Past the epochs it is laid over, the cosine goes on as PyTorch's does: back up to the rate it started at by twice
        // as many, and down again.
        var schedule = new CosineDecay(epochs: 8, minimum: 0.001);

        AssertRates(
            [0.1, 0.0962320368593087, 0.0855017856687341, 0.06944282990207196, 0.0505, 0.03155717009792806, 0.0154982143312659,
             0.004767963140691306, 0.001, 0.004767963140691307],
            schedule);
        Assert.Equal(0.050500000000000066, schedule.RateAt(20, 0.1), 1e-12);
        Assert.Equal(0.001, schedule.RateAt(40, 0.1), 1e-12);
    }

    [Fact]
    public void ACosineDecay_FallsToNothing_UnlessItsLeastRateIsSaid()
    {
        var schedule = new CosineDecay(epochs: 8);

        Assert.Equal(0.05, schedule.RateAt(4, 0.1), 1e-12);
        Assert.Equal(0, schedule.RateAt(8, 0.1), 1e-12);
    }

    [Fact]
    public void ALinearWarmup_RisesFromItsStartToTheRateOverItsEpochs_AsPyTorchsLinearLRDoes_AndHoldsTheRateAfter()
    {
        // PyTorch's LinearLR(start_factor=0.25, total_iters=4), as Fixtures/optimizers-pytorch.py printed it.
        AssertRates(
            [0.025, 0.043750000000000004, 0.06250000000000001, 0.08125000000000002, 0.10000000000000002, 0.10000000000000002,
             0.10000000000000002, 0.10000000000000002],
            new LinearWarmup(epochs: 4, start: 0.25));
        Assert.Equal(0.1, new LinearWarmup(epochs: 4, start: 0.25).RateAt(40, 0.1), 1e-12);
    }

    [Fact]
    public void ALinearWarmup_StartsAtAThirdOfTheRate_UnlessItsStartIsSaid()
    {
        // PyTorch's LinearLR as it is left: a third of the rate, over five epochs.
        AssertRates(
            [0.03333333333333333, 0.04666666666666667, 0.060000000000000005, 0.07333333333333335, 0.08666666666666668, 0.1, 0.1],
            new LinearWarmup(epochs: 5));
        Assert.Equal(1.0 / 3, new LinearWarmup(5).Start);
        Assert.Null(new LinearWarmup(5).Then);
    }

    [Fact]
    public void ALinearWarmup_HandsOverToTheScheduleThatFollowsIt_CountedFromTheEndOfTheRamp_AsPyTorchsSequentialLRDoes()
    {
        // SequentialLR([LinearLR(0.25, 4), CosineAnnealingLR(T_max=6, eta_min=0.001)], milestones=[4]).
        var schedule = new LinearWarmup(epochs: 4, start: 0.25, then: new CosineDecay(epochs: 6, minimum: 0.001));

        AssertRates(
            [0.025, 0.043750000000000004, 0.06250000000000001, 0.08125000000000002, 0.1, 0.09336825748732973, 0.07525000000000001,
             0.0505, 0.025750000000000012, 0.007631742512670284, 0.001, 0.007631742512670284],
            schedule);
        Assert.IsType<CosineDecay>(schedule.Then);
        Assert.Equal(4, schedule.Epochs);
        Assert.Equal(0.25, schedule.Start);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-0.25)]
    [InlineData(1.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AWarmupThatStartsAtNothing_BelowIt_OrAboveTheRate_IsRefused(double start)
    {
        // A rate of nothing is refused everywhere: a warm-up starts above it, at most at the rate itself.
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearWarmup(4, start));
    }

    [Fact]
    public void AWarmupOverNoEpochs_IsRefused_AndOneThatStartsAtTheRate_IsTheRate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LinearWarmup(0));
        Assert.All(Enumerable.Range(0, 6), epoch => Assert.Equal(0.1, new LinearWarmup(3, 1).RateAt(epoch, 0.1), 1e-15));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void AStepDecayEveryNoEpochs_IsRefused(int every)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StepDecay(every));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-0.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AFactorThatIsNotAPositiveNumber_IsRefused(double factor)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StepDecay(3, factor));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ExponentialDecay(factor));
    }

    [Fact]
    public void ACosineOverNoEpochs_OrToALeastRateBelowNothing_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineDecay(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineDecay(8, -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CosineDecay(8, double.NaN));
    }

    [Fact]
    public void AnEpochBeforeTheFirst_OrARateThatIsNotAPositiveNumber_IsRefused()
    {
        LearningRateSchedule[] schedules = [new ConstantRate(), new StepDecay(3), new ExponentialDecay(0.9), new CosineDecay(8), new LinearWarmup(3, then: new CosineDecay(8))];

        Assert.All(schedules, schedule =>
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => schedule.RateAt(-1, 0.1));
            Assert.Throws<ArgumentOutOfRangeException>(() => schedule.RateAt(0, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => schedule.RateAt(0, double.NaN));
        });
    }

    private static void AssertRates(double[] expected, LearningRateSchedule schedule)
    {
        for (var epoch = 0; epoch < expected.Length; epoch++)
        {
            Assert.Equal(expected[epoch], schedule.RateAt(epoch, 0.1), 1e-12);
        }
    }
}
