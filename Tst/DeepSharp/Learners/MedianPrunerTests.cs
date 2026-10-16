// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Optuna's median rule: a trial is stopped at an epoch when the best validation loss it has reached by then is worse than the
/// median of what the trials that finished reached at that same epoch of the same deal — once enough trials have finished and
/// the trial is past its warm-up, both of which the person says, since how long a slow starter needs is a fact of their run.
/// </summary>
public class MedianPrunerTests
{
    private static readonly MedianPruner Pruner = new(startup: 2, warmUp: 1);

    [Fact]
    public void ATrialWorseThanTheMedianOfTheFinishedAtTheSameEpoch_IsPruned_AndOneAtOrBelowItIsNot()
    {
        Trial[] earlier = [Finished(0, [0.9, 0.5]), Finished(1, [0.9, 0.7]), Finished(2, [0.9, 0.6])];

        Assert.True(Pruner.Prunes(new Progress(0, [0.9, 0.61], earlier)));
        Assert.False(Pruner.Prunes(new Progress(0, [0.9, 0.6], earlier)));
        Assert.False(Pruner.Prunes(new Progress(0, [0.55, 0.9], earlier)));   // its best so far, not its last
    }

    [Fact]
    public void TheMedianOfAnEvenCount_IsTheMeanOfTheMiddleTwo()
    {
        Trial[] earlier = [Finished(0, [0.9, 0.4]), Finished(1, [0.9, 0.6])];

        Assert.True(Pruner.Prunes(new Progress(0, [0.9, 0.51], earlier)));
        Assert.False(Pruner.Prunes(new Progress(0, [0.9, 0.5], earlier)));
    }

    [Fact]
    public void BeforeEnoughTrialsHaveFinished_OrWithinTheWarmUp_NothingIsPruned()
    {
        Trial[] one = [Finished(0, [0.1, 0.1])];
        Trial[] two = [Finished(0, [0.1, 0.1]), Finished(1, [0.1, 0.1])];

        Assert.False(Pruner.Prunes(new Progress(0, [5, 5], one)));
        Assert.False(Pruner.Prunes(new Progress(0, [5], two)));
        Assert.True(Pruner.Prunes(new Progress(0, [5, 5], two)));
    }

    [Fact]
    public void OnlyTrialsThatFinished_AndHeardThatEpochOfThatDeal_Count()
    {
        Trial[] earlier =
        [
            Finished(0, [0.9, 0.9]),
            Finished(1, [0.9, 0.9]),
            Finished(2, [0.9]) with { Pruned = true, Scores = [] },
        ];
        var otherDeal = Finished(3, [0.1, 0.1]) with { Heard = [[0.1, 0.1], [0.1, 0.1]] };

        Assert.False(Pruner.Prunes(new Progress(0, [0.9, 0.8], earlier)));
        Assert.False(Pruner.Prunes(new Progress(1, [0.9, 0.8], earlier)));
        Assert.True(Pruner.Prunes(new Progress(1, [0.9, 0.8], [.. earlier, otherDeal])));
    }

    [Fact]
    public void AStartupOrAWarmUpBelowNothing_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MedianPruner(startup: -1, warmUp: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MedianPruner(startup: 0, warmUp: -1));
    }

    // A trial that finished, having heard these losses on its one deal.
    private static Trial Finished(int number, double[] heard) =>
        new(number, new Candidate([new Setting("rate", 0.01, null)]), Declared(), [heard[^1]]) { Heard = [heard] };

    private static LearnNetworkStep Declared() => new(
        [new DeepSharp.Pipelines.PartDeclaration("dense", [new DeepSharp.Pipelines.PartSetting("units", DeepSharp.Pipelines.PartValue.Of(1))])],
        NetworkWords.Optimizers[1].Declared(),
        NetworkWords.Losses[0].Declared());
}
