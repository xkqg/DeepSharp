// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A study can stop a trial whose run is going worse than the trials before it were at the same epoch: a pruner hears every
/// epoch of every deal and may end the trial there. The validation rows stop a run and never move its numbers, so a pruner that
/// never prunes changes nothing, and a trial it stops is never the one the study keeps.
/// </summary>
public class StudyPruningTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Study Studied(int trials = 4, int deals = 1, Func<Progress, bool>? pruner = null) =>
        new(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 3))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
            Pruner = pruner,
        };

    [Fact]
    public void APrunerThatNeverPrunes_IsTheStudyWithoutOne_NumberForNumber()
    {
        var plain = Studied().Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var heard = Studied(pruner: _ => false).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(plain.Trials.Select(trial => trial.Scores), heard.Trials.Select(trial => trial.Scores));
        Assert.Equal(plain.Best.Number, heard.Best.Number);
        Assert.All(heard.Trials, trial => Assert.False(trial.Pruned));
    }

    [Fact]
    public void ATrialHearsTheValidationLossOfEveryEpochOfEveryDeal()
    {
        var studied = Studied(trials: 2, deals: 2).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.All(studied.Trials, trial =>
        {
            Assert.Equal(2, trial.Heard.Count);
            Assert.All(trial.Heard, deal => Assert.Equal(3, deal.Count));
        });

        // The winner's networks are the ones the best trial trained, so what it heard is their history.
        Assert.Equal(
            studied.Winner.Select(network => network.History!.Epochs.Select(epoch => epoch.ValidationLoss!.Value)),
            studied.Best.Heard);
    }

    [Fact]
    public void ATrialThePrunerStops_IsPruned_NeverFinished_AndNeverTheBest()
    {
        // The first trial runs whole; every trial after it is stopped at its first epoch.
        var studied = Studied(pruner: progress => progress.Earlier.Count > 0).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.False(studied.Trials[0].Pruned);
        Assert.True(studied.Trials[0].Finished);
        Assert.All(studied.Trials.Skip(1), trial =>
        {
            Assert.True(trial.Pruned);
            Assert.False(trial.Finished);
            Assert.Empty(trial.Scores);
            Assert.True(double.IsNaN(trial.Score));
            Assert.Single(Assert.Single(trial.Heard));
        });
        Assert.Equal(0, studied.Best.Number);
    }

    [Fact]
    public void APrunerHears_TheDeal_TheTrialsLossesSoFar_AndTheTrialsBeforeIt()
    {
        var heard = new List<Progress>();

        Studied(trials: 2, deals: 2, pruner: progress =>
        {
            heard.Add(progress);

            return false;
        }).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(12, heard.Count);
        Assert.Equal([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], heard.Select(progress => progress.Deal));
        Assert.Equal([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], heard.Select(progress => progress.SoFar.Count));
        Assert.Equal([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], heard.Select(progress => progress.Earlier.Count));
    }

    [Fact]
    public void APrunerBesideAScoreOfOnesOwn_IsRefused_ForThePrunerHearsTheValidationLoss()
    {
        var scored = new Study(Space, (candidate, network) => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1))
        {
            Pruner = _ => false,
            Score = trained => trained.ValidationLoss(),
        };

        var refused = Assert.Throws<InvalidOperationException>(() => scored.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains(nameof(Study.Pruner), refused.Message, StringComparison.Ordinal);
        Assert.Contains(nameof(Study.Score), refused.Message, StringComparison.Ordinal);
    }
}
