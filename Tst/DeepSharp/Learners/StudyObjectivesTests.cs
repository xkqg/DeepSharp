// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A study may judge a trial by more than one number, where no one trade-off alone chooses between candidates: an
/// <see cref="Study.Objectives"/> gives a vector instead of the one number a <see cref="Study.Score"/> or a
/// <see cref="Study.Judge"/> gives, and the trials that are not bettered on every one of them, by any other, are its
/// <see cref="EnsembleExtensions.ParetoFront"/>. It is refused beside anything else that would judge or stop a trial by a
/// number the objectives do not give.
/// </summary>
public class StudyObjectivesTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Study Studied(int trials = 4, Func<IReadOnlyList<TrainedNetwork>, IReadOnlyList<double>>? objectives = null) =>
        new(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Objectives = objectives ?? (networks => [networks[0].ValidationLoss(), networks[0].Network.Parameters().Sum(parameter => (double)parameter.Value.Shape.Count)]),
        };

    private static Trial Finished(int number, IReadOnlyList<double> mean) => new(number, new Candidate([]), null!, []) { ObjectiveScores = [mean] };

    [Fact]
    public void AnObjectivesStudy_FillsEveryTrialsObjectiveScores_OneVectorPerDeal_AndLeavesScoresEmpty()
    {
        var result = Studied(trials: 3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.All(result.Trials, trial =>
        {
            Assert.Empty(trial.Scores);
            Assert.Single(trial.ObjectiveScores);
            Assert.Equal(2, trial.ObjectiveScores[0].Count);
            Assert.True(trial.Finished);
        });
    }

    [Fact]
    public void TheBestOfAnObjectivesStudy_IsTheLexicographicMinimumOfTheMeanObjectiveVector_TheEarlierTrialOnATie()
    {
        var heard = new List<IReadOnlyList<double>>();
        var result = Studied(trials: 4, objectives: networks =>
        {
            var given = new double[][] { [0.5, 10.0], [0.2, 99.0], [0.2, 1.0], [0.9, 0.0] }[heard.Count];

            heard.Add(given);

            return given;
        }).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        // Trial 1's [0.2, 99.0] and trial 2's [0.2, 1.0] tie on the first value; the lexicographic rule looks at the second, and
        // trial 2 wins it outright — the earlier-trial tie-break is for a vector equal in every value, not merely its first.
        Assert.Equal(2, result.Best.Number);
    }

    [Fact]
    public void AnObjectivesTrial_IsNotFinished_WhenAValueIsNotANumber()
    {
        var trial = Finished(0, [double.NaN, 1.0]);

        Assert.False(trial.Finished);
    }

    [Fact]
    public void ParetoFront_HoldsTheTrialsNoOtherBettersOnEveryObjective_NeverOneAnotherDominates()
    {
        Trial[] trials =
        [
            Finished(0, [1.0, 5.0]), // dominated by trial 2: worse or equal on both, strictly worse on the first.
            Finished(1, [2.0, 1.0]), // on the front: lower on the second than every other trial.
            Finished(2, [0.5, 4.0]), // on the front: lower on the first than every other trial.
            Finished(3, [2.0, 6.0]), // dominated by trial 1 and trial 2 alike.
        ];
        var result = new StudyResult(trials, trials[1], []);

        var front = result.ParetoFront();

        Assert.Equal([1, 2], front.Select(trial => trial.Number));
    }

    [Fact]
    public void AnObjectivesStudy_IsRefusedBesideAScoreOfItsOwn()
    {
        var scored = new Study(Space, (candidate, network) => network.Dense(candidate.Whole("units")).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
            Score = _ => 0,
            Objectives = networks => [networks[0].ValidationLoss(), 0],
        };

        var refused = Assert.Throws<InvalidOperationException>(() => scored.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains(nameof(Study.Objectives), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnObjectivesStudy_IsRefusedBesideAJudgeOfItsOwn()
    {
        var judged = new Study(Space, candidate => [
            new NetworkDeclaration(Vocabularies.Torch).Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 1).Step(),
        ])
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
            Judge = networks => networks.Sum(network => network.ValidationLoss()),
            Objectives = networks => [networks[0].ValidationLoss(), 0],
        };

        var refused = Assert.Throws<InvalidOperationException>(() => judged.Run((candidate, deal) => [StudyTests.Passengers(deal)], TestContext.Current.CancellationToken));

        Assert.Contains(nameof(Study.Objectives), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnObjectivesStudy_IsRefusedBesideAPruner()
    {
        var pruned = new Study(Space, (candidate, network) => network.Dense(candidate.Whole("units")).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
            Objectives = networks => [networks[0].ValidationLoss(), 0],
            Pruner = _ => false,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => pruned.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains(nameof(Study.Pruner), refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnObjectivesStudy_IsRefusedBesideADealStopper()
    {
        var stopped = new Study(Space, (candidate, network) => network.Dense(candidate.Whole("units")).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
            Objectives = networks => [networks[0].ValidationLoss(), 0],
            DealStopper = _ => false,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => stopped.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains(nameof(Study.DealStopper), refused.Message, StringComparison.Ordinal);
    }
}
